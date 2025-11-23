import numpy as np
import pyqg
from functools import cached_property

def config_for(m):
    """Return the parameters needed to initialize a new
    pyqg.QGModel, except for nx and ny."""
    config = dict(H1 = m.Hi[0])
    for prop in ['L', 'W', 'dt', 'rek', 'g', 'beta', 'delta',
            'U1', 'U2', 'rd', 'dt', 'tmax', 'tavestart']:
        config[prop] = getattr(m, prop)
    return config

class Coarsener:
    """Common code for defining filtering and coarse-graining operators."""
    def __init__(self, high_res_model, low_res_nx, model_config=None):
        assert low_res_nx < high_res_model.nx
        assert low_res_nx % 2 == 0
        self.m1 = high_res_model
        self.m1._invert()
        if model_config is None:
            self.m2 = pyqg.QGModel(nx=low_res_nx, **config_for(high_res_model))
        else:
            self.m2 = pyqg.QGModel(**model_config)
        self.m2.q = self.coarsen(self.m1.q)
        self.m2._invert() # recompute psi, u, and v
        self.m2._calc_derived_fields()

    @property
    def q_forcing_total(self):
        for m in [self.m1, self.m2]:
            m._invert()  # find streamfunction from pv
            m._do_advection()  # use streamfunction to calculate advection tendency
            m._do_friction()  # apply friction
        return self.coarsen(self.m1.dqhdt) - self.to_real(self.m2.dqhdt)

    def to_real(self, var):
        """Convert variable to real space, if needed."""
        for m in [self.m1, self.m2]:
            if var.shape == m.qh.shape:
                return m.ifft(var)
        return var

    def to_spec(self, var):
        """Convert variable to spectral space, if needed."""
        for m in [self.m1, self.m2]:
            if var.shape == m.q.shape:
                return m.fft(var)
        return var

    def subgrid_forcing(self, var):
        """Compute subgrid forcing of a given `var` (as string)."""
        q1 = getattr(self.m1, var)
        q2 = getattr(self.m2, var)
        adv1 = self.coarsen(self.m1._advect(q1))
        adv2 = self.to_real(self.m2._advect(q2))
        return adv1 - adv2

    def subgrid_fluxes(self, var):
        """Compute subgrid fluxes (wrt. u and v) of a given `var`."""
        q1 = getattr(self.m1, var)
        q2 = getattr(self.m2, var)
        u_flux = self.coarsen(self.m1.ufull * q1) - self.m2.ufull * q2
        v_flux = self.coarsen(self.m1.vfull * q1) - self.m2.vfull * q2
        return u_flux, v_flux

    @property
    def ratio(self):
        """Ratio of high-res to low-res grid length."""
        return self.m1.nx / self.m2.nx

    def coarsen(self, var):
        """Filter and coarse-grain a variable (as array)."""
        raise NotImplementedError()

    @cached_property
    def ds1(self):
        """xarray representation of the high-res model."""
        return self.m1.to_dataset()

    @cached_property
    def ds2(self):
        """xarray representation of the low-res model."""
        return self.m2.to_dataset()

    # CALLED IN DATA GENERATION FILE: data.ipynb, when appending distribution variables
    @cached_property
    def ds(self):
        hires_t = self.m1  # hires at time t
        lowres_t = self.m2  # lores at time t
        hires_t1 = self.m1
        lowres_t1 = self.m2

        for m in [hires_t1, lowres_t1]:
            m._invert()
            m._do_advection()
            m._do_friction()
            m._do_external_forcing()  # apply external forcing

            if m.uv_parameterization is not None:
                m._do_uv_subgrid_parameterization()  # apply velocity subgrid forcing term, if present
            if m.q_parameterization is not None:
                m._do_q_subgrid_parameterization()  # apply potential vorticity subgrid forcing term, if present

            m._calc_diagnostics()
            m._forward_timestep()  # apply tendencies to step the model forward, (filter gets called here)
            m._print_status()

        self.m3 = pyqg.QGModel(nx=self.m2.nx, **config_for(self.m1))
        self.m3.q = self.coarsen(hires_t1.q)  # ground truth at t+1
        self.m3._invert()  # recompute psi, u, and v
        self.m3._calc_derived_fields()

        # --- OUTPUTS (5 total) ---
        # hires_t : hires model at time t              ---> hires_t1 : hires model at time t+1
        # lowres_t: lowres (hires c/f) model at time t ---> lowres_t1: lowres_t model at time t+1
        # m3: lowres (hires c/f) model at time t+1

        return hires_t.to_dataset(), lowres_t.to_dataset(), hires_t1.to_dataset(), lowres_t1.to_dataset(), self.m3.to_dataset()


class SpectralCoarsener(Coarsener):
    """Spectral truncation with a configurable filter."""
    def coarsen(self, var):
        # Truncate high-frequency indices & filter
        vh = self.to_spec(var)
        nk = self.m2.qh.shape[1]//2
        trunc = np.hstack((vh[:, :nk,:nk+1],
                           vh[:,-nk:,:nk+1]))
        filtered = trunc * self.spectral_filter / self.ratio**2
        return self.to_real(filtered)

    @property
    def spectral_filter(self):
        raise NotImplementedError()

class Operator1(SpectralCoarsener):
    """Spectral truncation with a sharp filter."""
    @property
    def spectral_filter(self):
        return self.m2.filtr

class Operator2(SpectralCoarsener):
    """Spectral truncation with a softer Gaussian filter."""
    @property
    def spectral_filter(self):
        return np.exp(-self.m2.wv**2 * (2*self.m2.dx)**2 / 24)

class Operator3(Coarsener):
    """Diffusion-based filter, then real-space coarsening."""
    def coarsen(self, var):
        import gcm_filters
        f = gcm_filters.Filter(dx_min=1,
            filter_scale=self.ratio,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR)
        d = self.m1.to_dataset().isel(time=-1)
        q = d.q*0 + self.to_real(var) # hackily convert to data array
        r = int(self.ratio)
        assert r == self.ratio
        return f.apply(q, dims=['y','x']).coarsen(y=r, x=r).mean().data

if __name__ == '__main__':
    from scipy.stats import pearsonr

    m1 = pyqg.QGModel(nx=256)

    for _ in range(10000):
        m1._step_forward()

    op1 = Operator1(m1, 64)
    op2 = Operator2(m1, 64)
    op3 = Operator3(m1, 64)

    for op in [op1, op2, op3]:
        q_forcing = op.subgrid_forcing('q')

        uq_flux, vq_flux = op.subgrid_fluxes('q')

        q_forcing2 = op.m2.ifft(
            op.m2.ik * op.m2.fft(uq_flux) +
            op.m2.il * op.m2.fft(vq_flux)
        )

        corr = pearsonr(q_forcing.ravel(), q_forcing2.ravel())[0]

        print(op.__class__.__name__, corr)

        assert corr > 0.5
