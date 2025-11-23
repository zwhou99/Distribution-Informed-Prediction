import os
os.environ["OMP_NUM_THREADS"] = "16"  # Limit the number of OpenMP threads for numerical stability/performance

import numpy as np
import pyqg
import torch
import argparse
import xarray as xr

from sklearn.mixture import GaussianMixture
from pyqg_parameterization_benchmarks.neural_networks import FCNNParameterization
from coarsening_ops import Operator1, Operator2, Operator3
from utils import *

# Fit Gaussian Mixture Model to normalized data
def fit_gmm(data, n_components=10):
    mu = data.mean()
    std = data.std()
    data_true_normalized = (data - mu) / std
    gmm = GaussianMixture(n_components, random_state=42)
    gmm.fit(data_true_normalized.reshape(-1, 1))
    return gmm, mu, std

# Fit GMMs to each level of low-res q fields (used for calibration)
def generate_low_total_dist_data(low_res, n_components=1, stable_start=31):
    n_levels = low_res['q'].sizes['lev']
    low_res_total_dist = [{} for _ in range(n_levels)]
    for lev in range(n_levels):
        q = np.array(low_res['q'].isel(time=slice(stable_start, None), lev=lev).values.flatten())
        gmm, mu, std = fit_gmm(q, n_components)
        low_res_total_dist[lev] = {
            'gmm': gmm,
            'mu': mu,
            'std': std,
        }
    return low_res_total_dist

# Perform KSD calibration via iterative gradient descent
def ksd_calibration(q_raw, dist_info, is_standardize=True, lamb=0.01, max_step=201, tolerance=20, bandwidth=3,
                    device='cpu'):
    q_shape = q_raw.shape
    opt_lev_list = []
    n_levels = q_raw.sizes['lev']

    for lev in range(n_levels):
        q = np.array(q_raw.isel(time=0, lev=lev).values.flatten())

        # Extract GMM and normalization info
        gmm = dist_info[lev]['gmm']
        means = torch.tensor(gmm.means_.flatten(), dtype=torch.float64, device=device)
        weights = torch.tensor(gmm.weights_.flatten(), dtype=torch.float64, device=device)
        covariances = torch.tensor(gmm.covariances_.flatten(), dtype=torch.float64, device=device)
        overall_mean = torch.dot(weights, means)
        overall_std = torch.sqrt(torch.dot(weights, means ** 2 + covariances) - overall_mean ** 2)

        mu = dist_info[lev]['mu']
        std = dist_info[lev]['std']
        q_normalized = (q - mu) / std

        # Initialize tensor for optimization
        y_test_hat_tensor = torch.tensor(q, dtype=torch.float64, device=device)
        if is_standardize:
            y_test_hat_tensor = (y_test_hat_tensor - torch.mean(y_test_hat_tensor)) / torch.std(y_test_hat_tensor)

        y_test_hat_tensor = y_test_hat_tensor.detach().clone().requires_grad_(True)
        y_test_hat_tensor_numpy = y_test_hat_tensor.detach().cpu().clone().numpy()

        with torch.no_grad():
            h = bandwidth * torch.std(y_test_hat_tensor).item()

        # Precompute GMM sample quantiles for WDL metric
        q_vec = torch.arange(1, 1000, dtype=torch.float64) / 1000
        quantiles = np.percentile(gmm.sample(10000)[0], q_vec * 100)

        raw_WDL = np.sum((np.quantile(q_normalized, q_vec) - quantiles) ** 2)
        stan_WDL = np.sum((np.quantile(y_test_hat_tensor_numpy, q_vec) - quantiles) ** 2)

        best_results = {'dist_wdl': stan_WDL, 'best_test_y_hat': y_test_hat_tensor_numpy}
        early_stopping_patience = 0
        n_test = len(y_test_hat_tensor)

        # Gradient descent loop
        for step in range(max_step):
            pairwise_values = u_q_neg(y_test_hat_tensor.unsqueeze(1), y_test_hat_tensor.unsqueeze(0), means,
                                      covariances, weights, h)
            torch.diagonal(pairwise_values)[:] = 0
            test_statistic = torch.sum(pairwise_values) / np.sqrt(n_test)
            test_statistic.backward()

            # Normalize and clip gradients
            gradient = y_test_hat_tensor.grad
            gradient = torch.clip(gradient, -5 * torch.median(torch.abs(gradient)),
                                  5 * torch.median(torch.abs(gradient)))
            with torch.no_grad():
                gradient /= torch.sqrt(torch.mean(gradient ** 2)) / torch.std(y_test_hat_tensor)
                y_test_hat_tensor -= lamb * gradient
            y_test_hat_tensor.grad.zero_()

            # Compute dist_WDL and check early stopping
            y_test_hat_numpy = y_test_hat_tensor.detach().cpu().clone().numpy()
            dist_WDL = np.sum((np.quantile(y_test_hat_numpy, q_vec) - quantiles) ** 2)
            if dist_WDL < best_results['dist_wdl']:
                best_results = {'dist_wdl': dist_WDL, 'best_test_y_hat': y_test_hat_numpy}
                early_stopping_patience = 0
            else:
                early_stopping_patience += 1
            if early_stopping_patience >= tolerance:
                break

        # Recover calibrated q for this level
        y_test_opt = best_results['best_test_y_hat']
        cali_WDL = np.sum((np.quantile(y_test_opt, q_vec) - quantiles) ** 2)
        print(f"lev: {lev}, step: {step}, raw_WDL: {raw_WDL:.4f}, stan_WDL: {stan_WDL:.4f}, cali_WDL: {cali_WDL:.4f}", flush=True)

        y_test_opt_unnormalized = y_test_opt * std + mu
        y_test_opt_reshaped = y_test_opt_unnormalized.reshape((1, q_shape[-2], q_shape[-1]))
        opt_lev_list.append(y_test_opt_reshaped)

    opt_lev = np.concatenate(opt_lev_list, axis=0)
    return opt_lev

# Run online simulation with optional KSD calibration applied after each step
def run_online_with_cali(high_res_config, low_res_config, operator_id=1,
                         sampling_freq=1000, stable_start=31, num_snapshots=87,
                         low_res_total_dist=None, lamb=0.01, max_step=201, tolerance=20, bandwidth=3, device='cpu'):

    # Initialize and stabilize high-res model
    high_res_model = pyqg.QGModel(**high_res_config)
    for i in range(stable_start * sampling_freq):
        high_res_model._step_forward()

    # Construct low-res model via operator
    ops = [Operator1, Operator2, Operator3]
    operator = ops[operator_id - 1]
    low_res_model = operator(high_res_model, low_res_config['nx'], low_res_config).m2

    # Initial snapshot
    t_cur = low_res_model.t + high_res_config['tavestart']
    low_cali = low_res_model.to_dataset().copy(deep=True).assign_coords(time=("time", [t_cur]))
    snap_lo = [low_cali]

    # Online time stepping with optional KSD calibration
    for i in range(num_snapshots - 1):
        for j in range(sampling_freq):
            low_res_model._step_forward()

        if low_res_total_dist:
            lo_cali_q = low_res_model.to_dataset().copy(deep=True)['q']
            cali_q = ksd_calibration(lo_cali_q, low_res_total_dist, is_standardize=True, lamb=lamb,
                                     max_step=max_step, tolerance=tolerance, bandwidth=bandwidth, device=device)
            # Replace q and recompute dependent fields
            low_res_model.q = cali_q
            low_res_model._invert()
            low_res_model._calc_derived_fields()

        t_cur = low_res_model.t + high_res_config['tavestart']
        low_cali = low_res_model.to_dataset().copy(deep=True).assign_coords(time=("time", [t_cur]))
        snap_lo.append(low_cali)

    # Combine snapshots into a single dataset
    low_res = xr.concat(snap_lo, dim='time')
    return low_res

# -------- MAIN EXECUTION --------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Calibration online.")
    parser.add_argument('--config', type=str, default='jet')
    parser.add_argument('--nx_hires', type=int, default=256)
    parser.add_argument('--nx_lores', type=int, default=64)
    parser.add_argument('--dt', type=float, default=3600.0)
    parser.add_argument('--sampling_freq', type=int, default=1000)
    parser.add_argument('--stable_start', type=int, default=31)
    parser.add_argument('--num_snapshots', type=int, default=87)
    parser.add_argument('--operator_id', type=int, default=3)
    parser.add_argument('--use_FCNN', type=str2bool, default=False)
    parser.add_argument('--input_id', type=int, default=1)
    parser.add_argument('--target_id', type=int, default=1)
    parser.add_argument('--use_ksd', type=str2bool, default=True)
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--lamb', type=float, default=0.01)
    parser.add_argument('--max_step', type=int, default=201)
    parser.add_argument('--tolerance', type=int, default=5)

    args = parser.parse_args()

    config = args.config
    nx_hires = args.nx_hires
    nx_lores = args.nx_lores
    dt = args.dt
    sampling_freq = args.sampling_freq
    stable_start = args.stable_start
    num_snapshots = args.num_snapshots
    operator_id = args.operator_id
    use_FCNN = args.use_FCNN
    input_id = args.input_id
    target_id = args.target_id
    use_ksd = args.use_ksd
    n_components = args.n_components
    lamb = args.lamb
    max_step = args.max_step
    tolerance = args.tolerance

    # High and low-res model configuration
    high_res_config = dict(
        nx=nx_hires,
        dt=dt,
        tmax=(num_snapshots + stable_start + 1) * dt * sampling_freq,
        tavestart=stable_start * dt * sampling_freq,
    )
    low_res_config = dict(
        nx=nx_lores,
        dt=dt,
        tmax=(num_snapshots + 1) * dt * sampling_freq,
        tavestart=0,
    )

    # Apply physical parameters for jet or eddy configurations
    if config == 'jet':
        high_res_config.update(dict(rek=7e-8, beta=1e-11, delta=0.1))
        low_res_config.update(dict(rek=7e-8, beta=1e-11, delta=0.1))
        if stable_start < 121:
            stable_start = 121
            high_res_config['tavestart'] = stable_start * dt * sampling_freq
            print("stable_start is set to 121 for jet config")
    elif config == 'eddy':
        if stable_start < 31:
            stable_start = 31
            high_res_config['tavestart'] = stable_start * dt * sampling_freq
            print("stable_start is set to 31 for eddy config")
    else:
        raise NotImplementedError(f"Configuration {config} is not implemented yet.")

    suffix = f"{config}_{operator_id}_{nx_hires}_{nx_lores}_{int(dt)}_{sampling_freq}_{stable_start}_{num_snapshots}"

    # FCNN parameterization support
    if use_FCNN:
        FCNN = FCNNParameterization(
            f'pyqg_parameterization_benchmarks/models/fcnn_filter{operator_id}_input{input_id}_target{target_id}')
        if target_id <= 2 or target_id == 4:
            low_res_config['q_parameterization'] = FCNN
        else:
            low_res_config['uv_parameterization'] = FCNN
        suffix += f'_fcnn_{input_id}_{target_id}'

    # KSD calibration distribution preparation
    if use_ksd:
        filename = (f"data/low_res_truth_{config}_{operator_id}_{nx_hires}_{nx_lores}_{int(dt)}_{sampling_freq}"
                    f"_{stable_start}_{num_snapshots}.zarr")
        low_res = xr.open_zarr(filename).isel(run=slice(0, 5))
        low_res_total_dist = generate_low_total_dist_data(low_res, n_components, stable_start)
        suffix += f'_cali_{n_components}_{lamb}_{max_step}_{tolerance}'
    else:
        low_res_total_dist = None

    # Logging
    print('config:', config)
    print(f'nx_hires: {nx_hires}, nx_lores: {nx_lores}')
    print(f'dt: {dt}, sampling_freq: {sampling_freq}')
    print(f'stable_start: {stable_start}, num_snapshots: {num_snapshots}')
    print('operator_id:', operator_id)
    if use_FCNN:
        print(f'use_FCNN, input_id: {input_id}, target_id: {target_id}')
    if use_ksd:
        print(f'calibration, n_components: {n_components}, lamb: {lamb}, max_step: {max_step}, tolerance: {tolerance}')

    # Run the full calibration simulation
    low_res = run_online_with_cali(high_res_config, low_res_config, operator_id, sampling_freq, stable_start,
                                   num_snapshots, low_res_total_dist, lamb, max_step, tolerance, device=device)

    # Save calibrated output
    append_run_to_zarr(low_res, f"data/online_low_res_{suffix}.zarr")
