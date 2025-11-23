import pyqg
from coarsening_ops import Operator1, Operator2, Operator3
from utils import *
import argparse
import xarray as xr

# Function to generate simulation data
def generate_data(model_config, nx_lores=64, sampling_freq=1000, num_snapshots=101):
    # Initialize the high-resolution model with the given configuration
    high_res_model = pyqg.QGModel(**model_config)

    # Lists to store snapshots
    snap_hi = []                       # High-resolution snapshots
    snap_lo_truth = [[] for _ in range(3)]   # Low-resolution truth snapshots using different coarsening operators
    snap_lo_model = [[] for _ in range(3)]   # Low-resolution model snapshots

    # Initialize coarsening operators and corresponding low-res models
    op1 = Operator1(high_res_model, nx_lores)
    op2 = Operator2(high_res_model, nx_lores)
    op3 = Operator3(high_res_model, nx_lores)
    low_res_models = [op1.m2, op2.m2, op3.m2]

    # Store the initial snapshot
    snap_hi.append(high_res_model.to_dataset())
    for j in range(3):
        low_res_model = low_res_models[j].to_dataset()
        low_res_model["time"].data[...] = high_res_model.t  # Sync time from high-res
        snap_lo_truth[j].append(low_res_model)
        snap_lo_model[j].append(low_res_model)

    # Time stepping and snapshot collection
    for i in range(num_snapshots - 1):
        # Advance both high-res and low-res models by `sampling_freq` steps
        for j in range(sampling_freq):
            high_res_model._step_forward()
            for k in range(len(low_res_models)):
                low_res_models[k]._step_forward()

        # Store current high-res snapshot
        snap_hi.append(high_res_model.to_dataset())

        # Store current low-res model output
        for j in range(3):
            low_res_model = low_res_models[j].to_dataset()
            low_res_model["time"].data[...] = high_res_model.t
            snap_lo_model[j].append(low_res_model)

        # Reconstruct low-res truth via coarsening from new high-res state
        op1 = Operator1(high_res_model, nx_lores)
        op2 = Operator2(high_res_model, nx_lores)
        op3 = Operator3(high_res_model, nx_lores)
        low_res_models = [op1.m2, op2.m2, op3.m2]

        for j in range(3):
            low_res_truth = low_res_models[j].to_dataset()
            low_res_truth["time"].data[...] = high_res_model.t
            snap_lo_truth[j].append(low_res_truth)

    # Concatenate snapshots into a full time series
    high_res = xr.concat(snap_hi, dim='time')
    low_res_model = [xr.concat(snap_lo_model[i], dim='time') for i in range(3)]
    low_res_truth = [xr.concat(snap_lo_truth[i], dim='time') for i in range(3)]
    return high_res, low_res_model, low_res_truth

# Main script entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Generation.")
    parser.add_argument('--config', type=str, default='jet')               # Simulation type: 'jet' or 'eddy'
    parser.add_argument('--nx_hires', type=int, default=256)               # High-resolution grid size
    parser.add_argument('--nx_lores', type=int, default=64)                # Low-resolution grid size
    parser.add_argument('--dt', type=float, default=3600.0)                # Time step in seconds
    parser.add_argument('--sampling_freq', type=int, default=1000)         # Steps per snapshot
    parser.add_argument('--stable_start', type=int, default=31)            # Steps to stabilize before recording
    parser.add_argument('--num_snapshots', type=int, default=87)           # Number of snapshots to generate

    args = parser.parse_args()

    config = args.config
    nx_hires = args.nx_hires
    nx_lores = args.nx_lores
    dt = args.dt
    sampling_freq = args.sampling_freq
    stable_start = args.stable_start
    num_snapshots = args.num_snapshots

    # Set up model configuration dictionary
    model_config = dict(
        nx=nx_hires,
        dt=dt,
        tmax=(num_snapshots + stable_start + 1) * dt * sampling_freq,   # Total simulation time
        tavestart=stable_start * dt * sampling_freq                     # Start time for diagnostics
    )

    # Update config-specific parameters
    if config == 'jet':
        model_config.update(dict(
            rek=7e-8,
            beta=1e-11,
            delta=0.1,
        ))
        # Ensure minimum stable start for jet config
        if stable_start < 61:
            stable_start = 61
            model_config.update(dict(
                tavestart=stable_start * dt * sampling_freq,
            ))
            print("stable_start is set to 61 for jet config")
    elif config == 'eddy':
        if stable_start < 31:
            stable_start = 31
            model_config.update(dict(
                tavestart=stable_start * dt * sampling_freq,
            ))
            print("stable_start is set to 31 for eddy config")
    else:
        raise NotImplementedError(f"config {config} is not implemented yet")

    # Generate data using the configured simulation
    high_res, low_res_model, low_res_truth = generate_data(model_config, nx_lores, sampling_freq,
                                                           num_snapshots + stable_start)

    # Save high-resolution data to Zarr
    suffix = f"{config}_{nx_hires}_{nx_lores}_{int(dt)}_{sampling_freq}_{stable_start}_{num_snapshots}"
    filename = f"data/high_res_{suffix}.zarr"
    append_run_to_zarr(high_res, filename)

    # Save low-resolution model and truth data for each operator
    for i in range(3):
        suffix = f"{config}_{i+1}_{nx_hires}_{nx_lores}_{int(dt)}_{sampling_freq}_{stable_start}_{num_snapshots}"
        filename = f"data/low_res_model_{suffix}.zarr"
        append_run_to_zarr(low_res_model[i], filename)
        filename = f"data/low_res_truth_{suffix}.zarr"
        append_run_to_zarr(low_res_truth[i], filename)

    print("Data generation complete.")
