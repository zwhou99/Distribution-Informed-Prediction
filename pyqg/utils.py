import os
import numpy as np
import xarray as xr
import torch
import argparse
import fsspec
import shutil
from filelock import FileLock



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def u_q_neg(x_1, x_2, mu, var, pi, h):
    # Compute log probabilities and weights for GMM components
    log_probs = torch.log(pi) - 0.5 * torch.log(var) - (x_1 - mu) ** 2 / 2 / var

    # Compute the minimum value for the constant so that the log weights are valid
    with torch.no_grad():
        cons = torch.min(torch.min(x_1.min(), x_2.min()), mu.min()) - 1

    log_weights1 = torch.log(x_1 - cons) - torch.log(var)
    log_weights2 = torch.log(mu - cons) - torch.log(var)
    log_sum_weights1 = torch.logsumexp(log_probs + log_weights1, dim=1, keepdim=True)
    log_sum_weights2 = torch.logsumexp(log_probs + log_weights2, dim=1, keepdim=True)
    log_sum_probs = torch.logsumexp(log_probs, dim=1, keepdim=True)
    sq1 = torch.exp(log_sum_weights1 - log_sum_probs) - torch.exp(log_sum_weights2 - log_sum_probs)

    # Compute log probabilities and weights for the second input
    log_probs = (torch.log(pi) - 0.5 * torch.log(var)).unsqueeze(1) - (x_2 - mu.unsqueeze(1)) ** 2 / 2 / var.unsqueeze(
        1)
    log_weights1 = torch.log(x_2 - cons) - torch.log(var).unsqueeze(1)
    log_weights2 = (torch.log(mu - cons) - torch.log(var)).unsqueeze(1)
    log_sum_weights1 = torch.logsumexp(log_probs + log_weights1, dim=0, keepdim=True)
    log_sum_weights2 = torch.logsumexp(log_probs + log_weights2, dim=0, keepdim=True)
    log_sum_probs = torch.logsumexp(log_probs, dim=0, keepdim=True)
    sq2 = torch.exp(log_sum_weights1 - log_sum_probs) - torch.exp(log_sum_weights2 - log_sum_probs)

    # sq1 = (x_1 - mu)/ var # for 1 gaussian
    # sq2 = (x_2 - mu)/ var # for 1 gaussian

    # Compute the final result based on bandwidth h
    if h == float('inf'):
        return sq1 * sq2
    else:
        return torch.exp(-(x_1 - x_2) ** 2 / 2 / (h ** 2)) * (
                sq1 * sq2 - (sq1 - sq2) * (x_1 - x_2) / h ** 2 + 1 / h ** 2 - (x_1 - x_2) ** 2 / h ** 4)



def get_dataset(path, base_url="https://g-402b74.00888.8540.data.globus.org"):
    mapper = fsspec.get_mapper(f"{base_url}/{path}.zarr")
    return xr.open_zarr(mapper, consolidated=True)



def append_run_to_zarr(ds_new, path, run_dim="run"):
    """
    Append a new run to a Zarr dataset along the 'run' dimension, using a temporary Zarr file
    to align chunks with the existing data. File locking ensures safe concurrent access.

    Parameters
    ----------
    ds_new : xr.Dataset or xr.DataArray
        New data to append (one run).
    path : str
        Path to the .zarr directory.
    run_dim : str, default="run"
        Name of the run dimension.
    """
    if isinstance(ds_new, xr.DataArray):
        ds_new = ds_new.to_dataset(name=ds_new.name or "data")

    lock_path = path.rstrip("/") + ".lock"
    with FileLock(lock_path):
        if os.path.exists(path):
            ds_old = xr.open_zarr(path, consolidated=False)

            if run_dim not in ds_old.dims:
                ds_old = ds_old.expand_dims({run_dim: [0]})

            next_run_id = ds_old.sizes[run_dim]
            ds_new = ds_new.expand_dims({run_dim: [next_run_id]})

            # Generate temporary Zarr path
            tmp_path = path[:-5] + '_tmp' + path[-5:]
            ds_new.to_zarr(tmp_path, mode="w")
            ds_new = xr.open_zarr(tmp_path, consolidated=False)

            ds_combined = xr.concat([ds_old, ds_new], dim=run_dim)
            ds_combined.to_zarr(path, mode="w")

            # Clean up the temporary Zarr folder
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
        else:
            ds_new = ds_new.expand_dims({run_dim: [0]})
            ds_combined = ds_new
            ds_combined.to_zarr(path, mode="w")

