import numpy as np
import xarray as xr
import fsspec
import argparse
from pyqg_parameterization_benchmarks.neural_networks import FCNNParameterization
from pyqg_parameterization_benchmarks.utils import FeatureExtractor

# Helper function to load Zarr dataset using fsspec from either cloud or local path
def get_dataset(path, base_url="https://g-402b74.00888.8540.data.globus.org"):
    mapper = fsspec.get_mapper(f"{base_url}/{path}.zarr")
    return xr.open_zarr(mapper, consolidated=True)

if __name__ == "__main__":
    # Argument parser for filter type, input configuration, and target type
    parser = argparse.ArgumentParser(description="Train an FCNN on the pyqg data.")
    parser.add_argument('--filter_id', type=int, default=3)   # Which filter to use (coarsening operator)
    parser.add_argument('--input_id', type=int, default=7)    # Binary-coded input variable selection
    parser.add_argument('--target_id', type=int, default=5)   # Target variable group

    args = parser.parse_args()

    filter_id = args.filter_id
    input_id = args.input_id
    target_id = args.target_id

    # Load the dataset
    data = get_dataset(f'eddy/forcing{filter_id}', base_url='pyqg_parameterization_benchmarks/datasets').isel(
        run=list(range(25, 275))).load()

    # Define possible input and target variable groups
    all_inputs = [['q'], ['u', 'v'], ['ddx(u)', 'ddx(v)', 'ddy(u)', 'ddy(v)']]
    all_targets = [['q_subgrid_forcing'], ['q_forcing_total'], ['u_subgrid_forcing', 'v_subgrid_forcing'],
                   ['uq_subgrid_flux', 'vq_subgrid_flux'], ['uu_subgrid_flux', 'uv_subgrid_flux', 'vv_subgrid_flux']]

    # Convert input_id (e.g. 1–7) to binary mask for selecting input groups
    inputs = []
    for i in range(3):
        if (input_id >> i) & 1:
            inputs.extend(all_inputs[i])

    # If gradients (3rd input group) are selected, compute and add them to the dataset
    if (input_id >> 2) & 1:
        extractor = FeatureExtractor(data)
        data['ddx(u)'] = extractor.extract_feature('ddx(u)')
        data['ddx(v)'] = extractor.extract_feature('ddx(v)')
        data['ddy(u)'] = extractor.extract_feature('ddy(u)')
        data['ddy(v)'] = extractor.extract_feature('ddy(v)')

    # Select target variables based on target_id
    targets = all_targets[target_id - 1]

    # If the target is total forcing, compute it as the difference of time derivatives
    if target_id == 2:
        data['q_forcing_total'] = data.dqdt_bar - data.dqbar_dt

    # Print selected configuration for verification
    print(inputs)
    print(targets)
    print(f'pyqg_parameterization_benchmarks/models/fcnn_filter{filter_id}_input{input_id}_target{target_id}')

    # Train the FCNN model; zero_mean=True is used for most targets except for fluxes
    if target_id >= 4:
        FCNNParameterization.train_on(data,
                                      f'pyqg_parameterization_benchmarks/models/fcnn_filter{filter_id}_input{input_id}_target{target_id}',
                                      inputs=inputs, targets=targets, zero_mean=False)
    else:
        FCNNParameterization.train_on(data,
                                      f'pyqg_parameterization_benchmarks/models/fcnn_filter{filter_id}_input{input_id}_target{target_id}',
                                      inputs=inputs, targets=targets, zero_mean=True)
