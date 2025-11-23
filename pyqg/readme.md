# pyqg

Code for the online hybrid emulator of quasi-geostrophic turbulence.

## Preview
There are 5 Python scripts and 1 Jupyter notebook in this folder.

## Dependencies
- Based on (and modifies) code from https://github.com/m2lines/pyqg_parameterization_benchmarks. Please read their instructions first to understand usage and expectations.

## Environment
- Python 3.10.14
- Install packages: `pip install -r requirements.txt`

## Code structure
- `utils.py`: save/load data helpers.
- `coarsening_ops.py`: defines operators for coarsening.
- `data.py`: generates training/evaluation data.
- `train_neural_network.py`: trains the FCNN.
- `cali_online.py`: runs online simulations.
- `pyqg_result.ipynb`: reproduces figures (1, 5, 6) from “Application to an Online Hybrid Emulator of Quasi-Geostrophic Turbulence”.

## Running the code
1) `python data.py` — generate required data.
2) `python train_neural_network.py` — train the FCNN.
3) `python cali_online.py` — run online simulations.
4) Open `pyqg_result.ipynb` to reproduce figures.
