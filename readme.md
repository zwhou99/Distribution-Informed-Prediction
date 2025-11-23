# Distribution-Informed Prediction

Code for calibrating geophysical predictions under constrained probabilistic distributions.

## Project layout
- `pyqg/`: Online hybrid emulator of quasi-geostrophic turbulence; generate data with `data.py`, train with `train_neural_network.py`, run simulations with `cali_online.py`, then inspect `pyqg_result.ipynb`.
- `toy_example_pco2/`: Toy example and air–sea pCO2 application; see notebook flow and data notes in its README.

## Environment
- Python 3.10–3.11
- Install dependencies: `pip install -r requirements.txt`
- GPU is recommended for training-heavy scripts if available.

## Getting started
1) Create/activate a virtual environment.
2) Install requirements (above).
3) Follow the per-folder README instructions to prepare data, run training, and reproduce figures/tables.
