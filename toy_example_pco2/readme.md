# Toy example & pCO2

Code for the toy example and the air–sea pCO2 application.

## Preview
There are 2 Python files and 3 Jupyter notebooks in this folder.

## Environment
- Python 3.11.5
- Install packages: `pip install -r requirements.txt`

## Data
- Place datasets under `data/`.
- CESM dataset: https://figshare.com/collections/Large_ensemble_pCO2_testbed/4568555
- Add any required preprocessing/filenames to match the loaders in `utils.py`.

## Code structure
- `model.py`: defines FFN.
- `utils.py`: loads CESM dataset.
- `toy_example.ipynb`: reproduces toy example (Tables 1–2, Figure 2).
- `pco2_train.ipynb`: prepares data/models for pCO2 application.
- `pco2_result.ipynb`: reproduces pCO2 results (Tables 3–8, Figures 3, 4, A1).

## Running the code
1) Open and run `toy_example.ipynb` directly.
2) For pCO2: run `pco2_train.ipynb` to generate necessary data/artifacts, then run `pco2_result.ipynb`.
