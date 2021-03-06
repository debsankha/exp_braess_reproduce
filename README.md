

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6363078.svg)](https://doi.org/10.5281/zenodo.6363078)


Code to reproduce the article on Braes Paradox in Electrical Power Grid Models
------------------------------------------------------------------------------

# How to use?
## Set up the python environment
1. Use a python 3.8 environment. Clone the repository
```bash
git clone --recurse-submodules https://github.com/debsankha/exp_braess_reproduce.git
cd exp_braess_reproduce
```
2. Install the package `braess_detection`:
```bash
pip install -e braess_detection`
```

## Generate data
```bash
python generate_statistics.py
```
## Plot
Run the notebook `reproduce_figure.ipynb`
