Code to reproduce the article on Braes Paradox in Electrical Power Grid Models
------------------------------------------------------------------------------

# How to use?
## Set up the python environment
1. Use a python 3.8 environment
```bash
git clone --recurse-submodules https://github.com/debsankha/exp_braess_reproduce.git
cd exp_braess_reproduce
```
2. Install the requirements `pip install -r braess_detection/requirements.txt`
3. Install the package itself `pip install -e braess_detection`

## Generate data
```bash
git clone --recurse-submodules https://github.com/debsankha/exp_braess_reproduce.git
cd exp_braess_reproduce
python generate_statistics.py
```
## Plot
Run the notebook `reproduce_figure.ipynb`
