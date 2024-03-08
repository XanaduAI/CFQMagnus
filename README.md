# Commutator-free quasi-Magnus operators

This repository contains the code to compute the commutator-free quasi-Magnus cost. The code is found in folder `CFQMagnus/`, while `coefficients/` contains the coefficients of the CFQMs, and `results/` contains the error incurred simulating the CFQM for time step `h`.

The main files can are `magnus_errors.py` , containing the primitives that we need to compute their cost; and `main.py`, where we compute the error per step of each CFQM.

The cost minimizaton and associated paper figure generation can be implemented using the figure generation files in `CFQMagnus/figure_generation`, including `time_scaling.py`, `error_scaling.py`, `time_n_scaling.py`, `error_weight.py` and `error_per_step.py` .

## Installation

Navigate to the root directory of this repository and issue

```shell
pip install -e .
```

You may also create a new conda environment before this, though the packages we use are standard (numpy, matplotlib,...).

## Usage

The workflow of this software library is the following:

1. Define the coefficients in `compute_coefficients.py`, which will be saved in folder `coefficients/`.
2. Execute `main.py`, which will use `magnus_errors.py` to compute and save a dictionary into a json with keys being the time step `h` and the value being the error.
   a. More specifically, our software is designed to compute the error of spin models so the dictionary `error_step` will have the format `step_error[n][s][m][h]`=e for non-split `split_step_error[s][m][h]`=e for split-operator CFQMs.
   b.  `n` represents the number of spins (only related to Trotter error, and applicable to non-split operators, otherwise that index is not there), `2s` is the order of the method, `m` is the number of exponentials of the CFQM, and `h` is the time step. If you are only interested in the error excluding the Trotter error, check functions `step_error_wout_trotter` for the non-split CFQMs, and `compute_step_error_split` for split-operator CFQMs.
3. Finally, once those `h->error` dictionaries have been computed, we can minimize the cost (maximize `h`) and plot the result for different systems.

Check the paper for more details on how these are computed.

## How to cite
