# Commutator-free quasi-Magnus operators

This repository contains the code to compute the commutator-free quasi-Magnus cost. The code is found in folder `CFQMagnus/`. The main files can are `magnus_errors.py`, containing the primitives that we need to compute their cost; and `main.py`, where we compute the error per step of each CFQM.

The figure generation can be implemented using the figure generation files in `CFQMagnus/figure_generation`, including `time_scaling.py`, `error_scaling.py`, `time_n_scaling.py`, `error_weight.py` and `error_per_step.py` .

## Installation

Navigate to the root directory of this repository and issue

```shell
pip install -e .
```

You may also create a new conda environment before this, though the packages we use are standard (numpy, matplotlib,...).
