# Commutator-free quasi-Magnus operators

This repository contains the code to compute the cost of implementing quantum simulation using Commutator-free quasi-Magnus (CFQM) operators, originally introduced in

> Blanes, S., & Moan, P. C. (2006). Fourth-and sixth-order commutator-free Magnus integrators for linear and non-linear dynamical systems.  *Applied Numerical Mathematics*,  *56* (12), 1519-1537.

such as for instance:

$$
\exp(\Omega^{[4]}(h)) = \exp(\alpha_1 h A_1 + \alpha_2 h A_2)\exp(\alpha_2 h A_1 + \alpha_1 h A_2),
$$

with $\alpha_1 = \frac{3-2\sqrt{3}}{12}$, $\alpha_2 = \frac{3+2\sqrt{3}}{12}$, and $A_1$ and $A_2$ defined as $i$ times the Hamiltonian evaluated at the roots of the Legendre polynomial of order 2 of the time segment of length $h$.

## Navigating the repository

This repository has the following structure

- The cost estimation cost is found in folder `CFQMagnus/`,
- while `coefficients/` contains the coefficients of the CFQMs,
- and `results/` saves the error-per-step incurred simulating the CFQM when using time step `h`.

The main files can are `magnus_errors.py` , containing the primitives that we need to compute their cost; and `main.py`, where we compute the error per step of each CFQM.

The cost minimizaton and associated paper figure generation can be implemented using the figure generation files in `CFQMagnus/figure_generation/`.

We additionally included a file `CFQMagnus/pennylane_simulation.py` providing an example of how to implement time simulation of a CFQM in [Pennylane](https://pennylane.ai/), using a Heisenberg model for the case.

## Installation

Navigate to the root directory of this repository and run

```shell
pip install -e .
```

You may also create a new conda environment before this, though the packages we use are standard (numpy, matplotlib,...).

## Usage

The workflow of this software library is the following:

1. Define the coefficients in `compute_coefficients.py`, and execute the script, which will save them as a json in folder `coefficients/`.
2. Execute `main.py`, which will use `magnus_errors.py` to compute and save a dictionary into a json with keys being the time step `h` and the value being the error, found in `results/`.

   a. More specifically, our software is designed to compute the error of spin models so the dictionary `error_step` will have the format `step_error[n][s][m][h]`=e for non-split and `split_step_error[s][m][h]`=e for split-operator CFQMs.

   b.  `n` represents the number of spins (only related to Trotter error, and applicable to non-split operators, otherwise that index is not there), `2s` is the order of the method, `m` is the number of exponentials of the CFQM, and `h` is the time step. If you are only interested in the error excluding the Trotter error, check functions `step_error_wout_trotter` for the non-split CFQMs, and `compute_step_error_split` for split-operator CFQMs.
3. Finally, once those `h->error` dictionaries have been computed, we can minimize the cost (equivalently maximize `h` under error bound constraint) and plot the result for different systems, using the figure generation scripts in `CFQMagnus/figure_generation/`.

Check the paper for more details on how these are computed.

## How to cite
