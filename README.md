# Commutator-free quasi-Magnus operators

This repository contains the code to compute the cost of implementing quantum simulation using Commutator-free quasi-Magnus (CFQM) operators, originally introduced in [Fourth- and sixth-order commutator-free Magnus integrators for linear and non-linear dynamical systems](https://www.sciencedirect.com/science/article/abs/pii/S0168927405002163) (see pdf [here](https://personales.upv.es/~serblaza/2006APNUM.pdf)) such as:

$$
\exp(\Omega^{[4]}(h)) = \exp(-i \alpha_1 h H(t_1) - i \alpha_2 h H(t_2))\exp(-i\alpha_2 h H(t_1) -i \alpha_1 h H(t_2)),
$$

with $\alpha_1 = \frac{3-2\sqrt{3}}{12}$, $\alpha_2 = \frac{3+2\sqrt{3}}{12}$, $H(t)$ the time-dependent Hamiltonian, and $t_1$ and $t_2$ defined as the roots of the Legendre polynomial of order 2 of the time segment of length $h$.

## Navigating the repository

This repository has the following structure

- The cost estimation cost is found in folder `CFQMagnus/`,
- while `coefficients/` contains the coefficients of the CFQMs (see `CFQMagnus/compute_coefficients.py` for their preparation),
- and `results/` saves the error-per-step incurred simulating the CFQM when using time step `h`.

The main files can are `magnus_errors.py` , containing the primitives that we need to compute their cost; and `main.py`, where we compute the error-per-step of each CFQM, which are saved to `results/`.

The cost minimizaton and associated paper figure generation can be implemented using the figure generation files in `CFQMagnus/figure_generation/`.

We additionally included a file `CFQMagnus/pennylane_simulation.py` providing an example of how to implement time simulation of a CFQM in [Pennylane](https://pennylane.ai/), using a Heisenberg model for the sake of an example.

## Installation

Navigate to the root directory of this repository and run

```shell
pip install -e .
```

You may also create a new conda environment before this, though the packages we use are standard ones and can be found in the `requirements.txt`.

## How to cite

Find the article in [Quantum simulation of time-dependent Hamiltonians via commutator-free quasi-Magnus operators](https://arxiv.org/abs/2403.13889)

```
@misc{casares2024quantum,
      title={Quantum simulation of time-dependent Hamiltonians via commutator-free quasi-Magnus operators},
      author={Pablo Antonio Moreno Casares and Modjtaba Shokrian Zini and Juan Miguel Arrazola},
      year={2024},
      eprint={2403.13889},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
