# Designing Quantum Phases with Foundation Neural-Network Quantum States

This repository contains the code and data for the paper [*Designing Strongly Correlated Quantum Phases of Matter with Foundation Neural-Network Quantum States*](https://arxiv.org/abs/2609.40209), which introduces a framework for inverse model design from first principles based on Foundation Neural-Network Quantum States.
The method identifies a quantum many-body Hamiltonian whose ground state maximizes a desired property, such as magnetism, superconductivity, charge order and others. 
In this repository, the approach is applied to the frustrated Heisenberg model on the square lattice, with eight next-nearest-neighbour couplings on a 2×2 unit cell.

The code is built on [`jax`](https://github.com/jax-ml/jax), [`flax`](https://github.com/google/flax) and [`NetKet`](https://github.com/netket/netket).

## Content of the repository

- **models**: the Foundation Vision Transformer, which takes the couplings and the spin configuration as input, with its sign structure initialized by a classical phase.
- **utils**: the components of the simulation:
  - the Hamiltonian with 8 couplings and the Hilbert space of couplings and spins;
  - the variational optimization of the foundation model over many systems at once (SR driver, Monte Carlo sampling, local estimators);
  - the natural-gradient optimization of the Edwards-Anderson loss in coupling space;
  - logging, checkpointing and early stopping.
- **scripts**: the script to reproduce the results of the paper (`main.py`), its configuration file (`vmc_config.yaml`) and the definition of the configuration options (`config.py`).
- **data**: the paper's figure data stored in `.npy` and `.npz` files.

## Usage

The dependencies are listed in `requirements.txt`. The simulation is configured in `scripts/vmc_config.yaml`, and every option can also be overridden from the command line:

```bash
python scripts/main.py --lattice.L 6 --hamiltonian.n_systems 1024
```

The script is written for a multi-GPU cluster. To run it on a single GPU or on a laptop, see the comment above `jax.distributed.initialize()` in `scripts/main.py`.

Results are saved in the Weights & Biases run folder when `wandb.log` is `True`, and in `output_dir` (default: `results/`) otherwise, as specified in `scripts/vmc_config.yaml`.

## Loading data

Example code for loading the data using NumPy (to run from the repository root):

```python
import numpy as np

with np.load('data/fig4/panel_c/q2.npz') as curve:
    x, y, error = curve['x'], curve['y'], curve['y_error']
boundary = np.load('data/fig3/panel_a_boundary_neel.npy')
x, y = boundary[:, 0], boundary[:, 1]
```
