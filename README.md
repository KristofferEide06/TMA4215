# TMA4215 — Project 1

Numerical investigation of polynomial and radial basis function interpolation for TMA4215 Numerical Mathematics.

The main report is available in [`notebooks/project_1.ipynb`](notebooks/project_1.ipynb). The notebook contains the numerical experiments, figures, mathematical analysis, and discussion, while the reusable implementation is organized as a Python package under `src/`.

## Topics

- Global Lagrange interpolation
- Equidistant and Chebyshev nodes
- Runge's phenomenon
- Maximum- and \(L^2\)-norm error estimates
- Analytical interpolation-error bounds
- Piecewise-polynomial interpolation
- Runtime comparisons
- Gaussian radial basis function interpolation
- Shape-parameter selection and matrix conditioning
- Gradient-descent optimization of RBF nodes and shape parameters

## Project structure
- `notebooks/`
  - `project_1.ipynb` — Main report and numerical experiments.

- `src/tma4215_project1/`
  - `functions.py` — Functions used in the interpolation experiments.
  - `interpolation/`
    - `methods.py` — Lagrange, piecewise-polynomial, and RBF interpolation.
    - `nodes.py` — Node generation and gradient-descent optimization.
    - `norms.py` — Numerical error-norm calculations.
  - `analysis/`
    - `computations/`
      - `lagrange_computations.py` — Lagrange error and convergence calculations.
      - `piecewise_computations.py` — Piecewise-interpolation calculations.
      - `rbf_computations.py` — RBF error and optimization calculations.
      - `runtime_computations.py` — Runtime measurements and comparisons.
    - `plots/`
      - `lagrange_plots.py` — Lagrange interpolation plots.
      - `piecewise_plots.py` — Piecewise-interpolation plots.
      - `rbf_plots.py` — RBF interpolation and optimization plots.

- `outputs/` — Generated figures and numerical results.
- `docs/` — Supporting documentation.
- `tests/` — Project tests.
- `pyproject.toml` — Package configuration and dependencies.
- `uv.lock` — Locked dependency versions.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

The principal dependencies are:

- NumPy
- Matplotlib
- Autograd
- JupyterLab

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/KristofferEide06/TMA4215.git
cd TMA4215/project_1
```

Install the project and its development dependencies:

```bash
uv sync --extra dev
```

## Running the report

Start JupyterLab from the `project_1` directory:

```bash
uv run jupyter lab notebooks/project_1.ipynb
```

The notebook locates the project source directory automatically and can therefore be run from either the project root or the `notebooks` directory.

To regenerate and save the figures, set the following variable near the beginning of the notebook:

```python
SAVE_FIGURES = True
```

Generated figures are then written to subdirectories under `outputs/`.

## Reproducibility

Dependency versions are recorded in `uv.lock`. To reproduce the project environment, run:

```bash
uv sync --frozen --extra dev
```

Then restart the notebook kernel and run all cells in order.

## AI declaration

A detailed declaration describing how AI tools were used during the project is included near the beginning of the main notebook.
