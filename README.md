# Monte Carlo Integration

A small Python library for multidimensional Monte Carlo integration over rectangular domains.

## Features

- Uniform and stratified sampling in N dimensions
- Error estimation using independent repetitions
- Automated tests and benchmarks

## Installation

From the project directory:

```bash
python -m pip install .
```

For development (includes pytest): `python -m pip install -e ".[dev]"`

## Example

```python
from mc_integration import montecarlo_integration, uniform

def f(x):
    return x[0] + x[1]

estimate, error = montecarlo_integration(
    f, [(0, 2), (0, 3)], samples=100,
    sampler=uniform, repeats=50
)
print(estimate, error)  # Exact integral: 15
```

Use `stratified_uniform` for stratified sampling; in `d` dimensions, `samples` must be a perfect `d`-th power.

## Tests and benchmarks

Run `python -m pytest -v` and `python benchmark.py` from the project directory.
