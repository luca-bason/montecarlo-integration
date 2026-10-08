# Monte Carlo Integration

Small Python library for multidimensional Monte Carlo integration.

The project is being developed as an exercise to explore:

- multidimensional Monte Carlo integration
- error estimation
- uniform sampling
- stratified sampling
- automated testing
- benchmarking

## Current features

- Uniform random sampling in N dimensions
- Stratified sampling in N dimensions
- Monte Carlo integration over rectangular domains
- Error estimation using repeated independent runs

## Example

```python
def f(x):
    return x[0] + x[1]

estimate, error = montecarlo_integration(
    f,
    [(0, 2), (0, 3)],
    100,
    sampler=uniform,
    repeats=50
)

print(estimate, error)