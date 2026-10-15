# Monte Carlo Integration
# Numerical integration using uniform and stratified random sampling.

import numpy as np
from itertools import product

# =============================================================================
# EXAMPLES
# =============================================================================

if __name__ == "__main__":

    def f(x):
        return x[0]**2

    def g(x):
        return x[0]+x[1]


    est_u, err_u = montecarlo_integration(g, [(0, 2), (0, 3)], 100,sampler=uniform,repeats=50)

    est_s, err_s = montecarlo_integration(g, [(0, 2), (0, 3)], 100,sampler=stratified_uniform,repeats=50)

    print("Uniform:", est_u, err_u)
    print("Stratified:", est_s, err_s)

# =============================================================================
# SAMPLING METHODS
# =============================================================================

# Shared NumPy random number generator.
rng = np.random.default_rng()

# Generate uniformly distributed random points within the integration domain.
# bounds: list of (lower, upper) intervals, one for each dimension.
# samples: total number of points (>= 2).
# Returns: array of shape (samples, dimensions).
def uniform(bounds,samples):

    if samples < 2:
        raise ValueError("the samples must be at least 2")
    
    points = np.empty((samples,len(bounds)))

    for i in range(samples):
        for j,bound in enumerate(bounds):
            points[i,j] = rng.uniform(bound[0],bound[1])

    return points

# Divide the domain into equally sized cells and sample one point per cell.
# samples must be a perfect d-th power, where d is the number of dimensions.
# Returns: array of shape (samples, dimensions).
def stratified_uniform(bounds,samples):

    if samples < 2:
        raise ValueError("the samples must be at least 2")

    dim = len(bounds)
    n_strata = round(samples**(1/dim))

    if n_strata**dim != samples:
        raise ValueError("in d dimensions stratified sampling samples must be a perfect d-th power")

    points = np.empty((samples, dim))

    for k,cell in enumerate(product(range(n_strata), repeat=dim)):

        for j, stratum in enumerate(cell):

            h = (bounds[j][1]-bounds[j][0])/n_strata
            lower = bounds[j][0] + stratum*h
            upper = bounds[j][0] + (stratum+1)*h

            points[k,j] = rng.uniform(lower,upper)

    return points


# =============================================================================
# INTEGRATION ESTIMATORS
# =============================================================================

# Estimate an integral using a single Monte Carlo run:
# I ≈ V * mean(f(x)), where V is the integration domain volume.
#
# function: callable accepting a NumPy array of coordinates.
# bounds: integration intervals for each dimension.
# samples: number of sampled points (>= 2).
# sampler: sampling method (default: uniform).
# Returns: estimated integral.
def estimate_once(function, bounds, samples, sampler=uniform):

    if samples < 2:
        raise ValueError("the samples must be at least 2")


    volume=1.
    for bound in bounds:

        if bound[0] >= bound[1]:
            raise ValueError("the inferior bound must be lower than the superior bound")
    
        volume*=(bound[1]-bound[0])


    points = sampler(bounds,samples)
    f_x = np.empty(samples)

    for i in range(samples):
        f_x[i] = function(points[i])


    mean_value=np.mean(f_x)
    integral=volume*mean_value

    return integral

# Estimate an integral and its standard error through repeated Monte Carlo runs.
#
# function, bounds, samples, sampler: same as estimate_once().
# repeats: number of independent estimates (>= 2, default: 10).
# Returns: (mean_integral, standard_error).
# Standard error = std(estimates, ddof=1) / sqrt(repeats).
def montecarlo_integration(function,bounds,samples,sampler=uniform,repeats=10):

    if repeats < 2:
        raise ValueError("repeats must be at least 2")

    estimates = np.empty(repeats)

    for i in range(repeats):
        estimates[i] = estimate_once(function,bounds,samples,sampler)

    mean_estimate = np.mean(estimates)


    std_value = np.std(estimates, ddof=1)
    error = std_value/np.sqrt(repeats)

    return mean_estimate,error

