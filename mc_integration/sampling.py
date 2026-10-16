import numpy as np
from itertools import product

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
