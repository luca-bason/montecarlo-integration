# Monte Carlo Integration
# Numerical integration using uniform and stratified random sampling.

import numpy as np
from .sampling import uniform, stratified_uniform


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

