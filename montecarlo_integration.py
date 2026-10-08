#montecarlo integration

import numpy as np
from itertools import product


rng = np.random.default_rng()

# genera punti uniformemente negli intervalli dati, n per ogni intervallo
def uniform(bounds,samples):

    if samples < 2:
        raise ValueError("the samples must be at least 2")
    
    points = np.empty((samples,len(bounds)))

    for i in range(samples):
        for j,bound in enumerate(bounds):
            points[i,j] = rng.uniform(bound[0],bound[1])

    return points



def stratified_uniform_nd(bounds,samples):

    if samples < 2:
        raise ValueError("the samples must be at least 2")

    dim = len(bounds)
    n_strata = round(samples**(1/dim))

    if n_strata**dim != samples:
        raise ValueError("For N dimensions stratified sampling samples must be a perfect Nth power")

    points = np.empty((samples, dim))

    for k,cell in enumerate(product(range(n_strata), repeat=dim)):

        for j, stratum in enumerate(cell):

            h = (bounds[j][1]-bounds[j][0])/n_strata
            lower = bounds[j][0] + stratum*h
            upper = bounds[j][0] + (stratum+1)*h

            points[k,j] = rng.uniform(lower,upper)

    return points


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



def f(x):
    return x[0]**2

def g(x):
    return x[0]+x[1]


est_u, err_u = montecarlo_integration(
    g, [(0, 2), (0, 3)], 100,
    sampler=uniform,
    repeats=50
)

est_s, err_s = montecarlo_integration(
    g, [(0, 2), (0, 3)], 100,
    sampler=stratified_uniform_nd,
    repeats=50
)

print("Uniform:", est_u, err_u)
print("Stratified:", est_s, err_s)