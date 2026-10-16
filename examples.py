import numpy as np
from mc_integration import (uniform, stratified_uniform,estimate_once,montecarlo_integration)


# =============================================================================
# EXAMPLES
# =============================================================================

def f(x):
    return x[0]**2

def g(x):
    return x[0]+x[1]


est_u, err_u = montecarlo_integration(g, [(0, 2), (0, 3)], 100,sampler=uniform,repeats=50)

est_s, err_s = montecarlo_integration(g, [(0, 2), (0, 3)], 100,sampler=stratified_uniform,repeats=50)

print("Uniform:", est_u, err_u)
print("Stratified:", est_s, err_s)