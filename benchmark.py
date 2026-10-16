import time
from mc_integration import (uniform, stratified_uniform,estimate_once,montecarlo_integration)
import numpy as np

def g(x):
    return x[0] + x[1]

bounds = [(0, 2), (0, 3)]
samples_size = [100,400,900]
rep = 10
exact_integral = 15

benchmark_runs = 10

for samples in samples_size:
    times_uniform = []
    times_strat_uniform = []
    errors_uniform = []
    errors_stratified = []

    for i in range(benchmark_runs):
        unif_start = time.perf_counter()
        est,err = montecarlo_integration(g,bounds,samples,sampler=uniform,repeats=rep)
        unif_end = time.perf_counter()
        unif_diff = unif_end - unif_start
        times_uniform.append(unif_diff)
        errors_uniform.append(abs(est - exact_integral))

    for i in range(benchmark_runs):
        unif_strat_start = time.perf_counter()
        est,err = montecarlo_integration(g,bounds,samples,sampler=stratified_uniform,repeats=rep)
        unif_strat_end = time.perf_counter()
        unif_strat_diff = unif_strat_end - unif_strat_start
        times_strat_uniform.append(unif_strat_diff)
        errors_stratified.append(abs(est - exact_integral))

    print("samples:",samples)
    unif_mean_time = np.mean(times_uniform)
    unif_std_time = np.std(times_uniform,ddof=1)
    unif_error_mean = np.mean(errors_uniform)
    print("uniform mean time:",unif_mean_time)
    print("uniform std time:",unif_std_time)
    print("uniform error mean:",unif_error_mean)

    strat_unif_mean_time = np.mean(times_strat_uniform)
    strat_unif_std_time = np.std(times_strat_uniform,ddof=1)
    strat_unif_error_mean = np.mean(errors_stratified)
    print("stratified uniform mean time:",strat_unif_mean_time)
    print("stratified uniform std time:",strat_unif_std_time)
    print("stratified uniform error mean:",strat_unif_error_mean)
