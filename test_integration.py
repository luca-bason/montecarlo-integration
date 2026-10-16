from mc_integration import (uniform, stratified_uniform,estimate_once,montecarlo_integration)
import pytest
import numpy as np


def f(x):
    return 1.0


def test_constant_function():
    estimate = estimate_once(f,[(0, 2), (0, 3)],10)
    assert estimate == 6

def test_invalid_samples():
    with pytest.raises(ValueError):
        estimate_once(f,[(0, 1)],1)

def test_uniform_shape():
    unif = uniform([(0, 1), (0, 2), (0, 3)],100)
    assert unif.shape == (100,3)

def test_uniform_bounds():
    bounds = [(0, 2), (3, 5)]
    samples =100
    unif = uniform(bounds,samples)
    for i in range(samples):
        for j,bound in enumerate(bounds):
            assert bound[0] <= unif[i,j] <= bound[1]

def test_stratified_invalid_samples():
    with pytest.raises(ValueError):
        stratified_uniform([(0, 1), (0, 1)],10)

def test_stratified_one_point_per_cell():
    bounds = [(0, 2), (0, 2)]
    samples = 4
    strat_unif = stratified_uniform(bounds,samples)
    cells = []

    for i in range(samples):
        x = strat_unif[i,0]
        y = strat_unif[i,1]
        cells.append((int(x),int(y)))

    assert set(cells) == {(0, 0), (0, 1), (1, 0), (1, 1)}

def test_constant_function_error():
    bounds = [(0, 2), (0, 3)]
    est,err = montecarlo_integration(f,bounds,100,sampler=uniform,repeats=5)
    assert est == 6
    assert err == 0

def test_standard_error():
    batches = [np.array([[0.0], [0.0]]),np.array([[1.0], [1.0]])]
    def controlled_sampler(bounds, samples):
        return batches.pop(0)

    def g(x):
        return x[0]

    est,err = montecarlo_integration(g,[(0, 1)],2,sampler=controlled_sampler,repeats=2)

    assert est == pytest.approx(0.5)
    assert err == pytest.approx(0.5)

def test_invalid_repeats():
    with pytest.raises(ValueError):
        montecarlo_integration(f,[(0, 1)],10,sampler=uniform,repeats=1)

def test_invalid_bounds():
    with pytest.raises(ValueError):
        estimate_once(f,[(2,2)],100)

def test_stratified_3d_bounds():
    bounds = [(0, 2), (1, 5), (-3, 3)]
    samples = 8
    strat_unif = stratified_uniform(bounds,samples)
    assert strat_unif.shape == (8,3)
    for i in range(samples):
        for j,bound in enumerate(bounds):
            assert bound[0] <= strat_unif[i,j] <= bound[1]

def test_stratified_integration():
    bounds = [(0, 2), (1, 4), (-1, 1)]
    est,err = montecarlo_integration(f,bounds,8,sampler=stratified_uniform,repeats=5)
    assert est == pytest.approx(12)
    assert err == pytest.approx(0)