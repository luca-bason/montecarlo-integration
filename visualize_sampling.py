import numpy as np
import matplotlib.pyplot as plt
from mc_integration import uniform,stratified_uniform

bounds = [(0, 2), (0, 3)]
samples = 100

unif_points = uniform(bounds,samples)

unif_x = unif_points[:, 0]
unif_y = unif_points[:, 1]

strat_unif_points = stratified_uniform(bounds,samples)

strat_unif_x = strat_unif_points[:, 0]
strat_unif_y = strat_unif_points[:, 1]

fig,axes = plt.subplots(1, 2,figsize=(10,4))

axes[0].scatter(unif_x,unif_y)
axes[0].set_title("Uniform sampling")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_xlim(0, 2)
axes[0].set_ylim(0, 3)

axes[1].scatter(strat_unif_x,strat_unif_y)
axes[1].set_title("Stratified uniform sampling")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_xlim(0, 2)
axes[1].set_ylim(0, 3)

n_strata = int(samples ** 0.5)
hx = (bounds[0][1] - bounds[0][0]) / n_strata
hy = (bounds[1][1] - bounds[1][0]) / n_strata
for i in range(n_strata-1):
    x_value = bounds[0][0]+(i+1)*hx
    y_value = bounds[1][0]+(i+1)*hy
    axes[1].axvline(x_value)
    axes[1].axhline(y_value)

plt.tight_layout()
plt.show()