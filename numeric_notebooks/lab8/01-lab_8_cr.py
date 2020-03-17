# %% [markdown]
# # Lab 8
### EOSC 511
#### Christopher Rodell
# %% [markdown]
# I. Do FIRST this subset of problem 3. Set loop=True, and compare the
# efficiency of the SOR method (currently implemented, suggest overrelaxation coefficient of 1.7) and the Jacobi iteration (you need to implement, suggest coefficient of 1; I find 1.7 unstable). Also compare to
# indexing the loops and doing Jacobi interation by setting loop=False.
# You can time functions using %timeit

# %%

import context
import numlabs.lab8.qg as qg
import sys


import matplotlib.pyplot as plt
import numpy as np

time = 10*86400

qg.main((time, True))


qg.main((time, False))



