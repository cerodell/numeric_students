# %% [markdown]
# # Lab 8
### EOSC 511
#### Christopher Rodell
# %% [markdown]
# I. Do FIRST this subset of problem 3. Set loop=True, and compare the
# efficiency of the SOR method (currently implemented, suggest overrelaxation coefficient of 1.7) and the Jacobi iteration (you need to implement, suggest coefficient of 1; I find 1.7 unstable). Also compare to
# indexing the loops and doing Jacobi interation by setting loop=False.
# You can time functions using %timeit
# $$
# \\
# $$
# *See qg.py script and look at functions qg_cr and relax_jacobi.
#  Sadly could not get this to work properly. I was having issue
#   get everything to converge. I kept getting nan values as the
#    iteration moves forward. Seemed to start well but then flopped :(
# Not sure where I was going wrong. I had attempted this by 
# resourcing code I found off Wikipedia.*
# $$
# \\
# $$
# *To answer the question regarding computational cost. Based 
# on the Cost of Schemes Table. The SOR is a bit more computational 
# expense than the simpler Jacobi method. Also, the SOR improves
#  convergence considerably as compared to the Jacobi method.* 
# $$
# \\
# $$
# *Looking at the SOR method using the loop set to true of false
#  we find that when setting to true the function takes a few 
#  one-thousandths of sec longer to run and also convergence
#   to something logical that plots. While false is ever so slightly
#    quicker the result is a convergence to values less than zero*
# %% [markdown]
# Plot using Successive Over-Relaxation with loop set to True
# %%
import context
import numlabs.lab8.qg as qg
import sys
import matplotlib.pyplot as plt
import numpy as np
import time


t = 10*86400
start = time.time()
qg.main((t, True))
end = time.time()
print(end - start, "Time elapsed for loop set to True")

# %% [markdown]
# Plot using Successive Over-Relaxation with loop set to False
# %%
start = time.time()
qg.main((t, False))
end = time.time()
print(end - start, "Time elapsed for loop set to False")


# %% [markdown]
# Plot using the Jacobi iteration
# See qg.py script and look at functions qg_cr and relax_jacobi. 
# Sadly could not get this to work properly. I was having issue
#  get everything to converge. I kept getting nan values as the iteration moves forward. Seemed to start well but then flopped :(
# Not sure where I was going worng. I had attempted this by resourcing code I found off Wikipedia. 
# %%
## See qg.py script if you want to see what I had tried
# qg.qg_cr((time, False))

# %% [markdown]
# Then using the most efficient scheme, choose one parameter 
# of the problem (eg depth, width of the ocean, vertical 
# viscosity, wind stress, latitude) vary it (3 or 4 choices)
#  and compare the solutions.

# *I am using the SOR scheme as I was unable to get the Jacobi
#  to work properly. Based on the lab the SOR improves
#  convergence considerably as compared to the Jacobi method. *
# $$
# \\
# $$
#  