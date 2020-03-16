# %% [markdown]
# # Lab 7b 
### EOSC 511
#### Christopher Rodell
# %% [markdown]
# ## Question 1. 
# **Hand-in an answer to question 7b from the lab itself.**
# ### Problem Seven
# -b) For grid C, write down the finite difference form of the shallow water equations.
# Shallow Water Equations
# ##### Full Equations, Eqn 1
# $$
# \frac{\partial u}{\partial t}-f v=-g \frac{\partial h}{\partial x}
# $$
# $$
# \frac{\partial u}{\partial t}-f v=-g \frac{h_{(i+\frac{1}{2}), j, n}-h_{(i-\frac{1}{2}), j, n}}{\Delta x}
# $$
# $$
# \frac{u_{i, j, (n+\frac{1}{2})}-u_{i, j, (n-\frac{1}{2})}}{\Delta t}-f v=-g \frac{h_{(i+\frac{1}{2}), j, n}-h_{(i-\frac{1}{2}), j, n}}{\Delta x}
# $$
# $$
# \\
# $$
# ##### Full Equations, Eqn 2
# $$
# \frac{\partial v}{\partial t}-f u=-g \frac{\partial h}{\partial y}
# $$
# $$
# \frac{\partial v}{\partial t}-f v=-g \frac{h_{i,(j+\frac{1}{2}), n}-h_{i,(j-\frac{1}{2}), n}}{\Delta y}
# $$
# $$
# \frac{v_{i, j, (n+\frac{1}{2})}-v_{i, j, (n-\frac{1}{2})}}{\Delta t}-fu =-g \frac{h_{i,(j+\frac{1}{2}), n}-h_{i,(j-\frac{1}{2}), n}}{\Delta y}
# $$
# $$
# \\
# $$
# ##### Full Equations, Eqn 3
# $$
# \frac{\partial h}{\partial t}+H \frac{\partial u}{\partial x}+H \frac{\partial v}{\partial y}=0
# $$
# $$
# \frac{d h}{d t}+ H\frac{u_{(i+\frac{1}{2}), j, n}-u_{(i-\frac{1}{2}), j, n}}{\Delta x} + H\frac{v_{i,(j+\frac{1}{2}), n}-v_{i,(j-\frac{1}{2}), n}}{\Delta y} = 0
# $$
# $$
# \frac{h_{i, j, (n+{\frac{1}{2}})} - h_{i, j, (n-\frac{1}{2})}}{\Delta t} + H\frac{u_{(i+\frac{1}{2}), j, n}-u_{(i-\frac{1}{2}), j, n}}{\Delta x} + H\frac{v_{i,(j+\frac{1}{2}), n}-v_{i,(j-\frac{1}{2}), n}}{\Delta y} = 0
# $$
# $$
# \\
# $$
# $$
# \\
# $$
# The finite difference approximation to the full shallow water equations on the C grid:
# $$
# \frac{u_{i, j, (n+\frac{1}{2})}-u_{i, j, (n-\frac{1}{2})}}{\Delta t}-f v=-g \frac{h_{(i+\frac{1}{2}), j, n}-h_{(i-\frac{1}{2}), j, n}}{\Delta x}
# $$
# $$
#  \frac{v_{i, j, (n+\frac{1}{2})}-v_{i, j, (n-\frac{1}{2})}}{\Delta t}-fu =-g \frac{h_{i,(j+\frac{1}{2}), n}-h_{i,(j-\frac{1}{2}), n}}{\Delta y} \\ 
# $$
# $$
# \frac{h_{i, j, (n+{\frac{1}{2}})} - h_{i, j, (n-\frac{1}{2})}}{\Delta t} + H\frac{u_{(i+\frac{1}{2}), j, n}-u_{(i-\frac{1}{2}), j, n}}{\Delta x} + H\frac{v_{i,(j+\frac{1}{2}), n}-v_{i,(j-\frac{1}{2}), n}}{\Delta y} = 0
# $$

# %% [markdown]
# ## Question 2
# **You will use the interactive1.py code in numlabs/lab7 for this question.
# If your experience with the Coriolis force is minimal, you can chose the
# “small” option. See the doc string for reasonable parameters.
# The code solves the high water level in the center (similar to rain.py)
# in two-dimensions with periodic boundary conditions on a flat bottom.
# The depth is set in the functions find depth*. Use grid-C and edit the
# find depth3 function.**
# - a) Choose an interesting but smooth topography (remembering that
# the domain is periodic in both space dimensions). Implement it
# in find depth3 correctly given the grid-C staggering.
# - b) Run your new code. Discuss any other changes you make to the
# code. You may want to change what and when it plots.
# - c) Explain the differences that the bottom topography makes.

# %%
import context
from IPython.display import Image
import IPython.display as display
import matplotlib.pyplot as plt
# %matplotlib 
import numpy as np

# import the 2-dimensional drop solver
from numlabs.lab7 import interactive1_cr
from numlabs.lab7 import interactive1

plt.close('all')
# grid A is grid 1, grid B is grid 2 and and grid C is grid 3
# ngrid is the number of grid points in x and y
# dt is the time step in seconds
# T is the time plotted is seconds to 4*3600 is 4 hours

interactive1_cr.interactive1(grid=3, ngrid=11, dt=150, T=4*3600)

interactive1.interactive1(grid=3, ngrid=11, dt=150, T=4*3600)


# interactive1_cr.interactive1(grid=4, ngrid=11, dt=150, T=4*3600)


plt.show('all')

# %%
