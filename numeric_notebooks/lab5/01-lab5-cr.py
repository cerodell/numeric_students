# %% [markdown]
# # Lab 5: Daisyworld
# ### Christopher Rordell
# ##### EOSC 511
# %% [markdown]
# ## Problem constant growth
# Note that though the daisy growth rate per unit time depends on the amount 
# of available fertile land, it is not otherwise coupled to the environment 
# (i.e. 𝛽𝑖 is note a function of temperature. Making the growth a function of
# bare ground, however, keeps the daisy population bounded and the daisy population 
# will eventually reach some steady state. The next python cell has a script that
# runs a fixed timestep Runge Kutte routine that calculates area coverage of white
# and black daisies for fixed growth rates 𝛽𝑤 and 𝛽𝑏. Try changing these growth 
# rates (specified in the derivs5 routine) and the initial white and black 
# concentrations (specified in the fixed_growth.yaml file discussed next).
#
# $$
# \\
# $$
#
# - 1\) For a given set of growth rates try various (non-zero) initial daisy populations.

# - 2\) For a given set of initial conditions try various growth rates. In particular, try rates that are both greater than and less than the death rate.

# - 3\) Can you determine when non-zero steady states are achieved? Explain.

# %% 