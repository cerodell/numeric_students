# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-language_info,-toc,-latex_envs
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Laboratory 7: Solving partial differential equations using an explicit, finite difference method.
# ## Christopher Rodell
# ### EOSC 511

# %% [markdown]
# ## Problem Two
# - a\) Find the CFL condition (in seconds) for 𝑑𝑡 for the Python example in Problem One. Test your value.
# $$
# d / d t>2 \sqrt{g H}
# $$
# $$
# \\
# $$
# Using the Stagered Grid.
# $$
# \begin{array}{l}{\frac{u_{i}(t+d t)-u_{i}(t-d t)}{2 d t}+g \frac{h_{i+1}(t)-h_{i}(t)}{d x}=0} \\ {\frac{h_{i}(t+d t)-h_{i}(t-d t)}{2 d t}+H \frac{u_{i}(t)-u_{i-1}(t)}{d x}=0}\end{array}
# $$
# %% [markdown]
# **CR CFL calculation**
# $$
# {g = 980}    \hspace{2mm} cm * s^{-2} \hspace{10mm} {H = 1}    \hspace{2mm} cm \hspace{10mm} {dx = 1}    \hspace{2mm} cm 
# $$
# $$
# d / d t>2 \sqrt{g H}
# $$
# $$
# \frac{dx}{2 \sqrt{g H}} > dt
# $$
# $$
# \\
# $$
# $$
# 0.0159 s > dt
# $$
# **If dt is greater than 0.0159 the model becomes unstable and explodes. SEE PLOT BELOW AS EXAMPLE dt = 0.018s**


# %%
import context
import matplotlib.pyplot as plt
import numpy as np
from numlabs.lab7 import rain_cr


rain_cr.rain([50,9])
plt.show()

# %%
