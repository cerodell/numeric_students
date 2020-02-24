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
