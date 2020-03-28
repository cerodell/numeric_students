import context
import numpy as np
from approximator import Approximator




## Create the grid and initial conditions
initialVals={'x': 10000, 'y': 10000,  'dx':100., 'dy':100 
                 ,'dt':10. , 'uf': 5., 'shape': (100,100), 'time': 1000
                   ,'yf_start': 50, 'yf_end': 52, 'xf_start': 50, 'xf_end':52, 
                         'R0': 1.2, 'a1': 1, 'a2': 0.5, 'a3': 1,  }


# yf_start, yf_end = 50, 52
# xf_start, xf_end = 50, 52
coeff = Approximator(initialVals)

# rk3 = coeff.rk3()

plot = coeff.plot_functions()


