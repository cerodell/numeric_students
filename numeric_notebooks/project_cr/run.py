import context
import numpy as np
from approximator import Approximator




## Create the grid and initial conditions
initialVals={'x': 10000, 'y': 10000,  'dx':100. 
                 ,'dt':10. , 'u0': 5., 'shape': (100,100), 'time': 10 }

coeff = Approximator(initialVals)

plot = coeff.plot_functions()
