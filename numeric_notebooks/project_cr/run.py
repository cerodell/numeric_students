import context
import numpy as np
from approximator import Approximator
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
import matplotlib.animation as animation




## Create the grid and initial conditions
initialVals={'x': 10000, 'y': 10000,  'dx':100., 'dy':100 
                 ,'dt':10. , 'uf': 5., 'shape': (100,100), 'time': 1000
                   ,'yf_start': 40, 'yf_end': 60, 'xf_start': 50, 'xf_end':52, 
                         'R0': 1.2, 'a1': 1, 'a2': 1, 'a3': 1,  }



coeff = Approximator(initialVals)

########################################################################

# rk3 = coeff.rk3()

########################################################################

plot = coeff.plot_functions()


########################################################################
########################################################################
# level = np.arange(np.min(coeff.world),np.max(coeff.world),1)
# fig,ax = plt.subplots()
# def animate(i):
#        ax.clear()
#        ax.contour(rk3[i,:,:],20, zorder = 10, cmap='Reds')
#        # ax.contourf(coeff.xx,coeff.yy, coeff.world,cmap='terrain', levels = level, zorder =1)

#        ax.set_title('%03d'%(i)) 
# interval = 0.01#in seconds     
# ani = animation.FuncAnimation(fig,animate,100,interval=interval*1e+3,blit=False)
# plt.show()
########################################################################
########################################################################

