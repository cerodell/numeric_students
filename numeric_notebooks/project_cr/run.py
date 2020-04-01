import context
import yaml
import numpy as np
from approximator import Approximator
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
import matplotlib.animation as animation




## Create the grid and initial conditions
# initialVals={'x': 1000, 'y': 1000,  'dx':10., 'dy':10. 
#                  ,'dt':1. , 'uf': 2., 'time': 2
#                    ,'yf_start': 40, 'yf_end': 60, 'xf_start': 50, 'xf_end':52, 
#                          'R0': 1.2, 'a1': 1., 'a2': .5, 'a3': 1,  }



coeff = Approximator("namelist.yaml")

########################################################################

# plot2D = coeff.plot_main()

########################################################################

plot3D = coeff.plot_Ter3D()

# plot3D = coeff.plot_Phi3D()
# plt.close('all')

########################################################################
########################################################################
# rk3 = coeff.rk3()

# level = np.arange(np.min(coeff.world),np.max(coeff.world),1)
# fig,ax = plt.subplots()
# def animate(i):
#        ax.clear()
#        ax.contour(coeff.xx,coeff.yy,rk3[i,:,:], zorder = 10, cmap='Reds')
#        ax.contourf(coeff.xx,coeff.yy, coeff.world,cmap='terrain', levels = level, zorder =1)

#        ax.set_title('%03d'%(i)) 
# interval = 0.01#in seconds     
# ani = animation.FuncAnimation(fig,animate,coeff.time,interval=interval*1e+3,blit=False)
# plt.show()
# ani.save('/Users/rodell/Desktop/fire_fail3.mp4', fps=30)

########################################################################
########################################################################

