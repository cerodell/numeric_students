import context
import numpy as np
from approximator import Approximator
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
import matplotlib.animation as animation




## Create the grid and initial conditions
initialVals={'x': 10000, 'y': 10000,  'dx':100., 'dy':100 
                 ,'dt':10. , 'uf': 5., 'time': 1000
                   ,'yf_start': 40, 'yf_end': 60, 'xf_start': 50, 'xf_end':52, 
                         'R0': 1.2, 'a1': 1, 'a2': 1, 'a3': 1,  }



coeff = Approximator(initialVals)

########################################################################

# plot2D = coeff.plot_main()

########################################################################

plot3D = coeff.plot_3D()


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

