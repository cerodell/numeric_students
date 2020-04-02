import context
import yaml
import numpy as np
from approximator import Approximator
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
import matplotlib.animation as animation


coeff = Approximator("namelist.yaml")

########################################################################
# 
# plot2D = coeff.plot_main()

########################################################################

# plot3D = coeff.plot_Ter3D()

plot3D = coeff.plot_Phi3D()
# plt.close('all')

########################################################################
########################################################################
# rk3 = coeff.rk3()

# level = np.arange(np.min(coeff.world),np.max(coeff.world),1)
# fig,ax = plt.subplots()
# def animate(i):
#        ax.clear()
#        ax.contour(coeff.xx,coeff.yy,rk3[i,:,:], zorder = 10, cmap='Reds', levels = 0)
#        ax.contourf(coeff.xx,coeff.yy, coeff.world,cmap='terrain', levels = level, zorder =1)

#        ax.set_title('%03d'%(i)) 
# interval = 0.01#in seconds     
# ani = animation.FuncAnimation(fig,animate,coeff.timevars.nsteps,interval=interval*1e+3,blit=False)
# plt.show()
# ani.save('/Users/rodell/Desktop/fire_fail3.mp4', fps=30)

########################################################################
########################################################################

