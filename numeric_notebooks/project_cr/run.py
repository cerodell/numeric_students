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
                         'R0': 1.2, 'a1': 1, 'a2': 0.5, 'a3': 1,  }


# yf_start, yf_end = 50, 52
# xf_start, xf_end = 50, 52
coeff = Approximator(initialVals)

rk3 = coeff.rk3()

print(np.max(rk3),'max')
print(np.min(rk3),'min')

# plot = coeff.plot_functions()

levels = np.arange(-1,1,0.1)
norm = colors.Normalize(vmin= -1, vmax =1)

fig,ax = plt.subplots()
def animate(i):
       ax.clear()
       C = ax.contourf(rk3[i,:,:],0, levels = levels, norm = norm)
       ax.set_title('%03d'%(i)) 
       
# clb = plt.colorbar.ColorbarBase(norm=norm, ax =ax)


interval = 0.01#in seconds     
ani = animation.FuncAnimation(fig,animate,1000,interval=interval*1e+3,blit=False)
plt.show()




# def animate(i): 
#     z = var[i,:,0,:].T
#     cont = plt.contourf(x, y, z, 25)
#     if (tslice == 0):
#         plt.title(r't = %1.2e' % t[i] )
#     else:
#         plt.title(r't = %i' % i)

#     return cont  

# anim = animation.FuncAnimation(fig, animate, frames=Nt)

# fig, ax = plt.subplots(1,1, figsize=(8,8))


