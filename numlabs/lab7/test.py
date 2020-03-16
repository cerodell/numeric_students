
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# # # Make the X, Y meshgrid instead of np.tile
# ngrid = 11
# H0 = 1000

# # xs = np.linspace(0, 2*np.pi, ngrid)
# # ys = np.linspace(0, 2*np.pi, ngrid)

# xs = np.linspace(0, L, ngrid)
# ys = np.linspace(0, L, ngrid)

# Hu, Hv = np.meshgrid(xs, ys) 
# amp =400*np.sin((Hu+Hu)) + H0
# # %matplotlib 
# L = 1000e3

# xs = np.linspace(0, L, ngrid)
# ys = np.linspace(0, L, ngrid)

# xx, yy = np.meshgrid(xs, ys) 

# fig = plt.figure()
# ax3d = fig.add_subplot(111, projection='3d')
# surf = ax3d.plot_surface(xx, yy, amp)

zz = np.arange(0,97,1)
for i in zz[0::6]:
    print(zz[i])
# Make the X, Y meshgrid instead of np.tile
# ngrid = 11
# H0 = 1000

# xs = np.linspace(0, H0, ngrid)

# Hu = xs
# Hv = Huu*np.ones_like(Hu)
# Hu, Hv = np.meshgrid(Hu, Hv) 


# # Z evaluation
# amp = np.sin(Hu+Hv)
# %matplotlib 


# fig = plt.figure()
# ax3d = fig.add_subplot(111, projection='3d')
# surf = ax3d.plot_surface(tau, phi, amp)
