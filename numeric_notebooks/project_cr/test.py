import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d



# A = np.array([[1, 2, 6, 7], [3, 4, 5, 2], [1,4, 8, 12], [13, 5, 7, 14]])
# dA = np.gradient(A)

# print(A)

# # phi_j = -self.u0 * ((np.roll(self.phi_j,-1) - np.roll(self.phi_j,1)) / (2 * self.dxy))

# print('#####################')

# dAcr = np.roll((np.roll(A,-1, axis=1)), 1, axis=0)
# print(dAcr)


# dBcr = np.roll(A,-1, axis=(0,1))
# print(dBcr)
timer = np.arange(10)

shape = (100,100)
scale = 100.0
octaves = 6
persistence = 0.4
lacunarity = 2.0

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=42)

lin_x = np.linspace(0,10000,shape[0],endpoint=False)
lin_y = np.linspace(0,10000,shape[1],endpoint=False)
xx,yy = np.meshgrid(lin_x,lin_y)
worldcr = np.abs(world*500-2)

print(worldcr.shape)

test = np.gradient(worldcr)
# print(test.shape)
# # 

# # worldcr = np.where(world>0, world, 0.001)
# worldcr = np.abs(world*500+200)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(xx,yy,worldcr,cmap='terrain')

plt.show()


# phi = np.ones(shape)

# yf_start, yf_end = 50, 52
# xf_start, xf_end = 50, 52


# phi_ij = np.ones(shape)
# phi_ij[yf_start:yf_end, xf_start:xf_end] = 0
# phi_ij = phi_ij
# phi_test  = np.gradient(phi)




