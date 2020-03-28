import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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
worldcr = np.abs(world*500+200)



test = np.gradient(worldcr)
# print(x.shape)
# 

# worldcr = np.where(world>0, world, 0.001)
# worldcr = np.abs(world*500+200)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(xx,yy,worldcr,cmap='terrain')

# plt.show()


phi = np.ones(shape)

phi_test  = np.gradient(phi)
