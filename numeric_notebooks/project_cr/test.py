import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

x, y = 10000, 10000
shape = (100,100)
lin_x = np.linspace(0,x,shape[0],endpoint=False)
lin_y = np.linspace(0,y,shape[1],endpoint=False)
xx,yy = np.meshgrid(lin_x,lin_y)

test = np.random.randint(1,10, size=shape)

def LoG(x, y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -1 / (np.pi * sigma ** 4) * (1 - temp) * np.exp(-temp)

half_N = x // 2
zz = LoG(xx - half_N, yy - half_N, sigma=20000) *10e18

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(xx, yy, -zz, cmap='coolwarm')
# plt.show()



# A = np.array([[1, 2, 6, 7], [3, 4, 5, 2], [1,4, 8, 12], [13, 5, 7, 14]])
# dA = np.gradient(A)

# print(A)

# # phi_j = -self.u0 * ((np.roll(self.phi_j,-1) - np.roll(self.phi_j,1)) / (2 * self.dxy))

# print('#####################')

# dAcr = np.roll((np.roll(A,-1, axis=1)), 1, axis=0)
# print(dAcr)


# dBcr = np.roll(A,-1, axis=(0,1))
# print(dBcr)
# timer = np.arange(10)

# shape = (100,100)

# test = np.random.randint(1,10, size=shape)
# scale = 100.0
# octaves = 6
# persistence = 0.4
# lacunarity = 2.0

# world = np.zeros(shape)
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         world[i][j] = noise.pnoise2(i/scale, 
#                                     j/scale, 
#                                     octaves=octaves, 
#                                     persistence=persistence, 
#                                     lacunarity=lacunarity, 
#                                     repeatx=1024, 
#                                     repeaty=1024, 
#                                     base=42)

# lin_x = np.linspace(0,10000,shape[0],endpoint=False)
# lin_y = np.linspace(0,10000,shape[1],endpoint=False)
# xx,yy = np.meshgrid(lin_x,lin_y)
# worldcr = np.abs(world*500-2)

# print(worldcr.shape)

# test = np.gradient(worldcr)



# 
# 
# 
# 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(xx,yy,worldcr,cmap='terrain')

# plt.show()






