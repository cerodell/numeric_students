import context
import noise
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple 
from mpl_toolkits.mplot3d import axes3d




class Approximator:

    #############################################
    # Initialize condtions
    #############################################
    def __init__(self, valueDict):
        """
        Create the grid and initial conditions
        """
        ############################################################
        ############ Initial conditions from dictionary ############
        ############################################################
        self.__dict__.update(valueDict)

        xshape = int(self.x/self.dx)
        yshape = int(self.y/self.dy)
        self.shape = (xshape, yshape)
        print(self.shape, "Domain is shape")
        ############################################################


        ############################################################
        ############## Define the function of phi ##################
        ############################################################
        # phi_ij = np.ones(self.shape)
        # phi_ij = np.random.randn(100,100)
        phi_ij = np.ones(self.shape)
        phi_ij[self.yf_start:self.yf_end, self.xf_start:self.xf_end] = -0.1
        self.phi_ij = phi_ij
        print(phi_ij[5,5], 'phi_ij Initial')
        ############################################################


        ############################################################
        ############## Define the terrian aka world ################
        ############################################################
        scale = 100.0
        octaves = 6
        persistence = 0.4
        lacunarity = 2.0
        world = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                world[i][j] = noise.pnoise2(i/scale, 
                                            j/scale, 
                                            octaves=octaves, 
                                            persistence=persistence, 
                                            lacunarity=lacunarity, 
                                            repeatx=1024, 
                                            repeaty=1024, 
                                            base=42)

        lin_x = np.linspace(0,self.x,self.shape[0],endpoint=False)
        lin_y = np.linspace(0,self.y,self.shape[1],endpoint=False)
        self.xx,self.yy = np.meshgrid(lin_x,lin_y)
        self.world = np.abs(world*500-2)
        ############################################################


        return

    #############################################
    ############# Advection Function ############
    #############################################
    def advect_fun(self):
        """
        Fire rate of spread, Rf, parameteriziation defiend by Rothermel (1972)
       
        Variables
        ----------
        Rf: Fire Rate of Spread (ms^-1)
        R0: Rate of spread in the absence of wind and terrain slope (ms^-1)

        uf: Midflame height horizontal wind vector (ms^-1)
        a1, a2, a3: Model coefficient for fuel characteristics defined by Anderson (1982) (?)
        normal = moral graident 
        k1, k2: Rf broken down into parts
        
        Returns
        -------
        Rf: Fire Rate of Spread
        """
        y, x = 50, 50
        normal = self.centdif() / np.abs(self.centdif())
        # print(normal[y,x], 'normal')
        # print(normal.shape, 'normal shape')

        k1 = 1 + self.a1 * np.power((self.uf * normal), self.a2)
        # print(k1[y,x], 'k1')
        k2 = self.a3 * np.power((self.dZ() * normal), 2)
        # print(k2[y,x], 'k2')

        Rf = self.R0 * (k1 + k2) * np.abs(self.centdif())
        # print(Rf[y,x], 'Rf')
        # print(np.max(Rf), 'Rf max')

        return Rf

    
    #############################################
    ###### Spatial discretization methods #######
    #############################################
    def centdif(self):
        """
        Centered difference spatial approximation

        Returns
        -------
        phi_ij: dimensionless
        """
        phi_ij = self.phi_ij

        k1 = np.roll(phi_ij , -1, axis = (0, 1))
        k2 = np.roll(np.roll(phi_ij , -1, axis = 1), 1, axis = 0)
        k3 = np.roll(np.roll(phi_ij , -1, axis = 0), 1, axis = 1)
        k4 = np.roll(phi_ij , 1, axis = (0, 1))

        phi_ij = (k1 - k2 - k3 - k4) / (4 * self.dx * self.dy)
        
        # print(phi_ij[54,50],"centdif phi")
        return phi_ij

    def dZ(self):
        """
        Centered difference spatial approximation

        Returns
        -------
        dZ: gradient of terrain (dz/dx,dz/dy)
        """
        z = self.world

        k1 = np.roll(z , -1, axis = (0, 1))
        k2 = np.roll(np.roll(z , -1, axis = 1), 1, axis = 0)
        k3 = np.roll(np.roll(z , -1, axis = 0), 1, axis = 1)
        k4 = np.roll(z , 1, axis = (0, 1))

        dZ = (k1 - k2 - k3 - k4) / (4 * self.dx * self.dy)
        
        # print(z[54,50],"centdif dZ")
        return dZ

    #############################################
    ######## Time discretization methods ########
    #############################################


    def rk3(self):
        """
        Runge-Kutta 3rd order Centred in Space
        
        Returns
        -------
        phi_n1: next time step of phi_ij

        """
        phi_OG = self.phi_ij

        phi_n1 = []
        x, y = 5, 5
        for n in range(self.time):
            print(n, 'time')
            phi_ij = self.phi_ij
            print(phi_ij[y,x], "phi_ij var")
            phi_str = phi_ij + (self.dt/3) * self.advect_fun()
            print(phi_str[y,x], 'phi_str')

            self.phi_ij = phi_str
            print(self.phi_ij[y,x], 'self phi_ij should be phi_str')

            phi_str_str  = phi_ij + (self.dt/2) * self.advect_fun()
            print(phi_str_str[y,x], 'phi_str_str')

            self.phi_ij = phi_str_str
            print(self.phi_ij[y,x], 'self phi_ij should be phi_str_str')

            phi_n  = phi_ij + self.dt * self.advect_fun()
            phi_n = np.array(phi_n)
            print(phi_n[y,x], "phi_n pre where")
            
            phi_n = np.where(phi_n < 0, phi_n, -0.1)
            print(phi_n[y,x], "phi_n post where")

            # phi_n = np.where(phi_n < 0, phi_n, 1)
            # print(phi_n[y,x], "phi_n post where")

            phi_n1.append(phi_n)
            self.phi_ij = phi_n

        phi_n1 = np.stack(phi_n1)
        print(phi_n1.shape)
        self.phi_ij = phi_OG

        return phi_n1


    #############################################
    ############ Ploting functions ##############
    #############################################
    def plot_functions(self):
        """
        Ploting function
        """
        rk3 = self.rk3()


        ## Test plot of terrain 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")


        # ax.plot_surface(self.xx,self.yy, rk3[-1,:,:],cmap='terrain')
  

        fig, ax = plt.subplots(1,1, figsize=(8,8))
        # cmap = cm.coolwarm
        # level = np.arange(-3,3,0.1)
        # v_line = np.arange(-3,3,0.8)
        # f2_ax3.set_title('MSLP Diff(YSU-Base)')
        # f2_ax3.coastlines('50m')
        # level = np.arange(np.min(self.world),np.max(self.world),1)
        # C = ax.contourf(self.xx,self.yy, self.world,cmap='terrain', levels = level, zorder =4)
        # CS = ax.contour(self.xx,self.yy, self.world,cmap='terrain')
                        # transform=crs.PlateCarree(), levels = v_line, colors = 'k', linewidths = 0.5)
        # f2_ax3.clabel(CS, fmt = '%1.1d', colors = 'k', fontsize=4) #contour line labels
        # rk3 = self.rk3()
        C = ax.contourf(self.xx,self.yy, rk3[-1,:,:], zorder =10, cmap ='Reds')



        # fig, ax = plt.subplots(1,1, figsize=(12,4))
    #     fig.suptitle("Runge-Kutta 3rd order Centred in Space  CR: 0.5", fontsize= plt_set.title_size, fontweight="bold")
    #     ax.plot(self.xx, self.Pj, color = 'blue', label = "Initial concentration", zorder = 10)
    #     ax.plot(self.xx,self.cideal, color = 'red', label = "Final Ideal", zorder = 8)
    #     Prk3 = self.rk3()
    #     ax.plot(self.xx,Prk3.T[:,-1], color = 'green', label = "RK3", zorder = 9)
    #     ax.set_xlabel('Grid Index (i)', fontsize = plt_set.label)
    #     ax.set_ylabel('Quantity', fontsize = plt_set.label)
    #     ax.xaxis.grid(color='gray', linestyle='dashed')
    #     ax.yaxis.grid(color='gray', linestyle='dashed')
    #     ax.set_ylim(-10,15)
    #     ax.legend()
    #     plt.show()
        plt.show()

        return







