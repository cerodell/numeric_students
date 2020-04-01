import context
import noise
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple 
from mpl_toolkits.mplot3d import axes3d




class Approx:

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
        self.world = world*6000
        ############################################################


        ############################################################
        ############## Define the function of phi ##################
        ############################################################
        # phi_ij = np.ones(self.shape)
        # phi_ij = np.random.randn(100,100)
        # phi_ij = np.ones(self.shape) 
        # phi_ij[self.yf_start:self.yf_end, self.xf_start:self.xf_end] = -0.1
        # self.phi_ij = phi_ij 
        # print(phi_ij[5,5], 'phi_ij Initial')

        def LoG(x, y, sigma):
            phi = (x ** 2 + y ** 2) / (2 * sigma ** 2)
            return -1 / (np.pi * sigma ** 4) * (1 - phi) * np.exp(-phi)

        half_N = self.x // 2
        zz = -LoG(self.xx - half_N, self.yy - half_N, sigma=100) * 30e10
        self.zz = np.where(zz > 0, zz, 0)
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
        print(k2[y,x], 'k2')
        zz =  self.dZ()
        print(zz[44,73], 'zz')


        Rf = self.R0 * (k1 + k2) * np.abs(self.centdif())
        # Rf = self.R0 * np.abs(self.centdif())
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
        x, y = 50, 50
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
            # print(phi_n[y,x], "phi_n post where")

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
    def plot_Ter3D(self):
        """
        ################################
        ## Terrain 3D Plot
        ################################        
        """
        plane1 = np.ones(self.shape) * 45
        # plane = self.phi_ij * 45
        fig = plt.figure(figsize=(12,6))
        fig.suptitle("Surface Function", fontsize= 16, fontweight="bold")
        ax = fig.add_subplot(111, projection="3d")
        ter = ax.plot_surface(self.xx,self.yy, self.world,cmap='terrain', zorder = 10)
        # ax.plot_surface(self.xx,self.yy, plane ,cmap='Reds', alpha = .8, zorder = 1)
        ax.plot_surface(self.xx,self.yy, plane1 ,cmap='Reds_r', alpha = .8, zorder = 1)

        ax.set_xlabel('Distance (X: m)', fontsize = 14)
        ax.set_ylabel('Distance (Y: m)', fontsize = 14)
        ax.set_zlabel('Height (Z: m)', fontsize = 14)
        fig.colorbar(ter, shrink=0.5, aspect=5)
        plt.show()

        return

    def plot_Phi3D(self):
        """
        ################################
        ## Phi 3D Plot
        ################################        
        """
        # plane1 = np.ones(self.shape) * 45
        # plane = self.phi_ij * 45
        fig = plt.figure(figsize=(12,6))
        fig.suptitle("Surface Function", fontsize= 16, fontweight="bold")
        ax = fig.add_subplot(111, projection="3d")
        ter = ax.plot_surface(self.xx,self.yy, self.zz,cmap='r', zorder = 10)
        # ax.plot_surface(self.xx,self.yy, plane ,cmap='Reds', alpha = .8, zorder = 1)
        # ax.plot_surface(self.xx,self.yy, plane1 ,cmap='Reds_r', alpha = .8, zorder = 1)

        ax.set_xlabel('Distance (X: m)', fontsize = 14)
        ax.set_ylabel('Distance (Y: m)', fontsize = 14)
        ax.set_zlabel('Height (Z: m)', fontsize = 14)
        fig.colorbar(ter, shrink=0.5, aspect=5)
        plt.show()

        return


    def plot_main(self):
        """
        ################################################
        ## Fire line overlayed on Terrain contourf Plot
        ################################################
        """

        rk3 = self.rk3()
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        fig.suptitle("Fire Line Propagation", fontsize= 16, fontweight="bold")
        level = np.arange(np.min(self.world),np.max(self.world),1)
        fire = ax.contour(self.xx,self.yy, rk3[-1,:,:], zorder =10, cmap ='Reds')
        ax.contourf(self.xx,self.yy, self.world,cmap='terrain', levels = level, zorder = 1)
        # fig.colorbar(fire, shrink=0.5, aspect=5)

        ax.set_xlabel('Distance (X: m)', fontsize = 14)
        ax.set_ylabel('Distance (Y: m)', fontsize = 14)
        plt.show()

        return


    def plot_test(self):
        """
        ################################################
        ## Play Plot fucntion
        ################################################
        """

        # fig, ax = plt.subplots(1,1, figsize=(8,8))
        # fig.suptitle("Runge-Kutta 3rd order Centred in Space  CR: 0.5", fontsize= plt_set.title_size, fontweight="bold")
        # Prk3 = self.rk3()
        # ax.plot(self.xx,Prk3.T[:,-1], color = 'green', label = "RK3", zorder = 9)
        # ax.set_xlabel('Grid Index (i)', fontsize = plt_set.label)
        # ax.set_ylabel('Quantity', fontsize = plt_set.label)
        # ax.xaxis.grid(color='gray', linestyle='dashed')
        # ax.yaxis.grid(color='gray', linestyle='dashed')
        # ax.set_ylim(-10,15)
        # ax.legend()
        # plt.show()

        return
