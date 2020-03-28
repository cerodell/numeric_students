import context
import noise
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple 
from mpl_toolkits.mplot3d import axes3d




class Approximator:

    #############################################
    # initialize condtions
    #############################################
    def __init__(self, valueDict):
        """
        Create the grid and initial conditions
        """
        ##  Defined conditions from dictonary
        self.__dict__.update(valueDict)

        scale = 100.0
        octaves = 6
        persistence = 0.4
        lacunarity = 2.0

        self.phi_ij = np.ones_like(self.shape)
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
        self.world = np.abs(world*500+200)

        return

    #############################################
    # spatial discretization methods
    #############################################
    def centdif(self):
        """
        Centered difference spatial approximation
        """
        # print(self.Pj[50],"centdif start")
        phi_j = -self.u0 * ((np.roll(self.phi_j,-1) - np.roll(self.phi_j,1)) / (2 * self.dx))
        
        # print(Pj[50],"centdif end")
        return phi_ij

    def backdif(self):
        """
        Backward difference spatial approximation
        """
        # print(self.Pj[50],"backdif start")
        phi_j = -self.u0 * ((self.phi_j - np.roll(self.phi_j,1)) / (self.dx))
        
        # print(Pj[50],"backdif end")
        return phi_ij
    
    #############################################
    # time discretization methods
    #############################################

    def rk3(self):
        """
        Runge-Kutta 3rd order Centred in Space
        """
        phi_OG = self.phi_ij

        phi_ijn1 = []

        for n in range(len(self.time)):
        
            phi_ij = self.phi_ij
            # print(Pj[50], "Pj var")
            P_str = Pj + (self.dt/3) * self.centdif()
            # print(P_str[50], 'P_str')

            self.Pj = P_str
            # print(self.Pj[50], 'self Pj should be Pjstr')

            P_str_str  = Pj + (self.dt/2) * self.centdif()
            # print(P_str_str[50], 'P_str_str')

            self.Pj = P_str_str
            # print(self.Pj[50], 'self Pj should be Pj_str_str')

            Pn  = Pj + self.dt * self.centdif()
            Pn = np.array(Pn)
            # print(Pn[50], "Pn pre append")
            Pjn_1.append(Pn)

            self.Pj = Pn
            # print(self.Pj[50], "self Pj or Pn")

        Pjn_1 = np.array(Pjn_1)
        self.Pj = Pj_OG

        return Pjn_1

    def plot_functions(self):

        ## Test plot of terrain
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(self.xx,self.yy, self.world,cmap='terrain')

        plt.show()
  
    #     fig, ax = plt.subplots(1,1, figsize=(12,4))
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

        return







