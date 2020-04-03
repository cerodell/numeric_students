'''
    File name: approximator.py
    Author: Christopher Rodell
    Date created: 3/27/2020
    Python Version: 3.7.6
    ------
    IMPORTANT: Ensure you working dirctiory has context.py within it.
    
    Essesntial packadges

    noise: used to generate terrain
    import noise
    https://anaconda.org/conda-forge/noise
    conda install -c conda-forge noise

'''

import context

import noise
import yaml
import numpy as np
from context import this_dir
import matplotlib.pyplot as plt
from collections import namedtuple 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Approximator:
    """
    A class used to fire simualtion using the level set method

    ...

    Attributes
    ----------
    namelist : yaml
        a yaml file containing initial condition and domain set up 

    Methods
    -------
    reinitialize()
        Reinitialize ensure phi doesnt blow up, as recommend from Munoz-Esparza et al.
    
    advect_fun()
        Fire rate of spread, Rf, parameteriziation defiend by Rothermel (1972)
    
    centdif()
        Centered difference spatial approximation for phi

    dZ()
        Centered difference spatial approximation for terrain

    rk3()
        Runge-Kutta 3rd order Centred in Space

    plot_Ter3D()
        Terrain 3D Plot

    plot_Phi3D()
        Phi 3D Plot

    plot_main()
        Fire line overlayed on Terrain contourf Plot

    plot_main_animate()
        Fire line overlayed on Terrain contourf Plot animated
   
    plot_test()
        Play Plot fucntion
    """

    #############################################
    # Initialize condtions
    #############################################
    def __init__(self, namelist):
        """
        Create the grid and initial conditions
        """
        ## yloc and xloc are used from print statments thoughout this class
        self.yloc, self.xloc = 50, 50

        ############################################################
        ######### Initial conditions from namelist.yaml ############
        ############################################################
        fileY = open(this_dir / namelist, 'r')
        config = yaml.load(fileY, Loader=yaml.SafeLoader)
        self.config = config

        timevars = namedtuple('timevars',config['timevars'].keys())
        self.timevars = timevars(**config['timevars'])

        spatialvars = namedtuple('spatialvars', config['spatialvars'].keys())
        self.spatialvars = spatialvars(**config['spatialvars'])

        fire = namedtuple('fire', config['fire'].keys())
        self.fire = fire(**config['fire'])

        firecoeff = namedtuple('firecoeff', config['firecoeff'].keys())
        self.firecoeff = firecoeff(**config['firecoeff'])

        xshape = int(self.spatialvars.x/self.spatialvars.dx)
        yshape = int(self.spatialvars.y/self.spatialvars.dy)
        self.shape = (xshape, yshape)
        print(self.shape, "Domain Shape")
        ############################################################


        ############################################################
        ############## Define the terrian aka world ################
        ############################################################
        scale = self.shape[0]
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

        lin_x = np.linspace(0,self.spatialvars.x,self.shape[0],endpoint=False)
        lin_y = np.linspace(0,self.spatialvars.y,self.shape[1],endpoint=False)
        self.xx,self.yy = np.meshgrid(lin_x,lin_y)
        self.world = np.abs(world*6000)
        ############################################################


        ############################################################
        ############## Define the function of phi ##################
        ############################################################
        phi_ij = np.ones(self.shape)
        # phi_ij = np.random.randint(1,20, size=self.shape)

        xf_start, xf_end, yf_start, yf_end = self.fire
        yfshape, xfshape = int(abs(yf_start-yf_end)), int(abs(xf_start-xf_end))

        phi_ij[yf_start:yf_end, xf_start:xf_end] = -3 
        
        self.phi_ij = phi_ij 
        # print(phi_ij[self.yloc,self.xloc], 'phi_ij Initial')
        ############################################################

        return

    def reinitialize(self,phi):
        """
        Reinitialize ensure phi doesnt blow up, as recommend from Munoz-Esparza et al.
        """

        random = np.random.uniform(low=3., high=13.3, size=(1,))
        phi_n = np.where(phi > 0, phi, -random)
        # print(phi_n[self.yloc,self.xloc], "phi_n post where")

        random2  = np.random.uniform(low=3., high=13.3, size=(1,))
        phi_n = np.where(phi_n < 10, phi_n, random2)
        # print(phi_n[self.yloc,self.xloc], "phi_n post post where")

        return phi_n

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
        uf, R0, a1, a2, a3 = self.firecoeff

        delta_phi = self.centdif()
        delta_z = self.dZ()

        # print(np.max(delta_phi), "cent diff in fire fun")
        normal = delta_phi / np.abs(delta_phi)
        # print(normal[self.yloc,self.xloc], 'normal')

        k1 = 1 + a1 * np.power((uf * normal), a2)
        # print(k1[self.yloc,self.xloc], 'k1')

        k2 = a3 * np.power((delta_z * normal), 2)
        # print(k2[self.yloc,self.xloc], 'k2')

        Rf = R0 * (k1 + k2) * np.abs(delta_phi)
        # print(Rf[self.yloc,self.xloc], 'Rf')
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
        x, y, dx, dy = self.spatialvars

        phi_ij = self.phi_ij
        # print(phi_ij[self.yloc,self.xloc],"centdif phi start")

        k1 = np.roll(phi_ij , -1, axis = (0, 1))
        # print(k1[self.yloc,self.xloc], "k1 cent")

        k2 = np.roll(np.roll(phi_ij , -1, axis = 1), 1, axis = 0)
        # print(k2[self.yloc,self.xloc], "k2 cent")

        k3 = np.roll(np.roll(phi_ij , -1, axis = 0), 1, axis = 1)
        # print(k3[self.yloc,self.xloc], "k3 cent")

        k4 = np.roll(phi_ij , 1, axis = (0, 1))
        # print(k4[self.yloc,self.xloc], "k4 cent")

        phi_ij = (k1 - k2 - k3 - k4) / (4 * dx * dy)
        # print(phi_ij[self.yloc,self.xloc],"centdif phi end")
        
        return phi_ij



    def dZ(self):
        """
        Centered difference spatial approximation

        Returns
        -------
        dZ: gradient of terrain (dz/dx,dz/dy)
        """
        x, y, dx, dy = self.spatialvars

        z = self.world
        # print(z[self.yloc,self.xloc],"z pre centdiff")

        k1 = np.roll(z , -1, axis = (0, 1))

        k2 = np.roll(np.roll(z , -1, axis = 1), 1, axis = 0)

        k3 = np.roll(np.roll(z , -1, axis = 0), 1, axis = 1)

        k4 = np.roll(z , 1, axis = (0, 1))

        dZ = (k1 - k2 - k3 - k4) / (4 * dx * dy) 
        # print(dZ[self.yloc,self.xloc],"centdif dZ")

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
        dt, nsteps = self.timevars
        phi_OG = self.phi_ij

        phi_n1 = []

        for n in range(nsteps):
            self.time = n * dt
            print(self.time, 'time (seconds)')
            phi_ij = self.phi_ij
            # print(phi_ij[self.yloc,self.xloc], "phi_ij var")

            phi_str = phi_ij - (dt/3) * self.advect_fun()
            # print(phi_str[self.yloc,self.xloc], 'phi_str')

            self.phi_ij = phi_str
            # print(self.phi_ij[self.yloc,self.xloc], 'self phi_ij should be phi_str')

            phi_str_str  = phi_ij - (dt/2) * self.advect_fun()
            # print(phi_str_str[self.yloc,self.xloc], 'phi_str_str')

            self.phi_ij = phi_str_str
            # print(self.phi_ij[self.yloc,self.xloc], 'self phi_ij should be phi_str_str')

            phi_n  = phi_ij - dt * self.advect_fun()
            phi_n = np.array(phi_n)
            # print(phi_n[self.yloc,self.xloc], "phi_n pre where")

            ## THIS IS VERY IMPORTANT!!!!!!!!!!
            ## Reinitialize phi to ensure the model doesnt expoled!
            phi_n = self.reinitialize(phi_n)

            phi_n1.append(phi_n)
            self.phi_ij = phi_n

        phi_n1 = np.stack(phi_n1)
        print(phi_n1.shape, "Phi Final Shape n,j,i (time,y,x)")
        
        self.phi_ij = phi_OG
        return phi_n1


    #############################################
    ############ Ploting functions ##############
    #############################################
    def plot_Ter3D(self):
        """
        Terrain 3D Plot     
        """
        plane = self.phi_ij * 45
        fig = plt.figure(figsize=(12,6))
        fig.suptitle("Surface Function", fontsize= 16, fontweight="bold")
        ax = fig.add_subplot(111, projection="3d")
        ter = ax.plot_surface(self.xx,self.yy, self.world,cmap='terrain', zorder = 10)
        ax.plot_surface(self.xx,self.yy, plane ,cmap='Reds', alpha = .8, zorder = 1)

        ax.set_xlabel('Distance (X: m)', fontsize = 14)
        ax.set_ylabel('Distance (Y: m)', fontsize = 14)
        ax.set_zlabel('Height (Z: m)', fontsize = 14)
        fig.colorbar(ter, shrink=0.5, aspect=5)
        plt.show()

        return

    def plot_Phi3D(self):
        """
        Phi 3D Plot
        """
        rk3 = self.rk3()

        fig = plt.figure(figsize=(12,6))
        fig.suptitle("Surface Function", fontsize= 16, fontweight="bold")
        ax = fig.add_subplot(111, projection="3d")
        phi = ax.plot_surface(self.xx,self.yy, rk3[-1,:,:],cmap='Reds', zorder = 10)

        ax.set_xlabel('Distance (X: m)', fontsize = 14)
        ax.set_ylabel('Distance (Y: m)', fontsize = 14)
        ax.set_zlabel('Height (Z: m)', fontsize = 14)
        fig.colorbar(phi, shrink=0.5, aspect=5)
        plt.show()

        return


    def plot_main(self):
        """
        Fire line overlayed on Terrain contourf Plot
        """
        fire_level = 0
        rk3 = self.rk3()
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        fig.suptitle("Fire Line Propagation", fontsize= 16, fontweight="bold")
        ax.set_title("Run Time: Seconds:  " + str(self.time) , fontsize= 8)
        level = np.arange(np.min(self.world),np.max(self.world),1)
        fire = [plt.Rectangle((0,0),1,1, fc = "red")]  
        ax.contour(self.xx,self.yy, rk3[-1,:,:], zorder =10, colors='red', levels = fire_level, linewidths = 2.5)
        ter = ax.contourf(self.xx,self.yy, self.world,cmap='terrain', levels = level, zorder = 1)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = plt.colorbar(ter, cax=cax)
        clb.set_label('Elevation (Z: meters)', fontsize = 12)
        clb.ax.tick_params(labelsize=12) 
        clb.set_alpha(.75)
        clb.draw_all()
        
        ax.legend(fire, ["Fire Line"])
        ax.set_xlabel('Distance (X: meters)', fontsize = 14)
        ax.set_ylabel('Distance (Y: mmeters)', fontsize = 14)
        plt.show()

        return

    def plot_main_animate(self):
        """
        Fire line overlayed on Terrain contourf Plot
        """
        nsteps = self.timevars.nsteps
        dt = self.timevars.dt
        fire_level = 0
        rk3 = self.rk3()
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        fig.suptitle("Fire Line Propagation", fontsize= 16, fontweight="bold")
        level = np.arange(np.min(self.world),np.max(self.world),1)
        fire = [plt.Rectangle((0,0),1,1, fc = "red")]  
        ter = ax.contourf(self.xx,self.yy, self.world,cmap='terrain', levels = level, zorder = 1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = plt.colorbar(ter, cax=cax)
        clb.set_label('Elevation (Z: meters)', fontsize = 12)
        clb.ax.tick_params(labelsize=12) 
        clb.set_alpha(.75)
        clb.draw_all()          
        ax.legend(fire, ["Fire Line"])
        ax.set_xlabel('Distance (X: meters)', fontsize = 14)
        ax.set_ylabel('Distance (Y: mmeters)', fontsize = 14)

        ax2 = ax.twinx()
        def animate(i):
            ax2.clear()
            ax2.set_title("Run Time: Seconds:  " + '%03d'%(i * dt ), fontsize= 8)
            ax2.contour(self.xx,self.yy, rk3[i,:,:], zorder =10, colors='red', levels = fire_level, linewidths = 2.5)
            ax2.set_yticklabels([])

        ani = animation.FuncAnimation(fig,animate,nsteps,interval= 100 ,blit=False)
        plt.show()

        # plt.close('all')
        # ani.save( str(this_dir) + 'fire.mp4', fps=30)

    def plot_test(self):
        """
        Play Plot fucntion
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







