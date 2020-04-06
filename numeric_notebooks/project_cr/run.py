'''
    File name: run.py
    Author: Christopher Rodell
    Date created: 3/27/2020
    Python Version: 3.7.6
    ------
    IMPORTANT: Ensure you working dirctiory has context.py within it.

    run.py calls on the approximator class and namelist.yaml to plot fire simualtion
'''

import context
from approximator import Approximator


coeff = Approximator("namelist.yaml")


########################################################################
""" #######  Fire line overlayed on Terrain contourf Plot ########## """
########################################################################

plot_main = coeff.plot_main()

########################################################################
""" #### Fire line overlayed on Terrain contourf Plot animated ##### """
########################################################################

# plot_animation = coeff.plot_main_animate()

########################################################################
""" ##################### Terrain 3D Plot ######################### """
########################################################################

# plot_ter_3D = coeff.plot_Ter3D()


########################################################################
""" ######################## Phi 3D Plot ########################### """
########################################################################

# plot_phi_3D = coeff.plot_Phi3D()


########################################################################
########################################################################