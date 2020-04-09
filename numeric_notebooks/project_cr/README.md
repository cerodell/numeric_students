# EOSC 511 Final Project
#### Christopher Rodell

A simple model of wildland propagation using the Level-Set Method.

## Important
Ensure your working dirctiory has context.py within it.<br /> 

**Essesntial packadges** <br /> 
`import noise` <br />
https://anaconda.org/conda-forge/noise <br />
```conda install -c conda-forge noise``` <br />
*Noise is used to generate terrain* <br />




## How to Run
1)  Define initial conditions and domain set up in the namelist.yaml file <br />
    *Default values are provided*
 
2) Open run.py 
	- choose a plotting option *default is plot_main*
	- other plotting option are defined within run.py
	- hit run :) 





#### NOTE
I have included an abort command to the model when the CFL condition is violated. You can turn this feature on or off by setting cfl = "Yes" or "No"  in run.py

*Default is set to "Yes"*


approximator.py contains the following methods/attributes

    """
    A class used to model wildland fire rate of spread using the level set method

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
        3rd order Runge-Kutta

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