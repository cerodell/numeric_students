# EOSC 511 Final Project
#### Christopher Rodell

A simple model of wildland propagation using the Level-Set Method.


## How to Run
1)  Define initial condition and domain set up in the namelist.yaml file *Default values are provided*
 
2) Open run.py 
	- choose a plotting option *default is plot_main*
	- other plotting option are defined within run.py
	- hit run :) 





NOTE
I have included an abort command to the model when the CFL condition is violated. You can turn this feature on or off by setting cfl = "Yes" or "No"  in run.py

*Default is set to "Yes"*

