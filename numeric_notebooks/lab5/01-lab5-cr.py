# %% [markdown]
# # Lab 5: Daisyworld
# ### Christopher Rordell
# ##### EOSC 511
# %% [markdown]
# ## Problem constant growth
# Note that though the daisy growth rate per unit time depends on the amount 
# of available fertile land, it is not otherwise coupled to the environment 
# (i.e. 𝛽𝑖 is note a function of temperature. Making the growth a function of
# bare ground, however, keeps the daisy population bounded and the daisy population 
# will eventually reach some steady state. The next python cell has a script that
# runs a fixed timestep Runge Kutte routine that calculates area coverage of white
# and black daisies for fixed growth rates 𝛽𝑤 and 𝛽𝑏. Try changing these growth 
# rates (specified in the derivs5 routine) and the initial white and black 
# concentrations (specified in the fixed_growth.yaml file discussed next).
#
# $$
# \\
# $$
#
# - 1\) For a given set of growth rates try various (non-zero) initial daisy populations.
# <img src="/Users/crodell/repos/numeric_students/numeric_notebooks/lab5/images/initial_daisy-populations_W2_B7.png">
# <img src="/Users/crodell/repos/numeric_students/numeric_notebooks/lab5/images//initial_daisy-initial_daisy-populations_W4_B2.png">
# $$
# \\
# $$
# **See Images Folder**
# $$
# \\
# $$
# - 2\) For a given set of initial conditions try various growth rates. In particular, try rates that are both greater than and less than the death rate.
# $$
# \\
# $$
# **See Images Folder**
# $$
# \\
# $$
# - 3\) Can you determine when non-zero steady states are achieved? Explain.
# $$
# \\
# $$
# **Yes, you can determine when a non-zero steady state occurs. 
# This happens when the slope of the function is zero at a 
# fractional value that is no zero. Another way to say this is..**
# $$
# \begin{aligned}
# &\frac{d A_{w}}{d t}=0\\
# &\frac{d A_{b}}{d t}=0
# \end{aligned}
# $$

# %% 
#
# 4.1  integrate constant growth rates with fixed timesteps
#
import context
from context import lab_dir
from numlabs.lab5.lab5_funs import Integrator
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


class Integ51(Integrator):
    def set_yinit(self):
        #
        # read in 'albedo_white chi S0 L albedo_black R albedo_ground'
        #
        uservars = namedtuple('uservars', self.config['uservars'].keys())
        self.uservars = uservars(**self.config['uservars'])
        #
        # read in 'whiteconc blackconc'
        #
        initvars = namedtuple('initvars', self.config['initvars'].keys())
        self.initvars = initvars(**self.config['initvars'])
        self.yinit = np.array(
            [self.initvars.whiteconc, self.initvars.blackconc])
        self.nvars = len(self.yinit)
        return None

    #
    # Construct an Integ51 class by inheriting first intializing
    # the parent Integrator class (called super).  Then do the extra
    # initialization in the set_yint function
    #
    def __init__(self, coeffFileName):
        super().__init__(coeffFileName)
        self.set_yinit()

    def derivs5(self, y, t):
        """y[0]=fraction white daisies
           y[1]=fraction black daisies

           Constant growty rates for white
           and black daisies beta_w and beta_b

           returns dy/dt
        """
        user = self.uservars
        #
        # bare ground
        #
        x = 1.0 - y[0] - y[1]

        # growth rates don't depend on temperature
        beta_b = 0.9  # growth rate for black daisies
        beta_w = 0.1  # growth rate for white daisies

        # create a 1 x 2 element vector to hold the derivitive
        f = np.empty([self.nvars], 'float')
        f[0] = y[0] * (beta_w * x - user.chi)
        f[1] = y[1] * (beta_b * x - user.chi)
        return f


theSolver = Integ51('fixed_growth.yaml')
timeVals, yVals, errorList = theSolver.timeloop5fixed()

plt.close('all')
# title_plot = 'initial_daisy-populations_W4_B2'
title_plot = 'growth_rate_W1_B9'

thefig, theAx = plt.subplots(1, 1)
thefig.suptitle(title_plot)
theLines = theAx.plot(timeVals, yVals)
theLines[0].set_marker('+')
theLines[1].set_linestyle('--')
theLines[1].set_color('k')
theLines[1].set_marker('*')
theAx.set_title('lab 5 interactive 1  constant growth rate')
theAx.set_xlabel('time')
theAx.set_ylabel('fractional coverage')
theAx.legend(theLines, ('white daisies', 'black daisies'), loc='best')
thefig.savefig(str(lab_dir)+'/images/' +str(title_plot))

thefig, theAx = plt.subplots(1, 1)
thefig.suptitle(title_plot)
theLines = theAx.plot(timeVals, errorList)
theLines[0].set_marker('+')
theLines[1].set_linestyle('--')
theLines[1].set_color('k')
theLines[1].set_marker('*')
theAx.set_title('lab 5 interactive 1 errors')
theAx.set_xlabel('time')
theAx.set_ylabel('error')
out = theAx.legend(theLines, ('white errors', 'black errors'), loc='best')
thefig.savefig(str(lab_dir)+'/images/error_' +str(title_plot))


# %% [markdown]

# ## Problem Coupling

# - 1\) For the current value of L (0.2) in the file coupling.yaml,
#  the final daisy steady state is zero. Why is it zero?
# $$
# \\
# $$
# **With L(0.2)  we see the final daisy steady-state to be zero because 
# there isn't enough solar energy reaching the surface of Dasiyword to make 
# it warm enough to allow growth for daisy. Meaning the intail Daisy on Daisyworld
#  all die off as the planet is too cold to sustain Daisylife.**
# $$
# \\
# $$
# - 2\) Find a value of L which leads to a non-zero steady state.
# $$
# \\
# $$
# **Its found that at L(0.6) is close to the lower limit where Daisyworld 
# reaches and non-zero steady state. L 0.6 allows just enough solar energy for the Daisys to grow.**
# $$
# \\
# $$
# - 3\) What happens to the emission temperature as L is varied? Make a plot of 𝐿 vs. 𝑇𝐸 for 10-15 values
#  of 𝐿. To do this, I overrode the value of L from the init file by passing a new value into the 
#  IntegCoupling constructor (see Appendix A). This allowed me to put
# $$
# \\
# $$
# **Referance Code Below**
# $$
# \\
# $$

# %%
import matplotlib.pyplot as plt


class IntegCoupling(Integrator):
    """rewrite the init and derivs5 methods to
       work with a single (grey) daisy
    """
    def set_yinit(self, newL):
                #
        # change the luminocity
        #
        self.config["uservars"]["L"] = newL # change solar incidence fraction
        #
        # make a new namedtuple factory called uservars that includes newL
        #
        uservars_fac = namedtuple('uservars', self.config['uservars'].keys())
        #
        # use the factory to make the augmented uservars named tuple
        #
        self.uservars = uservars_fac(**self.config['uservars'])
        #


        #
        # read in 'albedo_grey chi S0 L  R albedo_ground'
        #
        uservars = namedtuple('uservars', self.config['uservars'].keys())
        self.uservars = uservars(**self.config['uservars'])
        #
        # read in 'greyconc'
        #
        initvars = namedtuple('initvars', self.config['initvars'].keys())
        self.initvars = initvars(**self.config['initvars'])
        self.yinit = np.array([self.initvars.greyconc])
        self.nvars = len(self.yinit)
        return None

    def __init__(self, coeffFileName):
        super().__init__(coeffFileName)
        self.set_yinit()

    def __init__(self, coeffFileName, newL):
       super().__init__(coeffFileName)
       self.set_yinit(newL)

    def derivs5(self, y, t):
        """
           Make the growth rate depend on the ground temperature
           using the quadratic function of temperature

           y[0]=fraction grey daisies
           t = time
           returns f[0] = dy/dt
        """
        sigma = 5.67e-8  # Stefan Boltzman constant W/m^2/K^4
        user = self.uservars
        x = 1.0 - y[0]
        albedo_p = x * user.albedo_ground + y[0] * user.albedo_grey
        Te_4 = user.S0 / 4.0 * user.L * (1.0 - albedo_p) / sigma
        eta = user.R * user.L * user.S0 / (4.0 * sigma)
        temp_y = (eta * (albedo_p - user.albedo_grey) + Te_4)**0.25
        if (temp_y >= 277.5 and temp_y <= 312.5):
            beta_y = 1.0 - 0.003265 * (295.0 - temp_y)**2.0
        else:
            beta_y = 0.0

        # create a 1 x 1 element vector to hold the derivative
        f = np.empty([self.nvars], np.float64)
        f[0] = y[0] * (beta_y * x - user.chi)
        return f



# %%
import matplotlib.pyplot as plt

newL = np.arange(0,1,0.05)
theSolver = IntegCoupling("coupling.yaml", 0.2)
timeVals, yVals, errorList = theSolver.timeloop5fixed()

thefig, theAx = plt.subplots(1, 1)
theLines = theAx.plot(timeVals, yVals)
theAx.set_title('lab 5: interactive 2 Coupling with grey daisies')
theAx.set_xlabel('time')
theAx.set_ylabel('fractional coverage')
out = theAx.legend(theLines, ('grey daisies', ), loc='best')

# %%
