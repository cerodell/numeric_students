# %% [markdown]

# # Laboratory 3: Linear Algebra
# ## Chris Rodell





# #### Problem Two: Condition number for Dirichlet problem
# Boundary condtions for a Dirichlet problem are:
# $$
# u(0)=u(1)=0
# $$
# *Solution of an ODE Using Linear Algebra*
# - a\) Using Python, compute the condition number for the matrix 
# 𝐴1 from Equation Differential System Matrix for several values of
#  𝑁 between 5 and 50. ( Hint: This will be much easier if you write 
# a small Python function that outputs the matrix 𝐴 for a given value
#  of 𝑁.)
# %% 

import numpy as np
import matplotlib.pyplot as plt

def Dirichlet(N):
    """
    This function that outputs a 
    matrix A & A2 for a given value of N
    using Dirichlet boundary condition.
    """
    A = np.zeros((N,N))
    for index in range(1,N-1):
        A[index,index] = -2
        A[index,index + 1] = A[index,index - 1] = 1

    A[0,0] = A[-1,-1] = 1
    A2 = A[1:-1,1:-1]
    KA = np.linalg.cond(A)
    KA2 = np.linalg.cond(A2)

    return A, A2, KA, KA2

## Loop and calcualte the condition number (KA & KA2) for the matrix (A & A2)
N = np.arange(5,55,5)
A_list, A2_list, KA_list, KA2_list = [], [], [], []
Ashape_list, A2shape_list = [], []
for size in N:
    A, A2, KA, KA2 = Dirichlet(size)
    A_list.append(A)
    A2_list.append(A2)
    KA_list.append(KA)
    KA2_list.append(KA2)
    Ashape_list.append(A.shape[0])
    A2shape_list.append(A2.shape[0])


    ## Print condtion number of matrix A
    print(f"Matrix A Size: {A.shape} Condition number: {KA}")



A_list[0]

# %% [markdown]

# - b\) Can you conjecture how the condition number (K) of 𝐴 depends on 𝑁?
# $$
# K=\|A\|\left\|A^{-1}\right\|
# $$
# **As the size of the matrix increases the condition number
# becomes larger. Meaning that the A becomes ill-condition 
# as it grows in size (ie more difficult to solve). 
# Another way you could say this is the range in the 
# eigenvalue is increasing making the condition number 
# large as the matrix become larger and more difficult to solve**



# %% 

## Plt condition number (K) v N size of matirx 𝐴 
fig,ax=plt.subplots(1,figsize=(16,8))
ax.set_title("Condition number (K) v N size of matirx")
# ax.scatter(N,KA_list, marker = '+', color = 'red')
ax.plot(Ashape_list,KA_list, color = 'blue', lw = 1.5)
ax.grid(True)
ax.set(xlabel='N (size of one side of matrix)',ylabel='Condtion Number (K)')


# %% [markdown]

# - c\) Another way to write the system of equations is to substitute 
# the boundary conditions into the equations, and thereby reduce size 
# of the problem to one of 𝑁−1 equations in 𝑁−1 unknowns. The corresponding
# matrix is simply the 𝑁−1 by 𝑁−1 submatrix of 𝐴1 from Equation 
# Differential System Matrix.
#
# Does this change in the matrix make a significant difference in the condition number?
# 
#
# **The change in the matrix makes a difference in the condition 
# number but it is not significant. It only slightly improves the 
# easy in solving the equation.**
#

# %%
## Print condtion number of matrix A2 (the reduced matrix)
for size in range(len(N)):
    print(f"Matrix A2 (reduced) Size: {A2_list[size].shape} Condition number: {KA2_list[size]}")

# %% 

## Plt condition number (K) v N size of matirx A2 
fig,ax=plt.subplots(1,figsize=(16,8))
ax.set_title("Condition number (K) v N size of matirx A and A2 ")
# ax.plot(np.log(Ashape_list),KA_list, color = 'blue', lw = 1.5, label = "Matrix A")
ax.plot(Ashape_list,KA_list, color = 'blue', lw = 1.5, label = "Matrix A")
ax.plot(A2shape_list,KA2_list, color = 'red', lw = 1.5, label = "Matrix A2 (reduced)")
ax.grid(True)
ax.legend()
ax.set(xlabel='N (size of one side of matrix)',ylabel='Condtion Number (K)')



# %% [markdown]

# #### Problem Three: Condition number for Neumann problem
# Boundary condtions for a discrete Neumann problem are:
# $$
# u_{-1}-2 u_{0}+u_{1}=d^{2} f_{0} \quad \text { and } \quad u_{N-1}-2 u_{N}+u_{N+1}=d^{2} f_{N}
# $$
#
#
# - a\) Derive the matrix corresponding to the linear system to be solved in both
#  of these cases.
# **SEE ATTACHED PNG IF NO DISPLAY**
# <img src='matrix_q3.png' width='60%' />
#
# %%

def Neumann(N):
    """
    This function that outputs a 
    matrix A3 for a given value of N
    using Neumann Boundary Conditions
    """
    A = np.zeros((N,N))
    for index in range(1,N-1):
        A[index,index] = -2
        A[index,index + 1] = A[index,index - 1] = 1

    A[0,0] = A[-1,-1] =1
    A3 = A[1:-1,1:-1]
    A3[0,1] = A3[-1,-2] = 2
    c = np.full(A3.shape[0],1)

    c = (c/N)
    A3 =np.vstack([A3,c])

    KA3 = np.linalg.cond(A3)

    return A3, KA3

# %% [markdown]
# - b\) How does the conditioning of the resulting matrix depend
# on the the size of the system?
# $$
# \\
# $$
# **The condition of the matrix becomes worse as the size of the matrix increase. 
# You can see this graph below. A (N) size increase the condition number grows
#  more and more NOTE it's not exponential was tested by plotting the log of the
#  condition number and size of array.**
#

# %% 
## Loop and calcualte the condition number (KA & KA2) for the matrix (A & A2)
N = np.arange(5,55,5)
A3_list,  KA3_list, A3shape_list = [], [], []

for size in N:
    A3, KA3 = Neumann(size)
    A3_list.append(A3)
    KA3_list.append(KA3)
    A3shape_list.append(A3.shape[0])

    ## Print condtion number of matrix A
    print(f"Matrix A3 Size: {A3.shape} Condition number: {KA3}")


# %% [markdown]
#
# - c\) Is it better or worse than for Dirichlet boundary conditions?
# $$
# \\
# $$
# ** This depends on when you are comparing to the Dirichlet boundary conditions
#  to the Newmann boundary condition. At a small size matrix, both the Dirichlet
#  boundary condition have a smaller (easier to solve) condition number than the
#  Newmann boundary condition. However, as the size of the matrix values increases
#  the eigenvalue range is less for the Newmann boundary condition (by this I mean
#  has a smaller boundary condition number).**
#
# %%

## Plt condition number (K) v N size of matirx A3 
fig,ax=plt.subplots(1,figsize=(16,8))
ax.set_title("Condition number (K) v N size of matirx A3 ")
ax.plot(Ashape_list,KA_list, color = 'blue', lw = 1.5, label = "Matrix A")
ax.plot(A2shape_list,KA2_list, color = 'red', lw = 1.5, label = "Matrix A2")
ax.plot(A3shape_list,KA3_list, color = 'k', lw = 1.5, label = "Matrix A3")
ax.grid(True)
ax.legend()
ax.set(xlabel='N (size of one side of matrix)',ylabel='Condtion Number (K)')



# %% [markdown]
# #### Problem Five: Consider a long hallway in an office building. If we assume that any
# cigarrette smoke, mixes across the width of the hallway and vertically
# through the depth of the hallway much faster than it mixes along the
# hallway, we can write the diffusion of cigaratte smoke as an equation
# where S is the concentration of smoke, κ is the rate of diffusion of
# smoke, γ is the rate at which the smoke sticks to the walls or otherwise
# leaves the system, α(x) is the sources of smoke, t is the time and x is
# distance along the hallway.
#
#
# - a\) Write the appropriate equation for the steady state.
# 
#
# $$
# \partial S / \partial t=\kappa \frac{\partial^{2} S}{\partial x^{2}}-\gamma S+\alpha(x)
# $$
#
# $$
# \frac{\partial S}{\partial t}=0\quad   \text{Steady State}
# $$
#
# $$
# 0 =\kappa \frac{\partial^{2} S}{\partial x^{2}}-\gamma S+\alpha(x)
# $$
#
#
# $$
# \kappa\left(\frac{S_{i+1}-2 S_{i}+S_{i-1}}{d^{2}}\right)-\gamma S+\alpha_{i}=0
# $$
#
# $$
# S_{i+1}-S_{i}\left(2+\frac{\gamma d^{2}}{\kappa}\right)+S_{i-1}=\frac{d^{2}}{\kappa} \alpha_{i}
# $$
#
# $$
# \gamma = 0\quad   \text{No Sink}
# $$
#
# $$
# S_{i+1}-2 S_{i}+S_{i-1}=\frac{d^{2}}{\kappa} \alpha_{i}
# $$
# $$
# \\
# \\
# $$
# - b\) Discretize the hall into N segments and write the equation for the
# steady state as a matrix equation.
# $$
# \\
# \\
# $$
# **SEE ATTACHED PNG IF NO DISPLAY**
# <img src='matrix_q5.png' width='60%' />
#
#
# %%

def NeumannCig(N, gamma, alpha, x):
    """
    This function that outputs a 
    matrix A4 for a given value of N
    using Neumann Boundary Conditions
    """
    kappa = 0.05 
    dx    = N * 20 
    

    A = np.zeros((N,N))
    for index in range(1,N-1):
        A[index,index] = -2 + ((gamma * dx**2)/kappa)
        A[index,index + 1] = A[index,index - 1] = 1

    A[0,0] = A[-1,-1] =1
    A4 = A[1:-1,1:-1]
    A4[0,1] = A4[-1,-2] = 2

    b = (dx**2/(kappa*alpha*x))
    c = np.full(A4.shape[0],b)


    KA4 = np.linalg.cond(A4)
    if gamma == 0:
        S4 = np.inf
    else:
        S4 = np.linalg.solve(A4,c)

    return A4, KA4, S4


# %% [markdown]
# - c\) Taking α(x) = 0.005δ(x∗) kg m−1s
# −1 where you can choose the
# point x∗, κ = 0.05m2s−1, γ = (3600 s)−1
# , find the solution for
# your choice of N between 5 and 15. Take the length of the hall as
# 20 m.

# %% 

## Loop and calcualte the solution for the matrix (A4)
## gamma = 1/3600
## alpha = 0.005
N     = np.arange(7,18,1)
gamma = (1/3600)
alpha = 0.005
x     = 4  ## Distance down hallway
A4_list,  KA4_list = [], []

for size in N:
    A4, KA4, S4 = NeumannCig(size, gamma, alpha, x)
    A4_list.append(A4)
    KA4_list.append(KA4)

    ## Print condtion number of matrix A
    print(f"Matrix A4 Size: {A4.shape} the solution: {S4[0]}")


# %% [markdown]

# - d\) What is the condition number of the matrix?

# %%

## Loop and calcualte the condition number (KA4) for the matrix (A4)
## gamma = 1/3600
## alpha = 0.005
N = np.arange(7,18,1)
gamma = (1/3600)
alpha = 0.005
A4_list,  KA4_list = [], []

for size in N:
    A4, KA4, S4 = NeumannCig(size, gamma, alpha, x)
    A4_list.append(A4)
    KA4_list.append(KA4)

    ## Print condtion number of matrix A
    print(f"Matrix A4 Size: {A4.shape} Condition number: {KA4}")

# %% [markdown]

# - e\) If γ is 0 what is the condition number of the matrix? Physically
# why is there no single solution?
# $$
# \\
# $$
# **With γ set equal to zero means there is no sink (ie smoke will 
# continue to accumulate). This is not physical because in real life the
#  walls are permeable and smoke will "sink" into the walls (material dependent
# of course, but regardless the smoke will escape somewhere).**

# %%


## Loop and calcualte the condition number (KA4) for the matrix (A4)
## gamma = 0
## alpha = 0
N = np.arange(7,18,1)
gamma = 0
alpha = 0.005
A4_list,  KA4_list = [], []

for size in N:
    A4, KA4, S4 = NeumannCig(size, gamma, alpha, x)
    A4_list.append(A4)
    KA4_list.append(KA4)

    ## Print condtion number of matrix A
    print(f"Matrix A4 Size: {A4.shape} Condition number: {KA4}")

# %% [markdown]

# - f\) If γ is 0 and α is 0, why physically is there no single solution?
# $$
# \\
# $$
# **To continue with the idea above (e) if there is no smoke leaving the
#  system smoke will continue to accumulate however if there is no alpha 
#  than there is no "input " or smoke being introduced to the system in the
#  first place! So smoke can not accumulate without an input...not physical **
# %%

## Loop and calcualte the condition number (KA4) for the matrix (A4)
## gamma = 0
## alpha = 0
N = np.arange(7,18,1)
gamma = 0
alpha = 0
A4_list,  KA4_list = [], []

for size in N:
    A4, KA4, S4 = NeumannCig(size, gamma, alpha, x)
    A4_list.append(A4)
    KA4_list.append(KA4)

    ## Print condtion number of matrix A
    print(f"Matrix A4 Size: {A4.shape} Condition number: {KA4}")



