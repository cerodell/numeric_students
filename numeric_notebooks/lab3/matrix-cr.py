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
    matrix 𝐴 for a given value of N
    """
    A = np.zeros((N,N))
    for index in range(1,N-1):
        A[index,index] = -2
        A[index,index + 1] = A[index,index - 1] = 1

    A[0,0] = A[-1,-1] =1
    A2 = A[1:-1,1:-1]
    KA = np.linalg.cond(A)
    KA2 = np.linalg.cond(A2)

    return A, A2, KA, KA2

## Loop and calcualte the condition number (KA & KA2) for the matrix (A & A2)
N = np.arange(5,55,5)
A_list, A2_list, KA_list, KA2_list = [], [], [], []

for size in N:
    A, A2, KA, KA2 = Dirichlet(size)
    A_list.append(A)
    A2_list.append(A2)
    KA_list.append(KA)
    KA2_list.append(KA2)

    ## Print condtion number of matrix A
    print(f"Matrix A Size: {A.shape} Condition number: {KA}")



A_list[0]

# %% [markdown]

# - b\) Can you conjecture how the condition number (K) of 𝐴 depends on 𝑁?
# $$
# K=\|A\|\left\|A^{-1}\right\|
# $$
#  **As the size of the matrix increases the condition number
#  becomes larger. Meaning that the A becomes ill-condition 
#  as it grows in size (ie more difficult to solve).**


# %% 

## Plt condition number (K) v N size of matirx 𝐴 
fig,ax=plt.subplots(1,figsize=(16,8))
ax.set_title("Condition number (K) v N size of matirx A ")
# ax.scatter(N,KA_list, marker = '+', color = 'red')
ax.plot(N,KA_list, color = 'blue', lw = 1.5)
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
# **This change in the matrix makes a difference in the condition 
# number but it is not significant. It only slightly improves the 
# easy in solving the equaioiton.**
#

# %%
## Print condtion number of matrix A2 (the reduced matrix)
for size in range(len(N)):
    print(f"Matrix A2 (reduced) Size: {A2_list[size].shape} Condition number: {KA2_list[size]}")

# %% 

## Plt condition number (K) v N size of matirx A2 
fig,ax=plt.subplots(1,figsize=(16,8))
ax.set_title("Condition number (K) v N size of matirx A and A2 ")
ax.plot(N,KA_list, color = 'blue', lw = 1.5, label = "Matrix A")
ax.plot(N,KA2_list, color = 'red', lw = 1.5, label = "Matrix A2 (reduced)")
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
# %%

def Neumann(N):
    """
    This function that outputs a 
    matrix 𝐴 for a given value of N
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


## Loop and calcualte the condition number (KA & KA2) for the matrix (A & A2)
N = np.arange(5,55,5)
A3_list,  KA3_list = [], []

for size in N:
    A3, KA3 = Neumann(size)
    A3_list.append(A3)
    KA3_list.append(KA3)

    ## Print condtion number of matrix A
    print(f"Matrix A3 Size: {A3.shape} Condition number: {KA3}")

# %%

## Plt condition number (K) v N size of matirx A3 
fig,ax=plt.subplots(1,figsize=(16,8))
ax.set_title("Condition number (K) v N size of matirx A3 ")
ax.plot(N,KA_list, color = 'blue', lw = 1.5, label = "Matrix A")
ax.plot(N,KA2_list, color = 'red', lw = 1.5, label = "Matrix A2")
ax.plot(N,KA3_list, color = 'k', lw = 1.5, label = "Matrix A3")
ax.grid(True)
ax.legend()
ax.set(xlabel='N (size of one side of matrix)',ylabel='Condtion Number (K)')


# %%
