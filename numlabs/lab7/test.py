
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# # Make the X, Y meshgrid instead of np.tile
# ngrid = 11
# H0 = 1000
# L = 1000e3


# # xs = np.linspace(0, 2*np.pi, ngrid)
# # ys = np.linspace(0, 2*np.pi, ngrid)

# xs = np.linspace(0, L, ngrid)
# ys = np.linspace(0, L, ngrid)

# Hu, Hv = np.meshgrid(xs, ys) 
# amp =400*np.sin((Hu+Hu)) + H0
# # %matplotlib 

# xs = np.linspace(0, L, ngrid)
# ys = np.linspace(0, L, ngrid)

# xx, yy = np.meshgrid(xs, ys) 

# fig = plt.figure()
# ax3d = fig.add_subplot(111, projection='3d')
# surf = ax3d.plot_surface(xx, yy, amp)

# zz = np.arange(0,97,1)
# for i in zz[0::6]:
#     print(zz[i])
# Make the X, Y meshgrid instead of np.tile
# ngrid = 11
# H0 = 1000

# xs = np.linspace(0, H0, ngrid)

# Hu = xs
# Hv = Huu*np.ones_like(Hu)
# Hu, Hv = np.meshgrid(Hu, Hv) 


# # Z evaluation
# amp = np.sin(Hu+Hv)
# %matplotlib 


# fig = plt.figure()
# ax3d = fig.add_subplot(111, projection='3d')
# surf = ax3d.plot_surface(tau, phi, amp)



ITERATION_LIMIT = 1000

# initialize the matrix
A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])
# initialize the RHS vector


b = np.array([[1., -3., 1., 2.],
              [-1., 1., -1., 2.],
              [8., -7., 9., -4.],
              [1., 2., -1., 5.]])
# prints the system
print("System:")
for i in range(A.shape[0]):
    row = ["{}*x{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
    print(" + ".join(row), "=", b[i])
print()

x = np.zeros_like(b)
for it_count in range(ITERATION_LIMIT):
    print("Current solution:", x)
    x_new = np.zeros_like(x)

    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x[:i])
        s2 = np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]

    if np.allclose(x, x_new, atol=1e-10, rtol=0.):
        break

    x = x_new

print("Solution:")
print(x.shape)
error = np.dot(A, x) - b
print("Error:")
print(error)


# def sor_solver(A, b, omega, initial_guess, convergence_criteria):
#     """
#     This is an implementation of the pseudo-code provided in the Wikipedia article.
#     Arguments:
#         A: nxn numpy matrix.
#         b: n dimensional numpy vector.
#         omega: relaxation factor.
#         initial_guess: An initial solution guess for the solver to start with.
#         convergence_criteria: The maximum discrepancy acceptable to regard the current solution as fitting.
#     Returns:
#         phi: solution vector of dimension n.
#     """
#     phi = initial_guess[:]
#     residual = np.linalg.norm(np.matmul(A, phi) - b) #Initial residual
#     while residual > convergence_criteria:
#         for i in range(A.shape[0]):
#             sigma = 0
#             for j in range(A.shape[1]):
#                 if j != i:
#                     sigma += A[i][j] * phi[j]
#             phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma)
#         residual = np.linalg.norm(np.matmul(A, phi) - b)
#         print('Residual: {0:10.6g}'.format(residual))
#     return phi


# # An example case that mirrors the one in the Wikipedia article
# residual_convergence = 1e-8
# omega = 0.5 #Relaxation factor

# A = np.ones((4, 4))
# A[0][0] = 4
# A[0][1] = -1
# A[0][2] = -6
# A[0][3] = 0

# A[1][0] = -5
# A[1][1] = -4
# A[1][2] = 10
# A[1][3] = 8

# A[2][0] = 0
# A[2][1] = 9
# A[2][2] = 4
# A[2][3] = -2

# A[3][0] = 1
# A[3][1] = 0
# A[3][2] = -7
# A[3][3] = 5

# b = np.ones(4)
# b[0] = 2
# b[1] = 21
# b[2] = -12
# b[3] = -6

# initial_guess = np.zeros(4)

# phi = sor_solver(A, b, omega, initial_guess, residual_convergence)
# print(phi.shape, 'shape')

