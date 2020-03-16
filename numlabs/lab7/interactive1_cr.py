#!/usr/bin/env python
"""Interactive One.  Initialize a Gaussian bump in surface height and allow it evolve.  Note that under
rotational dynamics, the final steady state would be flow around the bump. If you chose the smaller domain size
even though the Coriolis force is still included, the scale is so small, its effect is negligible"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def interactive1(grid, ngrid, dt, T, small=False):  # return eta
    '''recommended values ngrid=11, dt=150, T=4*3600 (4 hours), small=False OR
    ngrid=11, dt=4, T=90, small=True
    '''
    
    stepper = {1: stepgrid1, 2: stepgrid2, 3: stepgrid3}
    find_depth = {1: find_depth1, 2: find_depth2, 3: find_depth3}

# define rotation speed, gravity and depth
    f = 1e-4
    g = 10
    H0 = 1000

# set up spatial scale L is total domain size
    L = 1000e3

# make the domain depth and width smaller if small is True
    if small:
        H0 = 10
        L = 1000
        
    dx = L/(ngrid-1)

    # z = np.arange(0,(L/2)*np.pi,(dx*10))   # start,stop,step
    # H0 = np.sin(z) + L
    # print(H0)

# write out dx/dt and sqrt(g*H) for comparison
    # print('dx/dt {0:.3f}\n'.format(dx/dt))
    # print('sqrt(g*H0) {0:.3f}\n'.format(np.sqrt(g*H0)))
    rdx = 1./dx

# set up Gaussian for forcing (its a bump in the center of the domain)
    x, y, spatial = findforcing(L, dx, ngrid)

# set up temporal scale T is total run time
    ntime = np.int(T/dt)

# initialize
    u, v, eta, up, vp, etap = initial(ngrid)

# set-up topography
    Hu, Hv = find_depth[grid](H0, ngrid)

####################################################################
################## Edits Made by Chirs Rodell#######################
####################################################################
    # main loop (leap-frog)
    u_list, v_list, eta_list, t_list = [], [], [], []
    for k in range(ntime):
        u, v, eta = stepper[grid](ngrid, f, g, Hu, Hv, dt, rdx, u, v, eta, up, vp, etap)
        # add forcing
        t = k * dt / 8640
        # print('forcing', t)

        eta = eta + 0.1 * (1 - np.exp(-t*t)) * spatial
        # periodic boundary conditions
        u, v, eta = periodicbc(ngrid, u, v, eta)

        # exchange values
        u, v, eta, up, vp, etap = exchange(u, v, eta, up, vp, etap)
        u_list.append(u), v_list.append(v), eta_list.append(eta), t_list.append(t)

    # plot contours
    # print('lenght of u list',len(u_list))
    lenght = range(len(u_list))
    for mplot in lenght[0::6]:
        plotit(grid, ngrid, dx, x, y, u_list[mplot], v_list[mplot], eta_list[mplot], t_list[mplot])

    return
####################################################################
####################################################################




def findforcing(L, dx, ngrid):
    '''define a two dimensional Gaussian of forcing with half-width L'''
    x = np.arange(0, L + 0.1*dx, dx)
    y = x # symmetrical case
    spatial = np.zeros((ngrid, ngrid))

    for i in range(ngrid):
        for j in range(ngrid):
            spatial[i,j] = np.exp(-((y[j] - L/2.) * (y[j] - L/2)
                                    +(x[i] - L/2.) * (x[i] - L/2))
                                    /(L*L / 48.))
    return x, y, spatial

def find_depth1(H0, ngrid):
    print ('Flat Bottom')
    Hu = H0*np.ones((ngrid, ngrid))
    Hv = H0*np.ones_like(Hu)
    return Hu, Hv

def find_depth2(H0, ngrid):
    print ('Flat Bottom')
    Hu = H0*np.ones((ngrid, ngrid))
    Hv = H0*np.ones_like(Hu)
    return Hu, Hv

####################################################################
################## Edits Made by Chirs Rodell#######################
####################################################################
def find_depth3(H0, ngrid):
    '''
    This function make a new terrian that is the shape 
    of a sine wave with 2 periods across the x domain
    '''
    from mpl_toolkits.mplot3d import axes3d
    print ('Sin wave Heights 1600 to 400')
    L = 1000e3

    xs = np.linspace(0, np.pi, ngrid)  ##one period sin wave
    ys = np.linspace(0, np.pi, ngrid)  ##one period sin wave
    # xs = np.linspace(0, L, ngrid)   ## Gental SLope
    # ys = np.linspace(0, L, ngrid)   ## Gental SLope

    xx, yy = np.meshgrid(xs, ys) 
    Hv = 600*np.sin(xx+xx) + H0
    Hu = 600*np.sin(xx+xx) + H0

    xL = np.linspace(0, L, ngrid)
    yL = np.linspace(0, L, ngrid)
    Lx, Ly = np.meshgrid(xL, yL) 
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_xlabel('$x$')  
    ax3d.set_ylabel('$y$')  
    ax3d.set_zlabel('$z$')   
    surf = ax3d.plot_surface(Lx, Ly, Hu)

    # print(Hu)


    return Hu, Hv
####################################################################
####################################################################

def initial(ngrid):
    '''initialize a ngrid x ngrid domain, u, v, and eta, all zero'''

    u = np.zeros((ngrid, ngrid))
    v = np.zeros_like(u)
    eta = np.zeros_like(u)

    up = np.zeros_like(u)
    vp = np.zeros_like(u)
    etap = np.zeros_like(u)

    return u, v, eta, up, vp, etap

def stepgrid1(ngrid, f, g, Hu, Hv, dt, rdx, u, v, eta, up, vp, etap):
    '''take a step forward using grid 1 (A)'''
    n = ngrid
    nm1 = ngrid-1
    nm2 = ngrid-2
    u[1:nm1, 1:nm1] = u[1:nm1, 1:nm1] + 2*dt * (f * vp[1:nm1, 1:nm1]
                                                - g * (etap[2:n, 1:nm1] - etap[0:nm2, 1:nm1]) * 0.5 * rdx)
    v[1:nm1, 1:nm1] = v[1:nm1, 1:nm1] + 2*dt * (-f * up[1:nm1, 1:nm1]
                                                - g *(etap[1:nm1,2:n] - etap[1:nm1, 0:nm2]) * 0.5 * rdx)
    eta[1:nm1, 1:nm1] = eta[1:nm1, 1:nm1] + 2*dt * (-(Hu[2:n, 1:nm1] * up[2:n, 1:nm1]
                                                      - Hu[0:nm2, 1:nm1] * up[0:nm2, 1:nm1]) * 0.5 * rdx
                                                    -(Hv[1:nm1, 2:n] * vp[1:nm1, 2:n]
                                                      - Hv[1:nm1, 0:nm2] * vp[1:nm1, 0:nm2]) * 0.5 * rdx)
    return u, v, eta

def stepgrid2(ngrid, f, g, Hu, Hv, dt, rdx, u, v, eta, up, vp, etap):
    '''take a step forward using grid 2 (B)'''
    n = ngrid
    nm1 = ngrid-1
    nm2 = ngrid-2
    u[1:nm1, 1:nm1] = u[1:nm1, 1:nm1] + 2*dt * (f * vp[1:nm1, 1:nm1]
                                                 -g * (etap[2:n, 1:nm1] - etap[1:nm1, 1:nm1]
                                                        + etap[2:n, 2:n] - etap[1:nm1, 2:n]) * 0.5 * rdx)
    v[1:nm1, 1:nm1] = v[1:nm1, 1:nm1] + 2*dt * (-f*up[1:nm1, 1:nm1]
                                                 - g * (etap[1:nm1, 2:n] - etap[1:nm1, 1:nm1]
                                                      + etap[2:n, 2:n] - etap[2:n, 1:nm1]) * 0.5 * rdx)
    eta[1:nm1, 1:nm1] = eta[1:nm1, 1:nm1] + 2*dt * (-(Hu[1:nm1, 1:nm1] * up[1:nm1, 1:nm1]
                                                      - Hu[0:nm2, 1:nm1] * up[0:nm2, 1:nm1]
                                                         + Hu[1:nm1, 0:nm2] * up[1:nm1, 0:nm2]
                                                         - Hu[0:nm2, 0:nm2] * up[0:nm2, 0:nm2]
                                                         + Hv[1:nm1, 1:nm1] * vp[1:nm1, 1:nm1]
                                                         - Hv[1:nm1, 0:nm2] * vp[1:nm1, 0:nm2]
                                                         + Hv[0:nm2, 1:nm1] * vp[0:nm2, 1:nm1]
                                                         - Hv[0:nm2, 0:nm2] * vp[0:nm2, 0:nm2]) * 0.5 * rdx)
    return u, v, eta

def stepgrid3(ngrid, f, g, Hu, Hv, dt, rdx, u, v, eta, up, vp, etap):
    '''take a step forward using grid 3 (C)'''
    n = ngrid
    nm1 = ngrid-1
    nm2 = ngrid-2
    vmid = np.zeros_like(u); umid = np.zeros_like(u) 
    vmid[1:nm1, 1:nm1] = (vp[1:nm1, 1:nm1] + vp[2:n, 1:nm1] + vp[1:nm1, 0:nm2] + vp[2:n, 0:nm2]) * 0.25
    umid[1:nm1, 1:nm1] = (up[1:nm1, 1:nm1] + up[1:nm1, 2:n] + up[0:nm2, 1:nm1] + up[0:nm2, 2:n]) * 0.25
    u[1:nm1, 1:nm1] = u[1:nm1, 1:nm1] + 2*dt * (f * vmid[1:nm1, 1:nm1]
                                                - g * (etap[2:n, 1:nm1] - etap[1:nm1, 1:nm1]) * rdx)
    v[1:nm1, 1:nm1] = v[1:nm1, 1:nm1] + 2*dt * (-f * umid[1:nm1, 1:nm1]
                                                - g *(etap[1:nm1, 2:n] - etap[1:nm1, 1:nm1]) * rdx)
    eta[1:nm1, 1:nm1] = eta[1:nm1, 1:nm1] + 2*dt * (-(Hu[1:nm1, 1:nm1] * up[1:nm1, 1:nm1]
                                                      - Hu[0:nm2, 1:nm1] * up[0:nm2, 1:nm1]) * rdx
                                                    -(Hv[1:nm1, 1:nm1] * vp[1:nm1, 1:nm1]
                                                      - Hv[1:nm1, 0:nm2] * vp[1:nm1, 0:nm2])* rdx)
    return u, v, eta

def periodicbc(ngrid, u, v, eta):
    '''do periodic boundary conditions'''
    eta[0, :] = eta[-2, :]
    eta[-1, :] = eta[1, :]
    eta[:, 0] = eta[:, -1]
    eta[:, -1] = eta[:, 1]

    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -1]
    u[:, -1] = u[:, 1]

    v[0, :] = v[-2, :]
    v[-1, :] = v[1, :]
    v[:, 0] = v[:, -1]
    v[:, -1] = v[:, 1]

    return u, v, eta

def exchange(u, v, eta, up, vp, etap):
    '''swap new and old values'''
    store = np.zeros_like(vp)  # make sure store is its own array, not just another name for vp
    store = vp
    vp = v
    v = store
    store = up
    up = u
    u = store
    store = etap
    etap = eta
    eta = store

    return u, v, eta, up, vp, etap

def plotit(grid, ngrid, dx, x, y, u, v, eta, time):
    '''Contour plots of u, v, eta'''

    shift = {1: (0, 0), 2: (0.5, 0.5), 3: (0.5, 0)}
    
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    fig.suptitle('forcing ' + str(round(time, 5)), fontsize=14)
    for i in range(2):
        ax[1,i].set_xlabel('x (km)')
        ax[i,0].set_ylabel('y (km)')
    ax[0,0].set_title('$\eta$')
    ax[0,1].set_title('velocity')
    ax[1,0].set_title('u')
    ax[1,1].set_title('v')
    ax[0,0].contour(x/1000, y/1000, eta.transpose())
    ax[1,0].contour((x + shift[grid][0] * dx)/1000,
                  (y + shift[grid][1] * dx)/1000, u.transpose())
    ax[1,1].contour((x + shift[grid][1] * dx)/1000,
                  (y + shift[grid][0] * dx)/1000, v.transpose())

    if grid == 3:
        ax[0,1].quiver(x[1:]/1000., y[1:]/1000,
                       0.5 * (u[1:,1:] + u[:-1,1:]).transpose(),
                       0.5 * (v[1:,1:] + v[1:,:-1]).transpose())
    else:
        ax[0,1].quiver(x/1000, y/1000, u.transpose(), v.transpose())
 


