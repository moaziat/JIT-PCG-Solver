import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy import sparse 
from preconditioned_cg import pcg, ssor_preconditioner
import time
#from tqdm import tqdm 
import matplotlib.pyplot as plt


'''
======= IMPORTANT =====================

The 2D Poisson equation is: ∂²p/∂x² + ∂²p/∂y² = b
I followed this course for discretization of this equation
Please check the following link:
https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/
    

'''




def sparse_poisson_pcg(p, dx, dy, b): 
    '''
    Putting the discretization of the PDE in matrix form we get
    diagonal = -2(dx**2 + dy**2) ==> coefficient of p[i, j]
    ex = dx**2 ==> coffecient of p[i +/- 1, j]
    ey = dy**2 ==>  coffecient of p[i, j +/- 1]
    '''


    nx, ny = p.shape
    N = nx * ny #total grid points
    
    #scaling coefficients to avoid numerical issues
    scale = dx*dy 
    diagonal = -2.0 * (dx**2 + dy ** 2 ) * np.ones(N) * scale
    ex = (1/dx**2) * np.ones(N-1) * scale 
    ey = (1/dy**2) * np.ones(N-nx) * scale

    #remove connections across boundary
    for i in range(nx-1, N-1, nx):
        ex[i] = 0

    diagonals = [diagonal, ex, ex, ey, ey]
    positions = [0, 1, -1, nx, -nx]

    #sparse matrix as an array
    A = sparse.diags(diagonals, positions, shape=(N, N)).toarray()

    M = ssor_preconditioner(A, omega=0.1)

    #adding small regulirization to diagonal of preconditioner 
    eps = 1e-8
    M += eps * np.eye(M.shape[0])



    p_vec = p.reshape(-1)
    b_vec = b.reshape(-1) * (dx**2 * dy**2)/(2 * (dx**2 + dy**2))
    p_new, residuals, converged = pcg( 
        A = A, 
        b = b_vec,
        M = M, 
        x0= p_vec,
        tol=1e-6, 
        max_iter=1000 
    )
    if not converged: 
        raise ValueError("PCG did not converge")

    #reshape back to 2D and apply boundary conditions
    p = p_new.reshape((ny, nx))

    #boundary conditions
    p[:, -1] = p[:, -2]
    p[0, :] = p[1, :]
    p[:, 0] = p[:, 1]
    p[-1, :] = 0

    return p, residuals

def cavity_flow_pcg(nt, u, v, dt, dx, dy, p, rho, nu):
    """Cavity flow solver using your PCG implementation"""
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros_like(p)
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p, residuals = sparse_poisson_pcg(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
        
        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        
        # Boundary conditions
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1    # lid velocity
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        
    return u, v, p

def build_up_b(b, rho, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

def cavity_flow_pcg(nt, u, v, dt, dx, dy, p, rho, nu):
    """
    Cavity flow solver using PCG. 
    The discretization, the boundary conditions are detailed in this jupyter notebook
    https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/14_Step_11.ipynb
    """
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros_like(p)
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p, residuals = sparse_poisson_pcg(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
        
        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        
        # Boundary conditions
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1    # lid velocity
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        
    return u, v, p


def plot_cavity_flow(u, v, p, 
                    dx=2/(41-1), dy=2/(41-1)):
    """
    Plot the cavity flow results
    
    Args:
        u: velocity in x direction
        v: velocity in y direction
        p: pressure
        dx, dy: grid spacing
    """
    nx, ny = u.shape[1], u.shape[0]
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot pressure field
    plt.subplot(131)
    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, p, cmap=cm.viridis)
    plt.title('Pressure field')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot velocity field (quiver plot)
    plt.subplot(132)
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    plt.title('Velocity field')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot streamlines
    plt.subplot(133)
    plt.streamplot(X, Y, u, v)
    plt.title('Streamlines')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    plt.show()
    
    # Plot velocity magnitude
    fig = plt.figure(figsize=(8, 6))
    velocity_magnitude = np.sqrt(u**2 + v**2)
    plt.contourf(X, Y, velocity_magnitude, alpha=0.5, cmap=cm.viridis)
    plt.colorbar(label='Velocity magnitude')
    plt.title('Velocity Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Set up parameters
    nx = 41
    ny = 41
    nt = 500
    nit = 50
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    
    rho = 1
    nu = .1
    dt = .001
    
    # Initial conditions
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    
    # Run cavity flow
    u, v, p = cavity_flow_pcg(nt, u, v, dt, dx, dy, p, rho, nu)
    
    # Plot results
    plot_cavity_flow(u, v, p)