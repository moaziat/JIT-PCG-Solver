import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy import sparse 
from preconditioned_cg import pcg, ssor_preconditioner,jacobi_preconditioner
import time
#from tqdm import tqdm 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve



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
    scale = 1.0 
    diagonal = -2.0 * (dx**2 + dy ** 2 ) * np.ones(N) 
    ex = (1/dx**2) * np.ones(N-1) 
    ey = (1/dy**2) * np.ones(N-nx) 

    #remove connections across boundary
    for i in range(nx-1, N-1, nx):
        ex[i] = 0

    diagonals = [diagonal, ex, ex, ey, ey]
    positions = [0, 1, -1, nx, -nx]

    #sparse matrix as an array
    A = sparse.diags(diagonals, positions, format='csr')
    A_dense = A.toarray()
    M = ssor_preconditioner(A_dense, omega=0.5)

    #adding small regulirization to diagonal of preconditioner 
    eps = 1e-8
    M += eps * np.eye(M.shape[0])



    p_vec = p.reshape(-1)
    b_vec = b.reshape(-1) 
    #initial guess
    x0 = p.reshape(-1)

    try:
        p_new = spsolve(A, b_vec)
        p = p_new.reshape((ny, nx))
        
        # Apply boundary conditions
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0
        
        return p, []

    except Exception as e:
        print(f"Sparse direct solve failed: {e}")
        print("Solving with PCG...")

    try:
        A_dense = A.toarray()
        p_new, residuals, converged = pcg( 
            A = A_dense, 
            b = b_vec,
            M = M, 
            x0= p_vec,
            tol=1e-6, 
            max_iter=1000 
        )
        if not converged: 
            raise ValueError("PCG did not converge")
    except Exception as e:
        print(f"PCG failed: {e}")
        print("Using previous solution")
        return p, []
    
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
    """Build the RHS of pressure equation with safety checks"""
    
    # Compute terms separately for better control
    du_dx = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
    dv_dy = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
    
    du_dy = (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy)
    dv_dx = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)
    
    # Clip values to prevent overflow
    max_val = 1e10
    du_dx = np.clip(du_dx, -max_val, max_val)
    dv_dy = np.clip(dv_dy, -max_val, max_val)
    du_dy = np.clip(du_dy, -max_val, max_val)
    dv_dx = np.clip(dv_dx, -max_val, max_val)
    
    b[1:-1, 1:-1] = rho * (
        1/dt * (du_dx + dv_dy) -
        du_dx**2 -
        2 * du_dy * dv_dx -
        dv_dy**2
    )
    
    return b
def cavity_flow_pcg(nt, u, v, dt, dx, dy, p, rho, nu, plot_interval=100):
    """Cavity flow solver with stability checks"""
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros_like(p)
    max_val = 1.0
    # CFL condition check
    cfl = dt * max(np.max(np.abs(u))/dx, np.max(np.abs(v))/dy)
    if cfl > 1.0:
        print(f"Warning: CFL condition not met. CFL = {cfl}")
        dt_suggested = 0.5 * min(dx, dy) / max(np.max(np.abs(u)), np.max(np.abs(v)))
        print(f"Suggested dt: {dt_suggested}")
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p_old = p.copy()
        p, residuals = sparse_poisson_pcg(p, dx, dy, b)
        
        # Check for pressure solve failure
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            print(f"Warning: Invalid pressure values at step {n}")
            p = p_old
            continue
            
        # Update u velocity
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         np.clip(un[1:-1, 1:-1] * dt / dx, -max_val, max_val) *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         np.clip(vn[1:-1, 1:-1] * dt / dy, -max_val, max_val) *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
        
        # Update v velocity
        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        np.clip(un[1:-1, 1:-1] * dt / dx, -max_val, max_val) *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        np.clip(vn[1:-1, 1:-1] * dt / dy, -max_val, max_val) *
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
        
        # Clip velocities to prevent instability
        u = np.clip(u, -max_val, max_val)
        v = np.clip(v, -max_val, max_val)
        
        if n % 10 == 0:
            print(f"Step {n}/{nt}")
            # Monitor maximum values
            print(f"Max u: {np.max(np.abs(u)):.2e}, Max v: {np.max(np.abs(v)):.2e}")
            print(f"Max p: {np.max(np.abs(p)):.2e}")
    
    return u, v, p

    """
    Cavity flow solver using PCG. 
    The discretization, the boundary conditions are detailed in this jupyter notebook
    https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/14_Step_11.ipynb
    """


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
    plt.contourf(X, Y, p, alpha=0.5, cmap='viridis')
    plt.colorbar()
    plt.contour(X, Y, p, cmap='viridis')
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
    plt.contourf(X, Y, velocity_magnitude, alpha=0.5, cmap='viridis')
    plt.colorbar(label='Velocity magnitude')
    plt.title('Velocity Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    # Adjusted parameters for better stability
    nx = 41
    ny = 41
    nt = 500
    
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    
    rho = 1.0
    nu = 0.1
    
    # Compute stable dt based on CFL condition
    dt = 0.25 * min(dx*dx/nu, dy*dy/nu)  # Conservative time step
    print(f"Using dt = {dt}")
    
    # Initial conditions with smooth startup
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    
    # Smooth startup for lid velocity
    x = np.linspace(0, 1, nx)
    u[-1, :] = 16 * x**2 * (1-x)**2  # Smooth polynomial
    
    try:
        print("Starting cavity flow simulation...")
        u, v, p = cavity_flow_pcg(nt, u, v, dt, dx, dy, p, rho, nu)
        print("Simulation completed successfully")
        plot_cavity_flow(u, v, p)
    except Exception as e:
        print(f"Simulation failed: {str(e)}")