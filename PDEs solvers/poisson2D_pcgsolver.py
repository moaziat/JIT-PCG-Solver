import numpy as np 
from scipy import sparse
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict
import time 


"""

======= IMPORTANT =====================

The 2D Poisson equation is: ∂²p/∂x² + ∂²p/∂y² = b
I followed this course for discretization of this equation
Please check the following link:
https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/
    
"""

def poisson2D_matrix(nx: int, ny: int, dx: float, dy: float) -> sparse.csr_matrix: 

    '''
    The 2D poisson equation can be discretized as follows
    pn(i,j) =(pni+1,j+pni−1,j)Δy2+(pni,j+1+pni,j−1)Δx2−bni,jΔx2Δy22(Δx2+Δy2)
    the goal is to put it as sparse matrix
    '''
    #Dimentionality of the problem
    N = nx * ny

    #Diagonal entries 
    main_diag = -2.0 * (1.0/dx**2 + 1.0/dy**2) * np.ones(N) 

    x_diag = np.ones(N-1) / dx**2 
    #remove conncetions across x-boundary
    x_diag[np.arange(nx-1, N-1, nx)] = 0 
    y_diag = np.ones(N-nx) / dy**2 

    #create sparse matrix
    diagonals = [main_diag, x_diag, x_diag, y_diag, y_diag]
    offsets = [0, 1, -1, nx, -nx]
    A = sparse.diags(diagonals, offsets, format='csr')

    return A


def boundary_conditions(p: np.ndarray, ny: int, nx: int) -> np.ndarray: 

    p[0:ny*nx:nx] = 0  
    p[nx-1:ny*nx:nx] = 0  
    p[0:nx] = 0  
    p[-nx:] = 0  
    return p

def pcg_poisson2D(A: sparse.csr_matrix, b: np.ndarray, M: sparse.csr_matrix,
                   ny: int, nx: int, max_iter: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, List[float], float]:
    
    #initial guess x0 (here we use x as a notation)
    x = np.zeros(A.shape[0])

    #inital residual r0 = b - Ax0
    r = b - A @ x 

    #preconditioner
    z = M @ r
    p = z.copy()

   
    
    # Initial residual norm
    rz = r.dot(z)
    initial_residual_norm = np.sqrt(r.dot(r))
    
    residual_history = [initial_residual_norm]
    start_time = time.time()

    for i in range(max_iter):
        x = boundary_conditions(x, ny, nx)
        Ap = A @ p
        alpha = rz / p.dot(Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        residual_norm = np.sqrt(r.dot(r))
        residual_history.append(residual_norm)
        
        relative_residual = residual_norm / initial_residual_norm
        if relative_residual < tol:
            break
            
        z = M @ r
        rz_new = r.dot(z)
        beta = rz_new / rz
        
        p = z + beta * p
        rz = rz_new
        
        # Enforce boundary conditions
        x = boundary_conditions(x, ny, nx)
    
    solve_time = time.time() - start_time
    return x, residual_history, solve_time


def jacobi_preconditioner(A: sparse.csr_matrix) -> sparse.csr_matrix:
   
    N = A.shape[0]
    diag = A.diagonal()
    return sparse.diags(1.0 / diag, format='csr')


def solver(nx: int, ny: int, dx:float, dy:float, b: np.ndarray) -> Dict: 


    A = poisson2D_matrix(nx, ny, dx, dy)
    precond_M = jacobi_preconditioner(A)
    x, hist, solve_time = pcg_poisson2D(A, b, precond_M, nx, ny)
    
    results = {}
    results['PCG (Jacobi)'] = {
        'solution': x.reshape(ny, nx),
        'history': hist,
        'time': solve_time
    }

    return results

def plot_solver(results: Dict, nx: int, ny: int):

    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(15, 5))
    
    # Create mesh grid for surface plot
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, results['PCG (Jacobi)']['solution'], 
                           cmap='viridis',
                           edgecolor='none',
                           alpha=0.8)
    
    # Customize surface plot
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Solution')
    ax1.view_init(30, 225)
    ax1.set_title(f'Solution Surface\nSolve Time: {results["PCG (Jacobi)"]["time"]:.3f}s')
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.8, aspect=10)
    cbar.set_label('Value')
    
    # Convergence history plot
    ax2 = fig.add_subplot(122)
    ax2.semilogy(results['PCG (Jacobi)']['history'], 'b-', linewidth=2, label='PCG (Jacobi)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual Norm')
    ax2.set_title(f'Convergence History\n({len(results["PCG (Jacobi)"]["history"])} iterations)')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Problem setup
    print("Setting up problem...")
    nx, ny = 50, 50
    xmin, xmax = 0, 2
    ymin, ymax = 0, 1
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    
    # Create source term (same as original problem)
    b = np.zeros(nx * ny)
    b[ny//4 * nx + nx//4] = 100
    b[3*ny//4 * nx + 3*nx//4] = -100
    
    # Solve with different methods
    results = solver(nx, ny, dx, dy, b)
    
    # Plot comparisons
    plot_solver(results, nx, ny)
