import numpy as np 
from scipy import sparse
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict
import time 
from scipy.sparse.linalg import spilu 
from conjugate_gradient import cg 
from preconditioners import jacobi_preconditioner, ssor_preconditioner, block_jacobi_2d




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

def solver(A: sparse.csr_matrix, b: np.ndarray, nx: int, ny: int, precond_func, **precond_kwargs) -> Dict:

    M = precond_func(A.data, A.indices, A.indptr, **precond_kwargs)
    start_time = time.time()
    x, hist = cg(A.data, A.indices, A.indptr, b, M, nx, ny)
    solve_time = time.time() - start_time
    results = {
        'solution':  x.reshape(ny, nx),
        'history': hist, 
        'time' : solve_time
    }

    return results


def plot_comparison(results: Dict[str, Dict], nx: int, ny: int):
    fig = plt.figure(figsize=(15, 8))
    
    # Solution surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    first_method = list(results.keys())[0]
    surf = ax1.plot_surface(X, Y, results[first_method]['solution'],
                           cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax1.set_title('Solution Surface')
    fig.colorbar(surf, ax=ax1)

    # Convergence history plot
    ax2 = fig.add_subplot(122)
    styles = ['-', '--', ':', '-.']
    colors = ['b', 'r', 'g', 'm']
    
    for (method, data), style, color in zip(results.items(), styles, colors):
        ax2.semilogy(data['history'], linestyle=style, color=color, linewidth=2,
                     label=f"{method}\n({len(data['history'])} iter, {data['time']:.3f}s)")
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual Norm')
    ax2.set_title('Convergence History Comparison')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    
    start_time = time.time()
    print("Setting up problem...")
    nx, ny = 500, 500

    
    xmin, xmax = 0, 2
    ymin, ymax = 0, 1
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    print("*====================================*")
    print("*=====  Grid size: N = ",(nx, ny),"==*" )
    print("*=====  Step size: ",[dx]," =========*" )
    print("*====================================*")





    # Create source term (same as original problem)
    b = np.zeros(nx * ny)
    b[ny//4 * nx + nx//4] = 100
    b[3*ny//4 * nx + 3*nx//4] = -100
    
    # Solve with different methods
    # Solve with different preconditioners
    A = poisson2D_matrix(nx, ny, dx, dy)
    results = {
        'Jacobi': solver(A, b, nx, ny, jacobi_preconditioner),
        'SSOR': solver(A, b, nx, ny, ssor_preconditioner),
        'Block jacobi': solver(A, b, nx, ny, block_jacobi_2d),
    
    }   
    exec_time = time.time() - start_time
    # Plot comparisons
    plot_comparison(results, nx, ny)

    exec_time = time.time() - start_time
    
    print("execution finished in", exec_time)