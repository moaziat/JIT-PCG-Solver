import numpy as np 
from scipy import sparse
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict
import time 
from scipy.sparse.linalg import spilu 
from conjugate_gradient import cg 



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

def solver(A: sparse.csr_matrix, b: np.ndarray, nx: int, ny: int, precond_func, **precond_kwargs) -> Dict:

    M = precond_func(A, **precond_kwargs)
    start_time = time.time()
    x, hist = cg(A.data, A.indices, A.indptr, b, M, nx, ny)
    solve_time = time.time() - start_time
    results = {
        'solution':  x.reshape(ny, nx),
        'history': hist, 
        'time' : solve_time
    }

    return results

def jacobi_preconditioner(A: sparse.csr_matrix) -> np.ndarray :
   return np.ones(A.shape[0])
if __name__ =="__main__":
    
    start_time = time.time()
    print("Setting up problem...")
    nx, ny = 500, 500
    xmin, xmax = 0, 2
    ymin, ymax = 0, 1
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    
    # Create source term (same as original problem)
    b = np.zeros(nx * ny)
    b[ny//4 * nx + nx//4] = 100
    b[3*ny//4 * nx + 3*nx//4] = -100
    
    # Solve with different methods
    # Solve with different preconditioners
    A = poisson2D_matrix(nx, ny, dx, dy)
    results = {
        'Jacobi': solver(A, b, nx, ny, jacobi_preconditioner),
    }
    exec_time = time.time() - start_time
    # Plot comparisons
    exec_time = time.time() - start_time
    print("execution finished in", exec_time)