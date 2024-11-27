import numpy as np 
from scipy import sparse
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict
import time 
from scipy.sparse.linalg import spilu 




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

def jacobi_preconditioner(A: sparse.csr_matrix) -> sparse.csr_matrix:
   
    return sparse.diags(1.0 / A.diagonal(), format='csr')

def SSOR_preconditioner(A: sparse.csr_matrix, omega: float=1.0) -> sparse.csc_matrix: 

    D = sparse.diags(A.diagonal(), format='csr')
    L = sparse.tril(A, k=-1, format='csr') 
    D_inv = sparse.diags(1.0/A.diagonal(), format='csr')
    M = (1.0 / (2.0-omega)) * (D/omega + L) @ D_inv @ (D/omega + L.T)
    return M

def ilu_preconditioner(A: sparse.csr_matrix, drop_tol: float = 0.01) -> sparse.csr_matrix:
    ilu = spilu(A, drop_tol=drop_tol)
    N = A.shape[0]
    M = sparse.linalg.LinearOperator((N, N), matvec=ilu.solve)
    return M

def boundary_conditions(p: np.ndarray, ny: int, nx: int) -> np.ndarray: 

    p[0:ny*nx:nx] = 0  
    p[nx-1:ny*nx:nx] = 0  
    p[0:nx] = 0  
    p[-nx:] = 0  
    return p

def pcg_poisson2D(A: sparse.csr_matrix, b: np.ndarray, M: sparse.csr_matrix,
                   ny: int, nx: int, max_iter: int = 20000, tol: float = 1e-10) -> Tuple[np.ndarray, List[float], float]:
    
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
    print("solve time", solve_time)
    return x, residual_history, solve_time

