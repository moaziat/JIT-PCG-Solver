import numpy as np 
from typing import Tuple, List
from scipy import sparse
from numba import jit, njit, prange
import time
from boundary_conditions import poisson2d_bc

@njit(parallel=True)
def dot_product(v1: np.ndarray, v2: np.ndarray) -> int: 
    return np.sum(v1 * v2)

@njit(parallel=True)
def norm(v: np.ndarray) -> int: 
    return np.sqrt(np.sum(v * v))

@njit(parallel=True)
def matvec_mul(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, x: np.ndarray) -> np.ndarray: 
    '''
    indptr maps the elements of data and indices to the rows of the sparse matrix
    '''

    n = len(indptr) - 1
    Ax = np.zeros(n)
    
    for i in prange(n): 
        row_sum = 0
        for j in range(indptr[i], indptr[i+1]): 
            row_sum += data[j] * x[indices[j]]
        Ax[i] = row_sum
    return Ax 




@njit(parallel=True)

def cg(A_value: np.ndarray, A_i: np.ndarray, A_indptr: np.ndarray, b: np.ndarray, M_data: np.ndarray, ny: int, nx: int, max_iter: int = 20000, tol: float=1e-10) -> Tuple[np.ndarray, List[float], float]: 
   
    n = len(A_indptr) - 1

    #initial guess
    x = np.zeros(n)
    x = poisson2d_bc(x, nx, ny)
    #initial residual 
    r = b - matvec_mul(A_value, A_i, A_indptr, x)

    #Apply preconditioner
    
    z = M_data * r
    p = z.copy()
    #initial values
    rz = dot_product(r, z)
    
    ini_residual_norm = norm(r)
    residual_history = [float(ini_residual_norm)]
    eps = 1e-15
    for i in range(max_iter): 
      

        Ap = matvec_mul(A_value, A_i, A_indptr, p)

        pAp = dot_product(p, Ap)
        
        if abs(pAp) < eps: 
            break 

        alpha = rz / pAp

        x +=  alpha * p 
        r -= alpha * Ap

        residual_norm = norm(r)
        residual_history.append(residual_norm)
        relative_residual = residual_norm / ini_residual_norm
        if relative_residual < tol: 
            break

        z = M_data * r


        #update search direction 
        rz_new = dot_product(r, z)
        if abs(rz) < eps:
            break
        beta = rz_new / rz 
        p = z + beta * p 
        rz = rz_new
        x = poisson2d_bc(x, ny, nx)

      
    return x, residual_history
