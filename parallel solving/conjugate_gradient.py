import numpy as np 
from typing import Tuple, List
from scipy import sparse
from numba import jit, njit
import time

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
     
    for i in range(n): 
        for j in range(indptr[i], indptr[i+1]): 
            Ax += data[j] * x[indices[j]]
    return Ax

@njit(prallel=True)
def pcg(A: sparse.csr_matrix, b: np.ndarray, M: sparse.csr_matrix, boundary_mat: np.ndarray, ny: int, nx: int, max_iter: int = 20000, tol: float=1e-10) -> Tuple[np.ndarray, List[float], float]: 

    A_value = A.data
    A_i = A.indices
    A_indptr = A.indptr 

    M_value = M.value
    M_i = M.indices
    M_indptr = M.indptr
    

    #initial guess
    x = np.zeros(A.shape[0])

    #initial residual 
    r = b - matvec_mul(A_value, A_i, A_indptr, x)

    #Apply preconditioner
    z = matvec_mul(M_value, M_i, M_indptr, r)
    p = z.copy()
    #initial values
    rz = dot_product(r, z)
    
    ini_residual_norm = norm(r)
    residual_history = [ini_residual_norm]

    start_time = time.time()

    for i in range(max_iter): 
        x = boundary_mat

        Ap = matvec_mul(A_value, A_i, A_indptr, p)

        pAp = dot_product(p, Ap)
        
        alpha = rz / pAp

        x +=  alpha * p 
        r -= alpha * Ap

        residual_norm = norm(r)
        residual_history.append(residual_norm)
        relative_residual = residual_norm / ini_residual_norm
        if relative_residual < tol: 
            break

        z = matvec_mul(M_value, M_i, M_indptr, r)


        #update search direction 
        rz_new = dot_product(r, z)
        beta = rz_new / rz 
        p = z + beta * p 
        rz = rz_new

        x = boundary_mat

    solve_time = time.time() - start_time*
    print(f"solve time: {solve_time} seconds")
    return x, residual_history, solve_time