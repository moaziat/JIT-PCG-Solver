import numpy as np 
from numba import njit, prange 
from scipy import sparse 

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