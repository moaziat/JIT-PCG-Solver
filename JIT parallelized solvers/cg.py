import numpy as np 
from typing import Tuple, List
from scipy import sparse
import time

def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.sum(v1 * v2)

def norm(v: np.ndarray) -> float:
    return np.sqrt(np.sum(v * v))

def matvec_mul(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Sparse matrix-vector multiplication in CSR format
    """
    n = len(indptr) - 1
    Ax = np.zeros(n)
    for i in range(n):
        for j in range(indptr[i], indptr[i+1]):
            Ax[i] += data[j] * x[indices[j]]
    return Ax

def pcg(A: sparse.csr_matrix, b: np.ndarray, M: np.ndarray = None, 
        max_iter: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, List[float], float]:
    """
    Preconditioned Conjugate Gradient solver
    
    Parameters:
    -----------
    A : sparse.csr_matrix
        System matrix (should be symmetric positive definite)
    b : np.ndarray
        Right-hand side vector
    M : np.ndarray, optional
        Preconditioner (diagonal matrix stored as 1D array)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance for relative residual
        
    Returns:
    --------
    x : np.ndarray
        Solution vector
    residual_history : list
        History of residual norms
    solve_time : float
        Total solve time in seconds
    """
    start_time = time.time()
    
    # Initialize
    n = A.shape[0]
    x = np.zeros(n)
    
    # Initial residual
    r = b - matvec_mul(A.data, A.indices, A.indptr, x)
    
    # Apply preconditioner if provided, otherwise use identity
    if M is None:
        M = np.ones(n)
    
    # Apply preconditioner
    z = M * r
    p = z.copy()
    
    # Initial values
    rz = dot_product(r, z)
    initial_residual = norm(r)
    residual_history = [initial_residual]
    
    for i in range(max_iter):
        # Matrix-vector product
        Ap = matvec_mul(A.data, A.indices, A.indptr, p)
        
        # Step length
        alpha = rz / dot_product(p, Ap)
        
        # Update solution and residual
        x += alpha * p
        r -= alpha * Ap
        
        # Check convergence
        residual_norm = norm(r)
        residual_history.append(residual_norm)
        
        relative_residual = residual_norm / initial_residual
        if relative_residual < tol:
            break
        
        # Apply preconditioner
        z = M * r
        
        # Update conjugate direction
        rz_new = dot_product(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new