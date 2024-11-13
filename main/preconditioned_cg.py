import numpy as np 
from typing import Tuple
from scipy import sparse
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import spsolve

def pcg(A: np.ndarray,
        b: np.ndarray, 
        M: np.ndarray, 
        x0: np.ndarray = None,
        tol: float = 1e-10, 
        max_iter: int = 1000) -> Tuple[np.ndarray, list, bool]: 
    
    n = len(b)
    if x0 is None: 
        x = np.zeros(n)
    else: 
        x = np.copy(x0)

    r = b - A @ x

    #solve My0 = r0 for y0
    y = np.linalg.solve(M, r)

    #initial search direction
    p = y.copy()

    residual_history = [np.linalg.norm(r)]
    initial_residual_norm = residual_history[0]
    

    for i in range(max_iter): 


        Ap = A @ p
        ry = r @ y
        pAp = p @ Ap
        eps = 1e-8
        if abs(pAp) < eps:  # Check for near-zero denominator
            print(f"Warning: Small pAp encountered at iteration {i}")
            break
        optim_step_size = ry / pAp
        
        x = x + optim_step_size * p 

        r = r - optim_step_size * Ap

        residual_norm = np.linalg.norm(r)
        residual_history.append(residual_norm)
        
        relative_residual = residual_norm / initial_residual_norm
        if relative_residual < tol:
            return x, residual_history, True
        
        y_new = np.linalg.solve(M, r)

        ry_new = r @ y_new
        beta = ry_new / ry

        p = y_new + beta * p 

        y = y_new
    return x, residual_history, False

def jacobi_preconditioner(A: np.ndarray) -> np.ndarray: 

    return np.diag(np.diag(A))

def ssor_preconditioner(A: np.ndarray, omega: float = 1.0) -> np.ndarray:

    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    
    D_inv = np.diag(1.0 / np.diag(A))
    #add regu to avoid division by 0
    eps = 1e-8 

    M = (1 / (2 - omega)) * (D/omega + L) @ D_inv @ (D/omega + L.T)
    return M

def block_jacobi_preconditioner(A: np.ndarray, block_size: int = 4) -> np.ndarray:

    n = A.shape[0]
    M = np.zeros_like(A)
    
    for i in range(0, n, block_size):
        end_idx = min(i + block_size, n)
        block = A[i:end_idx, i:end_idx]
        M[i:end_idx, i:end_idx] = block
        
    return M

def approximate_inverse_preconditioner(A: np.ndarray, drop_tol: float = 0.1) -> np.ndarray:

    n = A.shape[0]
    M = np.eye(n)
    A_sparse = sparse.csr_matrix(A)
    
    # Create sparse approximate inverse
    ilu = spilu(A_sparse, drop_tol=drop_tol)
    M = ilu.solve(np.eye(n))
    
    return M