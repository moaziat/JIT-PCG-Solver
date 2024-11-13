import numpy as np 
from typing import Tuple

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

