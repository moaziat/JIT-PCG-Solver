import numpy as np 
from scipy import sparse
from typing import Tuple

def conjugate_gradient(A: sparse.csr_matrix, b: np.ndarray, x0: np.ndarray = None, tol: float = 1e-10, max_iter: int = 1000) -> Tuple[np.ndarray, list, list]: 

    #dim of A 
    n = A.shape[0]

    #Initial guess
    if x0 is None:
        x = np.zeros(n)
    else: 
        x = x0.copy()

    #Initial residual r0 = b - Ax0
    r = b - A @ x

    #Initial search direction
    p = r.copy()

    #initial residual norm squared
    r_norm_sq = r.dot(r)


    for i in range(max_iter):
        
        Ap = A @ p

        #optimal step size
        opt_step_size = r_norm_sq / p.dot(Ap)

        #update solution : takes step of the size of optimal step in direction p
        x = x + opt_step_size @ p

        #update residual
        r = r - opt_step_size @ Ap
        new_norm_sq = r.dot(r)

        #conjugacy factor        
        conj_factor =  new_norm_sq/ r_norm_sq

        #update search direction
        p = r + conj_factor * p 

        #update for new iter
        r_norm_sq = new_norm_sq 



    
