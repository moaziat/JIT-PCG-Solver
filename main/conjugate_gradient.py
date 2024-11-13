import numpy as np 
from scipy import sparse
from typing import Tuple

def conjugate_gradient(A: sparse.csr_matrix, b: np.ndarray, x0: np.ndarray = None, tol: float = 1e-10, max_iter: int = 1000) -> Tuple[np.ndarray, list, list]: 

    #dim of A 
    n = A.shape[0]

    #Assure that the dimentionality of the problem
    if A.shape != (n, n): 
        raise  ValueError("dim of must be (n, n)")
    if b.shape != (n,): 
        raise ValueError("dim of b must be 1D-array; dim(b) = n")
    
    #Initial guess
    if x0 is None:
        x = np.zeros(n)
    else: 
        x = x0.copy()

    #Initial residual r0 = b - Ax0
    r = b - A @ x

    #Initial search direction
    p = r.copy()

    # initial residual norm for relative criterion 
    r_initial_norm = np.sqrt(r.dot(r))
    #initial residual norm squared
    r_norm_sq = r.dot(r)


    for i in range(max_iter):
        
        Ap = A @ p

        #optimal step size
        opt_step_size = r_norm_sq / p.dot(Ap)

        #update solution : takes step of the size of optimal step in direction p
        x = x + opt_step_size * p

        #update residual
        r = r - opt_step_size * Ap
        r_norm_sq_new = r.dot(r)

        '''
        Checking convergence using relative residual criterion 
        ||r(i)|| <= tol * ||r(0)|| e.g ||r(i)||/||r(0)|| <= tol
        '''
        
        relative_residual = np.sqrt(r_norm_sq_new) / r_initial_norm

        if relative_residual <= tol: 
            return x, i, relative_residual


       

        #conjugacy factor        
        conj_factor =  r_norm_sq_new/ r_norm_sq

        #update search direction
        p = r + conj_factor * p 

        #update for new iter
        r_norm_sq = r_norm_sq_new

    return x, max_iter, relative_residual



    
