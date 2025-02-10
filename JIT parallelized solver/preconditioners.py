import numpy as np
from numba import njit, prange

@njit(parallel=True)
def jacobi_preconditioner(A_value: np.ndarray, A_i: np.ndarray, A_indptr: np.ndarray) -> np.ndarray:
    #A_indptr: np.ndarray is an array of row pointers in csr format
    n = len(A_indptr) - 1
    M_data = np.ones(n, dtype=A_value.dtype)


    for i in prange(n): 

        row_start = A_indptr[i]
        row_end = A_indptr[i+1]
        
        for j in range(row_start, row_end):
            if A_i[j] == i: 
                if abs(A_value[j]) > 1e-14: 
                    M_data[i] = 1.0 / A_value[j]

                break
        if M_data[i] == 0: 
            M_data[i] == 1.0
    return M_data



@njit(parallel=True)
def ssor_preconditioner(A_value: np.ndarray, 
                       A_i: np.ndarray, 
                       A_indptr: np.ndarray,
                       omega: float = 1.0) -> np.ndarray:
    """
    Compute SSOR (Symmetric Successive Over-Relaxation) preconditioner.
    For stability, returns diagonal scaling that approximates SSOR effect.
    """
    n = len(A_indptr) - 1
    M_data = np.ones(n, dtype=A_value.dtype)
    
    # Compute modified diagonal elements
    for i in prange(n):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]
        
        diagonal = 0.0
        sum_lower = 0.0
        sum_upper = 0.0
        
        for j in range(row_start, row_end):
            col = A_i[j]
            if col < i:
                sum_lower += abs(A_value[j])
            elif col > i:
                sum_upper += abs(A_value[j])
            else:
                diagonal = A_value[j]
        
        # Compute scaling factor
        if abs(diagonal) > 1e-14:
            scaling = 1.0 / (abs(diagonal) + omega * (sum_lower + sum_upper))
            M_data[i] = scaling
            
    return M_data

@njit(parallel=True)
def block_jacobi_2d(A_value: np.ndarray, 
                   A_i: np.ndarray, 
                   A_indptr: np.ndarray) -> np.ndarray:
    """
    Simplified block Jacobi preconditioner for 2D problems.
    Returns a diagonal scaling that approximates block behavior.
    """
    n = len(A_indptr) - 1
    M_data = np.ones(n, dtype=A_value.dtype)
    
    for i in prange(n):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]
        
        diagonal = 0.0
        neighbor_sum = 0.0
        
        for j in range(row_start, row_end):
            if A_i[j] == i:
                diagonal = A_value[j]
            else:
                # Consider effect of neighboring points
                neighbor_sum += abs(A_value[j])
        
        if abs(diagonal) > 1e-14:
            # Use a more stable scaling factor
            M_data[i] = 1.0 / (diagonal + 0.1 * neighbor_sum)
            
    return M_data

@njit(parallel=True)
def apply_preconditioner(M_data: np.ndarray, r: np.ndarray) -> np.ndarray:
    return M_data * r 