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
@njit 

def apply_preconditioner(M_data: np.ndarray, r: np.ndarray) -> np.ndarray:
    return M_data * r 
