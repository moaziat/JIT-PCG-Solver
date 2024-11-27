import numpy as np
from numba import njit

@njit(parallel=True)
def poisson2d_bc(p: np.ndarray, ny: int, nx: int) -> np.ndarray: 

    p[0:ny*nx:nx] = 0  
    p[nx-1:ny*nx:nx] = 0  
    p[0:nx] = 0  
    p[-nx:] = 0  
    return p
