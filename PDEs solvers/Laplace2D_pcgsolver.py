import numpy as np 
from scipy import sparse
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict
import time 
from scipy.sparse.linalg import spilu 


def laplace2D_matrix(nx: int, ny: int, dx: float, dy: float) -> sparse.csr_matrix: 

    """
    Laplace's equation in 2D: d2p/dx2 + d2p/dy2 = 0
    Using centered finite difference
    Discretization ==>  (2/dx**2 + 2/dy**2)p(i,j) - 1/dx**2(p(i+1, j) + p(i-1,j)) - 1/dy**2(p(i,j+1), p(i,j-1)) = 0
    
    """
    N = nx * ny #dim of the grid
    
    #main diagonal; A(k,k) = 2/dx**2 + 2/dy**2 
    main_diag = 2.0 * (1.0/dx**2 + 1.0/dy**2) * np.ones(N)

    #off-diagonal in the x-direction A(k,k+1) = A(k, k-1) = -1/dx**2 
    x_diag = (-1.0/dx**2) * np.ones(N-1)

    #off-diagonal in the y-direction A(k+1,k) = A(k-1, k) = -1/dy**2
    y_diag = (-1.0/dy**2) * np.ones(N-nx)

    #remove connections across x-boundary; neighbors to (i, 0) and (i, nx-1) does not exist 
    x_diag[np.arange(nx - 1, N-1, nx)] = 0

    diagonals = [main_diag, x_diag, x_diag, y_diag, y_diag]

    
    offsets= [0, 1, -1, nx, -nx]
    A = sparse.diags(diagonals, offsets, format='csr')

    return A   

def boundary_conditions(p: np.ndarray, nx: int, ny: int, y: np.ndarray) -> np.ndarray: 

    p_2d = p.reshape(ny, nx)
    
    # Dirichlet conditions on x-boundaries
    p_2d[:, 0] = 0  # p = 0 @ x = 0
    p_2d[:, -1] = y  # p = y @ x = 2
    
    # Neumann conditions on y-boundaries (dp/dy = 0)
    p_2d[0, :] = p_2d[1, :]  # dp/dy = 0 @ y = 0
    p_2d[-1, :] = p_2d[-2, :]  # dp/dy = 0 @ y = 1
    
    return p_2d.flatten()

def pcg_laplace2D(A: sparse.csr_matrix, b: np.ndarray, M: sparse.csr_matrix,
                   ny: int, nx: int, y: np.ndarray, max_iter: int = 20000, tol: float = 1e-10) -> Tuple[np.ndarray, List[float], float]:
    
    #initial guess x0 (here we use x as a notation)
    x = np.zeros(A.shape[0])
    x = boundary_conditions(x, ny, nx, y)

    #inital residual r0 = b - Ax0
    r = b - A @ x 

    #preconditioner
    z = M @ r
    p = z.copy()

   
    
    # Initial residual norm
    rz = r.dot(z)
    initial_residual_norm = np.sqrt(r.dot(r))
    
    residual_history = [initial_residual_norm]
    start_time = time.time()

    for i in range(max_iter):
        x = boundary_conditions(x, ny, nx, y)
        Ap = A @ p
        alpha = rz / p.dot(Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        residual_norm = np.sqrt(r.dot(r))
        residual_history.append(residual_norm)
        
        relative_residual = residual_norm / initial_residual_norm
        if relative_residual < tol:
            break
            
        z = M @ r
        rz_new = r.dot(z)
        beta = rz_new / rz
        
        p = z + beta * p
        rz = rz_new
        
        # Enforce boundary conditions
        
    
    solve_time = time.time() - start_time
    return x, residual_history, solve_time


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
#def appro_inv_preconditioner(A: sparse.csr_matrix, drop_tol: float=0.1) -> sparse.csr_matrix:
    
def solver(A: sparse.csr_matrix, b: np.ndarray, nx: int, ny: int, y: np.ndarray, precond_func, **precond_kwargs) -> Dict: 


    
    M = precond_func(A, **precond_kwargs)
    x, hist, solve_time = pcg_laplace2D(A, b, M, nx, ny, y)
    
    results = {}
    results = {
        'solution': x.reshape(ny, nx),
        'history': hist,
        'time': solve_time
    }

    return results

def plot_comparison(results: Dict[str, Dict], nx: int, ny: int):
    fig = plt.figure(figsize=(15, 8))
    
    # Solution surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    first_method = list(results.keys())[0]
    surf = ax1.plot_surface(X, Y, results[first_method]['solution'],
                           cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Solution')
    ax1.view_init(30, 225)
    ax1.set_title('Solution Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.8, aspect=10)

    # Convergence history plot
    ax2 = fig.add_subplot(122)
    styles = ['-', '--', ':', '-.']
    colors = ['b', 'r', 'g', 'm']
    
    for (method, data), style, color in zip(results.items(), styles, colors):
        ax2.semilogy(data['history'], linestyle=style, color=color, linewidth=2,
                     label=f"{method}\n({len(data['history'])} iter, {data['time']:.3f}s)")
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual Norm')
    ax2.set_title('Convergence History Comparison')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Setting up problem...")
    nx, ny = 100, 100
    xmin, xmax = 0, 2
    ymin, ymax = 0, 1
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    
    A = laplace2D_matrix(nx, ny, dx, dy)
    b = np.zeros(nx * ny)
    y = np.linspace(0, 1, ny)
    
    # Solve with different preconditioners
    results = {
        'Jacobi': solver(A, b, nx, ny, y, jacobi_preconditioner),
        'SSOR': solver(A, b, nx, ny, y, SSOR_preconditioner, omega=1.0),
        'ILU': solver(A, b, nx, ny, y, ilu_preconditioner, drop_tol=0.01)
    }
    
    # Plot comparisons
    plot_comparison(results, nx, ny)