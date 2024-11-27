# JIT compiled preconditioned conjugate gradient (PCG)

### overview: 
There are many ways to solve PDEs. In this repository we are implementing the PCG, an iterative solverfor. For example here we are trying to solve 2D poisson ![equation](https://latex.codecogs.com/svg.latex?-\nabla^2%20u%20=%20b%20\quad%20\text{in}%20\quad%20\Omega%20\subset%20\mathbb{R}^2)

With centered finite difference method we discretize the equation into a sparse matrix problem, which is solved iteratively.