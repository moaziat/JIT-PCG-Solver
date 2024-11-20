
##### END GOAL
A small GNN model that computes the PCG
##### Level of difficulty: 
we gonna see, it's doable. 
###### Learning goals: 
discover some HPC concepts, write in C++.
  
##### progress
 - Implemented PCG solver for 2D poisson equation with different preconditioners 
 ![alt text](https://github.com/moaziat/smol-pcg/blob/master/PDEs%20solvers/poisson_results.png?raw=true)

  - Implemented PCG solver for 2D Laplace equation with different preconditioners 
 ![alt text](https://github.com/moaziat/smol-pcg/blob/master/PDEs%20solvers/laplace_results.png?raw=true)
 => Overall: ILU seems working best
    Computation is slow on i5-12th CPU for 500x500 grid 

