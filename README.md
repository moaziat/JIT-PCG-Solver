
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
    (this makes me think about to start learning HPC concepts before implementing any GNN)

## NEXT STEP
<<<<<<< HEAD
- CREATE OUR FIRST SMOL GNN? WHAT CHOICE OF GNN? OPTIMIZING WHAT? NUMBER OF ITERATIONS? CHOICE OF THE PRECONDITIONER? in this case I have to implement more solvers and test different preconditioners!
=======
- CREATE OUR FIRST SMOL GNN? WHAT CHOICE OF GNN? OPTIMAZING WHAT? NUMBER OF ITERATIONS? CHOICE OF THE PRECONDITIONER? in this case I have to implement more solvers and test different preconditioners!

-Else, start with some HPC concepts for solvers then transition to the core idea which is smol-GNN
>>>>>>> b2a8b91 (SSOR + ILU added) 
