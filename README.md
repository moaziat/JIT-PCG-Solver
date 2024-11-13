
##### END GOAL
A small GNN model that computes the PCG
##### Level of difficulty: 
we gonna see, it's doable. 
###### Learning goals: 
discover some HPC concepts, write in C++.
  
##### progress
 - read research literature + documented steps: 
  What have I understood? 
  Instead of solving Ax = b 
  We minimize the quadratic function f(x) = 1/2 * <x, x>A - transpose(b)x 
  meaning grad(f(x)) = 0 = Ax + b
  to minimize this we use the conjugate gradient algorithm.
  - Next steps? run tests, optimize the actual code (by reading more obviously), plot some vizs.