// Define characteristic length
lc = 1;

// Points
Point(1) = {0, 0, 0, lc};
Point(2) = {2, 0, 0, lc};
Point(3) = {2, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

// Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Line loop and surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Transfinite lines and surface
Transfinite Line {1, 3} = 20; // Divisions in x-direction
Transfinite Line {2, 4} = 10; // Divisions in y-direction
Transfinite Surface {1};
Recombine Surface {1};

// Generate mesh
Mesh 2;

