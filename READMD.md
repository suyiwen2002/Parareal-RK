Reference MATLAB implementation of Parareal time integration with independently selectable Runge–Kutta–type coarse and fine propagators for semilinear reaction–diffusion problems.
Requirements
MATLAB R2018b or later (scripts with local functions are used; 4.2 and
4.2.2 note R2016b as their own minimum).
Parallel Computing Toolbox — required by 4.2, 4.2.1, 4.2.2, 4.3,
4.3.1 and 4.4, which parallelise the fine propagator over coarse intervals
with parfor.
No other toolboxes are used; the spatial discretisation uses spdiags and
sparse direct solves only (MATLAB built-ins).
Memory: J = 160 in 2D is 25281 unknowns and runs comfortably on a
workstation; the 3D case in 4.4 (J = 160, 159^3 ≈ 4.0e6 unknowns) needs a
large-memory node. All timings reported in the paper were obtained on a
Shizi Cluster SS750-T3 node (2 × Intel Xeon Platinum 8350C, 64 cores,
512 GB RAM).
