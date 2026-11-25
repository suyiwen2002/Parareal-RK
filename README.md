Parareal-RK
MATLAB source code for the numerical experiments presented in the manuscript:
Iteration–Dependent Convergence and Parallel Performance of Parareal–Runge–Kutta Methods for Semilinear Parabolic Equations
Authors: Qi Zhang and Yiwen Su

1. Computational Environment
Software: MATLAB R2018b
Toolbox: Parallel Computing Toolbox
Server: Shizi Cluster SS750T3
CPU: Intel Xeon Platinum 8350C
Physical CPU cores: 64
Memory: 512 GB
2. Numerical Experiments
The program files are named according to the corresponding sections, figures, and tables in the manuscript.
Filename | Corresponding result
4.1figure1 | Figure 1
4.1figure2 | Figure 2
4.1table2 | Table 2
4.1table3 | Table 3
4.1table4 | Table 4
4.2figure3 | Figure 3
4.2figure4 | Figure 4
4.2table5 | Table 5
4.2table6 | Table 6
4.2table7 | Table 7
4.3figure5 | Figure 5
4.3figure6 | Figure 6
4.3table8 table9 | Tables 8 and 9
4.3table10 12 13 | Tables 10, 12, and 13
4.3table14 | Table 14
4.3table15,16 | Tables 15 and 16
4.3table17 | Table 17
3. Reproduction Instructions
Download the repository.
Open the corresponding source code in MATLAB.
For files without a .m extension, save a local copy with a valid MATLAB filename and the .m extension. For function files, use the main function name.
Set the numerical parameters according to the manuscript and run the program.
The experimental parameters and reference-solution settings are described in Section 4 of the manuscript.
4. Timing and Reproducibility
The parallel performance experiments use MATLAB parfor. Runtime statistics in Section 4.3 are calculated from five independent runs.
Numerical errors and convergence orders may exhibit small differences across computational environments. Runtime, speedup, and parallel efficiency depend on the hardware and software configurations.
