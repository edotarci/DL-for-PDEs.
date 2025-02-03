# Deep learning for PDEs 
It contains the final project (consisting of 3 subprojects) I completed for the course "Deep Learning for Scientific computing", thought at ETH by professor Siddharta Mishra in Fall 2024. The project focuses on leveraging Fourier Neural Operators (FNO) and regression methods to solving and discovering partial differential equations (PDEs). 

## Project Overview  

###  1. Fourier Neural Operator (FNO) for the 1D Wave Equation  
- Trains an FNO to approximate the solution of a **1D wave equation**.  
- Evaluates the model across different spatial resolutions and out-of-distribution (OOD) datasets.  
- Extends the training approach to a time-dependent setting.  

###  2. PDE Discovery with PDE-FIND  
- Uses sparse regression to reconstruct **unknown PDEs** from observed data.  
- Builds a library of candidate terms and selects the most relevant using sparsity-promoting techniques.  
- Applies the method to increasingly complex PDE datasets, including coupled equations.  

###  3. Neural Solver for the Allen-Cahn Equation  
- Develops a **foundation model** for solving the **Allen-Cahn equation**, a fundamental model in phase separation dynamics.  
- Investigates generalization across different parameter regimes and initial conditions.  
- Evaluates error metrics, convergence behavior, and physical properties.  
- Includes the proof of stability theorem for the equation.  

