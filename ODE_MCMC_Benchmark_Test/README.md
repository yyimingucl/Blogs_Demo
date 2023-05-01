## Benchmarking ODE Parameter Inference Problem with PyMC :rocket:
This repository contains code to benchmark the implementation speed for ODE parameter inference problem using PyMC with different solvers: [JAX ode solver](https://github.com/google/jax/blob/main/jax/experimental/ode.py), [Diffrax](https://github.com/patrick-kidger/diffrax), [the default scipy solver](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.ode.DifferentialEquation.html), and [SUNODE solver](https://github.com/pymc-devs/sunode).

Test Problems: 
- Lotka Volterra, 2D nonlinear system with 4 parameters 
- Lorenz 63, 3D linear system with 3 parameters 
- Simplified Enzymatic Reaction, 2D linear system with 2 parameters  

### Set up
To run the benchmarking code, you will need to have the following software installed on your system:

- Python 3.8 or higher
- pymc
- jax 
- diffrax 
- sunode
  
To install the required packages, you can run the following command:

`pip install -r requirements.txt`

### Usage 
To run the benchmarking code, you can use the following command: 

`python [test_model]_[solver].py`

This will run the benchmarking code for specified test_model+solver, print the results to the console and save the results to `./results/[test_model]/[solver].txt`. They will run a Metropolis-Hasting MCMC using PyMC and specified solver with 2 parallel chains, where each chain optimized with 500 tune samples and draw 1000 samples.

### Results
The shown bar plot is the mean and variance taking over 20 times for each benchmark.
#### Benchmark on Lotka Volterra
![image](https://github.com/yyimingucl/Blogs_Demo/blob/main/ODE_MCMC_Benchmark_Test/results/Lotka_Volterra/result.png)

#### Benchmark on Lorenz63
![image](https://github.com/yyimingucl/Blogs_Demo/blob/main/ODE_MCMC_Benchmark_Test/results/Lorenz63/result.png)

#### Benchmark on Enzymatic Reaction
![image](https://github.com/yyimingucl/Blogs_Demo/blob/main/ODE_MCMC_Benchmark_Test/results/Enzymatic_Reaction/result.png)

### Conclusion 
The benchmarking results show that JAX and Diffrax are around 1000 times faster than the PyMC default solver and 10 times faster than SUNODE, which is a remarkable improvement in speed :rocket:

It's also worth noting that the PyMC default solver was not numerically stable for some parameters and initial conditions, particularly in nonlinear cases like Lorenz63 and Lotka Volterra. This issue was largely alleviated with JAX, Diffrax, and SUNODE, with the warning for numerical instability rarely being raised.

Overall, the results suggest that JAX and Diffrax more are reliable and efficient solvers, and should be considered as alternatives to the PyMC default solver. The numerical stability of these solvers may also make them more suitable for certain types of problems, particularly those that involve nonlinear models.
