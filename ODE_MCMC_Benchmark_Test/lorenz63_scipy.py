# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.04.2023
# Description: Benchmark the Implementation Speed in ODE parameter inference 
# problem with PyMC + default ODE solver (scipy.integrate.odeint) for Lotka Volterra Model


import pymc as pm
import numpy as np
from utils import MakeSample
from scipy.integrate import odeint

SEED = 2023

print("PyMC Version: {}".format(pm.__version__))
print("Random Seed: {}".format(SEED))
print("="*50)
print(" ")

# Define ODE function: Lorenz 63
def ode_func(ys, t, args):
    sigma = args[0]
    rho = args[1]
    beta = args[2]

    x = ys[0]
    y = ys[1]
    z = ys[2]

    return [sigma * (y - x), 
            x * (rho - z) - y,
            x * y - beta * z ]

# Solver Settings
t0 = 0
t1 = 10
dt_obs = 0.5
dt = 0.01
times = np.arange(t0, t1, dt)
num_samples = int((t1-t0)/dt_obs)
sample_index = np.array([int(i*(dt_obs/dt)) for i in range(num_samples)])
# Create ODE solver
pymc_default_ode_solver = pm.ode.DifferentialEquation(ode_func, times=times, n_states=3, n_theta=3)


# Create Observations
true_parameters = [10., 28., 8/3]
true_initials = [0., 1., 0.]
obs = odeint(ode_func, y0=true_initials, t=times, args=(true_parameters,)).reshape([3,-1])

X_obs = obs[0, sample_index]+np.random.normal(0,0.5,size=20)
Y_obs = obs[1, sample_index]+np.random.normal(0,0.5,size=20)
Z_obs = obs[2, sample_index]+np.random.normal(0,0.5,size=20)


def get_scipy_model():
    with pm.Model() as scipy_model:
        sigma = pm.Uniform('sigma', 1, 51)
        rho = pm.Uniform('rho', 1, 51)
        beta = pm.Uniform('beta', 1, 10)

        x0 = pm.HalfNormal('x0', 1)
        y0 = pm.HalfNormal('y0', 1) 
        z0 = pm.HalfNormal('z0', 1)

        noise_sigma = pm.HalfNormal('noise', 1, shape=(3,))

        solution = pymc_default_ode_solver(y0=[x0, y0, z0], theta=[sigma, rho, beta], 
                                        return_sens=False).reshape([3,-1])
        
        X_hat = solution[0, sample_index]
        Y_hat = solution[1, sample_index]
        Z_hat = solution[2, sample_index]

        X_lik = pm.Normal("X_lik", mu=X_hat, sigma=noise_sigma[0], observed=X_obs)
        Y_lik = pm.Normal("Y_lik", mu=Y_hat, sigma=noise_sigma[1], observed=Y_obs)
        Z_lik = pm.Normal("Z_lik", mu=Z_hat, sigma=noise_sigma[2], observed=Z_obs)
    return scipy_model


if __name__ == "__main__":
    import json
    import os
    
    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_path = cur_path + '/results/Lorenz63/Scipy.txt'
 
    num_repeat = 20
    num_draws = 1000
    num_tunes = 500
    num_chains = 2

    print('[INFO] Benchmark Lorenz 63 with PyMC and Scipy Solver')
    print('[INFO] Repeat {} Runs'.format(num_repeat), \
          'Each Run with {} draws, {} tunes, and {} chains'.format(num_draws, num_tunes, num_chains))

    print(' ')
    Run_Times = []
    for i in range(num_repeat):
        print('[INFO] Run {}'.format(i+1))
        scipy_model = get_scipy_model()
        vars_list = sampled_vars_name = list(scipy_model.values_to_rvs.keys())[:-3]
        Run_Times.append(MakeSample(scipy_model, vars_list, num_draws, num_tunes, num_chains))
    
    print('[RETURN] Mean of Running Time: {}'.format(np.mean(Run_Times)))
    print('[RETURN] Std of Running Time: {}'.format(np.std(Run_Times)))
    print('[INFO] Results saved at {}'.format(save_path))
    with open(save_path, "w") as fp:
        json.dump(Run_Times, fp)
