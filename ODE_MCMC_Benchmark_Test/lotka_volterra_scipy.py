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

# Define ODE function: Lotka-Volterra 
def ode_func(y, t, args):
    alpha = args[0]
    beta = args[1]
    delta = args[2]
    gamma = args[3]
    s = y[0]
    p = y[1]
    return [alpha * s - beta * s * p, delta * s * p - gamma * p]

# Solver Settings
t0 = 0
t1 = 10
dt_obs = 0.5
dt = 0.01
times = np.arange(t0, t1, dt)
num_samples = int((t1-t0)/dt_obs)
sample_index = np.array([int(i*(dt_obs/dt)) for i in range(num_samples)])
# Create ODE solver
pymc_default_ode_solver = pm.ode.DifferentialEquation(ode_func, times=times, n_states=2, n_theta=4)


# Create Observations
true_parameters = [2., 1., 1., 4.]
true_initials = [5., 3.]
obs = odeint(ode_func, y0=true_initials, t=times, args=(true_parameters,)).reshape([2,-1])

S_obs = obs[0, sample_index] + np.random.normal(0,0.05,size=num_samples)
P_obs = obs[1, sample_index] + np.random.normal(0,0.05,size=num_samples)

def get_scipy_model():
    with pm.Model() as scipy_model:
        alpha = pm.Normal('alpha', 0, 1)
        beta = pm.Normal('beta', 0, 1)
        delta = pm.Normal('delta', 0, 1)
        gamma = pm.Normal('gamma', 0, 1)
        
        s0 = pm.Normal('s0', 0, 1)
        p0 = pm.Normal('p0', 0, 1) 
        sigma = pm.HalfNormal('noise', shape=(2,))

        solution = pymc_default_ode_solver(y0=[s0, p0], theta=[alpha,beta,delta,gamma], 
                                        return_sens=False).reshape([2,-1])
        
        S_hat = solution[0, sample_index]
        P_hat = solution[1, sample_index]

        S_lik = pm.Normal("S_lik", mu=S_hat, sigma=sigma[0], observed=S_obs)
        P_lik = pm.Normal("P_lik", mu=P_hat, sigma=sigma[1], observed=P_obs)
    return scipy_model


if __name__ == "__main__":
    import json
    import os
    
    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_path = cur_path + '/results/Lotka_Volterra/Scipy.txt'
 
    num_repeat = 20
    num_draws = 1000
    num_tunes = 500
    num_chains = 2

    print('[INFO] Benchmark Lotka Volterra with PyMC and Scipy Solver')
    print('[INFO] Repeat {} Runs'.format(num_repeat), \
          'Each Run with {} draws, {} tunes, and {} chains'.format(num_draws, num_tunes, num_chains))

    print(' ')
    Run_Times = []
    for i in range(num_repeat):
        print('[INFO] Run {}'.format(i+1))
        scipy_model = get_scipy_model()
        vars_list = sampled_vars_name = list(scipy_model.values_to_rvs.keys())[:-2]
        Run_Times.append(MakeSample(scipy_model, vars_list, num_draws, num_tunes, num_chains))
    
    print('[RETURN] Mean of Running Time: {}'.format(np.mean(Run_Times)))
    print('[RETURN] Std of Running Time: {}'.format(np.std(Run_Times)))
    print('[INFO] Results saved at {}'.format(save_path))
    with open(save_path, "w") as fp:
        json.dump(Run_Times, fp)
