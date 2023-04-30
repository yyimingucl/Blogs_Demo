# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.04.2023
# Description: Benchmark the Implementation Speed in ODE parameter inference 
# problem with PyMC + JAX ODE solver (8th order Runge-Kutta) for Lotka Volterra Model

import pymc as pm
import jax.numpy as jnp
import jax

from jax.config import config
config.update("jax_enable_x64", True)
import pytensor.tensor as pt
import numpy as np
from pytensor.link.jax.dispatch import jax_funcify 
from operators.jax_operator import JAX_ODEop, ODEGradop
from utils import MakeSample

SEED = 2023


print("JAX version: {}".format(jax.__version__))
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
    return jnp.stack([alpha * s - beta * s * p, 
                      delta * s * p - gamma * p])

# Solver Settings
t0 = 0
t1 = 10
dt_obs = 0.5
dt0 = 0.01
# Create Solver
LV_JAXODE_Op = JAX_ODEop(ode_func, t0, t1, dt_obs, state_dims=2,
                         num_parameters=4, dt0=dt0)

# Link the JAX function to Pytensor and enable the JAX Backend
@jax_funcify.register(JAX_ODEop)
def sol_op_jax_funcify(op, **kwargs):
    return LV_JAXODE_Op.get_sol

@jax_funcify.register(ODEGradop)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return LV_JAXODE_Op.get_sol._grad_op

# Create Observations
y0 = [5.,3.]
args = [2.,1.,1.,4.]
aug_args = jnp.asarray(y0+args)
obs = LV_JAXODE_Op.get_sol(aug_args).reshape([-1, 2])
S_obs = obs[:, 0]+np.random.normal(0,0.05,size=20)
P_obs = obs[:, 1]+np.random.normal(0,0.05,size=20)

def get_jax_model():
    with pm.Model() as jax_model:
        alpha = pm.Normal('alpha', 0, 1)
        beta = pm.Normal('beta', 0, 1)
        delta = pm.Normal('delta', 0, 1)
        gamma = pm.Normal('gamma', 0, 1)
        s0 = pm.Normal('s0', 0, 1)
        p0 = pm.Normal('p0', 0, 1) 
        sigma = pm.HalfNormal('noise', shape=(2,))

        params = pt.stack([[s0,p0,alpha,beta,delta,gamma]])
        solution = LV_JAXODE_Op(params).reshape([2,-1])
        
        S_hat = solution[0,:]
        P_hat = solution[1,:]

        S_lik = pm.Normal("S_lik", mu=S_hat, sigma=sigma[0], observed=S_obs)
        P_lik = pm.Normal("P_lik", mu=P_hat, sigma=sigma[1], observed=P_obs)
    return jax_model


if __name__ == "__main__":
    import json
    import os
    
    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_path = cur_path + '/results/Lotka_Volterra/JAX.txt'
 
    num_repeat = 20
    num_draws = 1000
    num_tunes = 500
    num_chains = 2
    print('[INFO] Benchmark Lotka Volterra with PyMC and JAX Solver')
    print('[INFO] Repeat {} Runs'.format(num_repeat), \
          'Each Run with {} draws, {} tunes, and {} chains'.format(num_draws, num_tunes, num_chains))

    print(' ')
    Run_Times = []
    for i in range(num_repeat):
        print('[INFO] Run {}'.format(i+1))
        jax_model = get_jax_model()
        vars_list = list(jax_model.values_to_rvs.keys())[:-2]
        Run_Times.append(MakeSample(jax_model, vars_list, num_draws, num_tunes, num_chains))
    
    print('[RETURN] Mean of Running Time: {}'.format(np.mean(Run_Times)))
    print('[RETURN] Std of Running Time: {}'.format(np.std(Run_Times)))
    print('[INFO] Results saved at {}'.format(save_path))
    with open(save_path, "w") as fp:
        json.dump(Run_Times, fp)
    