# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.04.2023
# Description: Benchmark the Implementation Speed in ODE parameter inference 
# problem with PyMC + JAX-Based Diffrax ODE solver (8th order Runge-Kutta) for Lorenz Model
# Diffrax - https://github.com/patrick-kidger/diffrax

import pymc as pm
import jax.numpy as jnp
import jax
import diffrax
from jax.config import config
config.update("jax_enable_x64", True)
import pytensor.tensor as pt
import numpy as np
from utils import MakeSample
from pytensor.link.jax.dispatch import jax_funcify 

from operators.diffrax_operator import Diffrax_ODEop, ODEGradop

SEED = 2023

print('Diffrax version: {}'.format(diffrax.__version__))
print("JAX version: {}".format(jax.__version__))
print("PyMC Version: {}".format(pm.__version__))
print("Random Seed: {}".format(SEED))

# Define ODE function: Lorenz 63
def ode_func(t, ys, args):
    
    sigma = args[0]
    rho = args[1]
    beta = args[2]

    x = ys[0]
    y = ys[1]
    z = ys[2]

    return jnp.stack([sigma * (y - x), 
                      x * (rho - z) - y,
                      x * y - beta * z ])

# Solver Settings
t0 = 0
t1 = 10
dt_obs = 0.5
dt0 = 0.01
solver = diffrax.Dopri8()
Lorenz_DiffraxODE_Op = Diffrax_ODEop(ode_func, t0, t1, dt_obs, state_dims=3, 
                                 num_parameters=3, dt0=dt0, solver=solver, )

# Link the JAX function to Pytensor and enable the JAX Backend
@jax_funcify.register(Diffrax_ODEop)
def sol_op_jax_funcify(op, **kwargs):
    return Lorenz_DiffraxODE_Op
@jax_funcify.register(ODEGradop)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return Lorenz_DiffraxODE_Op._grad_op


# Create Observations
y0 = [0., 1., 0.]
args = [10., 28., 8/3]
aug_args = jnp.asarray(y0+args)
obs = Lorenz_DiffraxODE_Op.get_sol(aug_args)
X_obs = obs[:, 0]+np.random.normal(0,0.5,size=20)
Y_obs = obs[:, 1]+np.random.normal(0,0.5,size=20)
Z_obs = obs[:, 2]+np.random.normal(0,0.5,size=20)

def get_diffrax_model():
    with pm.Model() as diffrax_model:
        sigma = pm.Uniform('sigma', 1, 51)
        rho = pm.Uniform('rho', 1, 51)
        beta = pm.Uniform('beta', 1, 10)

        x0 = pm.HalfNormal('x0', 1)
        y0 = pm.HalfNormal('y0', 1) 
        z0 = pm.HalfNormal('z0', 1)

        noise_sigma = pm.HalfNormal('noise', 1, shape=(3,))

        params = pt.stack([[x0, y0, z0, sigma, rho, beta]])
        solution = Lorenz_DiffraxODE_Op(params).reshape([3,-1])
        
        X_hat = solution[0,:]
        Y_hat = solution[1,:]
        Z_hat = solution[2,:]

        X_lik = pm.Normal("X_lik", mu=X_hat, sigma=noise_sigma[0], observed=X_obs)
        Y_lik = pm.Normal("Y_lik", mu=Y_hat, sigma=noise_sigma[1], observed=Y_obs)
        Z_lik = pm.Normal("Z_lik", mu=Z_hat, sigma=noise_sigma[2], observed=Z_obs)

    return diffrax_model


if __name__ == '__main__':
    import json
    import os
    
    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_path = cur_path + '/results/Lorenz63/Diffrax.txt'
 
    num_repeat = 20
    num_draws = 1000
    num_tunes = 500
    num_chains = 2
    print('[INFO] Benchmark Lorenz 63 with PyMC and Diffrax Solver')
    print('[INFO] Repeat {} Runs'.format(num_repeat), \
          'Each Run with {} draws, {} tunes, and {} chains'.format(num_draws, num_tunes, num_chains))

    print(' ')
    Run_Times = []
    for i in range(num_repeat):
        print('[INFO] Run {}'.format(i+1))
        diffrax_model = get_diffrax_model()
        vars_list = list(diffrax_model.values_to_rvs.keys())[:-3]
        Run_Times.append(MakeSample(diffrax_model, vars_list, num_draws, num_tunes, num_chains))
    
    print('[RETURN] Mean of Running Time: {}'.format(np.mean(Run_Times)))
    print('[RETURN] Std of Running Time: {}'.format(np.std(Run_Times)))
    print('[INFO] Results saved at {}'.format(save_path))
    with open(save_path, "w") as fp:
        json.dump(Run_Times, fp)
    