# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.04.2023
# Description: Benchmark the Implementation Speed in ODE parameter inference 
# problem with PyMC + SUNODE ODE solver for Lorenz63 Model
# SUNODE - https://github.com/pymc-devs/sunode

import numpy as np
import sunode
import sunode.wrappers.as_pytensor
import pymc as pm
import pytensor 
from utils import MakeSample, update_solver_options

SEED = 2023
print("PyMC Version: {}".format(pm.__version__))
print("Random Seed: {}".format(SEED))

# Define ODE function: Lorenz63
def ode_func(t, x, p):
    return {
        "x": p.sigma * ( x.y - x.x ),
        "y": x.x * ( p.rho - x.z ) -x.y,
        "z": x.x * x.y - p.beta * x.z
    }


def get_sunode_model(t, y=[None, None, None]):
    with pm.Model() as sunode_model:
        sigma = pm.Uniform("sigma", 1, 51)
        rho = pm.Uniform("rho", 1, 51)
        beta = pm.Uniform("beta", 1, 10)
        parameters = [sigma, rho, beta]  

        x0 = pm.HalfNormal('x0', 1.)   
        y0 = pm.HalfNormal('y0', 1.)  
        z0 = pm.HalfNormal('z0', 1.)
        y0 = [x0, y0, z0]

        x, _, problem, solver, _, _ = sunode.wrappers.as_pytensor.solve_ivp(y0={s.name.rstrip("0"): s for s in y0},
                                                                            params={p.name: p for p in parameters}|{"extra":np.array([1.])},
                                                                            rhs=ode_func, tvals=t, t0=t[0],
                                                                            solver_kwargs={"solver": "BDF", "adjoint_solver": "BDF"})

        for name, var in x.items():
            pm.Deterministic(name, var)
        
        update_solver_options(solver)
        
        noise_sigma = pm.HalfNormal('noise', 1)
        parameters.append(noise_sigma)

        pm.Normal("X_lik", mu=x['x'], sigma=noise_sigma, observed=y[0])
        pm.Normal("Y_lik", mu=x['y'], sigma=noise_sigma, observed=y[1])
        pm.Normal("Z_lik", mu=x['z'], sigma=noise_sigma, observed=y[2])

    solve_func = pytensor.function(parameters + y0, x, on_unused_input="ignore")
    return sunode_model, solve_func


if __name__ == "__main__":
    import json 
    import os
    
    # Create Observations 
    t = np.arange(0, 10, 0.5)

    true_parameters = {
        "sigma": 10.,
        "rho": 28.,
        "beta": 8/3,
        "noise": 1.
    }

    true_initial_state = {
        "x0": 0.,
        "y0": 1.,
        "z0": 0.
    }


    _, solve_func = get_sunode_model(t)
    x  = solve_func(**true_parameters, **true_initial_state)

    x_mean = x["x"]
    y_mean = x["y"]
    z_mean = x["z"]

    rng = np.random.default_rng(SEED)
    x_obs = x_mean + true_parameters["noise"] * rng.standard_normal(x_mean.shape)
    y_obs = y_mean + true_parameters["noise"] * rng.standard_normal(y_mean.shape)
    z_obs = z_mean + true_parameters["noise"] * rng.standard_normal(z_mean.shape)

    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_path = cur_path + '/results/Lorenz63/Sunode.txt'

    num_repeat = 20
    num_draws = 1000
    num_tunes = 500
    num_chains = 2

    print("[INFO] Benchmark Lorenz63 with PyMC and SUNODE Solver")
    print("[INFO] Repeat {} Runs".format(num_repeat),\
          "Each Run with {} draws, {} tunes, and {} chains".format(num_draws, num_tunes, num_chains))
    print(' ')

    Run_Times = []
    for i in range(num_repeat):
        print("[INFO] Run {}".format(i+1))
        sunode_model, _ = get_sunode_model(t=t, y=[x_obs, y_obs, z_obs])
        vars_list = list(sunode_model.values_to_rvs.keys())[:-3]
        Run_Times.append(MakeSample(sunode_model, vars_list, num_draws, num_tunes, num_chains))
    
    print("[RETURN] Mean of Running Time: {}".format(np.mean(Run_Times)))
    print("[RETURN] Std of Running Time: {}".format(np.std(Run_Times)))
    with open(save_path, "w") as fp:
        json.dump(Run_Times, fp)
