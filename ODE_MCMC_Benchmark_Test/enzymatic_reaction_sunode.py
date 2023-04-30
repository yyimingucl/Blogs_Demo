# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.04.2023
# Description: Benchmark the Implementation Speed in ODE parameter inference 
# problem with PyMC + SUNODE ODE solver for Enzymatic Reaction Model
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

# Define ODE function: Enzymatic Reaction
def ode_func(t, x, p):
    dpdt = p.vmax * ( x.s/ p.K_S + x.s )
    return {
        "s": -dpdt,
        "p": dpdt
    }


def get_sunode_model(t, y=[None, None]):
    with pm.Model() as sunode_model:
        vmax = pm.LogNormal("vmax", 0, 1)
        K_S = pm.LogNormal("K_S", 0, 1)
        parameters = [vmax, K_S]  

        s0 = pm.LogNormal('s0', mu=np.log(10), sigma=1)   
        p0 = pm.Normal('p0', 10., 2.)  
        y0 = [s0, p0]

        x, _, problem, solver, _, _ = sunode.wrappers.as_pytensor.solve_ivp(y0={s.name.rstrip("0"): s for s in y0},
                                                                            params={p.name: p for p in parameters}|{"extra":np.array([1.])},
                                                                            rhs=ode_func, tvals=t, t0=t[0],
                                                                            solver_kwargs={"solver": "BDF", "adjoint_solver": "BDF"})

        for name, var in x.items():
            pm.Deterministic(name, var)
        
        update_solver_options(solver)
        
        sigma = pm.HalfNormal('noise', 1)
        parameters.append(sigma)

        pm.Normal("S_lik", mu=x['s'], sigma=sigma, observed=y[0])
        pm.Normal("P_lik", mu=x['p'], sigma=sigma, observed=y[1])

    solve_func = pytensor.function(parameters + y0, x, on_unused_input="ignore")
    return sunode_model, solve_func


if __name__ == "__main__":
    import json 
    import os
    
    # Create Observations 
    t = np.arange(0, 10, 0.5)

    true_parameters = {
        "vmax": 0.5,
        "K_S": 2.,
        "noise": 1.
    }

    true_initial_state = {
        "s0": 10.,
        "p0": 2.,
    }


    _, solve_func = get_sunode_model(t)
    x  = solve_func(**true_parameters, **true_initial_state)

    s_mean = x["s"]
    p_mean = x["p"]


    rng = np.random.default_rng(SEED)
    s_obs = s_mean + true_parameters["noise"] * rng.standard_normal(s_mean.shape)
    p_obs = p_mean + true_parameters["noise"] * rng.standard_normal(p_mean.shape)

    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_path = cur_path + '/results/Enzymatic_Reaction/Sunode.txt'

    num_repeat = 20
    num_draws = 1000
    num_tunes = 500
    num_chains = 2

    print("[INFO] Benchmark Enzymatic Reaction with PyMC and SUNODE Solver")
    print("[INFO] Repeat {} Runs".format(num_repeat),\
          "Each Run with {} draws, {} tunes, and {} chains".format(num_draws, num_tunes, num_chains))
    print(' ')

    Run_Times = []
    for i in range(num_repeat):
        print("[INFO] Run {}".format(i+1))
        sunode_model, _ = get_sunode_model(t=t, y=[s_obs, p_obs])
        vars_list = list(sunode_model.values_to_rvs.keys())[:-2]
        Run_Times.append(MakeSample(sunode_model, vars_list, num_draws, num_tunes, num_chains))
    
    print("[RETURN] Mean of Running Time: {}".format(np.mean(Run_Times)))
    print("[RETURN] Std of Running Time: {}".format(np.std(Run_Times)))
    with open(save_path, "w") as fp:
        json.dump(Run_Times, fp)