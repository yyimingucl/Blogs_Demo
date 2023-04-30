# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.04.2023
# Description: Define utils function for recording running time

import pymc as pm
import time
import sunode

def timer(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        t1 = time.time()
        func(*args, **keyArgs)
        t2 = time.time()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return t2-t1
    return st_func

@timer
def MakeSample(model, vars_list, num_draws, num_tunes, num_chains=2):
    print("[INFO] Inferred Variables: {}".format(vars_list))
    with model:
        MH_sampler = pm.Metropolis(vars=vars_list)
        trace = pm.sample(draws=num_draws, tune=num_tunes, chains=num_chains,
                          step=[MH_sampler], progressbar=False)
        
# Update solver options
def update_solver_options(solver):
    lib = sunode._cvodes.lib
    lib.CVodeSStolerances(solver._ode, 1e-10, 1e-10)
    lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
    lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
    lib.CVodeSetMaxNumSteps(solver._ode, 100000)
    lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 100000)