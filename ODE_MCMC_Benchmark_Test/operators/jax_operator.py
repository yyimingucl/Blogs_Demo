# Author: Yiming Yang
# Contact: zcahyy1@ucl.ac.uk
# Created: 29.04.2023
# Description: Create the Pytensor Operator for using JAX ODE solver 
# with PyMC 

from jax.experimental.ode import odeint as jodeint
import jax.numpy as jnp
import numpy as np
import jax
from jax.config import config
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
config.update("jax_enable_x64", True)

class JAX_ODEop(Op):
    def __init__(self, ode_func, t0, t1, dt_obs, state_dims, num_parameters, dt0=0.01, vectorized=True):
        
        self.state_dims = state_dims
        self.num_parameters = num_parameters
        self.num_inferred = state_dims+num_parameters

        self.num_obs = len(np.arange(t0, t1, dt_obs))
        self.obs_timepoint_index = np.array([int(i*(dt_obs/dt0)) for i in range(self.num_obs)])
    
        print('[INFO] Dimension of Dynamical State {}'.format(self.state_dims))
        print('[INFO] Number of Parameters in ODE {}'.format(self.num_parameters))
        print('[INFO] Number of Parameters to be Inferred {}'.format(self.num_inferred))
        print('[INFO] Number of Observations {}'.format(self.num_obs))

        # Initialize the getting solution function with specified args.
        self.get_sol = self.initialize_get_solution(ode_func, state_dims, num_parameters, t0, t1, dt0)
    
        # JAX's vmap is all you need to vectorize the get_sol to work off of a list of parameter values
        self.vec_get_sol = self.get_sol if not vectorized else jax.jit(jax.vmap(self.get_sol))
        # JAX's autodifferentiation allows automatic construction of the vector-Jacobian product
        self.vjp_get_sol = jax.jit(lambda params,grad: jax.vjp(self.vec_get_sol, params)[1](grad)[0])
        # A separate Op to allow Pytensor to calculate the gradient via JAX's vjp
        self._grad_op = ODEGradop(self.vjp_get_sol, self.num_inferred)
    
    def initialize_get_solution(self, ode_func, state_dims, num_parameters, t0, t1, dt0):
        evaluate_time = np.arange(t0, t1, dt0)
        def get_sol(aug_args):
            y0 = aug_args[:state_dims] # initial conditions
            args = aug_args[state_dims:] # parameters
            # Solve the ODE with specified conditions (default solver is Dopri8 in jodeint)
            ref_sol = jodeint(ode_func, y0, evaluate_time, args, mxstep=100000000)
            return jnp.hstack(ref_sol[self.obs_timepoint_index, :])
        return get_sol

        
    def make_node(self, p):
        # Tells Pytensor what to expect in terms of the shape/number of inputs and outputs
        p = pt.as_tensor_variable(p)
        p_type = pt.tensor(dtype='float64', shape=(1,self.num_obs,self.state_dims))
        # node = Apply(self, [p], [p.type()])
        node = Apply(self, [p], [p_type])
        return node

    def perform(self, node, inputs, output):
        # Just calls the solver on the parameters
        params = inputs[0]
        output[0][0] = np.array(self.vec_get_sol(params))  # get the numerical solution of ODE states

    def grad(self, inputs, output):
        # Pytensor's gradient calculation
        params = inputs[0]
        grads = output[0] 
        return [self._grad_op(params, grads)]


class ODEGradop(Op):
    # Define gradient operator with respect to our defined ODE operator
    def __init__(self, vjp, num_inferred):
        self._vjp = vjp
        self.num_inferred = num_inferred
        
    def make_node(self, p, g):
        p = pt.as_tensor_variable(p)
        g = pt.as_tensor_variable(g)
        g_type = pt.tensor(dtype='float64', shape=(1, self.num_inferred))
        # node = Apply(self, [p, g], [g.type()])
        node = Apply(self, [p, g], [g_type])
        return node

    def perform(self, node, inputs_storage, output_storage):
        params = inputs_storage[0]
        grads = inputs_storage[1]
        out = output_storage[0]
        # Get the numerical vector-Jacobian product
        out[0] = np.array(self._vjp(params, grads))