import pandas
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro
from jax import random
import jax.numpy as jnp
import jax as jx
import numpy as np
import numpy as np
import numpyro.distributions as dist

# TODO:
# x eventually we may likely want to move back to torch,
#     because we want bi-level optimization using the torch optimizer,
#     so checking if jax is also available for torch would be a critical next step
# ---> perhaps just try torch functions?
# x vectorize softmax and architecture weights
# x add zero operation and test on zero case

# SIMULATION PARAMETERS

use_softmax = True
primitive_test = "exp"

inference_steps = 5000

# PREPARE DATA

def generate_x(start=-1, stop=1, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x

def transform_through_primitive_exp(x: np.ndarray):
    y = - 1 + np.exp(0.5 * x + 1) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_tanh(x: np.ndarray):
    y = - 1 + np.tanh(0.5 * x + 1) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y


GENERATORS = {
    "exp": transform_through_primitive_exp,
    "tanh": transform_through_primitive_tanh,
}

x = generate_x(num = 1000)
y = GENERATORS[primitive_test](x)


# DEFINE MODEL

def softmax(x1, x2):
    return jnp.divide(jnp.exp(x1), (jnp.exp(x1) + jnp.exp(x2)))

def model(x, y):

    b = numpyro.sample("b", dist.Normal(0., 1.))
    w_1 = numpyro.sample("w_exp", dist.Normal(-1., 1.))
    w_2 = numpyro.sample("w_tanh", dist.Normal(-1., 1.))

    a_exp = numpyro.sample("a_exp", dist.Normal(0., 1.))
    a_tanh = numpyro.sample("a_tanh", dist.Normal(0., 1.))

    b_exp = numpyro.sample("b_exp", dist.Normal(0., 1.))
    b_tanh = numpyro.sample("b_tanh", dist.Normal(0., 1.))
    sigma = numpyro.sample("sigma", dist.Uniform(0., 10.))

    mean = b + softmax(w_1, w_2) * jnp.exp(a_exp * x + b_exp) \
             + softmax(w_2, w_1) * jnp.tanh(a_tanh * x + b_tanh)

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)

# INFERENCE

guide = numpyro.infer.autoguide.AutoNormal(model)
optimizer = numpyro.optim.Adam(step_size=0.05)

svi = numpyro.infer.SVI(model,
          guide,
          optimizer,
          loss=numpyro.infer.Trace_ELBO())

svi_result = svi.run(random.PRNGKey(0), inference_steps, x, y)

w_exp = svi_result.params['w_exp_auto_loc']
w_tanh = svi_result.params['w_tanh_auto_loc']

a_exp = svi_result.params['a_exp_auto_loc']
a_tanh = svi_result.params['a_tanh_auto_loc']

b_exp = svi_result.params['b_exp_auto_loc']
b_tanh = svi_result.params['b_tanh_auto_loc']

print("softmax of exp is " + str(softmax(w_exp, w_tanh)))
print("coefficients of exp are" + str(a_exp) + " and " + str(b_exp))
print("softmax of tanh is " + str(softmax(w_tanh, w_exp)))
print("coefficients of tanh are" + str(a_tanh) + " and " + str(b_tanh))
