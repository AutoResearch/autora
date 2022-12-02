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
    y = - 1 + np.exp(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_tanh(x: np.ndarray):
    y = - 1 + np.tanh(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_none(x: np.ndarray):
    y = 0 + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_add(x: np.ndarray):
    y = x + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_subtract(x: np.ndarray):
    y = - x + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_logistic(x: np.ndarray):
    y = jx.nn.softmax(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_relu(x: np.ndarray):
    y = jx.nn.relu(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_sin(x: np.ndarray):
    y = jnp.sin(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_cos(x: np.ndarray):
    y = jnp.cos(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_ln(x: np.ndarray):
    y = jx.lax.log(x + 0.0000001) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_inverse(x: np.ndarray):
    y = jnp.divide(1, x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y


GENERATORS = {
    "none": transform_through_primitive_none,
    "exp": transform_through_primitive_exp,
    "tanh": transform_through_primitive_tanh,
    "add": transform_through_primitive_add,
    "subtract": transform_through_primitive_subtract,
    "logistic": transform_through_primitive_logistic,
    "relu": transform_through_primitive_relu,
    "sin": transform_through_primitive_sin,
    "cos": transform_through_primitive_cos,
    # "ln": transform_through_primitive_ln,
    "inverse": transform_through_primitive_inverse,
}

x = generate_x(num = 1000)
y = GENERATORS[primitive_test](x)

# DEFINE PRIMITIVES
OPS = {
    "exp": jnp.exp,
    "tanh": jnp.tanh,
    "none": lambda x: 0,
    "add": lambda x: x,
    "subtract": lambda x: -x,
    "logistic": jx.nn.softmax,
    "relu": jx.nn.relu,
    "sin": jnp.sin,
    "cos": jnp.cos,
    # "ln": lambda x: jx.lax.log(x + 0.0001),
    "inverse": lambda x: jnp.divide(1, x)
}

PRIMITIVES = (
    "none",
    "exp",
    "tanh",
    "add",
    "subtract",
    "logistic",
    "relu",
    "cos",
    "sin",
    # "ln",
    "inverse",
)

ops = list()
for primitive in PRIMITIVES:
    # OPS returns an nn module for a given primitive (defines as a string)
    op = OPS[primitive]
    ops.append(op)

# DEFINE MODEL

def softmax(target_primitive, arch_weights):

    sum = 0
    for primitive in PRIMITIVES:
        sum += jnp.exp(arch_weights[primitive])

    return jnp.divide(jnp.exp(arch_weights[target_primitive]), sum)

def model(x, y):

    arch_weights = dict()
    for primitive in PRIMITIVES:
        arch_weights[primitive] = numpyro.sample("w_" + primitive, dist.Normal(-1., 1.))

    # w_1 = numpyro.sample("w_exp", dist.Normal(0., 1.))
    # w_2 = numpyro.sample("w_tanh", dist.Normal(0., 1.))
    b = numpyro.sample("b", dist.Normal(0., 1.))
    sigma = numpyro.sample("sigma", dist.Uniform(0., 10.))
    mean = b # + softmax(w_1, w_2) * jnp.exp(x) + softmax(w_2, w_1) * jnp.tanh(x)

    for primitive in PRIMITIVES:
        if use_softmax:
            mean += softmax(primitive, arch_weights) * OPS[primitive](x)
        else:
            mean += arch_weights[primitive] * OPS[primitive](x)

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

arch_weights = dict()
for primitive in PRIMITIVES:
    arch_weights[primitive] = svi_result.params['w_' + primitive + '_auto_loc']

if use_softmax:
    for primitive in PRIMITIVES:
        print("softmax of " + primitive + " is " + str(softmax(primitive, arch_weights)))

    # print("w_exp: {0}".format(softmax("exp", arch_weights)))
    # print("w_tanh: {0}".format(softmax("tanh", arch_weights)))
else:
    for primitive in PRIMITIVES:
        print(primitive + " is " + str(arch_weights[primitive]))

for primitive in PRIMITIVES:
    print(primitive + " is " + str(arch_weights[primitive]))

