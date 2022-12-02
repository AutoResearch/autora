import pandas
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro
from jax import random
import jax.numpy as jnp
import jax as jx
import numpy as np
import numpy as np
import numpyro.distributions as dist
from inspect import signature

# TODO:
# (1) Try linear transformation for a couple basic operations
# (2) Add linear weights for each primitive using joint optimization
# (3) add bilevel optimization

# SIMULATION PARAMETERS

use_softmax = True
primitive_test = "linear_exp"

inference_steps = 5000

# PREPARE DATA

def generate_x(start=-1, stop=1, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x

def transform_through_primitive_none(x: np.ndarray):
    y = 0 + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_add(x: np.ndarray):
    y = x + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_subtract(x: np.ndarray):
    y = - x + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_exp(x: np.ndarray):
    y = - 1 + np.exp(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_exp(x: np.ndarray, a: float = 0.5, b: float = 1):
    y = np.exp(a * x + b) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_tanh(x: np.ndarray):
    y = - 1 + np.tanh(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_tanh(x: np.ndarray, a: float = 0.5, b: float = 1):
    y = np.tanh(a * x + b) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_logistic(x: np.ndarray):
    y = jx.nn.softmax(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_logistic(x: np.ndarray, a: float = 0.5, b: float = 1):
    y = jx.nn.softmax(a * x + b) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_relu(x: np.ndarray):
    y = jx.nn.relu(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_relu(x: np.ndarray, a: float = 0.5, b: float = 1):
    y = jx.nn.relu(a * x + b)  + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_sin(x: np.ndarray):
    y = jnp.sin(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_sin(x: np.ndarray, a: float = 0.5, b: float = 1):
    y = jnp.sin(a * x + b) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_cos(x: np.ndarray):
    y = jnp.cos(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_cos(x: np.ndarray, a: float = 0.5, b: float = 1):
    y = jnp.cos(a * x + b) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_ln(x: np.ndarray):
    y = jx.lax.log(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_ln(x: np.ndarray, a: float = 0.5, b: float = 1):
    y = jx.lax.log(a * x + b) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_inverse(x: np.ndarray):
    y = jnp.divide(1, x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_inverse(x: np.ndarray, a: float = 0.5, b: float = 1):
    y = jnp.divide(1, a * x + b)  + random.normal(random.PRNGKey(0), ([len(x)]))
    return y


GENERATORS = {
    "none": transform_through_primitive_none,
    "add": transform_through_primitive_add,
    "subtract": transform_through_primitive_subtract,
    "exp": transform_through_primitive_exp,
    "tanh": transform_through_primitive_tanh,
    "logistic": transform_through_primitive_logistic,
    "relu": transform_through_primitive_relu,
    "sin": transform_through_primitive_sin,
    "cos": transform_through_primitive_cos,
    "ln": transform_through_primitive_ln,
    "inverse": transform_through_primitive_inverse,
    "linear_exp": transform_through_primitive_linear_exp,
    "linear_tanh": transform_through_primitive_linear_tanh,
    "linear_logistic": transform_through_primitive_linear_logistic,
    "linear_relu": transform_through_primitive_linear_relu,
    "linear_sin": transform_through_primitive_linear_sin,
    "linear_cos": transform_through_primitive_linear_cos,
    "linear_ln": transform_through_primitive_linear_ln,
    "linear_inverse": transform_through_primitive_linear_inverse,
}

x = generate_x(num = 1000)
y = GENERATORS[primitive_test](x)

# DEFINE PRIMITIVES
OPS = {
    "none": lambda x: 0,
    "add": lambda x: x,
    "subtract": lambda x: -x,
    "exp": jnp.exp,
    "tanh": jnp.tanh,
    "logistic": jx.nn.softmax,
    "relu": jx.nn.relu,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "ln": lambda x: jx.lax.log(x),
    "inverse": lambda x: jnp.divide(1, x),
    "linear_exp": lambda x, a, b: jnp.exp(a * x + b),
    "linear_tanh": lambda x, a, b: jnp.tanh(a * x + b),
    "linear_logistic": lambda x, a, b: jx.nn.softmax(a * x + b),
    "linear_relu": lambda x, a, b: jx.nn.relu(a * x + b),
    "linear_sin": lambda x, a, b: jnp.sin(a * x + b),
    "linear_cos": lambda x, a, b: jnp.cos(a * x + b),
    "linear_ln": lambda x, a, b: jx.lax.log(a * x + b),
    "linear_inverse": lambda x, a, b: jnp.divide(1, a * x + b),
}

PRIMITIVES = (
    # "none",
    "linear_exp",
    "linear_tanh",
    # "linear_logistic",
    # "linear_relu",
    # "linear_cos",
    # "linear_sin",
    # "linear_inverse",
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
    coefficients = dict()
    for primitive in PRIMITIVES:
        arch_weights[primitive] = numpyro.sample("w_" + primitive, dist.Normal(-1., 1.))

        # get number of parameters for primitive
        num_params = len(signature(OPS[primitive]).parameters) - 1
        if num_params > 1:
            coefficients[primitive] = (numpyro.sample("a_" + primitive, dist.Normal(0., 1.)),
                                        numpyro.sample("b_" + primitive, dist.Normal(0., 1.)))

    b = numpyro.sample("b", dist.Normal(0., 1.))
    sigma = numpyro.sample("sigma", dist.Uniform(0., 10.))
    mean = b

    for primitive in PRIMITIVES:
        if use_softmax:
            if primitive in coefficients.keys():
                mean += softmax(primitive, arch_weights)\
                        * OPS[primitive](x, *coefficients[primitive])
            else:
                mean += softmax(primitive, arch_weights) * OPS[primitive](x)
        else:
            if primitive in coefficients.keys():
                mean += arch_weights[primitive] * OPS[primitive](x, *coefficients[primitive])
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
coefficients = dict()

for primitive in PRIMITIVES:
    arch_weights[primitive] = svi_result.params['w_' + primitive + '_auto_loc']
    num_params = len(signature(OPS[primitive]).parameters) - 1

    if num_params > 1:
        coefficients[primitive] = (svi_result.params['a_' + primitive + '_auto_loc'],
                                    svi_result.params['b_' + primitive + '_auto_loc'])

for primitive in PRIMITIVES:
    if use_softmax:
        print("softmax of " + primitive + " is " + str(softmax(primitive, arch_weights)))
    else:
        print(primitive + " is " + str(arch_weights[primitive]))

    if primitive in coefficients.keys():
        coefficients[primitive] = (svi_result.params['a_' + primitive + '_auto_loc'],
                                    svi_result.params['b_' + primitive + '_auto_loc'])
        print("coefficients of " + primitive + " are " + str(coefficients[primitive][0])
              + " and " + str(coefficients[primitive][1]))


