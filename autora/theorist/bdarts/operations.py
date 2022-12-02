import jax as jx
import jax.numpy as jnp

# DEFINE PRIMITIVES

PRIMITIVES = (
    "none",
    # "add",
    # "subtract",
    "linear",
    "linear_exp",
    "linear_tanh",
    # "linear_logistic",
    "linear_relu",
    "linear_cos",
    "linear_inverse",
)

OPS = {
    "none": lambda x: 0,
    "add": lambda x: x,
    "subtract": lambda x: -x,
    "mult": lambda x, a: a * x,
    "exp": jnp.exp,
    "logistic": lambda x: jnp.divide(1, 1 + jnp.exp(-x)),
    "relu": jx.nn.relu,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tanh": jnp.tanh,
    "ln": lambda x: jx.lax.log(x),
    "inverse": lambda x: jnp.divide(1, x),
    "linear": lambda x, a, b: a * x + b,
    "linear_exp": lambda x, a, b: jnp.exp(a * x + b),
    "linear_logistic": lambda x, a, b: jx.nn.softmax(a * x + b), # TODO: change to logistic
    "linear_relu": lambda x, a, b: jx.nn.relu(a * x + b),
    "linear_sin": lambda x, a, b: jnp.sin(a * x + b),
    "linear_cos": lambda x, a, b: jnp.cos(a * x + b),
    "linear_tanh": lambda x, a, b: jnp.tanh(a * x + b),
    "linear_ln": lambda x, a, b: jx.lax.log(a * x + b),
    "linear_inverse": lambda x, a, b: jnp.divide(1, a * x + b),
}


def softmax(target_primitive, arch_weights):

    sum = 0
    for primitive in arch_weights.keys():
        sum += jnp.exp(arch_weights[primitive])

    return jnp.divide(jnp.exp(arch_weights[target_primitive]), sum)
