import numpy as np
import jax.numpy as jnp
import jax as jx
from jax import random

def generate_x(start=-2, stop=2, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 0)
    return x

def transform_through_primitive_none(x: np.ndarray, seed: int = 0):
    y = 0 + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_add(x: np.ndarray, seed: int = 0):
    y = x + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_mult(x: np.ndarray, a: float = 0.5, seed: int = 0):
    y = a * x + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear(x: np.ndarray, a: float = 0.5, b: float = 1.,
                                       seed: int = 0):
    y = a * x + b + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_subtract(x: np.ndarray, seed: int = 0):
    y = - x + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_exp(x: np.ndarray, seed: int = 0):
    y = - 1 + np.exp(x) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_tanh(x: np.ndarray, seed: int = 0):
    y = - 1 + np.tanh(x) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear_exp(x: np.ndarray, a: float = 0.5, b: float = 1.,
                                           seed: int = 0):
    y = - 1 + np.exp(a * x + b) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear_tanh(x: np.ndarray, a: float = 0.5, b: float = 1.,
                                            seed: int = 0):
    y = - 1 + np.tanh(a * x + b) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_logistic(x: np.ndarray, seed: int = 0):
    y = jnp.divide(1, 1 + jnp.exp(-x)) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear_logistic(x: np.ndarray, a: float = 0.5, b: float = 1,
                                                seed: int = 0):
    y = jnp.divide(1, 1 + jnp.exp(-(a * x + b))) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_relu(x: np.ndarray, seed: int = 0):
    y = jx.nn.relu(x) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear_relu(x: np.ndarray, a: float = 0.5, b: float = 1,
                                            seed: int = 0):
    y = jx.nn.relu(a * x + b) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_sin(x: np.ndarray, seed: int = 0):
    y = jnp.sin(x) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear_sin(x: np.ndarray, a: float = 0.5, b: float = 1,
                                           seed: int = 0):
    y = jnp.sin(a * x + b) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_cos(x: np.ndarray, seed: int = 0):
    y = jnp.cos(x) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear_cos(x: np.ndarray, a: float = 0.5, b: float = 1,
                                           seed: int = 0):
    y = jnp.cos(a * x + b) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_ln(x: np.ndarray, seed: int = 0):
    y = jx.lax.log(x) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear_ln(x: np.ndarray, a: float = 0.5, b: float = 1,
                                          seed: int = 0):
    y = jx.lax.log(a * x + b) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_inverse(x: np.ndarray, seed: int = 0):
    y = jnp.divide(1, x) + random.normal(random.PRNGKey(seed), x.shape)
    return y

def transform_through_primitive_linear_inverse(x: np.ndarray, a: float = 0.5, b: float = 1,
                                               seed: int = 0):
    y = jnp.divide(1, a * x + b)  + random.normal(random.PRNGKey(seed), x.shape)
    return y


GENERATORS = {
    "none": transform_through_primitive_none,
    "add": transform_through_primitive_add,
    "subtract": transform_through_primitive_subtract,
    "mult": transform_through_primitive_mult,
    "linear": transform_through_primitive_linear,
    "exp": transform_through_primitive_exp,
    "logistic": transform_through_primitive_logistic,
    "relu": transform_through_primitive_relu,
    "sin": transform_through_primitive_sin,
    "cos": transform_through_primitive_cos,
    "tanh": transform_through_primitive_tanh,
    "ln": transform_through_primitive_ln,
    "inverse": transform_through_primitive_inverse,
    "linear_exp": lambda x, a, b: transform_through_primitive_linear_exp(x, a, b),
    "linear_logistic": lambda x, a, b: transform_through_primitive_linear_logistic(x, a, b),
    "linear_relu": lambda x, a, b: transform_through_primitive_linear_relu(x, a, b),
    "linear_sin": lambda x, a, b: transform_through_primitive_linear_sin(x, a, b),
    "linear_cos": lambda x, a, b: transform_through_primitive_linear_cos(x, a, b),
    "linear_tanh": lambda x, a, b: transform_through_primitive_linear_tanh(x, a, b),
    "linear_ln": lambda x, a, b: transform_through_primitive_linear_ln(x, a, b),
    "linear_inverse": lambda x, a, b: transform_through_primitive_linear_inverse(x, a, b),
}
