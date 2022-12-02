import jax as jx
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import pandas
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive

# TODO:
# - figure out why guide produces nan values. perhaps begin with simpler linear model
# and then add complexity

# SIMULATION PARAMETERS

use_softmax = True
primitive_test = "exp"

inference_steps = 5000

# PREPARE DATA


def generate_x(start=-1, stop=1, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x


def transform_through_primitive_exp(x: np.ndarray):
    y = -1 + np.exp(0.5 * x + 1) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y


def transform_through_primitive_tanh(x: np.ndarray):
    y = -1 + np.tanh(0.5 * x + 1) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y


GENERATORS = {
    "exp": transform_through_primitive_exp,
    "tanh": transform_through_primitive_tanh,
}

x = generate_x(num=1000)
y = GENERATORS[primitive_test](x)


# DEFINE MODEL


def softmax(x1, x2):
    return jnp.divide(jnp.exp(x1), (jnp.exp(x1) + jnp.exp(x2)))


def model(x, y):

    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    w_1 = numpyro.sample("w_exp", dist.Normal(-1.0, 1.0))
    w_2 = numpyro.sample("w_tanh", dist.Normal(-1.0, 1.0))

    a_exp = numpyro.sample("a_exp", dist.Normal(0.0, 1.0))
    a_tanh = numpyro.sample("a_tanh", dist.Normal(0.0, 1.0))

    b_exp = numpyro.sample("b_exp", dist.Normal(0.0, 1.0))
    b_tanh = numpyro.sample("b_tanh", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.Uniform(0.0, 10.0))

    mean = (
        b
        + softmax(w_1, w_2) * jnp.exp(a_exp * x + b_exp)
        + softmax(w_2, w_1) * jnp.tanh(a_tanh * x + b_tanh)
    )

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


def custom_guide(x, y):
    b_loc = numpyro.param("b_loc", np.float64(0))
    b_scale = numpyro.param("b_scale", np.float64(1), constraint=constraints.positive)

    w1_loc = numpyro.param("w1_loc", np.float64(0))
    w1_scale = numpyro.param("w1_scale", np.float64(1), constraint=constraints.positive)

    w2_loc = numpyro.param("w2_loc", np.float64(0))
    w2_scale = numpyro.param("w2_scale", np.float64(1), constraint=constraints.positive)

    a_exp_loc = numpyro.param("a_exp_loc", np.float64(0))
    a_exp_scale = numpyro.param(
        "a_exp_scale", np.float64(1), constraint=constraints.positive
    )

    a_tanh_loc = numpyro.param("a_tanh_loc", np.float64(0))
    a_tanh_scale = numpyro.param(
        "a_tanh_scale", np.float64(1), constraint=constraints.positive
    )

    b_exp_loc = numpyro.param("b_exp_loc", np.float64(0))
    b_exp_scale = numpyro.param(
        "b_exp_scale", np.float64(1), constraint=constraints.positive
    )

    b_tanh_loc = numpyro.param("b_tanh_loc", np.float64(0))
    b_tanh_scale = numpyro.param(
        "b_tanh_scale", np.float64(1), constraint=constraints.positive
    )

    sigma_loc = numpyro.param(
        "sigma_loc", np.float64(10), constraint=constraints.positive
    )
    sigma_scale = numpyro.param(
        "sigma_loc", np.float64(1), constraint=constraints.positive
    )

    b = numpyro.sample("b", dist.Normal(b_loc, b_scale))
    w_1 = numpyro.sample("w_exp", dist.Normal(w1_loc, w1_scale))
    w_2 = numpyro.sample("w_tanh", dist.Normal(w2_loc, w2_scale))

    a_exp = numpyro.sample("a_exp", dist.Normal(a_exp_loc, a_exp_scale))
    a_tanh = numpyro.sample("a_tanh", dist.Normal(a_tanh_loc, a_tanh_scale))

    b_exp = numpyro.sample("b_exp", dist.Normal(b_exp_loc, b_exp_scale))
    b_tanh = numpyro.sample("b_tanh", dist.Normal(b_tanh_loc, b_tanh_scale))
    sigma = numpyro.sample("sigma", dist.Uniform(sigma_loc, sigma_scale))

    return {
        "b": b,
        "w_1": w_1,
        "w_2": w_2,
        "a_exp": a_exp,
        "a_tanh": a_tanh,
        "b_exp": b_exp,
        "b_tanh": b_tanh,
        "sigma": sigma,
    }


# INFERENCE

# guide = numpyro.infer.autoguide.AutoNormal(model)
guide = custom_guide
optimizer = numpyro.optim.Adam(step_size=0.00005)  # 0.05

svi = numpyro.infer.SVI(model, guide, optimizer, loss=numpyro.infer.Trace_ELBO())

svi_result = svi.run(random.PRNGKey(0), inference_steps, x, y)


w_exp = svi_result.params["w_exp_auto_loc"]
w_tanh = svi_result.params["w_tanh_auto_loc"]

a_exp = svi_result.params["a_exp_auto_loc"]
a_tanh = svi_result.params["a_tanh_auto_loc"]

b_exp = svi_result.params["b_exp_auto_loc"]
b_tanh = svi_result.params["b_tanh_auto_loc"]

print("softmax of exp is " + str(softmax(w_exp, w_tanh)))
print("coefficients of exp are " + str(a_exp) + " and " + str(b_exp))
print("softmax of tanh is " + str(softmax(w_tanh, w_exp)))
print("coefficients of tanh are " + str(a_tanh) + " and " + str(b_tanh))
