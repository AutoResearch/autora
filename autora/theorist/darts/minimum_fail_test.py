import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random

seed = 0

# generate data


def generate_x(start=-2, stop=2, num=100):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x


def generate_y(x: np.ndarray):
    y = 0.5 * x + 1 + random.normal(random.PRNGKey(0), ([len(x)]))
    return y


x = generate_x(num=1000)
y = generate_y(x)

# define model


def model(x, y):
    # a = numpyro.sample("a", dist.Normal(1, 0.00000001))
    # b = numpyro.sample("b", dist.Normal(0., 0.00000001))
    a = numpyro.sample("a", dist.Normal(1, 1))
    b = numpyro.sample("b", dist.Normal(0.0, 1))
    y_hat = a * x + b
    numpyro.sample("y", dist.Normal(y_hat, 1.0), obs=y)


# do inference

svi = numpyro.infer.SVI(
    model,
    numpyro.infer.autoguide.AutoNormal(model),
    numpyro.optim.Adam(step_size=0.001),
    loss=numpyro.infer.Trace_ELBO(),
)

steps = 5000
# svi_result = svi.run(random.PRNGKey(0), steps, x, y)

state = svi.init(random.PRNGKey(seed), x, y)
for i in range(steps):
    state, loss = svi.update(state, x, y)
    params = svi.get_params(state)
    print(
        "step: "
        + str(i)
        + ", a: "
        + str(params["a_auto_loc"])
        + ", b: "
        + str(params["b_auto_loc"])
    )

# print("a: {0}".format(svi_result.params['a_auto_loc']))
# print("b: {0}".format(svi_result.params['b_auto_loc']))
