import jax as jx
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas
from jax import random
from matplotlib import pyplot as plt
from numpyro.infer import MCMC, NUTS, Predictive

# TODO:
# (1) try with guide
# (2) try with bilevel optimization

# SIMULATION PARAMETERS

use_softmax = True
primitive_test = "exp"

inference_steps = 200
inference_lr = 0.1

# PREPARE DATA


def generate_x(start=-2, stop=2, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x


def transform_through_primitive_exp(x: np.ndarray):
    y = -1 + np.exp(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y


def transform_through_primitive_tanh(x: np.ndarray):
    y = -1 + np.tanh(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y


GENERATORS = {
    "exp": transform_through_primitive_exp,
    "tanh": transform_through_primitive_tanh,
}

x = generate_x(num=1000)
y = GENERATORS[primitive_test](x)

# DEFINE PRIMITIVES
OPS = {
    "exp": jnp.exp,
    "tanh": jnp.tanh,
}

PRIMITIVES = (
    "exp",
    "tanh",
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


def coeff_model(x, y, arch_weights, b):

    a_exp = numpyro.sample("a_exp", dist.Normal(1.0, 1.0))
    a_tanh = numpyro.sample("a_tanh", dist.Normal(1.0, 1.0))

    sigma = numpyro.sample("sigma", dist.Uniform(0.0, 10.0))
    mean = b  # + softmax(w_1, w_2) * jnp.exp(x) + softmax(w_2, w_1) * jnp.tanh(x)

    for primitive in PRIMITIVES:
        if use_softmax:
            if primitive == "exp":
                mean += softmax(primitive, arch_weights) * OPS[primitive](a_exp * x)
            elif primitive == "tanh":
                mean += softmax(primitive, arch_weights) * OPS[primitive](a_tanh * x)
            else:
                mean += softmax(primitive, arch_weights) * OPS[primitive](x)
        else:
            if primitive == "exp":
                mean += arch_weights[primitive] * OPS[primitive](a_exp * x)
            else:
                mean += arch_weights[primitive] * OPS[primitive](a_tanh * x)

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


def arch_model(x, y, a_exp, a_tanh):

    arch_weights = dict()
    for primitive in PRIMITIVES:
        arch_weights[primitive] = numpyro.sample(
            "w_" + primitive, dist.Normal(-1.0, 1.0)
        )

    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.Uniform(0.0, 10.0))
    mean = b  # + softmax(w_1, w_2) * jnp.exp(x) + softmax(w_2, w_1) * jnp.tanh(x)

    for primitive in PRIMITIVES:
        if use_softmax:
            if primitive == "exp":
                mean += softmax(primitive, arch_weights) * OPS[primitive](a_exp * x)
            elif primitive == "tanh":
                mean += softmax(primitive, arch_weights) * OPS[primitive](a_tanh * x)
            else:
                mean += softmax(primitive, arch_weights) * OPS[primitive](x)
        else:
            if primitive == "exp":
                mean += arch_weights[primitive] * OPS[primitive](a_exp * x)
            else:
                mean += arch_weights[primitive] * OPS[primitive](a_tanh * x)

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


# INFERENCE

arch_guide = numpyro.infer.autoguide.AutoNormal(arch_model)
coeff_guide = numpyro.infer.autoguide.AutoNormal(coeff_model)

arch_optimizer = numpyro.optim.Adam(step_size=inference_lr)
coeff_optimizer = numpyro.optim.Adam(step_size=inference_lr)

arch_svi = numpyro.infer.SVI(
    arch_model, arch_guide, arch_optimizer, loss=numpyro.infer.Trace_ELBO()
)

coeff_svi = numpyro.infer.SVI(
    coeff_model, coeff_guide, coeff_optimizer, loss=numpyro.infer.Trace_ELBO()
)

arch_losses = list()
coeff_losses = list()

arch_weights = dict()
for primitive in PRIMITIVES:
    arch_weights[primitive] = -1
b = 0
a_exp = 1
a_tanh = 1

arch_state = arch_svi.init(random.PRNGKey(0), x, y, a_exp, a_tanh)
coeff_state = coeff_svi.init(random.PRNGKey(0), x, y, arch_weights, b)

for i in range(inference_steps):
    print(i)
    arch_state, loss = arch_svi.update(arch_state, x, y)
    arch_losses.append(loss)
    arch_params = arch_guide.get_params(arch_state)

    for primitive in PRIMITIVES:
        arch_weights[primitive] = arch_params["w_" + primitive]
    b = arch_params["b"]

    for j in range(inference_steps):
        coeff_state, loss = coeff_svi.update(coeff_state, x, y, arch_weights, b)
        coeff_losses.append(loss)

params = arch_svi.get_params(arch_state)

# plot losses
plt.plot(arch_losses)
plt.show()

arch_weights = dict()
for primitive in PRIMITIVES:
    arch_weights[primitive] = params["w_" + primitive + "_auto_loc"]

if use_softmax:
    for primitive in PRIMITIVES:
        print(
            "softmax of " + primitive + " is " + str(softmax(primitive, arch_weights))
        )

    # print("w_exp: {0}".format(softmax("exp", arch_weights)))
    # print("w_tanh: {0}".format(softmax("tanh", arch_weights)))
else:
    for primitive in PRIMITIVES:
        print(primitive + " is " + str(arch_weights[primitive]))

for primitive in PRIMITIVES:
    print(primitive + " is " + str(arch_weights[primitive]))


# print("### MCMC Inference")
# nuts_kernel = NUTS(model)
# mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500)
# mcmc.run(x, y)
# mcmc.print_summary()
