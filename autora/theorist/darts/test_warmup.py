import pandas
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro
import operator
from jax import random
import jax.numpy as jnp
import jax as jx
import numpy as np
import numpy as np
import numpyro.distributions as dist
from matplotlib import pyplot as plt

# TODO:
# (1) try test_numpyro_linear_basic with guide
# (2) try with bilevel optimization
#    x try more complex test case with coefficients, i.e. -1 + exp(0.5 * x + 1)
#    x consider re-initializing coefficient svi at every architecture step
#    x try reducing coeff update steps (should plot learning curve)
#    x rewrite so that coefficients are gneralized to a dictionary of coefficients
#    x try different seeds
#    x get final coefficient fitting to work
#    x move coefficient b into coeff model
#    x pass full ach params to coeff_model
#    x pass full coeff params to arch_model
#    x add functionality for sampling final architecture from posterior
#    x get to work with basic operations exp and tanh
#    - try add zero operation
#    - add coefficient for each operation
#    - priors are basically not working.
#    somehow numpyro.sample doesn't take into account the distribution when sampling
#    need to check with experts or online before proceeding.
#    - generalize to N nodes
#    - classify
#    - make sure to include priors for coefficients in operator file, separated by operation
#    - try implementing torch version
#    -

# SIMULATION PARAMETERS
seed = 0

use_softmax = True
primitive_test = "linear_tanh"
primitive_coeff_a = 0.5
primitive_coeff_b = 1

arch_inference_steps = 1 # 800
coeff_inference_steps_adjust = 1
coeff_inference_steps_init = 200 # 200
coeff_fitting_steps = 3000
inference_lr = 0.1 # 0.1
post_sampling_lr = 0.1

# PREPARE DATA

def generate_x(start=-2, stop=2, num=500):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x

def transform_through_primitive_none(x: np.ndarray):
    y = 0 + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_exp(x: np.ndarray):
    y = - 1 + np.exp(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_tanh(x: np.ndarray):
    y = - 1 + np.tanh(x) + random.normal(random.PRNGKey(0), ([len(x)]))
    return y

def transform_through_primitive_linear_exp(x: np.ndarray, a: float = 0.5, b: float = 1.):
    y = - 1 + np.exp(a * x + b) + random.normal(random.PRNGKey(seed), ([len(x)]))
    return y

def transform_through_primitive_linear_tanh(x: np.ndarray, a: float = 0.5, b: float = 1.):
    y = - 1 + np.tanh(a * x + b) + random.normal(random.PRNGKey(seed), ([len(x)]))
    return y




GENERATORS = {
    "none": transform_through_primitive_none,
    "exp": transform_through_primitive_exp,
    "tanh": transform_through_primitive_tanh,
    "linear_exp": lambda x, a, b: transform_through_primitive_linear_exp(x, a, b),
    "linear_tanh": lambda x, a, b: transform_through_primitive_linear_tanh(x, a, b),
}

x = generate_x(num = 1000)
if "linear" in primitive_test:
    y = GENERATORS[primitive_test](x, primitive_coeff_a, primitive_coeff_b)
else:
    y = GENERATORS[primitive_test](x)

# DEFINE PRIMITIVES
OPS = {
    "none": lambda x: 0,
    "exp": jnp.exp,
    "tanh": jnp.tanh,
    "linear_exp": lambda x, a, b: jnp.exp(a * x + b),
    "linear_tanh": lambda x, a, b: jnp.tanh(a * x + b),
}

PRIMITIVES = (
    "linear_exp",
    "linear_tanh",
    # "none",
    # "exp",
    # "tanh",
)

ops = list()
for primitive in PRIMITIVES:
    # OPS returns an nn module for a given primitive (defines as a string)
    op = OPS[primitive]
    ops.append(op)

# DEFINE MODEL

def softmax(target_primitive, arch_weights):

    sum = 0
    for primitive in arch_weights.keys():
        sum += jnp.exp(arch_weights[primitive])

    return jnp.divide(jnp.exp(arch_weights[target_primitive]), sum)

def coeff_model(x, y, arch_params, fixed_architecture = False, priors = None):

    coeff_weights = dict()

    # set up parameters
    if fixed_architecture:
        sampled_arch_weights = sample_architecture(arch_params, sampling_strategy="max")
        for arch_label in sampled_arch_weights:
            if "linear" in arch_label and "loc" in arch_label:
                a_label = "a_" + arch_label.removeprefix("w_").removesuffix("_auto_loc")
                b_label = "b_" + arch_label.removeprefix("w_").removesuffix("_auto_loc")
                a_label_loc = a_label + "_auto_loc"
                a_label_scale = a_label + "_auto_scale"
                b_label_loc = b_label + "_auto_loc"
                b_label_scale = b_label + "_auto_scale"

                # select priors for coefficient
                if priors is not None and a_label_loc in priors:
                    a_mean = priors[a_label_loc]
                else:
                    a_mean = 1.
                if priors is not None and a_label_scale in priors:
                    a_sd = jnp.abs(priors[a_label_scale])
                else:
                    a_sd = 1

                # select priors for offset
                if priors is not None and b_label_loc in priors:
                    b_mean = priors[b_label_loc]
                else:
                    b_mean = 0.

                if priors is not None and b_label_scale in priors:
                    b_sd = jnp.abs(priors[b_label_scale])
                else:
                    b_sd = 1

                primitive = arch_label.removeprefix("w_").removesuffix("_auto_loc")
                coeff_weights[primitive] = (numpyro.sample(a_label, dist.Normal(a_mean, a_sd)),
                                            numpyro.sample(b_label, dist.Normal(b_mean, b_sd)))

        # if len(coeff_weights) == 0:
        #     raise Warning("Architecture has no coefficients")
        #     return
        # a_exp = numpyro.sample("a_linear_exp", dist.Normal(1., 1.))
        # b_exp = numpyro.sample("b_linear_exp", dist.Normal(0., 1.))

    else:
        for primitive in PRIMITIVES:
            if "linear" in primitive:

                a_label = "a_" + primitive
                b_label = "b_" + primitive
                a_label_loc = a_label + "_auto_loc"
                a_label_scale = a_label + "_auto_scale"
                b_label_loc = b_label + "_auto_loc"
                b_label_scale = b_label + "_auto_scale"

                # select priors for coefficient
                if priors is not None and a_label_loc in priors:
                    a_mean = priors[a_label_loc]
                else:
                    a_mean = 1.
                if priors is not None and a_label_scale in priors:
                    a_sd = jnp.abs(priors[a_label_scale])
                else:
                    a_sd = 1.

                # select priors for offset
                if priors is not None and b_label_loc in priors:
                    b_mean = priors[b_label_loc]
                else:
                    b_mean = 0.

                if priors is not None and b_label_scale in priors:
                    b_sd = jnp.abs(priors[b_label_scale])
                else:
                    b_sd = 1.

                coeff_weights[primitive] = (numpyro.sample(a_label, dist.Normal(a_mean, a_sd)),
                                            numpyro.sample(b_label, dist.Normal(b_mean, b_sd)))

    # a_exp = numpyro.sample("a_exp", dist.Normal(1., 1.))
    # a_tanh = numpyro.sample("a_tanh", dist.Normal(1., 1.))
    #
    # b_exp = numpyro.sample("b_exp", dist.Normal(0., 1.))
    # b_tanh = numpyro.sample("b_tanh", dist.Normal(0., 1.))

    # select priors for offset
    b_label = 'b'
    b_label_loc = b_label + "_auto_loc"
    b_label_scale = b_label + "_auto_scale"
    if priors is not None and b_label_loc in priors:
        b_mean = priors[b_label_loc]
    if priors is not None and b_label_scale in priors:
        b_sd = jnp.abs(priors[b_label_scale])
    else:
        b_mean = 1.
        b_sd = 1.

    coeff_weights[b_label] = numpyro.sample(b_label, dist.Normal(b_mean, b_sd))
    sigma = numpyro.sample("sigma", dist.Uniform(0., 10.))
    mean = coeff_weights['b']  # + softmax(w_1, w_2) * jnp.exp(x) + softmax(w_2, w_1) * jnp.tanh(x)

    # define model
    arch_weights = dict()
    for primitive in PRIMITIVES:
        for arch_param in arch_params.keys():
            if primitive in arch_param and "loc" in arch_param:
                arch_weights[arch_param] = arch_params[arch_param]
                continue

    if fixed_architecture:
        # mean += OPS["linear_exp"](x, a_exp, b_exp)
        for primitive in PRIMITIVES:
            if primitive in coeff_weights.keys():
                mean += OPS[primitive](x, coeff_weights[primitive][0], coeff_weights[primitive][1])

    else:
        for primitive in PRIMITIVES:
            primitive_label = "w_" + primitive + "_auto_loc"
            if use_softmax:
                if primitive in coeff_weights.keys():
                    mean += softmax(primitive_label, arch_weights) * OPS[primitive](x, coeff_weights[primitive][0], coeff_weights[primitive][1])
                else:
                    mean += softmax(primitive_label, arch_weights) * OPS[primitive](x)
                # if primitive == "exp":
                #     mean += softmax(primitive, arch_weights) * OPS[primitive](a_exp * x + b_exp)
                # elif primitive == "tanh":
                #     mean += softmax(primitive, arch_weights) * OPS[primitive](a_tanh * x + b_tanh)
                # else:
                #     mean += softmax(primitive, arch_weights) * OPS[primitive](x)
            else:
                if primitive in coeff_weights.keys():
                    mean += arch_weights[primitive_label]  * OPS[primitive](x, coeff_weights[primitive][0], coeff_weights[primitive][1])
                else:
                    mean += arch_weights[primitive_label]  * OPS[primitive](x)
                # if primitive == "exp":
                #     mean += arch_weights[primitive] * OPS[primitive](a_exp * x)
                # else:
                #     mean += arch_weights[primitive] * OPS[primitive](a_tanh * x)

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)

def sample_architecture(arch_weights, sampling_strategy = "max"):
    sampled_arch_weights = list()

    if sampling_strategy == "max":
        weights = list()
        arch_labels = list()
        for primitive in PRIMITIVES:
            for key in arch_weights.keys():
                if primitive in key and "loc" in key:
                    weights.append(arch_weights[key])
                    arch_labels.append(key)

    elif sampling_strategy == "sample":
        weights = list()
        arch_labels = list()
        for primitive in PRIMITIVES:
            found_mean = False
            found_sd = False
            mean = 0.
            sd = 1.
            for key in arch_weights.keys():
                if primitive in key and "loc" in key:
                    mean = arch_weights[key]
                    found_mean = True
                    continue
                if primitive in key and "scale" in key:
                    sd = arch_weights[key]
                    found_sd = True
                    continue
                if found_mean and found_sd:
                    break

            if found_mean is False:
                raise Warning(
                    "No loc parameter found for primitive " + primitive + ". Using mean = 0.")
            if found_sd is False:
                raise Warning(
                    "No scale parameter found for primitive " + primitive + ". Using sd = 1.")
            # sample from gaussian distribution
            weights.append(np.random.normal(mean, sd))
            arch_labels.append("w_" + primitive + "_auto_loc")

    else:
        raise ValueError("Sampling strategy not implemented")

    # get index from highest weight
    primitive = arch_labels[np.argmax(weights)]
    sampled_arch_weights.append(primitive)

    return sampled_arch_weights

def arch_model(x, y, coeff_params): # a_exp, a_tanh, b_exp, b_tanh

    arch_weights = dict()
    for primitive in PRIMITIVES:
        arch_weights[primitive] = numpyro.sample("w_" + primitive, dist.Normal(-1., 1.))

    sigma = numpyro.sample("sigma", dist.Uniform(0., 10.))
    mean = coeff_params['b_auto_loc'] # + softmax(w_1, w_2) * jnp.exp(x) + softmax(w_2, w_1) * jnp.tanh(x)

    for primitive in PRIMITIVES:
        primitive_label_a = "a_" + primitive + "_auto_loc"
        primitive_label_b = "b_" + primitive + "_auto_loc"
        if use_softmax:
            if primitive_label_a in coeff_params.keys() and primitive_label_b in coeff_params.keys():
                mean += softmax(primitive, arch_weights) * \
                        OPS[primitive](x, coeff_params[primitive_label_a], coeff_params[primitive_label_b])
            else:
                mean += softmax(primitive, arch_weights) * OPS[primitive](x)

            # if primitive == "exp":
            #     mean += softmax(primitive, arch_weights) * OPS[primitive](a_exp * x + b_exp)
            # elif primitive == "tanh":
            #     mean += softmax(primitive, arch_weights) * OPS[primitive](a_tanh * x + b_tanh)
            # else:
            #     mean += softmax(primitive, arch_weights) * OPS[primitive](x)
        else:
            if primitive_label_a in coeff_params.keys() and primitive_label_b in coeff_params.keys():
                mean += arch_weights[primitive] * \
                        OPS[primitive](x, coeff_params[primitive_label_a], coeff_params[primitive_label_b])
            else:
                mean += arch_weights[primitive] * OPS[primitive](x)

            # if primitive == "exp":
            #     mean += arch_weights[primitive] * OPS[primitive](a_exp * x)
            # else:
            #     mean += arch_weights[primitive] * OPS[primitive](a_tanh * x)

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)


# PREPARE LOGGING

arch_losses = list()
coeff_losses = list()
log = dict()

for primitive in PRIMITIVES:
    log["w_" + primitive] = list()
    if "linear" in primitive:
        log["a_" + primitive] = list()
        log["b_" + primitive] = list()
log["b"] = list()


arch_params = dict()
coeff_params = dict()
arch_weights = dict()
coeff_weights = dict()

for primitive in PRIMITIVES:
    arch_params["w_" + primitive + "_auto_loc"] = -1.
    arch_params["w_" + primitive + "_auto_scale"] = 1.
    if "linear" in primitive:
        coeff_params["a_" + primitive + "_auto_loc"] = 1.
        coeff_params["a_" + primitive + "_auto_scale"] = 1.
        coeff_params["b_" + primitive + "_auto_loc"] = 0.
        coeff_params["b_" + primitive + "_auto_scale"] = 1.
        coeff_weights[primitive] = [1, 0]

coeff_params["b_auto_loc"] = 0.
coeff_weights['b'] = 0





# warm up

arch_priors = dict()
arch_priors["w_linear_exp_auto_loc"] = 1.
arch_priors["w_linear_tanh_auto_loc"] = 1.

coeff_priors = dict()
coeff_priors["a_linear_exp_auto_loc"] = 1.
coeff_priors["a_linear_exp_auto_scale"] = 1.
coeff_priors["b_linear_exp_auto_loc"] = 0.
coeff_priors["b_linear_exp_auto_scale"] = 1.
coeff_priors["a_linear_tanh_auto_loc"] = 1.
coeff_priors["a_linear_tanh_auto_scale"] = 1.
coeff_priors["b_linear_tanh_auto_loc"] = 0.
coeff_priors["b_linear_tanh_auto_scale"] = 0.000000000001
coeff_priors["b_auto_loc"] = 0.
coeff_priors["b_auto_scale"] = 1.0000001

coeff_priors_simple = dict()
coeff_priors_simple["b"] = 0.
coeff_priors_simple["b_linear_tanh"] = 0.

# TODO: CHECK IF TANH PRIOR IS ACTUALLY SET TO 10
# TODO: CONTROL THE INIT STATE DIRECTLY OR MODIFY PARAMETERS

# SET UP INFERENCE

loc_fn = numpyro.infer.initialization.init_to_value(values=coeff_priors_simple)
coeff_guide = numpyro.infer.autoguide.AutoNormal(coeff_model,
                                                 # init_scale=0.00001,
                                                 init_loc_fn=loc_fn,
                                                 )

coeff_optimizer = numpyro.optim.Adam(step_size=inference_lr)

coeff_svi = numpyro.infer.SVI(coeff_model,
          coeff_guide,
          coeff_optimizer,
          loss=numpyro.infer.Trace_ELBO())

coeff_state = coeff_svi.init(random.PRNGKey(seed), x, y,
                             arch_params = arch_priors,
                             priors = coeff_priors
                             )


warmup_steps = 100
for j in range(warmup_steps):
    coeff_state, loss = coeff_svi.update(coeff_state, x, y,
                                         arch_params = arch_priors,
                                         priors = coeff_priors,
                                         )

    coeff_params = coeff_svi.get_params(coeff_state)

    print_str = "coeff step: " + str(j) + ", loss: " + str(loss)
    for primitive in PRIMITIVES:
        if "linear" in primitive:
            prim_str = primitive.removeprefix("linear_")
            coeff_label = "a_" + primitive + "_auto_loc"
            print_str += ", a_" + prim_str + ": " + str(coeff_params[coeff_label])
    for primitive in PRIMITIVES:
        if "linear" in primitive:
            prim_str = primitive.removeprefix("linear_")
            coeff_label = "b_" + primitive + "_auto_loc"
            print_str += ", b_" + prim_str + ": " + str(coeff_params[coeff_label])
    coeff_label = "b_auto_loc"
    print_str += ", b: " + str(coeff_params[coeff_label])
    print(print_str)


