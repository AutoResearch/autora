import numpyro
import numpyro.distributions as dist
from bdarts_operations import OPS
from data_generators import *
from matplotlib import pyplot as plt
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive

from priors import guide_coeff_init, guide_arch_init, arch_priors, coeff_priors


# SIMULATION PARAMETERS
debug = False

seed = 0

primitive_test = "linear_relu"
primitive_coeff_a = 5
primitive_coeff_b = 1

arch_inference_steps = 200 # 200 # 800
coeff_inference_steps_adjust = 1 # 1
coeff_inference_steps_init = 200 # 200
coeff_fitting_steps = 3000 # 3000
inference_lr = 0.1 # 0.1
post_sampling_lr = 0.1

if debug:
    arch_inference_steps = 2
    coeff_inference_steps_adjust = 1
    coeff_inference_steps_init = 2
    coeff_fitting_steps = 2

# PREPARE DATA

x = generate_x(num = 100, start=-5, stop=5)
if "linear" in primitive_test:
    y = GENERATORS[primitive_test](x, primitive_coeff_a, primitive_coeff_b)
else:
    y = GENERATORS[primitive_test](x)

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
        # arch_params = sample_architecture(arch_params, sampling_strategy="max")
        for arch_label in arch_params:
            arch_label_primitive = arch_label.removeprefix("w_").removesuffix("_auto_loc")
            if "linear" in arch_label and "loc" in arch_label and arch_label_primitive in PRIMITIVES:
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

                primitive = arch_label.removeprefix("w_").removesuffix("_auto_loc")
                coeff_weights[primitive] = (numpyro.sample(a_label, dist.Normal(a_mean, a_sd)),
                                            numpyro.sample(b_label, dist.Normal(b_mean, b_sd)))

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
        for primitive in PRIMITIVES:
            if primitive in coeff_weights.keys():
                mean += OPS[primitive](x, coeff_weights[primitive][0], coeff_weights[primitive][1])

    else:
        for primitive in PRIMITIVES:
            primitive_label = "w_" + primitive + "_auto_loc"
            if primitive in coeff_weights.keys():
                mean += softmax(primitive_label, arch_weights) * OPS[primitive](x, coeff_weights[primitive][0], coeff_weights[primitive][1])
            else:
                mean += softmax(primitive_label, arch_weights) * OPS[primitive](x)


    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)

def sample_architecture(arch_weights, sampling_strategy = "max"):
    sampled_arch_weights = dict()

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
    sampled_arch_weights[primitive] = 1.0

    return sampled_arch_weights

def arch_model(x, y, coeff_params, priors = None): # a_exp, a_tanh, b_exp, b_tanh

    arch_weights = dict()
    for primitive in PRIMITIVES:
        w_loc = 1.
        w_scale = -1.
        if priors is not None:
            loc_label = "w_" + primitive + "_auto_loc"
            scale_label = "w_" + primitive + "_auto_scale"
            if loc_label in priors:
                w_loc = priors[loc_label]
            if scale_label in priors:
                w_scale = priors[scale_label]
        arch_weights[primitive] = numpyro.sample("w_" + primitive, dist.Normal(w_loc, w_scale))

    sigma = numpyro.sample("sigma", dist.Uniform(0., 10.))
    mean = coeff_params['b_auto_loc'] # + softmax(w_1, w_2) * jnp.exp(x) + softmax(w_2, w_1) * jnp.tanh(x)

    for primitive in PRIMITIVES:
        primitive_label_a = "a_" + primitive + "_auto_loc"
        primitive_label_b = "b_" + primitive + "_auto_loc"
        if primitive_label_a in coeff_params.keys() and primitive_label_b in coeff_params.keys():
            mean += softmax(primitive, arch_weights) * \
                    OPS[primitive](x, coeff_params[primitive_label_a],
                                   coeff_params[primitive_label_b])
        else:
            if primitive_label_a in coeff_params.keys() and primitive_label_b in coeff_params.keys():
                mean += arch_weights[primitive] * \
                        OPS[primitive](x, coeff_params[primitive_label_a], coeff_params[primitive_label_b])
            else:
                mean += arch_weights[primitive] * OPS[primitive](x)

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)

# SET UP GUIDE INITIALIZATION

coeff_loc_fn = numpyro.infer.initialization.init_to_value(values=guide_coeff_init)
arch_loc_fn = numpyro.infer.initialization.init_to_value(values=guide_arch_init)

# SET UP INFERENCE

arch_guide = numpyro.infer.autoguide.AutoNormal(arch_model,
                                                init_loc_fn=arch_loc_fn)
coeff_guide = numpyro.infer.autoguide.AutoNormal(coeff_model,
                                                 init_loc_fn=coeff_loc_fn)

arch_optimizer = numpyro.optim.Adam(step_size=inference_lr)
coeff_optimizer = numpyro.optim.Adam(step_size=inference_lr)

arch_svi = numpyro.infer.SVI(arch_model,
          arch_guide,
          arch_optimizer,
          loss=numpyro.infer.Trace_ELBO())

coeff_svi = numpyro.infer.SVI(coeff_model,
          coeff_guide,
          coeff_optimizer,
          loss=numpyro.infer.Trace_ELBO())

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

arch_weights = dict()
coeff_weights = dict()
coeff_weights['b'] = 0
for primitive in OPS:
    if "linear" in primitive:
        coeff_weights[primitive] = [1, 0]

# RUN INFERENCE
coeff_params = coeff_priors
arch_params = arch_priors

arch_state = arch_svi.init(random.PRNGKey(seed), x, y, coeff_params, arch_priors)
coeff_state = coeff_svi.init(random.PRNGKey(seed), x, y, arch_params, False, coeff_priors)

# warm up

# warmup_steps = 1000
# for i in range(warmup_steps):
#     coeff_state, loss = coeff_svi.update(coeff_state, x, y, arch_params)

for i in range(arch_inference_steps):
    arch_state, loss = arch_svi.update(arch_state, x, y, coeff_params, arch_priors)
    arch_losses.append(loss)
    arch_params = arch_svi.get_params(arch_state)


    for primitive in PRIMITIVES:
        arch_weights[primitive] = arch_params["w_" + primitive + "_auto_loc"]

    # re-initialize coefficient weights
    # coeff_state = coeff_svi.init(random.PRNGKey(seed), x, y, arch_weights, b)

    if i == 0:
        coeff_inference_steps = coeff_inference_steps_init
    else:
        coeff_inference_steps = coeff_inference_steps_adjust


    for j in range(coeff_inference_steps):
        coeff_state, loss = coeff_svi.update(coeff_state, x, y, arch_params, False, coeff_priors)
        coeff_losses.append(loss)

    coeff_params = coeff_svi.get_params(coeff_state)
    for primitive in PRIMITIVES:
        if primitive in coeff_weights.keys():
            coeff_weights[primitive] = (coeff_params["a_" + primitive + "_auto_loc"],
                                        coeff_params["b_" + primitive + "_auto_loc"])
    coeff_weights['b'] = coeff_params["b_auto_loc"]

    for primitive in PRIMITIVES:
        log["w_" + primitive].append(arch_weights[primitive])
        if "linear" in primitive:
            log["a_" + primitive].append(coeff_weights[primitive][0])
            log["b_" + primitive].append(coeff_weights[primitive][1])
    log["b"].append(coeff_weights['b'])

    print_str = "arch step: " + str(i) + ", coeff step: " + str(j) + ", loss: " + str(loss)
    for primitive in PRIMITIVES:
        prim_str = primitive.removeprefix("linear_")
        print_str += ", " + prim_str + ": " + str(arch_weights[primitive])
    for primitive in PRIMITIVES:
        if "linear" in primitive:
            prim_str = primitive.removeprefix("linear_")
            print_str += ", a_" + prim_str + ": " + str(coeff_weights[primitive][0])
    for primitive in PRIMITIVES:
        if "linear" in primitive:
            prim_str = primitive.removeprefix("linear_")
            print_str += ", b_" + prim_str + ": " + str(coeff_weights[primitive][1])
    print_str += ", b: " + str(coeff_weights['b'])
    print(print_str)

arch_params = arch_svi.get_params(arch_state)
coeff_params = arch_svi.get_params(coeff_state)

# SAMPLE FINAL MODEL

coeff_sampled_optimizer = numpyro.optim.Adam(step_size=post_sampling_lr)
coeff_sampled_guide = numpyro.infer.autoguide.AutoNormal(coeff_model)

coeff_priors = coeff_params
coeff_sampled_svi = numpyro.infer.SVI(coeff_model,
          coeff_sampled_guide,
          coeff_sampled_optimizer,
          loss=numpyro.infer.Trace_ELBO())

arch_weights_sampled = sample_architecture(arch_params, sampling_strategy="max")
svi_result = coeff_sampled_svi.run(random.PRNGKey(0), coeff_fitting_steps, x, y, arch_weights_sampled,
                                   fixed_architecture = True,
                                   priors = coeff_priors)

arch_labels = list()
for arch_label in arch_weights_sampled.keys():
    arch_labels.append(arch_label.removeprefix("w_").removesuffix("_auto_loc"))

coeff_sampled_params = svi_result.params
coeff_sampled_losses = svi_result.losses
b = coeff_sampled_params["b_auto_loc"]
keys = list()
for key in coeff_sampled_params.keys():
    keys.append(key)
if "linear" in keys[0]:
    print("FINAL MODEL: " +
          str(b) + " + " +
          arch_labels[0] + "(" +
          str(coeff_sampled_params["a_" + arch_labels[0] + "_auto_loc"]) + "x + " +
          str(coeff_sampled_params["b_" + arch_labels[0] + "_auto_loc"]) + ")")
else:
    print("FINAL MODEL: " +
          str(b) + " + " +
          arch_labels[0] + "(x)")

# GET PREDICTIONS

samples = coeff_guide.sample_posterior(random.PRNGKey(1), coeff_params, (10000,))
# prior_predictive = Predictive(coeff_model, num_samples=200)
prior_predictive = Predictive(coeff_model, samples)
prior_samples = prior_predictive(random.PRNGKey(1), x, None, arch_weights_sampled, True, coeff_priors)

mean_mu = jnp.mean(prior_samples['obs'], axis=0)
hpdi_mu = hpdi(prior_samples['obs'], 0.9, 0)


# PLOT RESULTS

# plot prediction
plt.plot(x.T, mean_mu.T)
plt.plot(x.T, y.T, 'o')
plt.fill_between(x.flatten().T, hpdi_mu[0].flatten().T, hpdi_mu[1].flatten().T, alpha=0.3, interpolate=True)
plt.title("prediction")
plt.show()

# plot losses
plt.plot(arch_losses)
plt.title("architecture loss over architecture search")
plt.show()

# plot losses
plt.plot(coeff_losses)
plt.title("coefficient loss over architecture search")
plt.show()

# plot losses
plt.plot(coeff_sampled_losses)
plt.title("coeff losses over coefficient fitting")
plt.show()

# plot architecture weights
fig, ax = plt.subplots()
lines = list()
for primitive in PRIMITIVES:
    lines.append(ax.plot(log["w_" + primitive], label=primitive)[0])
# line1, = ax.plot(log["w_exp"], label='w_exp')
# line2, = ax.plot(log["w_tanh"], label='w_tanh')
ax.set_xlabel('architecture epoch')
ax.set_title('architecture weights')
ax.set_ylabel('weight')
ax.legend(handles=lines) # [line1, line2]
plt.show()

# plot coefficients
fig, ax = plt.subplots()
lines = list()
for primitive in PRIMITIVES:
    if "linear" in primitive:
        lines.append(ax.plot(log["a_" + primitive], label="a_" + primitive)[0])
ax.set_xlabel('architecture epoch')
ax.set_title('coefficients')
ax.set_ylabel('weight')
ax.legend(handles=lines)
plt.show()

# plot offsets
fig, ax = plt.subplots()
lines = list()
for primitive in PRIMITIVES:
    if "linear" in primitive:
        lines.append(ax.plot(log["b_" + primitive], label="b_" + primitive)[0])
lines.append(ax.plot(log["b"], label='b')[0])
ax.set_xlabel('architecture epoch')
ax.set_title('offsets')
ax.set_ylabel('weight')
ax.legend(handles=lines)
plt.show()

arch_weights = dict()
for primitive in PRIMITIVES:
    arch_weights[primitive] = arch_params['w_' + primitive + '_auto_loc']


for primitive in PRIMITIVES:
    print("softmax of " + primitive + " is " + str(softmax(primitive, arch_weights)))

for primitive in PRIMITIVES:
    print(primitive + " is " + str(arch_weights[primitive]))

