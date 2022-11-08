import numpyro
import numpyro.distributions as dist
from data_generators import *
from operations import *
from typing import Dict, List
from numpyro.infer import Predictive


class Model(object):
    """
    A learner operating on the parameters of a Bayesian DARTS model.
    This learner handles training the coefficients
    (architecture weights).
    """

    def __init__(
        self,
        guide_init: Dict[str, float] = dict(),
        lr: float = 0.01,
        primitives: List[str] = list(),
        priors: Dict[str, float] = dict()
    ):
        self.primitives = primitives
        self.priors = priors
        self.params = priors
        self.current_loss = 0

        # set up parameter guide
        loc_fn = numpyro.infer.initialization.init_to_value(values=guide_init)
        guide = numpyro.infer.autoguide.AutoNormal(coeff_model,
                                                        init_loc_fn=loc_fn)
        self.guide = guide

        # set up parameter optimizer
        optimizer = numpyro.optim.Adam(step_size=lr)
        self.optimizer = optimizer

        # set up parameter inference
        self.svi = numpyro.infer.SVI(coeff_model,
                                     self.guide,
                                     self.optimizer,
                                     loss=numpyro.infer.Trace_ELBO())


    def initialize_inference(self,
                             x: jnp.ndarray,
                             y: jnp.ndarray,
                             arch_params: Dict[str, float] = dict(),
                             fix_architecture: bool = False,
                             seed: int = 0):

        self.state = self.svi.init(random.PRNGKey(seed),
                                             x,
                                             y,
                                             self.primitives,
                                             arch_params,
                                             fix_architecture,
                                             self.priors)


    def update(self,
               x: jnp.ndarray,
               y: jnp.ndarray,
               arch_params: Dict[str, float] = dict(),
               fix_architecture: bool = False):

        self.state, loss = self.svi.update(self.state,
                                               x,
                                               y,
                                               self.primitives,
                                               arch_params,
                                               fix_architecture,
                                               self.priors)

        self.params = self.svi.get_params(self.state)
        self.current_loss = loss

        return loss, self.params

    def run(self, iterations,
               X: jnp.ndarray,
               y: jnp.ndarray,
               arch_weights: Dict[str, float] = dict(),
               fix_architecture: bool = False):

        svi = numpyro.infer.SVI(coeff_model,
                                self.guide,
                                self.optimizer,
                                loss=numpyro.infer.Trace_ELBO())

        svi_result = svi.run(random.PRNGKey(0), iterations, x, y,
                                           arch_weights,
                                           fixed_architecture=fix_architecture,
                                           priors=self.priors)

        return svi_result.losses, svi_result.params



    def get_posterior_samples(self, x: jnp.ndarray, arch_params: Dict[str, float] = dict(), num_samples: int = 1000):

        samples = self.guide.sample_posterior(random.PRNGKey(1), self.params, (num_samples,))
        prior_predictive = Predictive(coeff_model, samples)
        posterior_samples = prior_predictive(random.PRNGKey(1), x, None, arch_params, True,
                                         self.priors)
        return posterior_samples

def coeff_model(x, y, PRIMITIVES, arch_params, fixed_architecture = False, priors = None):

    coeff_weights = dict()

    # set up parameters
    if fixed_architecture:
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

