from typing import Dict, List, Tuple

import numpyro
import numpyro.distributions as dist
from data_generators import *
from operations import *


class Architect(object):
    """
    A learner operating on the architecture weights of a Bayesian DARTS model.
    This learner handles training the weights associated with mixture operations
    (architecture weights).
    """

    def __init__(
        self,
        guide_init: Dict[str, float] = dict(),
        lr: float = 0.01,
        primitives: List[str] = list(),
        priors: Dict[str, float] = dict(),
    ):
        self.primitives = primitives
        self.priors = priors
        self.params = priors
        self.current_loss = 0

        # set up architecture guide
        loc_fn = numpyro.infer.initialization.init_to_value(values=guide_init)
        guide = numpyro.infer.autoguide.AutoNormal(arch_model, init_loc_fn=loc_fn)
        self.guide = guide

        # set up architecture optimizer
        optimizer = numpyro.optim.Adam(step_size=lr)
        self.optimizer = optimizer

        # set up architecture inference
        self.svi = numpyro.infer.SVI(
            arch_model, self.guide, self.optimizer, loss=numpyro.infer.Trace_ELBO()
        )

    def initialize_inference(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        coefficients: Dict[str, float] = dict(),
        seed: int = 0,
    ):

        self.state = self.svi.init(
            random.PRNGKey(seed), x, y, self.primitives, coefficients, self.priors
        )

    def update(
        self, x: jnp.ndarray, y: jnp.ndarray, coefficients: Dict[str, float] = dict()
    ):

        self.state, loss = self.svi.update(
            self.state, x, y, self.primitives, coefficients, self.priors
        )

        self.params = self.svi.get_params(self.state)
        self.current_loss = loss

        return loss, self.params

    def sample(self, sampling_strategy: str = "max"):
        sampled_arch_weights = dict()

        if sampling_strategy == "max":
            weights = list()
            arch_labels = list()
            for primitive in self.primitives:
                for key in self.params.keys():
                    if primitive in key and "loc" in key:
                        weights.append(self.params[key])
                        arch_labels.append(key)

        elif sampling_strategy == "sample":
            weights = list()
            arch_labels = list()
            for primitive in self.primitives:
                found_mean = False
                found_sd = False
                mean = 0.0
                sd = 1.0
                for key in self.params.keys():
                    if primitive in key and "loc" in key:
                        mean = self.params[key]
                        found_mean = True
                        continue
                    if primitive in key and "scale" in key:
                        sd = self.params[key]
                        found_sd = True
                        continue
                    if found_mean and found_sd:
                        break

                if found_mean is False:
                    raise Warning(
                        "No loc parameter found for primitive "
                        + primitive
                        + ". Using mean = 0."
                    )
                if found_sd is False:
                    raise Warning(
                        "No scale parameter found for primitive "
                        + primitive
                        + ". Using sd = 1."
                    )
                # sample from gaussian distribution
                weights.append(np.random.normal(mean, sd))
                arch_labels.append("w_" + primitive + "_auto_loc")

        else:
            raise ValueError("Sampling strategy not implemented")

        # get index from highest weight
        primitive = arch_labels[np.argmax(weights)]
        sampled_arch_weights[primitive] = 1.0

        return sampled_arch_weights


def arch_model(
    x, y, PRIMITIVES, coeff_params, priors=None
):  # a_exp, a_tanh, b_exp, b_tanh

    arch_weights = dict()
    for primitive in PRIMITIVES:
        w_loc = 1.0
        w_scale = -1.0
        if priors is not None:
            loc_label = "w_" + primitive + "_auto_loc"
            scale_label = "w_" + primitive + "_auto_scale"
            if loc_label in priors:
                w_loc = priors[loc_label]
            if scale_label in priors:
                w_scale = priors[scale_label]
        arch_weights[primitive] = numpyro.sample(
            "w_" + primitive, dist.Normal(w_loc, w_scale)
        )

    sigma = numpyro.sample("sigma", dist.Uniform(0.0, 10.0))
    mean = coeff_params[
        "b_auto_loc"
    ]  # + softmax(w_1, w_2) * jnp.exp(x) + softmax(w_2, w_1) * jnp.tanh(x)

    for primitive in PRIMITIVES:
        primitive_label_a = "a_" + primitive + "_auto_loc"
        primitive_label_b = "b_" + primitive + "_auto_loc"
        if (
            primitive_label_a in coeff_params.keys()
            and primitive_label_b in coeff_params.keys()
        ):
            mean += softmax(primitive, arch_weights) * OPS[primitive](
                x, coeff_params[primitive_label_a], coeff_params[primitive_label_b]
            )
        else:
            if (
                primitive_label_a in coeff_params.keys()
                and primitive_label_b in coeff_params.keys()
            ):
                mean += arch_weights[primitive] * OPS[primitive](
                    x, coeff_params[primitive_label_a], coeff_params[primitive_label_b]
                )
            else:
                mean += arch_weights[primitive] * OPS[primitive](x)

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)
