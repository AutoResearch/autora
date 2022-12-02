from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from numpyro.diagnostics import hpdi

from autora.theorist.bdarts.architect import Architect
from autora.theorist.bdarts.model import Model
from autora.theorist.bdarts.operations import OPS, PRIMITIVES, softmax
from autora.theorist.bdarts.priors import (
    arch_priors,
    coeff_priors,
    guide_arch_init,
    guide_coeff_init,
)


@dataclass(frozen=True)
class _BDARTSResult:
    """A container for passing fitted DARTS results around."""

    architecture: Architect
    model: Model


def _general_bdarts(
    X: np.ndarray,
    y: np.ndarray,
    primitives: List = PRIMITIVES,
    arch_lr: float = 0.1,
    coeff_lr: float = 0.1,
    coeff_lr_for_sampled_model: float = 0.1,
    arch_inference_steps: int = 200,
    coeff_inference_steps_init: int = 200,
    coeff_inference_steps_adjust: int = 1,
    coeff_inference_steps_for_sampled_model: int = 3000,
    arch_priors: Dict[str, float] = arch_priors,
    coeff_priors: Dict[str, float] = coeff_priors,
    guide_arch_init: Dict[str, float] = guide_arch_init,
    guide_coeff_init: Dict[str, float] = guide_coeff_init,
    execution_monitor: Callable = (lambda *args, **kwargs: None),
    sampling_strategy: str = "max",
    seed: int = 0,
):
    """
    Fit a Bayesian DARTS model to the given data.
    Parameters
    ----------
    X : np.ndarray
        The input data.
    y : np.ndarray
        The output data.
    Returns
    -------
    _BDARTSResult
        A container for the fitted Bayesian DARTS architecture and model.
    """

    # initialize the architect and model
    architect = Architect(
        guide_init=guide_arch_init,
        lr=arch_lr,
        primitives=primitives,
        priors=arch_priors,
    )

    architect.initialize_inference(X, y, coeff_priors, seed=seed)

    model = Model(
        guide_init=guide_coeff_init,
        lr=coeff_lr,
        primitives=primitives,
        priors=coeff_priors,
    )

    model.initialize_inference(X, y, arch_priors, seed=seed)

    # PREPARE LOGGING

    coeff_weights = dict()
    coeff_weights["b"] = 0
    for primitive in OPS:
        if "linear" in primitive:
            coeff_weights[primitive] = [1, 0]

    for epoch in range(arch_inference_steps):

        # update architecture
        arch_loss, arch_params = architect.update(X, y, coeff_weights)

        if epoch == 0:
            coeff_inference_steps = coeff_inference_steps_init
        else:
            coeff_inference_steps = coeff_inference_steps_adjust

        for j in range(coeff_inference_steps):
            # update coefficients
            coeff_loss, coeff_params = model.update(X, y, arch_params)

        log = _log_general_darts(log, arch_params, coeff_params, epoch, arch_loss)

        execution_monitor(model, architect, epoch, arch_loss, coeff_loss)

        _print_status(epoch, primitives, arch_params, coeff_params, arch_loss)

    # SAMPLE FINAL MODEL

    sampled_model = Model(
        lr=coeff_lr_for_sampled_model, primitives=primitives, priors=coeff_params
    )

    sampled_arch_weights = architect.sample(sampling_strategy=sampling_strategy)

    coeff_sampled_losses, coeff_sampled_params = sampled_model.run(
        iterations=coeff_inference_steps_for_sampled_model,
        x=X,
        y=y,
        arch_weights=sampled_arch_weights,
        fix_architecture=True,
    )
    log["coeff_sampled_losses"] = coeff_sampled_losses

    _print_model(sampled_arch_weights, coeff_sampled_params)

    results = _BDARTSResult(architecture=architect, model=sampled_model)

    return results


def _print_status(epoch, primitives, arch_params, coeff_params, loss):

    arch_weights = dict()
    coeff_weights = dict()
    coeff_weights["b"] = 0
    for primitive in primitives:
        if "linear" in primitive:
            coeff_weights[primitive] = [1, 0]

    for primitive in primitives:
        arch_weights[primitive] = arch_params["w_" + primitive + "_auto_loc"]

    for primitive in primitives:
        if primitive in coeff_weights.keys():
            coeff_weights[primitive] = (
                coeff_params["a_" + primitive + "_auto_loc"],
                coeff_params["b_" + primitive + "_auto_loc"],
            )
    coeff_weights["b"] = coeff_params["b_auto_loc"]

    print_str = (
        "arch step: " + str(epoch) + ", loss: " + str(loss)
    )

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
    print_str += ", b: " + str(coeff_weights["b"])
    print(print_str)


def _print_model(arch_weights, coeff_params):

    arch_labels = list()
    for arch_label in arch_weights.keys():
        arch_labels.append(arch_label.removeprefix("w_").removesuffix("_auto_loc"))

    b = coeff_params["b_auto_loc"]
    keys = list()
    for key in coeff_params.keys():
        keys.append(key)
    if "linear" in keys[0]:
        print(
            "FINAL MODEL: "
            + str(b)
            + " + "
            + arch_labels[0]
            + "("
            + str(coeff_params["a_" + arch_labels[0] + "_auto_loc"])
            + "x + "
            + str(coeff_params["b_" + arch_labels[0] + "_auto_loc"])
            + ")"
        )
    else:
        print("FINAL MODEL: " + str(b) + " + " + arch_labels[0] + "(x)")


class BDARTSExecutionMonitor:
    """
    A monitor of the execution of the DARTS algorithm.
    """

    def __init__(self, primitives: List):
        """
        Initializes the execution monitor.
        """
        self.arch_weight_history = list()
        self.coeff_history = list()
        self.offset_history = list()
        self.arch_loss_history = list()
        self.coeff_loss_history = list()
        self.epoch_history = list()
        self.primitives = list()
        self.log = dict()

        for primitive in primitives:
            self.log["w_" + primitive] = list()
            if "linear" in primitive:
                self.log["a_" + primitive] = list()
                self.log["b_" + primitive] = list()
        self.log["b"] = list()

        self.coeff_weights = dict()
        self.coeff_weights["b"] = 0
        for primitive in OPS:
            self.coeff_weights[primitive] = [1, 0]
            if "linear" in primitive:
                pass

        self.model = None
        self.architect = None

    def execution_monitor(
        self,
        model: Model,
        architect: Architect,
        epoch: int,
        **kwargs: Any,
    ):
        """
        A function to monitor the execution of the Bayesian DARTS algorithm.

        Arguments:
            model: The Bayesian DARTS model  containing the weights each operation
                in the mixture architecture
            architect: The architect object used to construct the mixture architecture.
            epoch: The current epoch of the training.
            **kwargs: other parameters which may be passed from the DARTS optimizer
        """

        self.model = model
        self.architect = architect

        # collect data for visualization
        self.epoch_history.append(epoch)
        # nd array (1, 6, 5) (5 primitives, 6 edges)
        # self.arch_weight_history.append(
        #     model.arch_parameters()[0].detach().numpy().copy()[np.newaxis, :]
        # )
        arch_params = architect.params
        coeff_params = model.params

        for primitive in PRIMITIVES:
            # arch_weights[primitive] = arch_params["w_" + primitive + "_auto_loc"]
            if primitive in self.coeff_weights.keys():
                self.coeff_weights[primitive] = (
                    coeff_params["a_" + primitive + "_auto_loc"],
                    coeff_params["b_" + primitive + "_auto_loc"],
                )
        self.coeff_weights["b"] = coeff_params["b_auto_loc"]

        for primitive in PRIMITIVES:
            self.log["w_" + primitive].append(
                arch_params["w_" + primitive + "_auto_loc"]
            )
            if "linear" in primitive:
                self.log["a_" + primitive].append(self.coeff_weights[primitive][0])
                self.log["b_" + primitive].append(self.coeff_weights[primitive][1])
        self.log["b"].append(self.coeff_weights["b"])

        self.arch_loss_history.append(architect.current_loss)
        self.param_loss_history.append(model.current_loss)
        self.primitives = model.primitives

    def display(self, x: np.ndarray = None, y: np.ndarray = None):
        """
        A function to display the execution monitor. This function will generate some plots.
        """

        if self.model is None or self.architect is None:
            raise Exception("The execution monitor has not been initialized with a model. " +\
                            "It requires at least one call for logging.")

        arch_params = self.architect.params

        # plot prediction

        if x is not None and y is not None:
            posterior_samples = self.model.get_posterior_samples(x,
                                                                 arch_params=self.architect.params)

            mean_mu = jnp.mean(posterior_samples["obs"], axis=0)
            hpdi_mu = hpdi(posterior_samples["obs"], 0.9, 0)


            plt.plot(x.T, mean_mu.T)
            plt.plot(x.T, y.T, "o")
            plt.fill_between(
                x.flatten().T,
                hpdi_mu[0].flatten().T,
                hpdi_mu[1].flatten().T,
                alpha=0.3,
                interpolate=True,
            )
            plt.title("prediction")
            plt.show()

        # plot losses
        plt.plot(self.arch_loss_history)
        plt.title("architecture loss over architecture search")
        plt.show()

        # plot losses
        plt.plot(self.coeff_loss_history)
        plt.title("coefficient loss over architecture search")
        plt.show()

        # # plot losses
        # plt.plot(coeff_sampled_losses)
        # plt.title("coeff losses over coefficient fitting")
        # plt.show()

        # plot architecture weights
        fig, ax = plt.subplots()
        lines = list()
        for primitive in PRIMITIVES:
            lines.append(ax.plot(self.log["w_" + primitive], label=primitive)[0])
        ax.set_xlabel("architecture epoch")
        ax.set_title("architecture weights")
        ax.set_ylabel("weight")
        ax.legend(handles=lines)  # [line1, line2]
        plt.show()

        # plot coefficients
        fig, ax = plt.subplots()
        lines = list()
        for primitive in PRIMITIVES:
            if "linear" in primitive:
                lines.append(ax.plot(self.log["a_" + primitive], label="a_" + primitive)[0])
        ax.set_xlabel("architecture epoch")
        ax.set_title("coefficients")
        ax.set_ylabel("weight")
        ax.legend(handles=lines)
        plt.show()

        # plot offsets
        fig, ax = plt.subplots()
        lines = list()
        for primitive in PRIMITIVES:
            if "linear" in primitive:
                lines.append(ax.plot(self.log["b_" + primitive], label="b_" + primitive)[0])
        lines.append(ax.plot(self.log["b"], label="b")[0])
        ax.set_xlabel("architecture epoch")
        ax.set_title("offsets")
        ax.set_ylabel("weight")
        ax.legend(handles=lines)
        plt.show()

        arch_weights = dict()
        for primitive in PRIMITIVES:
            arch_weights[primitive] = arch_params["w_" + primitive + "_auto_loc"]

        for primitive in PRIMITIVES:
            print(
                "softmax of "
                + primitive
                + " is "
                + str(softmax(primitive, arch_weights))
            )

        for primitive in PRIMITIVES:
            print(primitive + " is " + str(arch_weights[primitive]))

# TODO: write function call to general_bdarts with the execution monitor
# TODO: debug and compare step-by-step with bilevel_clean.py
