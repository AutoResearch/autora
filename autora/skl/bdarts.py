from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from autora.theorist.bdarts.architect import Architect
from autora.theorist.bdarts.model import Model
from autora.theorist.bdarts.operations import OPS, PRIMITIVES
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

    arch_losses = list()
    coeff_losses = list()
    log = dict()

    for primitive in PRIMITIVES:
        log["w_" + primitive] = list()
        if "linear" in primitive:
            log["a_" + primitive] = list()
            log["b_" + primitive] = list()
    log["b"] = list()

    coeff_weights = dict()
    coeff_weights["b"] = 0
    for primitive in OPS:
        if "linear" in primitive:
            coeff_weights[primitive] = [1, 0]

    for epoch in range(arch_inference_steps):

        # update architecture
        arch_loss, arch_params = architect.update(X, y, coeff_weights)
        arch_losses.append(arch_loss)

        if epoch == 0:
            coeff_inference_steps = coeff_inference_steps_init
        else:
            coeff_inference_steps = coeff_inference_steps_adjust

        for j in range(coeff_inference_steps):
            # update coefficients
            coeff_loss, coeff_params = model.update(X, y, arch_params)
            coeff_losses.append(coeff_loss)

        log = _log_general_darts(log, arch_params, coeff_params, epoch, arch_loss)

        execution_monitor(model, architect, epoch)

        _print_status(primitives, arch_params, coeff_params, arch_loss)

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


def _print_status(primitives, arch_params, coeff_params, loss):

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
        "arch step: " + str(i) + ", coeff step: " + str(j) + ", loss: " + str(loss)
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


def _log_general_darts(log, arch_params, coeff_params, i, loss, verbose=True):

    arch_weights = dict()
    coeff_weights = dict()
    coeff_weights["b"] = 0
    for primitive in OPS:
        if "linear" in primitive:
            coeff_weights[primitive] = [1, 0]

    for primitive in PRIMITIVES:
        arch_weights[primitive] = arch_params["w_" + primitive + "_auto_loc"]
        if primitive in coeff_weights.keys():
            coeff_weights[primitive] = (
                coeff_params["a_" + primitive + "_auto_loc"],
                coeff_params["b_" + primitive + "_auto_loc"],
            )
    coeff_weights["b"] = coeff_params["b_auto_loc"]

    for primitive in PRIMITIVES:
        log["w_" + primitive].append(arch_weights[primitive])
        if "linear" in primitive:
            log["a_" + primitive].append(coeff_weights[primitive][0])
            log["b_" + primitive].append(coeff_weights[primitive][1])
    log["b"].append(coeff_weights["b"])

    if verbose:
        print_str = "arch step: " + str(i) + ", loss: " + str(loss)
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

    return log


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

        # collect data for visualization
        self.epoch_history.append(epoch)
        # nd array (1, 6, 5) (5 primitives, 6 edges)
        # self.arch_weight_history.append(
        #     model.arch_parameters()[0].detach().numpy().copy()[np.newaxis, :]
        # )
        arch_params = architect.params
        coeff_params = model.params

        for primitive in PRIMITIVES:
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

    def display(self):
        """
        A function to display the execution monitor. This function will generate two plots:
        (1) A plot of the training loss vs. epoch,
        (2) a plot of the architecture weights vs. epoch, divided into subplots by each edge
        in the mixture architecture.
        """

        # TODO: check what is stored in the log function. technically, all of that logging
        #  should happen in the call of the execution monitor
        # TODO: first make sure that all information available is either in the init or the call
        # to execution monitor
        # TODO: then run through display function and relplace with own variables

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
            print(
                "FINAL MODEL: "
                + str(b)
                + " + "
                + arch_labels[0]
                + "("
                + str(coeff_sampled_params["a_" + arch_labels[0] + "_auto_loc"])
                + "x + "
                + str(coeff_sampled_params["b_" + arch_labels[0] + "_auto_loc"])
                + ")"
            )
        else:
            print("FINAL MODEL: " + str(b) + " + " + arch_labels[0] + "(x)")

        # GET PREDICTIONS
        # RAPLACE THIS WITH FUNCTION CALL OF MODEL CLASS

        samples = coeff_guide.sample_posterior(
            random.PRNGKey(1), coeff_params, (10000,)
        )
        # prior_predictive = Predictive(coeff_model, num_samples=200)
        prior_predictive = Predictive(coeff_model, samples)
        prior_samples = prior_predictive(
            random.PRNGKey(1), x, None, arch_weights_sampled, True, coeff_priors
        )

        mean_mu = jnp.mean(prior_samples["obs"], axis=0)
        hpdi_mu = hpdi(prior_samples["obs"], 0.9, 0)

        # PLOT RESULTS

        # plot prediction
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
                lines.append(ax.plot(log["a_" + primitive], label="a_" + primitive)[0])
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
                lines.append(ax.plot(log["b_" + primitive], label="b_" + primitive)[0])
        lines.append(ax.plot(log["b"], label="b")[0])
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
