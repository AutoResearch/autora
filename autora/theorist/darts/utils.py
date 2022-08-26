import csv
import glob
import os
import shutil
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn as nn

from autora.theorist.darts.model_search import Network
from autora.variable import ValueType


def create_output_file_name(
    file_prefix: str,
    log_version: int = None,
    weight_decay: float = None,
    k: int = None,
    seed: int = None,
    theorist: str = None,
) -> str:
    """
    Creates a file name for the output file of a theorist study.

    Arguments:
        file_prefix: prefix of the file name
        log_version: log version of the theorist run
        weight_decay: weight decay of the model
        k: number of nodes in the model
        seed: seed of the model
        theorist: name of the DARTS variant
    """

    output_str = file_prefix

    if theorist is not None:
        output_str += "_" + str(theorist)

    if log_version is not None:
        output_str += "_v_" + str(log_version)

    if weight_decay is not None:
        output_str += "_wd_" + str(weight_decay)

    if k is not None:
        output_str += "_k_" + str(k)

    if k is not None:
        output_str += "_s_" + str(seed)

    return output_str


def assign_slurm_instance(
    slurm_id: int,
    arch_weight_decay_list: List,
    num_node_list: List,
    seed_list: List,
) -> Tuple:
    """
    Determines the meta-search parameters based on the slum job id.

    Arguments:
        slurm_id: slurm job id
        arch_weight_decay_list: list of weight decay values
        num_node_list: list of number of nodes
        seed_list: list of seeds
    """

    seed_id = np.floor(
        slurm_id / (len(num_node_list) * len(arch_weight_decay_list))
    ) % len(seed_list)
    k_id = np.floor(slurm_id / (len(arch_weight_decay_list))) % len(num_node_list)
    weight_decay_id = slurm_id % len(arch_weight_decay_list)

    return (
        arch_weight_decay_list[int(weight_decay_id)],
        int(num_node_list[int(k_id)]),
        int(seed_list[int(seed_id)]),
    )


def sigmid_mse(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Returns the MSE loss for a sigmoid output.

    Arguments:
        output: output of the model
        target: target of the model
    """
    m = nn.Sigmoid()
    output = m(output)
    loss = torch.mean((output - target) ** 2)
    return loss


def compute_BIC(
    output_type: ValueType,
    model: torch.nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Returns the Bayesian information criterion for a DARTS model.

    Arguments:
        output_type: output type of the dependent variable
        model: model to compute the BIC for
        input: input of the model
        target: target of the model
    """

    # compute raw model output
    classifier_output = model(input)

    # compute associated probability
    m = get_output_format(output_type)
    prediction = m(classifier_output).detach()

    k, _, _ = model.countParameters()  # for most likely architecture

    if output_type == ValueType.CLASS:
        target_flattened = torch.flatten(target.long())
        llik = 0
        for idx in range(len(target_flattened)):
            lik = prediction[idx, target_flattened[idx]]
            llik += np.log(lik)
        n = len(target_flattened)  # number of data points

        BIC = np.log(n) * k - 2 * llik
        BIC = BIC

    elif output_type == ValueType.PROBABILITY_SAMPLE:
        llik = 0
        for idx in range(len(target)):

            # fail safe if model doesn't produce probabilities
            if prediction[idx] > 1:
                prediction[idx] = 1
            elif prediction[idx] < 0:
                prediction[idx] = 0

            if target[idx] == 1:
                lik = prediction[idx]
            elif target[idx] == 0:
                lik = 1 - prediction[idx]
            else:
                raise Exception("Target must contain either zeros or ones.")
            llik += np.log(lik)
        n = len(target)  # number of data points

        BIC = np.log(n) * k - 2 * llik
        BIC = BIC[0]

    else:
        raise Exception(
            "BIC computation not implemented for output type "
            + str(ValueType.PROBABILITY)
            + "."
        )

    return BIC

    # old


def compute_BIC_AIC(
    soft_targets: np.array, soft_prediction: np.array, model: Network
) -> Tuple:
    """
    Returns the Bayesian information criterion (BIC) as well as the
    Aikaike information criterion (AIC) for a DARTS model.

    Arguments:
        soft_targets: soft target of the model
        soft_prediction: soft prediction of the model
        model: model to compute the BIC and AIC for
    """

    lik = np.sum(
        np.multiply(soft_prediction, soft_targets), axis=1
    )  # likelihood of data given model
    llik = np.sum(np.log(lik))  # log likelihood
    n = len(lik)  # number of data points
    k, _, _ = model.count_parameters()  # for most likely architecture

    BIC = np.log(n) * k - 2 * llik

    AIC = 2 * k - 2 * llik

    return BIC, AIC


def cross_entropy(pred: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """
    Returns the cross entropy loss for a soft target.

    Arguments:
        pred: prediction of the model
        soft_targets: soft target of the model
    """
    # assuming pred and soft_targets are both Variables with shape (batchsize, num_of_classes),
    # each row of pred is predicted logits and each row of soft_targets is a discrete distribution.
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


class AvgrageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        """
        Initializes the average meter.
        """
        self.reset()

    def reset(self):
        """
        Resets the average meter.
        """
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val: float, n: int = 1):
        """
        Updates the average meter.

        Arguments:
            val: value to update the average meter with
            n: number of times to update the average meter
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple = (1,)) -> List:
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Arguments:
        output: output of the model
        target: target of the model
        topk: values of k to compute the accuracy at
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters_in_MB(model: Network) -> int:
    """
    Returns the number of parameters for a model.

    Arguments:
        model: model to count the parameters for
    """
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


def save(model: torch.nn.Module, model_path: str, exp_folder: str = None):
    """
    Saves a model to a file.

    Arguments:
        model: model to save
        model_path: path to save the model to
        exp_folder: general experiment directory to save the model to
    """
    if exp_folder is not None:
        os.chdir("exps")  # Edit SM 10/23/19: use local experiment directory
    torch.save(model.state_dict(), model_path)
    if exp_folder is not None:
        os.chdir("..")  # Edit SM 10/23/19: use local experiment directory


def load(model: torch.nn.Module, model_path: str):
    """
    Loads a model from a file.
    """
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(
    path: str,
    scripts_to_save: List = None,
    parent_folder: str = "exps",
    results_folder: str = None,
):
    """
    Creates an experiment directory and saves all necessary scripts and files.

    Arguments:
        path: path to save the experiment directory to
        scripts_to_save: list of scripts to save
        parent_folder: parent folder for the experiment directory
        results_folder: folder for the results of the experiment
    """
    os.chdir(parent_folder)  # Edit SM 10/23/19: use local experiment directory
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))

    if results_folder is not None:
        try:
            os.mkdir(os.path.join(path, results_folder))
        except OSError:
            pass

    if scripts_to_save is not None:
        try:
            os.mkdir(os.path.join(path, "scripts"))
        except OSError:
            pass
        os.chdir("..")  # Edit SM 10/23/19: use local experiment directory
        for script in scripts_to_save:
            dst_file = os.path.join(
                parent_folder, path, "scripts", os.path.basename(script)
            )
            shutil.copyfile(script, dst_file)


def read_log_files(results_path: str, winning_architecture_only: bool = False) -> Tuple:
    """
    Reads the log files from an experiment directory and returns the results.

    Arguments:
        results_path: path to the experiment results directory
        winning_architecture_only: if True, only the winning architecture is returned
    """

    current_wd = os.getcwd()

    os.chdir(results_path)
    filelist = glob.glob("*.{}".format("csv"))

    model_name_list = list()
    loss_list = list()
    BIC_list = list()
    AIC_list = list()

    # READ LOG FILES

    print("Reading log files... ")
    for file in filelist:

        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            for row in readCSV:
                if winning_architecture_only is False or "sample0" in row[0]:
                    model_name_list.append(row[0])
                    loss_list.append(float(row[1]))
                    BIC_list.append(float(row[2].replace("[", "").replace("]", "")))
                    AIC_list.append(float(row[3].replace("[", "").replace("]", "")))

    os.chdir(current_wd)

    return (model_name_list, loss_list, BIC_list, AIC_list)


def get_best_fitting_models(
    model_name_list: List,
    loss_list: List,
    BIC_list: List,
    topk: int,
) -> Tuple:
    """
    Returns the topk best fitting models.

    Arguments:
        model_name_list: list of model names
        loss_list: list of loss values
        BIC_list: list of BIC values
        topk: number of topk models to return
    """

    topk_losses = sorted(zip(loss_list, model_name_list), reverse=False)[:topk]
    res = list(zip(*topk_losses))
    topk_losses_names = res[1]

    topk_BICs = sorted(zip(BIC_list, model_name_list), reverse=False)[:topk]
    res = list(zip(*topk_BICs))
    topk_BICs_names = res[1]

    return (topk_losses_names, topk_BICs_names)


def format_input_target(
    input: torch.tensor, target: torch.tensor, criterion: Callable
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Formats the input and target for the model.

    Args:
        input: input to the model
        target: target of the model
        criterion: criterion to use for the model

    Returns:
        input: formatted input and target for the model

    """

    if isinstance(criterion, nn.CrossEntropyLoss):
        target = target.squeeze()

    return (input, target)


LOSS_FUNCTION_MAPPING = {
    ValueType.REAL: nn.MSELoss(),
    ValueType.PROBABILITY: sigmid_mse,
    ValueType.PROBABILITY_SAMPLE: sigmid_mse,
    ValueType.PROBABILITY_DISTRIBUTION: cross_entropy,
    ValueType.CLASS: nn.CrossEntropyLoss(),
    ValueType.SIGMOID: sigmid_mse,
}


def get_loss_function(outputType: ValueType):
    """
    Returns the loss function for the given output type of a dependent variable.

    Arguments:
        outputType: output type of the dependent variable
    """

    return LOSS_FUNCTION_MAPPING.get(outputType, nn.MSELoss())


OUTPUT_FORMAT_MAPPING = {
    ValueType.REAL: nn.Identity(),
    ValueType.PROBABILITY: nn.Sigmoid(),
    ValueType.PROBABILITY_SAMPLE: nn.Sigmoid(),
    ValueType.PROBABILITY_DISTRIBUTION: nn.Softmax(dim=1),
    ValueType.CLASS: nn.Softmax(dim=1),
    ValueType.SIGMOID: nn.Sigmoid(),
}


def get_output_format(outputType: ValueType):
    """
    Returns the output format (activation function of the final output layer)
    for the given output type of a dependent variable.

    Arguments:
        outputType: output type of the dependent variable
    """

    return OUTPUT_FORMAT_MAPPING.get(outputType, nn.MSELoss())


OUTPUT_STR_MAPPING = {
    ValueType.REAL: "",
    ValueType.PROBABILITY: "Sigmoid",
    ValueType.PROBABILITY_SAMPLE: "Sigmoid",
    ValueType.PROBABILITY_DISTRIBUTION: "Softmax",
    ValueType.CLASS: "Softmax",
    ValueType.SIGMOID: "Sigmoid",
}


def get_output_str(outputType: ValueType) -> str:
    """
    Returns the output string for the given output type of a dependent variable.

    Arguments:
        outputType: output type of the dependent variable
    """

    return OUTPUT_STR_MAPPING.get(outputType, "")
