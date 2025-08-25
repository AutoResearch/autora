import logging
from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .mcmc import Tree
from .parallel import Parallel

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def run(
    pms: Parallel, num_steps: int, thinning: int = 100
) -> Tuple[Tree, float, List[float]]:
    """

    Args:
        pms: Parallel Machine Scientist (BMS is essentially a wrapper for pms)
        num_steps: number of epochs / mcmc step & tree swap iterations
        thinning: number of epochs between recording model loss to the trace

    Returns:
        model: The equation which best describes the data
        model_len: (defined as description length) loss function score
        desc_len: Record of loss function score over time

    """
    desc_len, model, model_len = [], pms.t1, np.inf
    for n in tqdm(range(num_steps)):
        pms.mcmc_step()
        pms.tree_swap()
        if num_steps % thinning == 0:  # sample less often if we thin more
            desc_len.append(pms.t1.E)  # Add the description length to the trace
        if pms.t1.E < model_len:  # Check if this is the MDL expression so far
            model, model_len = deepcopy(pms.t1), pms.t1.E
        _logger.debug("Finish iteration {}".format(n))
    return model, model_len, desc_len


def present_results(model: Tree, model_len: float, desc_len: List[float]) -> None:
    """
    Prints out the best equation, its description length,
    along with a plot of how this has progressed over the course of the search tasks

    Args:
        model: The equation which best describes the data
        model_len: The equation loss (defined as description length)
        desc_len: Record of equation loss over time

    Returns: Nothing

    """
    print("Best model:\t", model)
    print("Desc. length:\t", model_len)
    plt.figure(figsize=(15, 5))
    plt.plot(desc_len)
    plt.xlabel("MCMC step", fontsize=14)
    plt.ylabel("Description length", fontsize=14)
    plt.title("MDL model: $%s$" % model.latex())
    plt.show()
