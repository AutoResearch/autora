import logging
from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .mcmc import Tree
from .parallel import Parallel

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def run(
    pms: Parallel, num_steps: int, thinning: int = 1, verbose=True
) -> Tuple[Tree, float, List[float]]:
    """

    Args:
        pms: Parallel Machine Scientist (BMS is essentially a wrapper for pms)
        num_steps: number of epochs / mcmc step & tree swap iterations
        thinning: number of epochs between recording model loss to the trace

    Returns:
        model: The equation which best describes the data_closed_loop
        model_len: (defined as description length) loss function score
        desc_len: Record of loss function score over time

    """
    desc_len, model, model_len = [], pms.t1, np.inf
    from datetime import datetime
    laps = []
    times = []
    splits = []
    start_time = datetime.now()
    for n in tqdm(range(num_steps)):
        # for tree in pms.trees.values():
        #     tree.fit_par = {}
        #     tree.representative = {}
        lap, split = pms.mcmc_step()
        laps.append(lap)
        splits.append(split)
        pms.tree_swap()
        stop = datetime.now()
        # laps.append((stop - start).total_seconds())
        times.append((datetime.now() - start_time).total_seconds())
        if num_steps % thinning == 0:  # sample less often if we thin more
            desc_len.append(pms.t1.E)  # Add the description length to the trace
        if pms.t1.E < model_len:  # Check if this is the MDL expression so far
            model, model_len = deepcopy(pms.t1), pms.t1.E
        _logger.debug("Finish iteration {}".format(n))
    if verbose:

        laps_over_time(laps)
        # splits_over_time(splits)
        speed_over_time(times)
    # loss_over_time(desc_len, times)
    return model, model_len, desc_len


def splits_over_time(splits):
    type_dict = {0:'No move',1:'prune root',2:'replace root',3:'node replacement',4:'et replacement'}
    tree_a_splits = []
    tree_b_splits = []
    for split in splits:
        tree_a_splits.append(split[0])
        tree_b_splits.append(split[-1])
    a_splits_type = [[] for _ in range(5)]
    b_splits_type = [[] for _ in range(5)]
    for split, type in tree_a_splits:
        a_splits_type[type].append(split)
    for split, type in tree_b_splits:
        b_splits_type[type].append(split)
    numpy_a_splits = np.zeros((5, len(tree_a_splits)))
    numpy_b_splits = np.zeros((5, len(tree_b_splits)))
    numpy_a_counter = np.zeros((5, len(tree_a_splits)))
    numpy_b_counter = np.zeros((5, len(tree_b_splits)))
    a_type_list = []
    b_type_list = []
    for idx, (split, type) in enumerate(tree_a_splits):
        a_type_list.append(type)
        if idx == 0:
            numpy_a_splits[type, 0] = split
            numpy_a_counter[type, 0] = 1
        else:
            if type not in range(5):
                print(type)
            for t in range(5):
                if t == type:
                    numpy_a_splits[t, idx] = split
                    numpy_a_counter[t, idx] = numpy_a_counter[t, idx-1] + 1
                else:
                    numpy_a_splits[t, idx] = numpy_a_splits[t, idx-1]
                    numpy_a_counter[t, idx] = numpy_a_counter[t, idx-1]
    for idx, (split, type) in enumerate(tree_b_splits):
        b_type_list.append(type)
        if idx == 0:
            numpy_b_splits[type, 0] = split
            numpy_b_counter[type, 0] = 1
        else:
            if type not in range(5):
                print(type)
            for t in range(5):
                if t == type:
                    numpy_b_splits[t, idx] = split
                    numpy_b_counter[t, idx] = numpy_b_counter[t, idx-1] + 1
                else:
                    numpy_b_splits[t, idx] = numpy_b_splits[t, idx-1]
                    numpy_b_counter[t, idx] = numpy_b_counter[t, idx-1]
    span = int(len(a_type_list)/5)
    a_type_rate = np.zeros((5, len(a_type_list)-span))
    b_type_rate = np.zeros((5, len(b_type_list)-span))
    for i in range(span, len(a_type_list)):
        for t in range(5):
            a_type_rate[t, i-span] = (numpy_a_counter[t, i] - numpy_a_counter[t, i-span])/span
            b_type_rate[t, i-span] = (numpy_b_counter[t, i] - numpy_b_counter[t, i - span]) / span
    for i in range(5):
        plt.plot(numpy_a_splits[i], c=(0, 0, (1+i)/(1+5), 1), label=type_dict[i])
        plt.plot(numpy_b_splits[i], c=((1+i)/(1+5), 0, 0, 1), label=type_dict[i])
    plt.legend(loc='upper left')
    plt.title('splits by move type')
    plt.show()
    rolling_a_splits = np.zeros((5, numpy_a_splits.shape[1]))
    rolling_b_splits = np.zeros((5, numpy_b_splits.shape[1]))
    for i in range(0, span):
        for t in range(5):
            rolling_a_splits[t, i] = numpy_a_splits[t, 0:i].mean()
            rolling_b_splits[t, i] = numpy_b_splits[t, 0:i].mean()
    for i in range(span, len(numpy_a_splits[0])):
        for t in range(5):
            rolling_a_splits[t, i] = numpy_a_splits[t, i-span:i].mean()
            rolling_b_splits[t, i] = numpy_b_splits[t, i-span:i].mean()
    for i in range(5):
        plt.plot(rolling_a_splits[i], c=(0, 0, (1+i)/(1+5), 1), label=type_dict[i])
        plt.plot(rolling_b_splits[i], c=((1+i)/(1+5), 0, 0, 1), label=type_dict[i])
    plt.legend(loc='upper left')
    plt.title('rolling average splits by move type (raw)')
    plt.show()
    numpy_a_splits[numpy_a_splits > 0.02] = 0.015
    numpy_b_splits[numpy_b_splits > 0.02] = 0.015
    for i in range(0, span):
        for t in range(5):
            rolling_a_splits[t, i] = numpy_a_splits[t, 0:i].mean()
            rolling_b_splits[t, i] = numpy_b_splits[t, 0:i].mean()
    for i in range(span, len(numpy_a_splits[0])):
        for t in range(5):
            rolling_a_splits[t, i] = numpy_a_splits[t, i-span:i].mean()
            rolling_b_splits[t, i] = numpy_b_splits[t, i-span:i].mean()
    for i in range(5):
        plt.plot(rolling_a_splits[i], c=(0, 0, (1+i)/(1+5), 1), label=type_dict[i])
        plt.plot(rolling_b_splits[i], c=((1+i)/(1+5), 0, 0, 1), label=type_dict[i])
    plt.legend(loc='upper left')
    plt.title('rolling average splits by move type (filtered)')
    plt.show()
    for i in range(5):
        plt.plot(numpy_a_counter[i], c=(0, 0, (1+i)/(1+5), 1), label=type_dict[i])
        plt.plot(numpy_b_counter[i], c=((1+i)/(1+5), 0, 0, 1), label=type_dict[i])
    plt.legend(loc='upper left')
    plt.title('cumulative frequency of move by move type')
    plt.show()
    for i in range(5):
        plt.plot(a_type_rate[i], c=(0, 0, (1+i)/(1+5), 1), label=type_dict[i])
        plt.plot(b_type_rate[i], c=((1+i)/(1+5), 0, 0, 1), label=type_dict[i])
    plt.legend(loc='upper left')
    plt.title('rolling frequency of move type')
    plt.show()


def laps_over_time(laps):
    np_laps = np.array(laps)
    np_avg = np.zeros(np_laps.shape)
    np_laps[np_laps > 0.02] = 0.02
    print(np_laps.mean(axis=0))
    # avg_span = 200
    avg_span = int(200*(len(laps)/3000))
    for i in range(avg_span):
        for j in range(len(np_laps[0])):
            for k in range(i):
                np_avg[i, j] += np_laps[k, j]/(i+1)
    for i in range(avg_span, len(np_laps)):
        for j in range(len(np_laps[0])):
            for k in range(avg_span):
                np_avg[i, j] += np_laps[i-k, j]/avg_span
    print(np_avg[0])
    for j in range(20):
        plt.plot(range(len(laps)), np_avg[:, j], c=(j/20, 0, 1-j/20,1), label=str(j+1))
    plt.legend(loc="upper left")
    # plt.plot(range(len(laps)), np_avg)
    plt.xlabel("Epochs passed")
    plt.ylabel("Average (200) Lap Time (Seconds)")
    plt.title("Lap Durations over Epochs")
    plt.show()


def speed_over_time(times):
    deltas = []
    for t in range(20, len(times)):
        deltas.append((times[t] - times[t-20])/20)
    plt.plot(range(len(deltas)), [1/d for d in deltas])
    plt.xlabel("Epochs passed")
    plt.ylabel("Average Epoch Training Speed (Epochs/Second)")
    plt.title("Training Speed over Epochs")
    plt.show()


def epochs_over_time(times):
    plt.plot(times, range(len(times)))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Epochs")
    plt.title("Epochs over Time")
    plt.show()


def loss_over_time(desc_len, times):
    assert len(desc_len) == len(times)
    plt.plot(times, desc_len)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Description Length")
    plt.title("Loss Over Training")
    plt.show()


def present_results(model: Tree, model_len: float, desc_len: List[float]) -> None:
    """
    Prints out the best equation, its description length,
    along with a plot of how this has progressed over the course of the search tasks

    Args:
        model: The equation which best describes the data_closed_loop
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


def predict(model: Tree, x: pd.DataFrame, y: pd.DataFrame) -> dict:
    """
    Maps independent variable data_closed_loop onto expected dependent variable data_closed_loop

    Args:
        model: The equation / function that best maps x onto y
        x: The independent variables of the data_closed_loop
        y: The dependent variable of the data_closed_loop

    Returns: Predicted values for y given x and the model as trained
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(model.predict(x), y)

    all_y = np.append(y, model.predict(x))
    y_range = all_y.min().item(), all_y.max().item()
    plt.plot(y_range, y_range)

    plt.xlabel("MDL model predictions", fontsize=14)
    plt.ylabel("Actual values", fontsize=14)
    plt.show()
    return model.predict(x)
