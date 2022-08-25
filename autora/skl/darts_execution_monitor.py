from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


def create_basic_execution_monitor():

    arch_weight_history = list()
    loss_history = list()
    epoch_history = list()
    primitives = list()

    def execution_monitor_(network, architect, epoch, **kwargs):
        # collect data for visualization
        epoch_history.append(epoch)
        arch_weight_history.append(
            network.arch_parameters()[0].detach().numpy().copy()[np.newaxis, :]
        )
        loss_history.append(architect.current_loss)

        nonlocal primitives
        primitives = network.primitives

    def display_execution_monitor_():
        nonlocal primitives
        loss_fig, loss_ax = plt.subplots(1, 1)
        loss_ax.plot(loss_history)

        arch_weight_history_array = np.vstack(arch_weight_history)
        num_epochs, num_edges, num_primitives = arch_weight_history_array.shape

        subplots_per_side = int(np.ceil(np.sqrt(num_edges)))

        arch_fig, arch_axes = plt.subplots(
            subplots_per_side, subplots_per_side, sharex=True, sharey=True
        )

        for (edge_i, ax) in zip(range(num_edges), arch_axes.reshape(-1)):
            for primitive_i in range(num_primitives):
                print(f"{edge_i}, {primitive_i}, {ax}")
                ax.plot(
                    arch_weight_history_array[:, edge_i, primitive_i],
                    label=f"{primitives[primitive_i]}",
                )

            ax.legend(loc="upper center")

        return SimpleNamespace(
            loss_fig=loss_fig,
            loss_ax=loss_ax,
            arch_fig=arch_fig,
            arch_axes=arch_axes,
        )

    return execution_monitor_, display_execution_monitor_
