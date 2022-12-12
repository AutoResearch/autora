from itertools import product
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from autora.cycle import Cycle


def get_variable_index(
    cycle: Cycle,
) -> Tuple[List[Tuple[int, str, str]], List[Tuple[int, str, str]]]:
    """
    Extracts information about independent and dependent variables from the cycle object.
    Returns a list of tuples of (index, name, units). The index is in reference to the column number
    in the observed value arrays.
    Args:
        cycle: AER Cycle object that has been run

    Returns: Tuple of 2 lists of tuples

    """
    l_iv = [
        (i, s.name, s.units)
        for i, s in enumerate(cycle.data.metadata.independent_variables)
    ]
    n_iv = len(l_iv)
    l_dv = [
        (i + n_iv, s.name, s.units)
        for i, s in enumerate(cycle.data.metadata.dependent_variables)
    ]
    return l_iv, l_dv


def observed_to_df(cycle: Cycle) -> pd.DataFrame:
    """
    Concatenates observation data of cycles into a single dataframe with a field "cycle" with the
    cycle index.
    Args:
        cycle: AER Cycle object that has been run

    Returns: Dataframe

    """
    l_observations = cycle.data.observations
    l_agg = []

    for i, data in enumerate(l_observations):
        l_agg.append(pd.DataFrame(data).assign(cycle=i))

    return pd.concat(l_agg)


def min_max_observations(cycle: Cycle) -> List[Tuple[float, float]]:
    """
    Returns minimum and maximum of observed values for each independent variable.
    Args:
        cycle: AER Cycle object that has been run

    Returns: List of tuples

    """
    l_return = []
    iv_index = range(len(cycle.data.metadata.independent_variables))
    l_observations = cycle.data.observations
    # Get min and max of observation data
    # Min and max by cycle - All IVs
    l_mins = [np.min(s, axis=0) for s in l_observations]  # Arrays by columns
    l_maxs = [np.max(s, axis=0) for s in l_observations]
    # Min and max for all cycles by IVs
    for idx in iv_index:
        glob_min = np.min([s[idx] for s in l_mins])
        glob_max = np.max([s[idx] for s in l_maxs])
        l_return.append((glob_min, glob_max))

    return l_return


def generate_condition_space(cycle: Cycle, steps: int = 50) -> np.array:
    """
    Generates condition space based on the minimum and maximum of all observed data in AER Cycle.
    Args:
        cycle: AER Cycle object that has been run
        steps: Number of steps to define the condition space

    Returns: np.array

    """
    l_min_max = min_max_observations(cycle)
    l_space = []

    for min_max in l_min_max:
        l_space.append(np.linspace(min_max[0], min_max[1], steps))

    if len(l_space) > 1:
        return np.array(list(product(*l_space)))
    else:
        return l_space[0].reshape(-1, 1)


def generate_mesh_grid(cycle: Cycle, steps: int = 50) -> np.ndarray:
    """
    Generates a mesh grid based on the minimum and maximum of all observed data in AER Cycle.
    Args:
        cycle: AER Cycle object that has been run
        steps: Number of steps to define the condition space

    Returns: np.ndarray

    """
    l_min_max = min_max_observations(cycle)
    l_space = []

    for min_max in l_min_max:
        l_space.append(np.linspace(min_max[0], min_max[1], steps))

    return np.meshgrid(*l_space)


def theory_predict(cycle: Cycle, conditions: Sequence) -> dict:
    """
    Gets theory predictions over conditions space and saves results of each cycle to a dictionary.
    Args:
        cycle: AER Cycle object that has been run
        conditions: Condition space. Should be an array of grouped conditions.

    Returns: dict

    """
    d_predictions = {}
    for i, theory in enumerate(cycle.data.theories):
        d_predictions[i] = theory.predict(conditions)

    return d_predictions


def check_replace_default_kw(default: dict, user: dict) -> dict:
    """
    Combines the key/value pairs of two dictionaries, a default and user dictionary. Unique pairs
    are selected and user pairs take precedent over default pairs if matching keywords. Also works
    with nested dictionaries.

    Returns: dict
    """
    # Copy dict 1 to return dict
    d_return = default.copy()
    # Loop by keys in dict 2
    for key in user.keys():
        # If not in dict 1 add to the return dict
        if key not in default.keys():
            d_return.update({key: user[key]})
        else:
            # If value is a dict, recurse to check nested dict
            if isinstance(user[key], dict):
                d_return.update(
                    {key: check_replace_default_kw(default[key], user[key])}
                )
            # If not a dict update the default value with the value from dict 2
            else:
                d_return.update({key: user[key]})

    return d_return


def plot_results_panel_2d(
    cycle: Cycle,
    iv_name: Optional[str] = None,
    dv_name: Optional[str] = None,
    steps: int = 50,
    wrap: int = 4,
    spines: bool = False,
    subplot_kw: dict = {},
    scatter_kw1: dict = {},
    scatter_kw2: dict = {},
    line_kw: dict = {},
) -> plt.figure:
    """
    Generates a multi-panel plot with each panel showing results of an AER cycle. Observed data
    is plotted as a scatter plot with the current cycle colored differently than observed data from
    previous cycles. The current cycle's theory is plotted as a line over the range of the observed
    data.
    Args:
        cycle: AER Cycle object that has been run
        iv_name: Independent variable name. Name should match the name instantiated in the cycle
                    object. Default will select the first.
        dv_name: Single dependent variable name. Name should match the names instantiated in the
                    cycle object. Default will select the first DV.
        steps: Number of steps to define the condition space to plot the theory.
        wrap: Number of panels to appear in a row. Example: 9 panels with wrap=3 results in a
                3x3 grid.
        spines: Show axis spines for 2D plots, default False.
        subplot_kw: Dictionary of keywords to pass to matplotlib 'subplot' function
        scatter_kw1: Dictionary of keywords to pass to matplotlib 'scatter' function that plots the
                    data points from previous cycles.
        scatter_kw2: Dictionary of keywords to pass to matplotlib 'scatter' function that plots the
                    data points from the current cycle.
        line_kw: Dictionary of keywords to pass to matplotlib 'plot' function that plots the theory
                 line.

        TODO: 1. Save to file feature.

    Returns: matplotlib figure

    """
    n_cycles = len(cycle.data.theories)

    # ---Figure and plot params---
    # Set defaults, check and add user supplied keywords
    # Default keywords
    subplot_kw_defaults = {
        "gridspec_kw": {"bottom": 0.16},
        "sharex": True,
        "sharey": True,
    }
    scatter_kw1_defaults = {
        "color": "black",
        "s": 2,
        "alpha": 0.6,
        "label": "Previous Data",
    }
    scatter_kw2_defaults = {
        "color": "tab:orange",
        "s": 2,
        "alpha": 0.6,
        "label": "New Data",
    }
    line_kw_defaults = {"label": "Theory"}
    # Combine default and user supplied keywords
    d_kw = {}
    for d1, d2, key in zip(
        [
            subplot_kw_defaults,
            scatter_kw1_defaults,
            scatter_kw2_defaults,
            line_kw_defaults,
        ],
        [subplot_kw, scatter_kw1, scatter_kw2, line_kw],
        ["subplot_kw", "scatter_kw1", "scatter_kw2", "line_kw"],
    ):
        assert isinstance(d1, dict)
        assert isinstance(d2, dict)
        d_kw[key] = check_replace_default_kw(d1, d2)

    # ---Extract IVs and DV metadata and indexes---
    ivs, dvs = get_variable_index(cycle)
    if iv_name:
        iv = [s for s in ivs if s[1] == iv_name][0]
    else:
        iv = [ivs[0]][0]
    if dv_name:
        dv = [s for s in dvs if s[1] == dv_name][0]
    else:
        dv = [dvs[0]][0]
    iv_label = f"{iv[1]} {iv[2]}"
    dv_label = f"{dv[1]} {dv[2]}"

    # Create a dataframe of observed data from cycle
    df_observed = observed_to_df(cycle)

    # Generate IV space
    condition_space = generate_condition_space(cycle, steps=steps)
    # Get theory predictions over space
    d_predictions = theory_predict(cycle, condition_space)

    # Subplot configurations
    if n_cycles < wrap:
        shape = (1, n_cycles)
    else:
        shape = (int(np.ceil(n_cycles / wrap)), wrap)
    fig, axs = plt.subplots(*shape, **d_kw["subplot_kw"])

    # Loop by panel
    for i, ax in enumerate(axs.flat):
        if i + 1 <= n_cycles:

            # ---Plot observed data---
            # Independent variable values
            x_vals = df_observed.loc[:, iv[0]]
            # Dependent values masked by current cycle vs previous data
            dv_previous = np.ma.masked_where(
                df_observed["cycle"] >= i, df_observed[dv[0]]
            )
            dv_current = np.ma.masked_where(
                df_observed["cycle"] != i, df_observed[dv[0]]
            )
            # Plotting scatter
            ax.scatter(x_vals, dv_previous, **d_kw["scatter_kw1"])
            ax.scatter(x_vals, dv_current, **d_kw["scatter_kw2"])

            # ---Plot Theory---
            conditions = condition_space[:, iv[0]]
            ax.plot(conditions, d_predictions[i], **d_kw["line_kw"])

            # Label Panels
            ax.text(0.05, 1, f"Cycle {i}", ha="left", va="top", transform=ax.transAxes)

            if not spines:
                ax.spines.right.set_visible(False)
                ax.spines.top.set_visible(False)

        else:
            ax.axis("off")

    # Super Labels
    fig.supxlabel(iv_label, y=0.07)
    fig.supylabel(dv_label)

    # Legend
    fig.legend(
        ["Previous Data", "New Data", "Theory"],
        ncols=3,
        bbox_to_anchor=(0.5, 0),
        loc="lower center",
    )

    return fig


def plot_results_panel_3d(
    cycle: Cycle,
    iv_names: Optional[List[str]] = None,
    dv_name: Optional[str] = None,
    steps: int = 50,
    wrap: int = 4,
    view: Optional[Tuple[float, float]] = None,
    subplot_kw: dict = {},
    scatter_kw1: dict = {},
    scatter_kw2: dict = {},
    surface_kw: dict = {},
) -> plt.figure:
    """
    Generates a multi-panel plot with each panel showing results of an AER cycle. Observed data
    is plotted as a scatter plot with the current cycle colored differently than observed data from
    previous cycles. The current cycle's theory is plotted as a line over the range of the observed
    data.
    Args:

        cycle: AER Cycle object that has been run
        iv_names: List of up to 2 independent variable names. Names should match the names
                    instantiated in the cycle object. Default will select up to the first two.
        dv_name: Single DV name. Name should match the names instantiated in the cycle object.
                    Default will select the first DV
        steps: Number of steps to define the condition space to plot the theory.
        wrap: Number of panels to appear in a row. Example: 9 panels with wrap=3 results in a
                3x3 grid.
        view: Tuple of elevation angle and azimuth to change the viewing angle of the plot.
        subplot_kw: Dictionary of keywords to pass to matplotlib 'subplot' function
        scatter_kw1: Dictionary of keywords to pass to matplotlib 'scatter' function that plots the
                    data points from previous cycles.
        scatter_kw2: Dictionary of keywords to pass to matplotlib 'scatter' function that plots the
                    data points from the current cycle.
        surface_kw: Dictionary of keywords to pass to matplotlib 'plot_surface' function that plots
                    the theory plane.

    Returns: matplotlib figure

    """
    n_cycles = len(cycle.data.theories)

    # ---Figure and plot params---
    # Set defaults, check and add user supplied keywords
    # Default keywords
    subplot_kw_defaults = {
        "gridspec_kw": {"bottom": 0.16},
        "subplot_kw": {"projection": "3d"},
    }
    scatter_kw1_defaults = {"color": "black", "s": 2, "label": "Previous Data"}
    scatter_kw2_defaults = {"color": "tab:orange", "s": 2, "label": "New Data"}
    surface_kw_defaults = {"alpha": 0.5, "label": "Theory"}
    # Combine default and user supplied keywords
    d_kw = {}
    for d1, d2, key in zip(
        [
            subplot_kw_defaults,
            scatter_kw1_defaults,
            scatter_kw2_defaults,
            surface_kw_defaults,
        ],
        [subplot_kw, scatter_kw1, scatter_kw2, surface_kw],
        ["subplot_kw", "scatter_kw1", "scatter_kw2", "surface_kw"],
    ):
        assert isinstance(d1, dict)
        assert isinstance(d2, dict)
        d_kw[key] = check_replace_default_kw(d1, d2)

    # ---Extract IVs and DV metadata and indexes---
    ivs, dvs = get_variable_index(cycle)
    if iv_names:
        iv = [s for s in ivs if s[1] == iv_names]
    else:
        iv = ivs[:2]
    if dv_name:
        dv = [s for s in dvs if s[1] == dv_name][0]
    else:
        dv = [dvs[0]][0]
    iv_labels = [f"{s[1]} {s[2]}" for s in iv]
    dv_label = f"{dv[1]} {dv[2]}"

    # Create a dataframe of observed data from cycle
    df_observed = observed_to_df(cycle)

    # Generate IV Mesh Grid
    x1, x2 = generate_mesh_grid(cycle, steps=steps)

    # Get theory predictions over space
    d_predictions = theory_predict(cycle, np.column_stack((x1.ravel(), x2.ravel())))

    # Subplot configurations
    if n_cycles < wrap:
        shape = (1, n_cycles)
    else:
        shape = (int(np.ceil(n_cycles / wrap)), wrap)

    fig, axs = plt.subplots(*shape, **d_kw["subplot_kw"])

    # Loop by panel
    for i, ax in enumerate(axs.flat):
        if i + 1 <= n_cycles:

            # ---Plot observed data---
            # Independent variable values
            l_x = [df_observed.loc[:, s[0]] for s in iv]
            # Dependent values masked by current cycle vs previous data
            dv_previous = np.ma.masked_where(
                df_observed["cycle"] >= i, df_observed[dv[0]]
            )
            dv_current = np.ma.masked_where(
                df_observed["cycle"] != i, df_observed[dv[0]]
            )
            # Plotting scatter
            ax.scatter(*l_x, dv_previous, **d_kw["scatter_kw1"])
            ax.scatter(*l_x, dv_current, **d_kw["scatter_kw2"])

            # ---Plot Theory---

            ax.plot_surface(
                x1, x2, d_predictions[i].reshape(x1.shape), **d_kw["surface_kw"]
            )
            # Axis labels
            ax.set_xlabel(iv_labels[0])
            ax.set_ylabel(iv_labels[1])
            ax.set_zlabel(dv_label)

            # Viewing angle
            if view:
                ax.view_init(*view)

        else:
            ax.axis("off")

    # Legend
    handles, labels = axs.flatten()[0].get_legend_handles_labels()
    legend_elements = [
        handles[0],
        handles[1],
        Patch(facecolor=handles[2].get_facecolors()[0]),
    ]
    fig.legend(
        handles=legend_elements,
        labels=labels,
        ncols=3,
        bbox_to_anchor=(0.5, 0),
        loc="lower center",
    )

    return fig
