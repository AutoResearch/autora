import logging
import typing

from graphviz import Digraph

from autora.theorist.darts.operations import Genotype, get_operation_label

_logger = logging.getLogger(__name__)


def plot(
    genotype: Genotype,
    filename: str,
    file_format: str = "pdf",
    view_file: bool = None,
    full_label: bool = False,
    param_list: typing.Tuple = (),
    input_labels: typing.Tuple = (),
    out_dim: int = None,
    out_fnc: str = None,
):
    """
    Generates a graphviz plot for a DARTS model based on the genotype of the model.

    Arguments:
        genotype: the genotype of the model
        filename: the filename of the output file
        file_format: the format of the output file
        view_file: if True, the plot will be displayed in a window
        full_label: if True, the labels of the nodes will be the full name of the operation
            (including the coefficients)
        param_list: a list of parameters to be included in the labels of the nodes
        input_labels: a list of labels to be included in the input nodes
        out_dim: the number of output nodes of the model
        out_fnc: the (activation) function to be used for the output nodes
    """

    g = darts_model_plot(
        genotype=genotype,
        full_label=full_label,
        param_list=param_list,
        input_labels=input_labels,
        out_dim=out_dim,
        out_fnc=out_fnc,
    )

    if view_file is None:
        if file_format == "pdf":
            view_file = True
        else:
            view_file = False

    g.render(filename, view=view_file, format=file_format)


def darts_model_plot(
    genotype: Genotype,
    full_label: bool = False,
    param_list: typing.Sequence = (),
    input_labels: typing.Sequence = (),
    out_dim: int = None,
    out_fnc: str = None,
    decimals_to_display: int = 2,
) -> Digraph:
    """
    Generates a graphviz plot for a DARTS model based on the genotype of the model.

    Arguments:
        genotype: the genotype of the model
        full_label: if True, the labels of the nodes will be the full name of the operation
            (including the coefficients)
        param_list: a list of parameters to be included in the labels of the nodes
        input_labels: a list of labels to be included in the input nodes
        out_dim: the number of output nodes of the model
        out_fnc: the (activation) function to be used for the output nodes
        decimals_to_display: number of decimals to include in parameter values on plot
    """

    format_string = "{:." + "{:.0f}".format(decimals_to_display) + "f}"

    graph = Digraph(
        edge_attr=dict(fontsize="20", fontname="times"),
        node_attr=dict(
            style="filled",
            shape="rect",
            align="center",
            fontsize="20",
            height="0.5",
            width="0.5",
            penwidth="2",
            fontname="times",
        ),
        engine="dot",
    )
    graph.body.extend(["rankdir=LR"])

    for input_node in input_labels:
        graph.node(input_node, fillcolor="#F1EDB9")  # fillcolor='darkseagreen2'
    # assert len(genotype) % 2 == 0

    # determine number of steps (intermediate nodes)
    steps = 0
    for op, j in genotype:
        if j == 0:
            steps += 1

    for i in range(steps):
        graph.node("k" + str(i + 1), fillcolor="#BBCCF9")  # fillcolor='lightblue'

    params_counter = 0
    n = len(input_labels)
    start = 0
    for i in range(steps):
        end = start + n
        _logger.debug(start, end)
        # for k in [2*i, 2*i + 1]:
        for k in range(
            start, end
        ):  # adapted this iteration from get_genotype() in model_search.py
            _logger.debug(genotype, k)
            op, j = genotype[k]
            if j < len(input_labels):
                u = input_labels[j]
            else:
                u = "k" + str(j - len(input_labels) + 1)
            v = "k" + str(i + 1)
            params_counter = k
            if op != "none":
                op_label = op
                if full_label:
                    params = param_list[
                        start + j
                    ]  # note: genotype order and param list order don't align
                    op_label = get_operation_label(
                        op, params, decimals=decimals_to_display
                    )
                    graph.edge(u, v, label=op_label, fillcolor="gray")
                else:
                    graph.edge(
                        u,
                        v,
                        label="(" + str(j + start) + ") " + op_label,
                        fillcolor="gray",
                    )  # '(' + str(k) + ') '
        start = end
        n += 1

    # determine output nodes

    out_nodes = list()
    if out_dim is None:
        out_nodes.append("out")
    else:
        biases = None
        if full_label:
            params = param_list[params_counter + 1]
            if len(params) > 1:
                biases = params[1]  # first node contains biases

        for idx in range(out_dim):
            out_str = ""
            # specify node ID
            if out_fnc is not None:
                out_str = out_str + out_fnc + "(r_" + str(idx)
            else:
                out_str = "(r_" + str(idx)

            if out_dim == 1:
                if out_fnc is not None:
                    out_str = "P(detected) = " + out_fnc + "(x"
                else:
                    # out_str = 'dx_1 = (x'
                    out_str = "P_n = (x"

            # if available, add bias
            if biases is not None:
                out_str = out_str + " + " + format_string.format(biases[idx]) + ")"
            else:
                out_str = out_str + ")"

            # add node
            graph.node(out_str, fillcolor="#CBE7C7")  # fillcolor='palegoldenrod'
            out_nodes.append(out_str)

    for i in range(steps):
        u = "k" + str(i + 1)
        if full_label:
            params_org = param_list[params_counter + 1 + i]  # count from k
            for out_idx, out_str in enumerate(out_nodes):
                params = list()
                params.append(params_org[0][out_idx])
                op_label = get_operation_label(
                    "classifier", params, decimals=decimals_to_display
                )
                graph.edge(u, out_str, label=op_label, fillcolor="gray")
        else:
            for out_idx, out_str in enumerate(out_nodes):
                graph.edge(u, out_str, label="linear", fillcolor="gray")

    return graph
