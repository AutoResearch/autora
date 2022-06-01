import sys

from graphviz import Digraph

from AER_theorist.darts.operations import *


def plot(
    genotype,
    filename,
    fileFormat="pdf",
    viewFile=None,
    full_label=False,
    param_list=(),
    input_labels=(),
    out_dim=None,
    out_fnc=None,
):

    decimals_to_display = 2
    format_string = "{:." + "{:.0f}".format(decimals_to_display) + "f}"

    g = Digraph(
        format=fileFormat,
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
    g.body.extend(["rankdir=LR"])

    for input_node in input_labels:
        g.node(input_node, fillcolor="#F1EDB9")  # fillcolor='darkseagreen2'
    # assert len(genotype) % 2 == 0

    # determine number of steps (intermediate nodes)
    steps = 0
    for op, j in genotype:
        if j == 0:
            steps += 1

    for i in range(steps):
        g.node(str(i), fillcolor="#BBCCF9")  # fillcolor='lightblue'

    params_counter = 0
    n = len(input_labels)
    start = 0
    for i in range(steps):
        end = start + n
        # for k in [2*i, 2*i + 1]:
        for k in range(
            start, end
        ):  # adapted this iteration from get_genotype() in model_search.py
            op, j = genotype[k]
            if j < len(input_labels):
                u = input_labels[j]
            else:
                u = str(j - len(input_labels))
            v = str(i)
            params_counter = k
            if op is not "none":
                op_label = op
                if full_label:
                    params = param_list[
                        start + j
                    ]  # note: genotype order and param list order don't align
                    op_label = get_operation_label(
                        op, params, decimals=decimals_to_display
                    )
                    g.edge(u, v, label=op_label, fillcolor="gray")
                else:
                    g.edge(
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
            g.node(out_str, fillcolor="#CBE7C7")  # fillcolor='palegoldenrod'
            out_nodes.append(out_str)

    for i in range(steps):
        if full_label:
            params_org = param_list[params_counter + 1 + i]  # count from k
            for out_idx, out_str in enumerate(out_nodes):
                params = list()
                params.append(params_org[0][out_idx])
                op_label = get_operation_label(
                    "classifier", params, decimals=decimals_to_display
                )
                g.edge(str(i), out_str, label=op_label, fillcolor="gray")
        else:
            for out_idx, out_str in enumerate(out_nodes):
                g.edge(str(i), out_str, label="linear", fillcolor="gray")

    if viewFile is None:
        if fileFormat == "pdf":
            viewFile = True
        else:
            viewFile = False

    g.render(filename, view=viewFile)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval("genotypes.{}".format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
