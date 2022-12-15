import numpy as np
from utils.node import Node
from scipy.stats import invgamma, norm

# TODO: it's only a demo page now that cannot function well.
def grow(node, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b):
    depth = node.depth

    # deciding the number of child nodes
    if node.depth > 0:
        prob = 1 / np.power((1 + depth), -beta)

        test = np.random.uniform(0, 1, 1)
        if test > prob:  # terminal
            node.feature = np.random.randint(0, nfeature, size=1)
            node.type = 0
        else:
            op_ind = np.random.choice(np.arange(len(Ops)), p=Op_weights)
            node.operator = Ops[op_ind]
            node.type = Op_type[op_ind]
            node.op_ind = op_ind

    else:  # root node, sure to split
        op_ind = np.random.choice(np.arange(len(Ops)), p=Op_weights)
        node.operator = Ops[op_ind]
        node.type = Op_type[op_ind]
        node.op_ind = op_ind

    # grow recursively
    if node.type == 0:
        node.feature = np.random.randint(0, nfeature, size=1)

    elif node.type == 1:
        node.left = Node(depth + 1)
        node.left.parent = node
        if node.operator == "ln":  # linear parameters
            node.a = norm.rvs(loc=1, scale=np.sqrt(sigma_a))
            node.b = norm.rvs(loc=0, scale=np.sqrt(sigma_b))
        grow(node.left, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)

    else:  # node.type=2
        node.left = Node(depth + 1)
        node.left.parent = node
        # node.left.order = len(Tree)
        node.right = Node(depth + 1)
        node.right.parent = node
        # node.right.order = len(Tree)
        grow(node.left, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
        grow(node.right, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)