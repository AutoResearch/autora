from autora_bsr.node import *
import numpy as np


def test_basic_inv_operation():
    root = Node(0)
    root.operator = inv_op
    root.type = NodeType.UNARY
    root.left = Node(1)
    root.left.type = NodeType.LEAF
    root.left.feature = 0
    test_x = np.array([[1, 2, 3], [4, 5, 6]])
    test_y = 1 / test_x[:, 0]
    assert (test_y - root.evaluate(test_x) < 1e-5).all()
