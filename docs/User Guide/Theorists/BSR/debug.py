import numpy as np
from autora.skl.bsr import BSRRegressor
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

X = np.random.uniform(-3, 3, (1000, 1))
# y = np.sin(X[:, 0]) + np.cos(X[:, 1])
y = np.cos(X[:, 0]) + X[:, 0] ** 3

bsr_estimator = BSRRegressor(
    tree_num=3,
    itr_num=100,
    val=100,
    beta=-1,
    show_log=True
)

bsr_estimator.fit(X, y)
pred_y = bsr_estimator.predict(X)

show = False
if show:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], pred_y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], y)

    plt.show()
