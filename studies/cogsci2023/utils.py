import numpy as np
import scipy
from logit_theorist import LogitRegression
from MLP_theorist import MLP_theorist

from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.pooler import grid_pool, poppernet_pool
from autora.experimentalist.sampler import (
    dissimilarity_sampler,
    falsification_sampler,
    model_disagreement_sampler,
    nearest_values_sampler,
    random_sampler,
    uncertainty_sampler,
)
from autora.skl.bms import BMSRegressor
from autora.skl.darts import DARTSRegressor
from autora.variable import ValueType
from autora.theorist.bms.prior import get_priors


def sigmoid(x):
    return scipy.special.expit(x)


def fit_theorist(X, y, theorist_name, metadata, theorist_epochs=None):

    output_type = metadata.dependent_variables[0].type

    if theorist_name == "BMS" or theorist_name == "BMS Fixed Root":
        if theorist_epochs is not None:
            epochs = theorist_epochs
        else:
            epochs = 15
        theorist = BMSRegressor(epochs=epochs)
    elif theorist_name == "DARTS 2 Nodes":
        if theorist_epochs is not None:
            epochs = theorist_epochs
        else:
            epochs = 500
        theorist = DARTSRegressor(
            max_epochs=epochs, output_type=output_type, num_graph_nodes=2
        )
    elif theorist_name == "DARTS 3 Nodes":
        if theorist_epochs is not None:
            epochs = theorist_epochs
        else:
            epochs = 5
        theorist = DARTSRegressor(
            max_epochs=epochs, output_type=output_type, num_graph_nodes=3
        )
    elif theorist_name == "DARTS 4 Nodes":
        if theorist_epochs is not None:
            epochs = theorist_epochs
        else:
            epochs = 500
        theorist = DARTSRegressor(
            max_epochs=epochs, output_type=output_type, num_graph_nodes=4
        )
    elif theorist_name == "MLP":
        if theorist_epochs is not None:
            epochs = theorist_epochs
        else:
            epochs = 5000
        theorist = MLP_theorist(epochs=epochs, output_type=output_type, verbose=True)
    elif theorist_name == "Logistic Regression":
        theorist = LogitRegression()
    else:
        raise ValueError(f"Theorist {theorist_name} not implemented.")

    found_theory = False
    while not found_theory:
        try:
            DV_type = metadata.dependent_variables[0].type
            if DV_type == ValueType.PROBABILITY and theorist_name == "BMS Fixed Root":
                theorist.fit(X, y, root=sigmoid)
            else:
                theorist.fit(X, y)
            found_theory = True
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print("Trying again....")

    return theorist


def get_seed_experimentalist(X_allowed, metadata, num_samples):
    # # set up seed experimentalist
    # experimentalist_seed = Pipeline(
    #     [
    #         ("grid", grid_pool),  # generate grid pool based on allowed values for independent variables
    #         ("random", random_sampler),  # randomly draw samples from the grid
    #         ("nearest_values", nearest_values_sampler),  # match drawn samples to data_closed_loop points in training set
    #     ],
    #     {
    #         "grid__ivs": metadata.independent_variables,
    #         "random__n": num_samples,
    #         "nearest_values__allowed_values": X_allowed,
    #         "nearest_values__n": num_samples,
    #     },
    # )

    experimentalist_seed = Pipeline(
        [
            ("pool", X_allowed),
            ("random", random_sampler),
        ],
        {"random__n": num_samples},
    )

    return experimentalist_seed


def get_experimentalist(
    experimentalist_name, X, y, X_allowed, metadata, theorist, num_samples
):

    # popper experimentalist
    if experimentalist_name == "popper":
        experimentalist = Pipeline(
            [
                ("pool", poppernet_pool),
                ("nearest_values", nearest_values_sampler),
            ],
            {
                "pool__model": theorist,  # theorist.model_
                "pool__x_train": X,
                "pool__y_train": y,
                "pool__metadata": metadata,
                "pool__n": num_samples,
                "nearest_values__allowed_values": X_allowed,
                "nearest_values__n": num_samples,
            },
        )

    elif experimentalist_name == "falsification":
        experimentalist = Pipeline(
            [
                ("pool", X_allowed),
                ("random", random_sampler),
                ("falsification", falsification_sampler),
            ],
            {
                "random__n": X_allowed.shape[0],
                "falsification__model": theorist,  # theorist.model_
                "falsification__x_train": X,
                "falsification__y_train": y,
                "falsification__metadata": metadata,
                "falsification__n": num_samples,
            },
        )

    elif experimentalist_name == "popper dissimiarlity":
        experimentalist = Pipeline(
            [
                ("pool", poppernet_pool),
                ("dissimilarity", dissimilarity_sampler),
                ("nearest_values", nearest_values_sampler),
            ],
            {
                "pool__model": theorist,
                "pool__x_train": X,
                "pool__y_train": y,
                "pool__metadata": metadata,
                "pool__n": num_samples * 10,
                "dissimilarity__X_ref": X,
                "dissimilarity__n": num_samples,
                "dissimilarity__inverse": True,
                "dissimilarity__metric": "euclidean",
                "nearest_values__allowed_values": X_allowed,
                "nearest_values__n": num_samples,
            },
        )

    # random experimentalist
    elif experimentalist_name == "random":
        experimentalist = Pipeline(
            [
                ("pool", X_allowed),
                ("random", random_sampler),
            ],
            {"random__n": num_samples},
        )

    # dissimilarity experimentalist
    elif experimentalist_name == "dissimilarity":
        experimentalist = Pipeline(
            [
                ("pool", X_allowed),
                ("random", random_sampler),
                ("dissimilarity", dissimilarity_sampler),
            ],
            {
                "random__n": X_allowed.shape[0],
                "grid__ivs": metadata.independent_variables,
                "dissimilarity__X_ref": X,
                "dissimilarity__n": num_samples,
                "dissimilarity__inverse": False,
                "dissimilarity__metric": "euclidean",
                "dissimilarity__integration": "product",
            },
        )

    # dissimilarity experimentalist
    elif experimentalist_name == "inverse dissimilarity":
        experimentalist = Pipeline(
            [
                ("pool", X_allowed),
                ("random", random_sampler),
                ("dissimilarity", dissimilarity_sampler),
            ],
            {
                "random__n": X_allowed.shape[0],
                "dissimilarity__X_ref": X,
                "dissimilarity__n": num_samples,
                "dissimilarity__inverse": True,
                "dissimilarity__metric": "euclidean",
            },
        )

    elif experimentalist_name == "model disagreement":
        experimentalist = Pipeline(
            [
                ("pool", X_allowed),
                ("random", random_sampler),
                ("model_disagreement", model_disagreement_sampler),
            ],
            {
                "random__n": X_allowed.shape[0],
                "model_disagreement__models": theorist.models_[0:2],
                "model_disagreement__n": num_samples,
            },
        )

    # random experimentalist
    elif experimentalist_name == "least confident":

        if metadata.dependent_variables[0].type != ValueType.PROBABILITY:
            raise ValueError(
                "Least confident sampling only implemented for probability DVs."
            )

        experimentalist = Pipeline(
            [
                ("pool", X_allowed),
                ("random", random_sampler),
                ("uncertainty", uncertainty_sampler),
            ],
            {
                "random__n": X_allowed.shape[0],
                "uncertainty__n": num_samples,
                "uncertainty__model": theorist,
                "uncertainty__measure": "least_confident",
            },
        )

    else:
        raise ValueError(f"Experimentalist {experimentalist_name} not implemented.")

    return experimentalist


def get_MSE(theorist, x, y_target):
    y_prediction = theorist.predict(x)

    if y_target.shape[1] == 1:
        y_target = y_target.flatten()

    MSE = np.mean(np.square(y_target - y_prediction))

    return MSE


def get_DL(theorist, theorist_name, x, y_target):
    # DL = BIC/2 + PRIORS
    # BIC = n * log(MSE) + k * log(n)
    k = 0  # number of parameters
    prior = 0.0
    prior_par, _ = get_priors()
    if theorist_name == "BMS" or theorist_name == "BMS Fixed Root":
        parameters = set(
            [p.value for p in theorist.ets[0] if p.value in theorist.parameters]
        )
        k = 1 + len(parameters)
        for op, nop in list(theorist.nops.items()):
            try:
                prior += prior_par["Nopi_%s" % op] * nop
            except KeyError:
                pass
            try:
                prior += prior_par["Nopi2_%s" % op] * nop**2
            except KeyError:
                pass
    elif theorist_name == "Logistic Regression":
        k = 1 + theorist.n_features_in
        prior += prior_par["Nopi_/"]
        prior += prior_par["Nopi_+"]
        for i in range(k - 1):
            prior += prior_par["Nopi_+"]
            prior += prior_par["Nopi_*"]
            prior += prior_par["Nopi_**"]
    else:
        raise
    n = len(x)  # number of observations
    mse = get_MSE(theorist, x, y_target)
    bic = n * np.log(mse) + k * np.log(n)
    return bic / 2.0 + prior
