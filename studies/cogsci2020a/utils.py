import numpy as np

from autora.skl.bms import BMSRegressor
from autora.skl.darts import DARTSRegressor

from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.pooler import grid_pool, poppernet_pool
from autora.experimentalist.sampler import nearest_values_sampler, random_sampler, \
    summed_dissimilarity_sampler, model_disagreement_sampler

def get_theorist(theorist_name, epochs):
    if theorist_name == "BMS":
        theorist = BMSRegressor(epochs=epochs)
    elif theorist_name == "DARTS":
        theorist = DARTSRegressor(max_epochs=epochs)
    else:
        raise ValueError(f"Theorist {theorist_name} not implemented.")
    return theorist

def get_seed_experimentalist(X_allowed, metadata, num_samples):
    # set up seed experimentalist
    experimentalist_seed = Pipeline(
        [
            ("grid", grid_pool),  # generate grid pool based on allowed values for independent variables
            ("random", random_sampler), # randomly draw samples from the grid
            ("nearest_values", nearest_values_sampler),  # match drawn samples to data points in training set
        ],
        {"grid__ivs": metadata.independent_variables,
         "random__n": num_samples,
         "nearest_values__allowed_values": X_allowed,
         "nearest_values__n": num_samples}
    )
    return experimentalist_seed

def get_experimentalist(experimentalist_name, X, y, X_allowed, metadata, theorist, num_samples):

    # popper experimentalist
    if experimentalist_name == "popper":
        experimentalist = Pipeline(
            [
                ("pool", poppernet_pool),
                ("nearest_values", nearest_values_sampler),
            ],
            {"pool__model": theorist.model_,
             "pool__x_train": X,
             "pool__y_train": y,
             "pool__metadata": metadata,
             "pool__n": num_samples,
             "nearest_values__allowed_values": X_allowed,
             "nearest_values__n": num_samples}
        )

    elif experimentalist_name == "popper_dissimilarity":
        experimentalist = Pipeline(
            [
                ("pool", poppernet_pool),
                ("dissimilarity", summed_dissimilarity_sampler),
                ("nearest_values", nearest_values_sampler),
            ],
            {"pool__model": theorist.model_,
             "pool__x_train": X,
             "pool__y_train": y,
             "pool__metadata": metadata,
             "pool__n": num_samples*10,
             "dissimilarity__X_ref": X,
             "dissimilarity__n": num_samples,
             "nearest_values__allowed_values": X_allowed,
             "nearest_values__n": num_samples}
        )

    # random experimentalist
    elif experimentalist_name == "random":
        experimentalist = Pipeline(
            [
                ("pool", X_allowed),
                ("random", random_sampler),
            ],
            {"random__n": num_samples}
        )

    # dissimilarity experimentalist
    elif experimentalist_name == "dissimilarity":
        experimentalist = Pipeline(
            [
                ("grid", grid_pool),
                ("dissimilarity", summed_dissimilarity_sampler),
            ],
            {
                "grid__ivs": metadata.independent_variables,
                "dissimilarity__X_ref": X,
                "dissimilarity__n": num_samples,
             }
        )

    elif experimentalist_name == "model disagreement":
        experimentalist = Pipeline(
            [
                ("pool", X_allowed),
                ("model_disagreement", model_disagreement_sampler),
            ],
            {
                "model_disagreement__models": theorist.models_[0:2],
                "model_disagreement__n": num_samples}
        )
    else:
        raise ValueError(f"Experimentalist {experimentalist_name} not implemented.")

    return experimentalist


def get_MSE(model, x, y_target):
    y_prediction = model.predict(x)

    if y_target.shape[1] == 1:
        y_target = y_target.flatten()

    MSE = np.mean(np.square(y_target-y_prediction))

    return MSE

