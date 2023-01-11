import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0

# shepard-luce choice parameters
tva_resolution = 10
maximum_category_similarity = 10
minimum_category_similarity = 0
minimum_feature_similarity = 0
maximum_feature_similarity = 10
focus = 0.8

# Theory of Visual Attention according to Equation (9) in
# Logan, G. D., & Gordon, R. D. (2001).
# Executive control of visual attention in dual-task situations. Psychological review, 108(2), 393.

# NOTE: Second input variable y_similarity_cat_1 has (intentionally) no effect on the outcome variable

def tva_metadata():

    x_similarity_cat_1 = IV(
        name="x_similarity_cat_1",
        allowed_values=np.linspace(maximum_category_similarity, minimum_category_similarity,
                                   tva_resolution),
        value_range=(maximum_category_similarity, minimum_category_similarity),
        units="similarity",
        variable_label="Similarity of X with Category 1",
        type=ValueType.REAL
    )

    y_similarity_cat_1 = IV(
        name="y_similarity_cat_1",
        allowed_values=np.linspace(maximum_category_similarity, minimum_category_similarity,
                                   tva_resolution),
        value_range=(maximum_category_similarity, minimum_category_similarity),
        units="similarity",
        variable_label="Similarity of Y with Category 1",
        type=ValueType.REAL
    )

    y_similarity_target = IV(
        name="y_similarity_target",
        allowed_values=np.linspace(minimum_feature_similarity, maximum_feature_similarity,
                                   tva_resolution),
        value_range=(minimum_feature_similarity, maximum_feature_similarity),
        units="similarity",
        variable_label="Similarity of Y with Target Feature",
        type=ValueType.REAL
    )

    choose_A1 = DV(
        name="choose_category_1_for_x",
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Choosing 1 for X",
        type=ValueType.PROBABILITY
    )

    metadata = VariableCollection(
        independent_variables=[x_similarity_cat_1,
                               y_similarity_cat_1,
                               y_similarity_target],
        dependent_variables=[choose_A1],
    )

    return metadata

def tva_experiment(X: np.ndarray,
                             focus: float = focus,
                             std = added_noise):

    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):

        x_similarity_cat_1 = x[0]
        x_similarity_cat_2 = maximum_category_similarity - x_similarity_cat_1
        y_similarity_cat_1 = x[1]
        y_similarity_cat_2 = maximum_category_similarity - y_similarity_cat_1

        x_similarity_feat_A = maximum_feature_similarity
        x_similarity_feat_B = maximum_feature_similarity - x_similarity_feat_A
        y_similarity_feat_A = x[2]
        y_similarity_feat_B = maximum_feature_similarity - y_similarity_feat_A

        focus_feature_A = focus
        focus_feature_B = 1 - focus_feature_A

        w_x = focus_feature_A * x_similarity_feat_A + focus_feature_B * x_similarity_feat_B
        w_y = focus_feature_A * y_similarity_feat_A + focus_feature_B * y_similarity_feat_B

        x_is_cat_1 = x_similarity_cat_1 * w_x / (w_x + w_y) + np.random.normal(0, std)
        x_is_cat_2 = x_similarity_cat_2 * w_x / (w_x + w_y)
        y_is_cat_1 = y_similarity_cat_1 * w_y / (w_x + w_y)
        y_is_cat_2 = y_similarity_cat_2 * w_y / (w_x + w_y)

        p_x_is_cat_1 = x_is_cat_1 / (x_is_cat_1 + x_is_cat_2 + y_is_cat_1 + y_is_cat_2)

        Y[idx] = p_x_is_cat_1

    return Y

def tva_data(metadata):

    x_similarity_cat_1 = metadata.independent_variables[0].allowed_values
    y_similarity_cat_1 = metadata.independent_variables[1].allowed_values
    y_similarity_target = metadata.independent_variables[2].allowed_values

    X = np.array(np.meshgrid(x_similarity_cat_1,
                             y_similarity_cat_1,
                             y_similarity_target)).T.reshape(-1,3)

    y = tva_experiment(X, std=0)

    return X, y

def plot_tva(model = None):
    import matplotlib.pyplot as plt
    metadata = tva_metadata()

    x_similarity_cat_1 = np.linspace(metadata.independent_variables[0].value_range[0],
                                    metadata.independent_variables[0].value_range[1],
                                    100)

    y_similarity_cat_1 = 0.5
    y_similarity_target_list = [0, 5, 10]

    for y_similarity_target in y_similarity_target_list:

        X = np.zeros((len(x_similarity_cat_1), 3))

        X[:,0] = x_similarity_cat_1
        X[:,1] = y_similarity_cat_1
        X[:,2] = y_similarity_target

        y = tva_experiment(X, std=0)
        plt.plot(x_similarity_cat_1.reshape((len(x_similarity_cat_1), 1)), y,
                 label=f"Distractor Similarity = {y_similarity_target} (Original)")

        if model is not None:
            y = model.predict(X)
            plt.plot(x_similarity_cat_1.reshape((len(x_similarity_cat_1), 1)), y,
                     label=f"Distractor Similarity = {y_similarity_target} (Recovered)")

    x_limit = [np.min(x_similarity_cat_1), np.max(x_similarity_cat_1)]
    y_limit = [0, 1]
    x_label = "Similarity Between X and Category 1"
    y_label = "Probability of Selecting Category 1 for X"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=4, fontsize="medium")
    plt.title("Theory of Visual Attention", fontsize="x-large")
    plt.show()


# X, y = tva_data(tva_metadata())
# plot_tva()
