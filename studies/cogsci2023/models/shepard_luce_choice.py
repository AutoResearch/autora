import numpy as np
from autora.variable import DV, IV, ValueType, VariableCollection

# general meta parameters
added_noise = 0.01

# shepard-luce choice parameters
shepard_luce_resolution = 5
maximum_similarity = 10
minimum_similarity = 1/shepard_luce_resolution
focus = 0.8

# Shepard-Luce Choice Rule according to Equation (4) in
# Logan, G. D., & Gordon, R. D. (2001).
# Executive control of visual attention in dual-task situations. Psychological review, 108(2), 393.
# Or Equation (5) in Luce, R. D. (1963). Detection and recognition.

def shepard_luce_metadata():
    similarity_category_A1 = IV(
        name="similarity_category_A1",
        allowed_values=np.linspace(minimum_similarity, maximum_similarity, shepard_luce_resolution),
        value_range=(minimum_similarity, maximum_similarity),
        units="similarity",
        variable_label="Similarity with Category A1",
        type=ValueType.REAL
    )

    similarity_category_A2 = IV(
        name="similarity_category_A2",
        allowed_values=np.linspace(minimum_similarity, maximum_similarity, shepard_luce_resolution),
        value_range=(minimum_similarity, maximum_similarity),
        units="similarity",
        variable_label="Similarity with Category A2",
        type=ValueType.REAL
    )

    similarity_category_B1 = IV(
        name="similarity_category_B1",
        allowed_values=np.linspace(minimum_similarity, maximum_similarity, shepard_luce_resolution),
        value_range=(minimum_similarity, maximum_similarity),
        units="similarity",
        variable_label="Similarity with Category B1",
        type=ValueType.REAL
    )

    similarity_category_B2 = IV(
        name="similarity_category_B2",
        allowed_values=np.linspace(minimum_similarity, maximum_similarity, shepard_luce_resolution),
        value_range=(minimum_similarity, maximum_similarity),
        units="similarity",
        variable_label="Similarity with Category B2",
        type=ValueType.REAL
    )

    focus_category_A = IV(
        name="focus_category_A",
        allowed_values=[0, 1],
        value_range=(0, 1),
        units="focus",
        variable_label="Focus on Category A",
        type=ValueType.REAL
    )


    choose_A1 = DV(
        name="choose_A1",
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Choosing A1",
        type=ValueType.PROBABILITY
    )

    metadata = VariableCollection(
        independent_variables=[similarity_category_A1,
                               similarity_category_A2,
                               similarity_category_B1,
                               similarity_category_B2,
                               focus_category_A],
        dependent_variables=[choose_A1],
    )

    return metadata

def shepard_luce_experiment(X: np.ndarray,
                             focus: float = focus,
                             std = added_noise):

    Y = np.zeros((X.shape[0],1))
    for idx, x in enumerate(X):

        similarity_A1 = x[0]
        similarity_A2 = x[1]
        similarity_B1 = x[2]
        similarity_B2 = x[3]
        focus_A = x[4]

        if focus_A == 1:
            actual_focus_A = focus
        elif focus_A == 0:
            actual_focus_A = 1 - focus

        y = (similarity_A1 * actual_focus_A + np.random.normal(0, std))/ \
            (similarity_A1 * actual_focus_A +
             similarity_A2 * actual_focus_A +
             similarity_B1 * (1 - actual_focus_A) +
             similarity_B2 * (1 - actual_focus_A))
        # probability can't be negative or larger than 1 (the noise can make it so)
        if y <= 0:
            y = 0.0001
        elif y >= 1:
            y = 0.9999
        Y[idx] = y

    return Y

def shepard_luce_data(metadata):

    similarity_A1 = metadata.independent_variables[0].allowed_values
    similarity_A2 = metadata.independent_variables[1].allowed_values
    similarity_B1 = metadata.independent_variables[2].allowed_values
    similarity_B2 = metadata.independent_variables[3].allowed_values
    focus_A = metadata.independent_variables[4].allowed_values

    X = np.array(np.meshgrid(similarity_A1,
                             similarity_A2,
                             similarity_B1,
                             similarity_B2,
                             focus_A)).T.reshape(-1,5)

    # remove all conditions from X where the focus is 0 and the similarity of A1 is 0 or the similarity of A2 is 0
    # X = X[~((X[:,4] == 0) & ((X[:,0] == 0) | (X[:,1] == 0)))]
    # X = X[~((X[:,4] == 1) & ((X[:,2] == 0) | (X[:,3] == 0)))]
    X = X[~((X[:,0] == 0) & (X[:,1] == 0) & (X[:,2] == 0) & (X[:,3] == 0))]

    y = shepard_luce_experiment(X, std=0)

    return X, y

def plot_shepard_luce(model = None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    metadata = shepard_luce_metadata()

    similarity_A1 = np.linspace(metadata.independent_variables[0].value_range[0],
                                metadata.independent_variables[0].value_range[1],
                                100)

    similarity_A2 = 0.5 # 1 - similarity_A1

    similarity_B1_list = [0.5, 0.75, 1]
    similarity_B2 = 0
    focus = 1.0

    colors = mcolors.TABLEAU_COLORS
    col_keys = list(colors.keys())
    for idx, similarity_B1 in enumerate(similarity_B1_list):
        # similarity_B2 = 1 - similarity_B1
        X = np.zeros((len(similarity_A1), 5))

        X[:,0] = similarity_A1
        X[:,1] = similarity_A2
        X[:,2] = similarity_B1
        X[:,3] = similarity_B2
        X[:,4] = focus

        y = shepard_luce_experiment(X, std=0)
        plt.plot(similarity_A1.reshape((len(similarity_A1), 1)), y,
                 label=f"Similarity to B1 = {similarity_B1} (Original)",
                 c=colors[col_keys[idx]])

        if model is not None:
            y = model.predict(X)
            plt.plot(similarity_A1, y, label=f"Similarity to B1 = {similarity_B1} (Recovered)",
                     c=colors[col_keys[idx]], linestyle="--")

    x_limit = [np.min(similarity_A1), np.max(similarity_A1)]
    y_limit = [0, 1]
    x_label = "Similarity to Category A1"
    y_label = "Probability of Selecting Category A1"

    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.xlabel(x_label, fontsize="large")
    plt.ylabel(y_label, fontsize="large")
    plt.legend(loc=4, fontsize="medium")
    plt.title("Shepard-Luce Choice Ratio", fontsize="x-large")
    plt.show()


# X, y = shepard_luce_data(shepard_luce_metadata())
# plot_shepard_luce()
