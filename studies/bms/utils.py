import numpy as np

from autora.theorist.bms.prior import get_priors


def get_MSE(theorist, x, y_target):
    y_prediction = theorist.predict(x)

    if y_target.shape[1] == 1:
        y_target = y_target.flatten()
    if y_prediction.shape[1] == 1:
        y_prediction = y_prediction.flatten()

    MSE = np.mean(np.square(y_target - y_prediction))

    return MSE


def get_LL(mse):
    # LL = log(MSE)
    # BMS seems to make this assumption `\_0_/`
    return np.log(mse)


def get_BIC(theorist, theorist_name, mse, num_obs):
    # BIC = n * LL + k * log(n)
    num_params = get_num_params(theorist, theorist_name)
    return num_obs * get_LL(mse) + num_params * np.log(num_obs)


def get_DL(theorist, theorist_name, mse, num_obs):
    # DL = BIC/2 + PRIORS
    bic = get_BIC(theorist, theorist_name, mse, num_obs)
    prior = get_prior(theorist, theorist_name)
    return bic / 2.0 + prior


def get_num_params(theorist, theorist_name):
    if "BMS" in theorist_name:
        parameters = set(
            [
                p.value
                for p in theorist.model_.ets[0]
                if p.value in theorist.model_.parameters
            ]
        )
        k = 1 + len(parameters)
    elif theorist_name == "Logit Regression":
        k = (1 + theorist.model_.coef_.shape[0]) * 2
    elif "DARTS" in theorist_name:
        model_str = theorist.model_repr()
        k = 1 + model_str.count(".")
    elif theorist_name == "MLP":
        return 1201
    else:
        print(theorist_name)
        raise
    return k


def get_prior(theorist, theorist_name):
    prior = 0.0
    prior_par, _ = get_priors()
    if "BMS" in theorist_name:
        for op, nop in list(theorist.model_.nops.items()):
            try:
                if op in ["+", "/", "exp", "-"]:
                    prior += prior_par["Nopi_%s" % op] * (nop + 1)
                else:
                    prior += prior_par["Nopi_%s" % op] * nop
            except KeyError:
                pass
            try:
                if op in ["+", "/", "exp", "-"]:
                    prior += prior_par["Nopi2_%s" % op] * (nop + 1) ** 2
                else:
                    prior += prior_par["Nopi2_%s" % op] * nop**2

            except KeyError:
                pass
        if theorist_name == "BMS Fixed Root":
            prior += prior_par["Nopi_/"] + prior_par["Nopi_+"] + prior_par["Nopi_exp"]
    elif theorist_name == "Logit Regression":
        k = (1 + theorist.model_.coef_.shape[0]) * 2
        prior += prior_par["Nopi_log"]
        prior += prior_par["Nopi_/"]
        prior += prior_par["Nopi_-"]
        # the part raised to the exponential
        prior += prior_par["Nopi_+"] * (k - 1)
        prior += prior_par["Nopi_*"] * (k - 1)
        prior += prior_par["Nopi_**"] * (k - 1)
        try:
            prior += prior_par["Nopi2_+"] * (k - 1) ** 2
            prior += prior_par["Nopi2_*"] * (k - 1) ** 2
            prior += prior_par["Nopi2_**"] * (k - 1) ** 2
        except KeyError:
            pass
    elif "DARTS" in theorist_name:
        model_str = theorist.model_repr()
        for op in theorist.primitives:
            try:
                prior += prior_par["Nopi_%s" % op] * model_str.count(op)
            except KeyError:
                pass
            try:
                prior += prior_par["Nopi2_%s" % op] * model_str.count(op) ** 2
            except KeyError:
                pass
    elif theorist_name == "MLP":
        print("Description Length not applicable to MLP")
    else:
        print(theorist_name)
        raise
    return prior


def mountain(x, y):
    return np.maximum(x, y)


def threshold2(x, y):
    return np.where(x > 0, x, y)


def threshold3(x, y, z):
    return np.where(x > 0, y, z)
