from sklearn.linear_model import LinearRegression
import numpy as np

class LogitRegression(LinearRegression):

    def fit(self, x, p):
        p = np.asarray(p)
        # The logit function is the inverse of the sigmoid or logistic function, and transforms
        # a continuous value (usually probability pp) in the interval [0,1] to the real line
        # (where it is usually the logarithm of the odds). The logit function is log(p / (1 - p))
        y = np.log(p / (1 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)
