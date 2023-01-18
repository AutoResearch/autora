from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class LogitRegression(LinearRegression):

    def fit(self, x, p, interaction_terms=True):
        p = np.asarray(p)
        self.interaction_terms = interaction_terms
        # The logit function is the inverse of the sigmoid or logistic function, and transforms
        # a continuous value (usually probability pp) in the interval [0,1] to the real line
        # (where it is usually the logarithm of the odds). The logit function is log(p / (1 - p))
        y = np.log(p / (1 - p))
        if self.interaction_terms:
            poly = PolynomialFeatures(interaction_only=True, include_bias=False)
            x = poly.fit_transform(x)
        return super().fit(x, y)

    def predict(self, x):
        if self.interaction_terms:
            poly = PolynomialFeatures(interaction_only=True, include_bias=False)
            x = poly.fit_transform(x)
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)
