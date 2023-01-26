from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class LogitRegression(LinearRegression):

    __name__ = 'Logit Regression'

    def fit(self, x, p, interaction_terms=True, logit_transform=True):
        p = np.asarray(p)
        self.interaction_terms = interaction_terms
        self.logit_transform = logit_transform
        x_org = x.copy()
        # The logit function is the inverse of the sigmoid or logistic function, and transforms
        # a continuous value (usually probability pp) in the interval [0,1] to the real line
        # (where it is usually the logarithm of the odds). The logit function is log(p / (1 - p))
        if self.logit_transform:
            y = np.log(p / (1 - p))
        else:
            y = p
        if self.interaction_terms:
            poly = PolynomialFeatures(interaction_only=True, include_bias=False)
            x = poly.fit_transform(x)

        self.model_ = super().fit(x, y)
        self.models_ = [self.model_]

        # fit a linear model to the logit-transformed data_closed_loop
        regr = LinearRegression()
        basic_model = regr.fit(x_org, y)
        self.models_.append(basic_model)

        return self

    def predict(self, x):
        if self.interaction_terms:
            poly = PolynomialFeatures(interaction_only=True, include_bias=False)
            x = poly.fit_transform(x)
        y = super().predict(x)
        if self.logit_transform:
            out = 1 / (np.exp(-y) + 1)
        else:
            out = y
        return out
