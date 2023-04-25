import statsmodels.formula.api as smf


class FlemingTheorist:

    def __init__(self):
        self.conditions = None
        self.model = None

    __name__ = 'Fleming Theorist'

    def fit(self, data, X_cond, y_col):
        self.conditions = X_cond

        # logit transform y data to perform linear regression instead
        # data[y_col] = self.logit(x=data[y_col])
        # data[['E', 'E2', 'R', 'R2']] = self.sigmoid(data[['E', 'E2', 'R', 'R2']])

        # Specify grouping variable for hierarchical logistic regression
        grouping_variable = 'g'
        fixed = [k for k in self.conditions.keys() if self.conditions[k] == 'f']
        variable = [k for k in self.conditions.keys() if self.conditions[k] == 'v']

        formula = y_col + ' ~ 1'
        random_formula = ''
        if len(fixed) > 0:
            formula += ' + ' + ' + '.join(fixed)
        if len(variable) > 0:
            random_formula = ' + '.join(variable)

        # Fit mixed-effects logistic regression model to data
        self.model = smf.mixedlm(formula, data, groups=data[grouping_variable]).fit()

        # Print summary of model results
        self.model = smf.mixedlm(formula, data,
                                 re_formula=random_formula,
                                 groups=data[grouping_variable]).fit()

        return self

    def predict(self, X):
        # return self.sigmoid(self.model.predict(X))
        return self.model.predict(X)
