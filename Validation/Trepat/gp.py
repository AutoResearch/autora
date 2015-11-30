import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcess
from sklearn import cross_validation
from matplotlib import pyplot as pl

from iodata import read_data

# Read the data
data, x, y = read_data(in_fname='cadhesome_protein_all.csv')
x = x[['CDH3']]
print x
print y

# Prepare for LOOCV
ypred, spred = [], []
loo = cross_validation.LeaveOneOut(len(y))
for train_index, test_index in loo:
    print train_index, test_index
    xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

    if float(xtest.iloc[0]) < 80 and float(xtest.iloc[0]) > 70:
        xf, yf = xtrain, ytrain

    # Instanciate a Gaussian Process model
    gp = GaussianProcess(
        regr='linear',
        corr='squared_exponential', theta0=.001,
        thetaL=1e-10, thetaU=1e10,
        nugget=.1,
        random_start=100,
    )

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(xtrain, ytrain)
    #gp.fit(x, y)
    print gp.get_params()

    # Make the prediction on the test set
    yipred, MSE = gp.predict(xtest, eval_MSE=True)
    sigma = np.sqrt(MSE)
    ypred.append(float(yipred))
    spred.append(float(sigma))

# Display results
ypred = pd.Series(ypred)
print y
print ypred
print 'MSE =', np.sqrt(np.sum((ypred - y)**2) / float(len(y)))

"""
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = pl.figure()
pl.plot(ypred, y, 'ro')
pl.plot((100, 500), (100, 500))
pl.show()
"""

# Fit a GP
print xf
print yf
gp = GaussianProcess(
    regr='linear',
    corr='squared_exponential', theta0=.1,
    thetaL=1e-10, thetaU=1e20,
    nugget=.1,
    optimizer='Welch',
    random_start=1000,
)
gp.fit(x, y)


xmesh = np.atleast_2d(np.linspace(0, 140, 1000)).T
y_pred, MSE = gp.predict(xmesh, eval_MSE=True)
sigma = np.sqrt(MSE)
fig = pl.figure()
#pl.plot(xmesh, y_pred, 'r:', label=u'$S_{xx}$')
pl.plot(x, y, 'ro', label=u'Observations')
pl.plot(x, ypred, 'go', label=u'LOO predictions')
pl.plot(xmesh, y_pred, 'b-', label=u'Fit')
pl.fill(np.concatenate([xmesh, xmesh[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$CDH3$')
pl.ylabel('$S_{xx}$')
#pl.ylim(-10, 20)
pl.legend(loc='upper left')

pl.show()
