import sys
import os
import pandas as pd
from copy import deepcopy
from sklearn import cross_validation

sys.path.append('../')
from mcmc import *

# -----------------------------------------------------------------------------
def BIC_SA(x, y, variables, prior_par, npar=None, ns=1000, fn_label='data',
           T_ini=5., T_fin=0.001, T_sched=0.95):
    """Find the formula with the lowest BIC using simulated annealing.
    
    """
    # Initialize
    if npar == None:
        npar = 2 * len(variables)
    t = Tree(
        variables=variables,
        parameters=['a%d' % i for i in range(npar)],
        x=x, y=y,
        prior_par=prior_par,
        BT=T_ini,
    )
    progressFileName = 'bicsa_%s_progress.dat' % fn_label
    traceFileName = 'bicsa_%s_trace.dat' % fn_label
    try:
        os.remove(progressFileName)
    except OSError:
        pass
    try:
        os.remove(traceFileName)
    except OSError:
        pass
    # SA
    while t.BT > T_fin:
        t.get_bic(reset=True, fit=False)
        t.get_energy(bic=False, reset=True)
        print t.BT, t.bic
        t.mcmc(burnin=1, thin=1, samples=ns,
               tracefn=traceFileName, progressfn=progressFileName,
               reset_files=False,
               verbose=False)
        t.BT *= T_sched
    # Done
    return t


# -----------------------------------------------------------------------------
def post_SA(x, y, variables, prior_par, npar=None, ns=1000, fn_label='data',
           T_ini=5., T_fin=0.001, T_sched=0.95):
    """Find the formula with the largest posterior (lowest energy) using simulated annealing.
    
    """
    # Initialize
    if npar == None:
        npar = 2 * len(variables)
    t = Tree(
        variables=variables,
        parameters=['a%d' % i for i in range(npar)],
        x=x, y=y,
        prior_par=prior_par,
        BT=T_ini,
        PT=1.,
    )
    progressFileName = 'postsa_%s_progress.dat' % fn_label
    traceFileName = 'postsa_%s_trace.dat' % fn_label
    try:
        os.remove(progressFileName)
    except OSError:
        pass
    try:
        os.remove(traceFileName)
    except OSError:
        pass
    # Thermalize the system to PT=BT=1
    curT = T_ini
    minE, minF = np.inf, None
    while curT > 1.:
        # Print the "true" (T=1) energy 
        t.BT = 1.
        t.get_bic(reset=True, fit=False)
        t.get_energy(bic=False, reset=True)
        print curT, t.E, minE
        # Keep track of the best energy so far
        if t.E < minE:
            minE = deepcopy(t.E)
            minF = deepcopy(t)
        # Reset T and MCMC
        t.BT = curT
        t.get_bic(reset=True, fit=False)
        t.get_energy(bic=False, reset=True)
        t.mcmc(burnin=1, thin=1, samples=ns,
               tracefn=traceFileName, progressfn=progressFileName,
               reset_files=False,
               degcorrect=False,
               verbose=False)
        if t.E < minE:
            minE = t.E
            minF = deepcopy(t)
        curT *= T_sched
    # Continue lowering PT=BT until T_fin
    reachedMin = False
    curT = 1.
    while not reachedMin:
        while t.BT > T_fin:
            # Print the "true" (T=1) energy 
            t.BT = t.PT = 1.
            t.get_bic(reset=True, fit=False)
            t.get_energy(bic=False, reset=True)
            print curT, t.E, minE
            # Keep track of the best energy so far
            if t.E < minE:
                minE = deepcopy(t.E)
                minF = deepcopy(t)
            # Reset T and MCMC
            t.BT = t.PT = curT
            t.get_bic(reset=True, fit=False)
            t.get_energy(bic=False, reset=True)
            t.mcmc(burnin=1, thin=1, samples=ns,
                   tracefn=traceFileName, progressfn=progressFileName,
                   reset_files=False,
                   degcorrect=False,
                   verbose=False)
            curT *= T_sched
        # Check if we missed the best visited formula
        t.BT = t.PT = 1.
        t.get_bic(reset=True, fit=False)
        t.get_energy(bic=False, reset=True)
        if t.E > minE:
            t = deepcopy(minF)
            curT = .2
        else:
            reachedMin = True
    # Done
    return t

# -----------------------------------------------------------------------------
def model_averaging_valid(x, y, variables, prior_par, npar=None,
                          ns=100, fn_label='data',
                          method='kfold', k=2):
    """Validate model averaging using k-fold (method="kfold") or leave-k-out (method="lko").

    """
    if method == 'lko':
        ttsplit = cross_validation.LeavePOut(len(y), k)
    elif method == 'kfold':
        ttsplit = cross_validation.KFold(len(y), n_folds=k)
    else:
        raise ValueError
    if npar == None:
        npar = 2*len(variables)
    mse, mae = [], []
    for train_index, test_index in ttsplit:
        xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        t = Tree(
            variables=variables,
            parameters=['a%d' % i for i in range(npar)],
            x=xtrain, y=ytrain,
            prior_par=prior_par,
            BT=1.0, PT=1.0,
        )
        ypred = t.trace_predict(xtest, samples=ns, thin=5000, write_files=False)
        ypredmean = ypred.mean(axis=1)
        ypredmedian = ypred.median(axis=1)

        print ytest
        print ypredmean
        print ypredmedian

        mse.append(np.mean((ytest - ypredmean)**2))
        mae.append(np.mean(np.abs(ytest - ypredmedian)))

    return mse, mae
        
