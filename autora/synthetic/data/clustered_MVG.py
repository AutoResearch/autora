import scipy.stats as ss
import numpy as np
from functools import partial
from ..inventory import SyntheticExperimentCollection, register


class multivariate_gaussian:

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self):

        return ss.multivariate_normal.rvs(mean=self.loc, cov=self.scale)

    def sample_conditioned(self, fixed_dims, values, return_full=False):
        assert len(fixed_dims) < len(self.loc), "Too many fixed dimensions!"

        dims_set = set(fixed_dims)
        free_dims = [i for i in range(len(self.loc)) if i not in dims_set]


        sigma_22 = self.scale[fixed_dims, :][:, fixed_dims]
        sigma_11 = self.scale[free_dims, :][:, free_dims]
        sigma_12 = self.scale[free_dims, :][:, fixed_dims]

        mu_1 = self.loc[free_dims]
        mu_2 = self.loc[fixed_dims]

        tmp = sigma_12 @ np.linalg.inv(sigma_22)
        sigma_bar = sigma_11 - tmp @ sigma_12.T
        mubar = mu_1 + tmp @ (values - mu_2)

        sample = ss.multivariate_normal.rvs(mean=mubar, cov=sigma_bar)

        if return_full:
            res = np.zeros(len(self.loc))
            res[free_dims] = sample
            res[fixed_dims] = values
            return res

        else:
            return sample

    def marginal_pdf(self, dims, values):
        sigma_new = self.scale[dims, :][:, dims]
        mu_new = self.loc[dims]

        return ss.multivariate_normal.pdf(values, mean=mu_new, cov=sigma_new)


class clustered_multivariate_gaussian:

    def __init__(self, locs, scales, cluster_priors):

        self.locs = locs
        self.scales = scales

        self.cluster_priors = cluster_priors
        self.clusters = []
        for i in range(len(locs)):
            self.clusters.append(multivariate_gaussian(self.locs[i], self.scales[i]))
                
    def sample(self, cond_dims=None, cond_vals=None, return_full=True, cluster_probs=None):

        # idea - pick a random cluster, then sample a value
        cluster_ind = np.random.choice(np.arange(len(self.locs)),
                                       p=self.cluster_priors if cluster_probs is None else cluster_probs)
        if cond_dims is not None:
            return self.clusters[cluster_ind].sample_conditioned(cond_dims, cond_vals, return_full)
        else:
            return self.clusters[cluster_ind].sample()

    def sample_conditioned(self, fixed_dims, values, return_full=True):

        # compute (marginal) likelihoods of observed values, compute posteriors
        # for clusters, then sample from these clusters in a conditional way

        posteriors = self.cluster_priors * np.array([c.marginal_pdf(fixed_dims, values) for c in self.clusters])
        posteriors = posteriors / np.sum(posteriors)

        return self.sample(cond_dims=fixed_dims, cond_vals=values, return_full=return_full, cluster_probs=posteriors)


def clustered_MVG(
        name = "Clustered Multivariate Gaussian",
        locs = [], # means of the clusters
        scales = [], # scales of the clusters
        # locs and scales must have the same length
        cluster_priors = [], # prior weights of the clusters; if not specified, clusters get equal weight
        n_samples = 1 # N of points to sample per each experimental condition
        ):
    
    
    if len(cluster_priors)==0: # cluster priors are not provided -> assign uniform priors
            cluster_priors = np.full(len(locs), 1 / len(locs))

    params = dict(locs = locs,
                  scales = scales,
                  cluster_priors = cluster_priors,
                  n_samples = n_samples,
                  name = name,
                  )    
    
    def experiment_runner(D: np.ndarray, V = np.ndarray):
        # D - an array of arrays representing dimensions to control on in each experiment
        # V - an array of arrays representing values along the controlled dimensions for each experiment
        # D and V must have the same shape 
        
        MVG = clustered_multivariate_gaussian(locs, scales, cluster_priors)
        Obs = []
        for i in len(D):
            for _ in n_samples: # N of samples per each experiment
                Obs.append(MVG.sample(cond_dims=D[i],cond_vals=V[i]))
        return Obs


    ground_truth = partial(experiment_runner)
        
    collection = SyntheticExperimentCollection(
        name=name,
        params=params,
        metadata=None,
        domain=None,
        experiment_runner=experiment_runner,
        ground_truth=ground_truth,
        plotter=None,
    )
    return collection

register("clustered_MVG", clustered_MVG)
