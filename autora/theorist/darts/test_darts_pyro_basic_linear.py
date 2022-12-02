import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyro.infer import Predictive

import pyro
import pyro.distributions as dist

# SIMULATION PARAMETERS
num_samples = 1000
svi_iters = 5000
svi_lr = .05

plot_posterios = False
run_mcmc_sampling = False

# PREPARE DATA

def generate_x(start=-5, stop=5, num=1000):
    x = np.expand_dims(np.linspace(start=start, stop=stop, num=num), 1)
    return x

def transform_through_primitive_exp(x: np.ndarray):
    y = -1 + np.exp(x) + torch.randn(x.shape)
    return y

x = torch.tensor(generate_x(num = num_samples).flatten())
y = torch.tensor(transform_through_primitive_exp(x)).flatten()

def softmax(x1, x2):
    return torch.divide(torch.exp(x1), (torch.exp(x1) + torch.exp(x2)))

# DEFINE MODEL

def model(x, y):
    # a = pyro.sample("a", dist.Normal(0., 1.))
    b = pyro.sample("b", dist.Normal(0., 1.))
    w_1 = pyro.sample("w_exp", dist.Normal(-1., 1.))
    w_2 = pyro.sample("w_tanh", dist.Normal(-1., 1.))

    a_exp = pyro.sample("a_exp", dist.Normal(0., 1.))
    a_tanh = pyro.sample("a_tanh", dist.Normal(0., 1.))

    b_exp = pyro.sample("b_exp", dist.Normal(0., 1.))
    b_tanh = pyro.sample("b_tanh", dist.Normal(0., 1.))

    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    mean = b + softmax(w_1, w_2) * torch.exp(a_exp * x + b_exp) \
             + softmax(w_2, w_1) * torch.tanh(a_tanh * x + b_tanh)

    with pyro.plate("data", len(x)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

# DEFINE GUIDE

# def guide(x, y):
#     a_loc = pyro.param('a_loc', torch.tensor(0.))
#     a_scale = pyro.param('a_scale', torch.tensor(1.),
#                          constraint=constraints.positive)
#     sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
#                              constraint=constraints.positive)
#     weights_loc = pyro.param('weights_loc', torch.randn(1))
#     weights_scale = pyro.param('weights_scale', torch.ones(1),
#                                constraint=constraints.positive)
#     a = pyro.sample("a", dist.Normal(a_loc, a_scale))
#     b = pyro.sample("b", dist.Normal(weights_loc[0], weights_scale[0]))
#     sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
#     mean = a + b * x


# INFERENCE VIA STOCHASTIC VARIATIONAL INFERENCE

guide = pyro.infer.autoguide.AutoNormal(model)
optimizer = pyro.optim.ClippedAdam({"lr": svi_lr,
                                    "clip_norm": 1.0,
                                   "betas": (0.5, 0.999),
                                    "weight_decay": 0,
                                    })

# optimizer = pyro.optim.AdagradRMSProp({"eta": 1.0,
#                                       "delta": 1e-4,
#                                       "t": 0.9
#                                        })

optimizer = pyro.optim.Adam({"lr": svi_lr})

svi = pyro.infer.SVI(model,
          guide,
          optimizer,
          loss=pyro.infer.Trace_ELBO())

pyro.clear_param_store()
smoke_test = ('CI' in os.environ)
num_iters = svi_iters if not smoke_test else 2
for i in range(num_iters):
    elbo = svi.step(x, y)
    if i % 500 == 0:
        print("Elbo loss: {}".format(elbo))

num_samples = 1000
predictive = Predictive(model, guide=guide, num_samples=num_samples)
svi_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
               for k, v in predictive(y, x).items()
               if k != "obs"}

# Utility function to print latent sites' quantile information.
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

for site, values in summary(svi_samples).items():
    print("Site: {}".format(site))
    print(values, "\n")


if run_mcmc_sampling:

    # INFERENCE VIA MCMC

    print("### MCMC Inference")
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(x, y)
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    for site, values in summary(hmc_samples).items():
        print("Site: {}".format(site))
        print(values, "\n")


sites = ["a", "b", "sigma"]

if plot_posterios:
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
    for i, ax in enumerate(axs.reshape(-1)):
        site = sites[i]
        sns.distplot(svi_samples[site], ax=ax, label="SVI (DiagNormal)")
        if run_mcmc_sampling:
            sns.distplot(hmc_samples[site], ax=ax, label="HMC")
        ax.set_title(site)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()

if run_mcmc_sampling:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle("Cross-section of the Posterior Distribution", fontsize=16)
    sns.kdeplot(hmc_samples["a"], hmc_samples["b"], ax=axs[0], shade=True, label="HMC")
    sns.kdeplot(svi_samples["a"], svi_samples["b"], ax=axs[0], label="SVI (DiagNormal)")
    x_lim = (summary(hmc_samples)['a']['mean'].array[0] - 3 *  summary(hmc_samples)['a']['std'].array[0],
            summary(hmc_samples)['a']['mean'].array[0] + 3 *  summary(hmc_samples)['a']['std'].array[0])
    y_lim = (summary(hmc_samples)['b']['mean'].array[0] - 3 *  summary(hmc_samples)['b']['std'].array[0],
            summary(hmc_samples)['b']['mean'].array[0] + 3 *  summary(hmc_samples)['b']['std'].array[0])
    axs[0].set(xlabel="a", ylabel="b", xlim=x_lim, ylim=y_lim)
    # sns.kdeplot(hmc_samples["bR"], hmc_samples["bAR"], ax=axs[1], shade=True, label="HMC")
    # sns.kdeplot(svi_samples["bR"], svi_samples["bAR"], ax=axs[1], label="SVI (DiagNormal)")
    # axs[1].set(xlabel="bR", ylabel="bAR", xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
    # handles, labels = axs[1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    plt.show()
