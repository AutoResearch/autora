import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
import pandas as pd
from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

# INFO: load data for problems from the official repo
data_uri = (
    "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
)
df_dev = pd.read_csv(data_uri, sep=";")
df_dev = df_dev[df_dev["age"] >= 18]


def model(df_dev):
    alpha = numpyro.sample("alpha", dist.Normal(60, 10))
    beta = numpyro.sample("beta", dist.LogNormal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 10))
    mu = numpyro.deterministic(
        "mu", alpha + beta * (df_dev["height"] - df_dev["height"].mean()).values
    )
    numpyro.sample("W", dist.Normal(mu, sigma), obs=df_dev["weight"].values)


# quadratic approximation part
guide = AutoNormal(model)
svi = SVI(model, guide, optim.Adam(1), Trace_ELBO(), df_dev=df_dev)
svi_result = svi.run(random.PRNGKey(0), 1000)

params = svi_result.params

# display summary of quadratic approximation
samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)


# INFO: good
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
# predictive = Predictive(guide.model, params=params, posterior_samples=samples)
predictive = Predictive(model, samples)
samples_predictive = predictive(rng_key_, df_dev)
print(samples_predictive.keys())
