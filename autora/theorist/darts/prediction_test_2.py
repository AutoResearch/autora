import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random

x = np.expand_dims(np.linspace(start=-1, stop=1, num=20), 1)
y = 3 * x + 1 + np.random.normal(size=(20, 1))

def model(x, y):
    a = numpyro.sample("a", dist.Normal(1, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0., 5.))
    mu = a * x + b
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

guide = AutoNormal(model)
svi = SVI(model, guide, optim.Adam(1), Trace_ELBO())
svi_result = svi.run(random.PRNGKey(0), 1000, x, y)

params = svi_result.params
samples = guide.sample_posterior(random.PRNGKey(1), params, (100,))
predictive = Predictive(model, samples)
samples_predictive = predictive(random.PRNGKey(0), x, None)
print(samples_predictive['y'].T)

