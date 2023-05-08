# Upload Conditions for the Super-Experiment

## Researcher Environment
```python
"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
    Can be used in conjunction with the super_experiment in examples/test_subject_environment
"""

from autora.variable import VariableCollection, Variable
from autora.runner.firebase_prolific import firebase_runner
from autora.experimentalist.pipeline import make_pipeline
import numpy as np
from sklearn.linear_model import LinearRegression
from autora.workflow.cycle import Cycle

# *** Set up meta data *** #
# independent variable is coherence (0 - 1)
# dependent variable is accuracy (0 - 1)
metadata = VariableCollection(
    independent_variables=[Variable(name="x", allowed_values=range(1))],
    dependent_variables=[Variable(name="y", value_range=(0, 1))], )

# *** Set up the theorist *** #
# The ground truth might actually not be a linear dependency,
# but might look somehow like this f(x) = 1 - e ^ (-x).
# Feel free to implement your own theorist here
theorist = LinearRegression()

# *** Set up the experimentalist *** #
# Also feel free to set up a more elaborate experimentalist here. This is just a random sampler
uniform_random_rng = np.random.default_rng(180)


def uniform_random_sampler():
    return uniform_random_rng.uniform(low=0, high=1, size=3)


experimentalist = make_pipeline([uniform_random_sampler])

# *** Set up the runner *** #
# Here fill in your own credentials
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key

firebase_credentials = {
    "type": "tyoe",
    "project_id": "project_id",
    "private_key_id": "private_key_id",
    "private_key": "private_key",
    "client_email": "client_email",
    "client_id": "client_id",
    "auth_uri": "auth_uri",
    "token_uri": "token_uri",
    "auth_provider_x509_cert_url": "auth_provider_x509_cert_url",
    "client_x509_cert_url": "client_x509_cert_url"
}

# simple experiment runner that runs the experiment on firebase
experiment_runner = firebase_runner(
    firebase_credentials=firebase_credentials,
    time_out=100,
    sleep_time=5)

# *** Set up the cycle *** #
cycle = Cycle(
    metadata=metadata,
    theorist=theorist,
    experimentalist=experimentalist,
    experiment_runner=experiment_runner,
    monitor=lambda state: print(f"Generated {len(state.theories)} theories"))

# run the cycle (we will be running 3 cycles with 3 conditions each)
cycle.run(num_cycles=3)


# *** Report the data *** #
# If you changed the theorist, also change this part
def report_linear_fit(m: LinearRegression, precision=4):
    s = f"y = {np.round(m.coef_[0].item(), precision)} x " \
        f"+ {np.round(m.intercept_.item(), 4)}"
    return s


print(report_linear_fit(cycle.data.theories[0]))
print(report_linear_fit(cycle.data.theories[-1]))

```

## Test Subject Environment
```javascript
// ATTENTION: To be able to import from "super-experiment", first install it in your test_subject_environment folder
// via the following command in the terminal: ``npm install super-experiment``
// this works together with the super_experiment_workflow.py in the examples/research_environment folder


import {instructions, block, createTrialSequence, accuracyBlock} from "super-experiment";

/**
 * This runs a super_experiment with coherence as condition and outputs the accuracy of the block as observation.
 * @param id (from autora: a id between 0 - number of participants per circle (can be used for between counterbalancing)
 * @param condition (from autora: the condition (in this case a single value between 0 and 1 is expected)
 * @return {Promise<*>} (to autora: the observation (in this case the average accuracy)
 */
const main = async (id, condition) => {
    // Show instructions (wait till participant presses a button)
    await instructions()
    // create a trial sequence of ten trials from a single coherence (all else being randomized, but not counterbalanced)
    const trial_sequence = createTrialSequence(10, {'coh': condition})
    // run the block with the created trial sequence
    const data = await block(trial_sequence)
    // calculate the accuracy from the data
    const observation = accuracyBlock(data)
    // return the observation
    return await observation
}

export default main
```
