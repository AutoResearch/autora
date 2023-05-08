# Upload Full JavaScript experiments

## Researcher Environment
```python
"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
    Can be used in conjunction with the stroop_experiment in examples/test_subject_environment
"""

from autora.variable import VariableCollection, Variable
from autora.runner.firebase_prolific import firebase_runner
from autora.experimentalist.pipeline import make_pipeline
import numpy as np
from sklearn.linear_model import LinearRegression
from autora.workflow.cycle import Cycle
from sweetbean.sequence import Block, Experiment
from sweetbean.stimulus import TextStimulus

# *** Set up meta data *** #
# independent variable is coherence (0 - 1)
# dependent variable is accuracy (0 - 1)
metadata = VariableCollection(
    independent_variables=[Variable(name="x", allowed_values=[i for i in range(4, 33)])],
    dependent_variables=[Variable(name="y", value_range=(-1, 1))])

# *** Set up the theorist *** #
# The ground truth might actually not be a linear dependency,
# but might look somehow like this f(x) = 1 - e ^ (-x).
# Feel free to implement your own theorist here
theorist = LinearRegression()

# *** Set up the experimentalist *** #
# Also feel free to set up a more elaborate experimentalist here. This is just a random sampler that samples between 4 and 32 training size
uniform_random_rng = np.random.default_rng(seed=180)


def uniform_random_sampler():
    return uniform_random_rng.uniform(low=4, high=33, size=3)


def to_experiment(conditions):
    """
    here we convert the numbers from the uniform sampler into full experiments
    """
    def create_experiment(condition):

        text = TextStimulus(
            duration=2000, text=f"press a if {condition} is larger then 20, b if not.", color="pink", choices=["a", "b"]
        )
        block = Block([text])
        experiment = Experiment([block])
        return experiment.to_js_string(as_function=True, is_async=True)

    return [create_experiment(con) for con in conditions]


experimentalist = make_pipeline([uniform_random_sampler, to_experiment])

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
import { initJsPsych } from 'jspsych';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';

global.initJsPsych = initJsPsych;
global.jsPsychHtmlKeyboardResponse = htmlKeyboardResponse

/**
 * This is the main function where you program your experiment. For example, you can install jsPsych via node and
 * use functions from there
 * @param id this is a number between 0 and number of participants. You can use it for example to counterbalance between subjects
 * @param condition this is a condition (for example uploaded to the database with the experiment runner in autora)
 * @returns {Promise<*>} after running the experiment for the subject return the observation in this function, it will be uploaded to autora
 */
const main = async (id, condition) => {
    const observation = await eval(condition + "\nrunExperiment();");
    // Here we get the average reaction time
    const rt_array = observation.select('rt')['values']
    let sum_rt = 0;
    for(let i = 0; i < rt_array.length; i++) {
        sum_rt += rt_array[i];
    }
    let avg = sum_rt / rt_array.length;
    return avg
}


export default main
```
