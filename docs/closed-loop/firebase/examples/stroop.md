# Stroop with JsPsych

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
    return uniform_random_rng.integers(low=4, high=33, size=3)


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
// To use the jsPsych package first install jspsych using `npm install jspsych`
// In this example the html-keyboard-response plugin is used. Install it via `npm install @jspsych/plugin-html-keyboard-respons`
// Here is documentation on how to program a jspsych experiment using npm:
// https://www.jspsych.org/7.3/tutorials/hello-world/#option-3-using-npm
// can be used with the stroop_workflow.py in the examples/research_environment folder

import {initJsPsych} from 'jspsych';
import 'jspsych/css/jspsych.css'
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';


/**
 * This is the main function where you program your experiment. For example you can install jsPsych via node and
 * use functions from there
 * @param id this is a number between 0 and number of participants. You can use it for example to counterbalance between subjects
 * @param condition this is a condition (4-32. Here we want to find out how the training length impacts the accuracy in a testing phase)
 * @returns {Promise<*>} the accuracy in the post-trainging phase relative to the pre-training phase
 */
const main = async (id, condition) => {
    const jsPsych = initJsPsych()

    // constants
    const FIXATION_DURATION = 800
    const SOA_DURATION = 400
    const STIMULUS_DURATION = 2000
    const FEEDBACK_DURATION = 800
    const PRE_TRAIN_TRIALS = 10
    const POST_TRAIN_TRIALS = 10

    // key to response mapping (this could be something different based on id for counterbalancing. One could use the
    // remainder of id / 24 to determine the counterbalance condition)
    const keyToResponseMapping = {
        'c': 'red',
        'd': 'blue',
        'n': 'green',
        'j': 'yellow'
    }


    // For convenience, we first define a function that returns a trial (as sequence of fixation, soa, stimulus and feedback)
    const trial = (color, word, phase) => {
        const stimulus_timeline = []
        // FIXATION
        stimulus_timeline.push(
            {
                type: htmlKeyboardResponse,
                stimulus: "+",
                trial_duration: FIXATION_DURATION
            })

        // SOA
        stimulus_timeline.push({
            type: htmlKeyboardResponse,
            stimulus: "",
            trial_duration: SOA_DURATION,
        })

        // STIMULUS
        stimulus_timeline.push({
            type: htmlKeyboardResponse,
            stimulus: `<div style="color: ${color}">${word}</div>`,
            choices: ['c', 'd', 'n', 'j'],
            trial_duration: STIMULUS_DURATION,
            response_ends_trial: true,
            on_finish: function (data) { // here we set the correct (based on response and keymapping) and the phase as entry in the data
                const key = jsPsych.data.getLastTrialData()['trials'][0]['response']
                data['correct'] = keyToResponseMapping[key] === color
                data['phase'] = phase
            }
        })

        // FEEDBACK
        if (phase === 'training') {
            stimulus_timeline.push({
                type: htmlKeyboardResponse,
                stimulus: () => { // stimulus depends on last correct
                    const correct = jsPsych.data.getLastTrialData()['trials'][0]['correct']
                    if (correct) {
                        return 'CORRECT'
                    }
                    return 'FALSE'
                },
                trial_duration: FEEDBACK_DURATION
            })
        }
        return stimulus_timeline

    }

    // Here we set up functions to randomly select words and colors

    // create lists for colors and words
    const colors = ['red', 'green', 'blue', 'yellow']
    const words = ['RED', 'GREEN', 'BLUE', 'YELLOW']

    // get a random color form the list
    const rand_color = () => {
        return colors[Math.floor(Math.random() * colors.length)];
    }

    // get a random word from the list
    const rand_word = () => {
        return words[Math.floor(Math.random() * words.length)];
    }

    // MAKE THE EXPERIMENT TIMELINE

    // Instructions
    let instructions = [
        {
            type: htmlKeyboardResponse,
            stimulus: 'In the following experiment you are asked to name the colors (not the meaning) of the words<br>Press >> Space << to continue',
            choices: [' ']
        }
    ]

    for (let k in keyToResponseMapping) {
        instructions.push({
            type: htmlKeyboardResponse,
            stimulus: `If the color of the word is ${keyToResponseMapping[k]}, press >> ${k} << <br>Press >> ${k} >> to continue`,
            choices: [k]
        })
    }
    instructions.push(
        {
            type: htmlKeyboardResponse,
            stimulus: 'The experiment will start now<br>Press >> Space << to continue',
            choices: [' ']
        }
    )

    // Pre-training
    let pretraining = [];
    for (let i = 0; i < PRE_TRAIN_TRIALS; i++) {
        pretraining = pretraining.concat(trial(rand_color(), rand_word(), 'pre-training'));
    }

    // training
    let training = [];
    for (let i = 0; i < condition; i++) {
        training = training.concat(trial(rand_color(), rand_word(), 'training'));
    }

    // post-training
    let posttraining = []
    for (let i = 0; i < POST_TRAIN_TRIALS; i++) {
        posttraining = posttraining.concat(trial(rand_color(), rand_word(), 'post-training'));
    }

    // a pause trial
    const pause = {
        type: htmlKeyboardResponse,
        stimulus: 'The next block will start now<br>Press >> Space << to continue',
        choices: [' ']
    }

    // this is the timeline: instructions, pretraining, pause, training, pause, posstraining
    const timeline = [...instructions, ...pretraining, ...[pause], ...training, ...[pause], ...posttraining]

    // run the experiment and wait it to finish
    await jsPsych.run(timeline)

    // calculate accuracy before and after training
    const preTrainAcc = jsPsych.data.get().filter({'phase': 'pre-training', 'correct': true}).count() / PRE_TRAIN_TRIALS
    const postTrainAcc = jsPsych.data.get().filter({'phase': 'post-training', 'correct': true}).count() / POST_TRAIN_TRIALS


    // return difference between before and after training as observation
    return (postTrainAcc - preTrainAcc)
}


export default main
```
