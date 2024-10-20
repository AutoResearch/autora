# Workflow Explanation

In this section, we will take a deeper dive into the workflow of the closed-loop bandit task. You can find an overview of the study [here](index.md). We will explain the different components of the workflow and how they interact with each other.

- Navigate to the workflow ``autora_workflow.py`` in the ``researcher_hub`` folder.

## Workflow Variables

The workflow has a few variables that you can adjust to customize the workflow to your needs. Here are the most important ones:

```python
# *** CONSTANTS *** #

RUNNER_TYPE = 'firebase'  # Options: synthetic, firebase, prolific

TRIALS_PER_PARTICIPANTS = 100
SAMPLES_PER_CYCLE = 1
PARTICIPANTS_PER_CYCLE = 40
CYCLES = 4
INITIAL_REWARD_PROBABILITY_RANGE = [.2, .8]
SIGMA_RANGE = [.2, .2]

EPOCHS = 10 # 100
```

- `RUNNER_TYPE`: This variable determines which runner to use. You can choose between `synthetic`, `firebase`, and `prolific`. The `synthetic` runner is used to test the workflow with simulated data. The `firebase` runner is used to run the workflow with the actual web experiment on Firebase. The `prolific` runner is used to run the workflow on Firebase and recruit participants via Prolific.
- `TRIALS_PER_PARTICIPANTS`: The number of trials each participant will have to complete in the two-armed bandit experiment.
- `SAMPLES_PER_CYCLE`: The number of new experiment conditions (i.e., reward sequences) to generate in each cycle. Here, we generate one new experiment condition per cycle and let all participants perform the experiment with the same reward sequence.
- `PARTICIPANTS_PER_CYCLE`: The number of participants to recruit in each cycle. 
- `CYCLES`: The number of cycles to run the workflow. In each cycle, we generate new experiment conditions, recruit participants, and discover a learning rule.
- `INITIAL_REWARD_PROBABILITY_RANGE`: The range of initial reward probabilities for the two boxes. The reward probabilities are randomly sampled from this range for each new experiment condition.
- `SIGMA_RANGE`: The range of the standard deviation of the Gaussian noise added to the reward probabilities. This noise models the stochasticity of the rewards.
- `EPOCHS`: The number of epochs to train the recurrent neural network (RNN) on the behavioral data. The RNN is used to model the choice behavior of participants.

## Seeding the Workflow

The workflow starts by seeding the random number generator to ensure reproducibility:

```python
seed = 11

# for reproducible results:
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
```

## Experimental Variables

This study includes one independent and one dependent variable.
```python
variables = VariableCollection(
    independent_variables=[Variable(name="reward-trajectory")],
    dependent_variables=[Variable(name="choice-trajectory")]
)
```

### Independent Variable

The independent variable corresponds to the actual reward trajectory for the two (red and blue) boxes. It is a 2 x ``n_trials`` vector with entries between 0 and 1. Each row corresponds to the reward probability of one box. 

### Dependent Variable

The dependent variable corresponds to the choice trajectory of the participant. It is a 2 x ``n_trials`` vector with entries 0 or 1, indicating the choice of the participant for each of the two boxes. Thus, every column is 1 one-hot encoding of the choice, indicating whether the participant chose the red or blue box.

## AutoRA State

In this example, we choose a non-standard state for the AutoRA workflow. We use a `RnnState` that extends the `StandardState` with additional models. 
```python
@dataclass(frozen=True)
class RnnState(StandardState):
    models_additional:  List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend"},
    )

# initialize the state:
state = RnnState(variables=variables)
```

The `RnnState` is a dataclass that is used to store the state of the AutoRA workflow. You can learn more about non-standard states in the [User Guide](https://autoresearch.github.io/autora/core/docs/cycle/Dynamically%20Extending%20and%20Altering%20the%20State/).

## Experimentalists

In this study, we will leverage two experimentalist, one for initializing the first experiment conditions and one for generating new experiment conditions in each cycle. 

### Random Sampler

The `RandomPooler` is used to initialize the first experiment conditions. It randomly samples reward trajectories from the initial reward probability range and sigma range.

```python
@on_state()
def pool_on_state(num_samples, n_trials=TRIALS_PER_PARTICIPANTS):
    """
    This is creates `num_samples` randomized reward-trajectories of length `n_trials`
    """
    sigma = np.random.uniform(SIGMA_RANGE[0], SIGMA_RANGE[1])
    trajectory_array = bandit_random_pool(
        num_rewards=2,
        sequence_length=n_trials,
        initial_probabilities=[INITIAL_REWARD_PROBABILITY_RANGE, INITIAL_REWARD_PROBABILITY_RANGE],
        sigmas=[sigma, sigma],
        num_samples=num_samples
    )
    trajectory_df = pd.DataFrame({'reward-trajectory': trajectory_array})
    return Delta(conditions=trajectory_df)
```

Note that it is important to use the `@on_state()` decorator to ensure that the pooler is only called when the state is updated. This wrapper ensures that the new reward trajectory ``trajectory_df`` is stored int the ``conditions`` field of the AutoRA ``state``.

### Model Disagreement Sampler

The `ModelDisagreementPooler` is used to generate new experiment conditions in each cycle. It uses the best two models discovered so far to generate new reward trajectories that best distinguish between the two models.

Since the predictions of a model has a non-standard format (it isn't a single number but a 2 x ``n_trials`` array), we need to create a custom distance function. 

The prediction for a model is a list of two-dimensional vectors, e.g., ``array([[0.5, 0.5], [0.68..., 0.31...], ...])``, indicating the reward probability for the boxes on each trial. The following code specifies how to compute a distance between two proposed reward trajectories specified in this format:

```python
def custom_distance(prob_array_a, prob_array_b):
    return np.mean([(prob_array_a[0] - prob_array_b[0])**2 + (prob_array_a[1] - prob_array_b[1])**2])
```

This distance function is then used in model disagreement sampler, which is also wrapped in an `@on_state()` decorator:

```python
@on_state()
def model_disagreement_on_state(
        conditions, models, models_additional, num_samples):
    conditions = model_disagreement_sampler_custom_distance(
        conditions=conditions['reward-trajectory'],
        models=[models[-1], models_additional[-1]],
        distance_fct=custom_distance,
        num_samples=num_samples,
    )
    return Delta(conditions=conditions)
```

Note that this sampler takes two models as input, which are the best two models discovered so far. It then generates new reward trajectories that best distinguish between the two models.

## Experiment Runners

The experiment runners are responsible for generating data. Here, we provide three different experiment runners. You will be able to switch between these runners by changing the `RUNNER_TYPE` variable at the beginning of the workflow.

### Synthetic Runner

The `SyntheticRunner` is used to generate synthetic data for testing the workflow. It generates synthetic data by simulating a known reinforcement learning rule (Q-learning) on the reward trajectories. The runner is part of the ``autora-synthetic`` package.

```python
runner = q_learning()
```

The synthetic runner is also wrapped in an `@on_state()` decorator, allowing it to add to the ``experiment_data`` field in the AutoRA ``state``. 

```python
@on_state()
def runner_on_state_synthetic(conditions):
    choices, choice_probabilities = runner.run(conditions, return_choice_probabilities=True)
    experiment_data = pd.DataFrame({
        'reward-trajectory': conditions['reward-trajectory'].tolist(),
        'choice-trajectory': choices,
        'choice-probability-trajectory': choice_probabilities
    })
    return Delta(experiment_data=experiment_data)
```

The wrapper takes as input the reward trajectories ``conditions``, and generates the corresponding choices. Because this is a synthetic model, we can also extract the choice probability for each option (which are unknown if the choices are generated from a human participant).

### Firebase Runner

The `FirebaseRunner` is used to run the actual web experiment on Firebase. It uploads new experiment conditions to the Firestore database and downloads the results of the experiment. 

#### Credentials

The firebase credentials are used to configure the runner (yours should look different compared to the ones below).

```python
# firebase credentials
firebase_credentials = {
  "type": "service_account",
  "project_id": "closed-loop-bandit",
  "private_key_id": "1eYOURCREDENTIALS7a43c",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvYOURCREDENTIALSu/VNaT\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-jkvmg@closed-loop-bandit.iam.gserviceaccount.com",
  "client_id": "106074818984686391054",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-jkvmg%40closed-loop-bandit.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```

#### Runner Specification

The firebase runner is defined by its credentials, which are used to authenticate with the Firebase project. In addition, we can specify the ``time_out`` (in minutes) and ``sleep_time`` (in seconds) for the runner. The former specifies the minutes a participant has available to complete the experiment, and the latter specifies the seconds between checks for new data. 

```python 

sleep_time = 30

study_completion_time = 5

experiment_runner_firebase = firebase_runner(
    firebase_credentials=firebase_credentials,
    time_out=study_completion_time,
    sleep_time=sleep_time)
```

#### The Actual Web Experiment

The actual experiment is written with the ``jsPsych`` in ``JavaScript`` and located in ``testing_zone/src/design/main.js``. The experiment is a simple two-armed bandit task where participants have to choose between a red and a blue box. The reward probabilities of the boxes change over time, and participants must learn to select the box that provides the most reward.

Here, we won't explain how to program jsPsych experiments. Please refer to the [jsPsych documentation](https://www.jspsych.org/) for more information.

Note that the ``main`` function of the ``main.js`` file starts like this

```javascript
const main = async (id, condition) => {

    const timeline_variables = JSON.parse(condition)
    ...
```

The function takes an `id` and a `condition` as input. The `id` is the participant id, and the `condition` contains the reward trajectory that the participant should interact with. However, the `condition` also contains other things. It is a JSON string that is parsed into a JavaScript object, containing the entire timeline of the experiment. This timeline specifies which boxes are presented where and which one is rewarded.

Our workflow uses a helper function to ensure that the `condition` is correctly formatted and uploaded to the Firebase database.

#### Helper Function for Uploading Experiment Conditions to Web Experiment

The following helper function transforms a condition (the output of an experimentalist) to a trial sequence, that is readable by the jsPsych experiment. jsPsych expects a trial sequence in the following format: ``trial_sequence = {'feature_a': [1, 2, 3, ...], 'feature_b': ['red', 'green', ...], ...}``.

```python
def _condition_to_trial_sequence(conditions):
    """
    Transforms conditions created by the experimentalist in a list of trial sequences
    """
    trial_sequences = []
    for c in conditions['reward-trajectory'].tolist():
        sequence = []
        if len(c) % 2:
            print('WARNING: trajectory has an odd number of entries. ')

        # create a counterbalanced position list:
        _n = len(c) // 2
        pos_list = [['left', 'right']] * _n + [['right', 'left']] * _n
        random.shuffle(pos_list)

        # a condition c is a list of values (c = [[0, 1], [1, 1], [0, 0], ...])
        for idx, trial in enumerate(c):
            sequence.append({'values': trial.tolist(), 'pos': pos_list[idx]})
        trial_sequences.append(sequence)
    return trial_sequences
```

Note that this function also counterbalances the position of the boxes (left or right) to avoid any biases due to the position of the boxes.

#### Helper Function for Processing Experimental Data

We also have to transform the data returned by the ``jsPsych`` experiment to our ``experiment_data``. This is achieved by the following function:

```python
def _jsPsych_to_experiment_data(data):
    result = []
    # For the output format of the jsPsych script, see the return value in testing_zone/src/design/main.js
    for item in data:
        parsed = json.loads(item)
        condition = json.loads(parsed['condition'])
        observation = parsed['observation']
        c_subj = {'reward-trajectory': [], 'choice-trajectory': []}
        for c, o in zip(condition, observation['values']):
            t = c['values']
            c_subj['reward-trajectory'].append(t)
            if o == 0:
                c_subj['choice-trajectory'].append([1, 0])
            else:
                c_subj['choice-trajectory'].append([0, 1])

        c_subj['reward-trajectory'] = np.array(c_subj['reward-trajectory'])
        c_subj['choice-trajectory'] = np.array(c_subj['choice-trajectory'])
        result.append(c_subj)
    return result
```

Here, we extract both the choices and the reward sequence from the data returned by the experiment. The choices are stored in the ``chocie-trajectory`` field, and the reward sequence is stored in the ``reward-trajectory`` field. Critically, both fields are designed as lists of one-hot vectors, with each vector specifying the choice and reward for the two options in each trial, respectively.

### Wrapper for Firebase Runner

Ultimately, we will wrap the Firebase runner in an `@on_state()` decorator. This wrapper allows the runner to add to the ``experiment_data`` field in the AutoRA ``state``. It makes use of the helper functions defined above.

```python
@on_state()
def runner_on_state_firebase(conditions):
    trial_sequences = _condition_to_trial_sequence(conditions)
    data = experiment_runner_firebase(trial_sequences)
    experiment_data = _jsPsych_to_experiment_data(data)
    return Delta(experiment_data=experiment_data)
```

### Prolific Runner

The `firebase_prolific_runner` is used to recruit participants via Prolific. It uploads the experiment to Firebase and uses Prolific to recruit participants for the experiment.

We begin with specifying some variables for the runner...

```python
# time in seconds between checks
sleep_time = 30

# Study name: This will be the name that will appear on prolific, participants that have participated in a study with the same name will be
# excluded automatically
study_name = 'my autora experiment'

# Study description: This will appear as study description on prolific
study_description= 'Two bandit experiment'

# Study Url: The url of your study (you can find this in the Firebase Console)
study_url = 'www.my-autora-experiment.com'

# Study completion time (minutes): The estimated time a participant will take to finish your study. We use the compensation suggested by Prolific to calculate how much a participant will earn based on the completion time.
study_completion_time = 5

# Prolific Token: You can generate a token on your Prolific account
prolific_token = 'my prolific token'

# Completion code: The code a participant gets to prove they participated. If you are using the standard set up (with cookiecutter), please make sure this is the same code that you have provided in the .env file of the testing zone.
completion_code = 'my completion code'
```

...and then define the runner based on these variables. 

```python
experiment_runner_prolific = firebase_prolific_runner(
            firebase_credentials=firebase_credentials,
            sleep_time=sleep_time,
            study_name=study_name,
            study_description=study_description,
            study_url=study_url,
            study_completion_time=study_completion_time,
            prolific_token=prolific_token,
            completion_code=completion_code,
        )
```

Similar to the firebase runner, we can wrap it in an `@on_state()` decorator to add to the ``experiment_data`` field in the AutoRA ``state``.

```python
@on_state()
def runner_on_state_prolific(conditions):
    trial_sequences = _condition_to_trial_sequence(conditions)
    data = experiment_runner_prolific(trial_sequences)
    experiment_data = _jsPsych_to_experiment_data(data)
    return Delta(experiment_data=experiment_data)
```

## Theorists

Our goal is to automatically discover a learning rule that best explains the choice behavior of all participants. We will accomplish this using a method that combines recurrent neural network modeling with equation discovery.

![rnn_theorist.png](img/rnn_theorist.png)

This method, available in the ``autora[theorist-rnn-sindy-rl]`` package, fits the choice behavior of participants with a recurrent neural network, which computes values for both of the two options (the red box and the blue box), and then chooses between the options based on their relative value. Once the neural network is fit, the method applies an equation discovery technique called "Sparse Identification of Non-Linear Systems" ([SINDy](https://pysindy.readthedocs.io/en/latest/examples/2_introduction_to_sindy/example.html)) to extract the learning rule that the network has discovered.

Here, we define two theorist, each with a different parameterization of the method above:

```python
theorist = RNNSindy(2, epochs=EPOCHS, polynomial_degree=2)
theorist_additional = RNNSindy(2, epochs=EPOCHS, polynomial_degree=1)
```

The firs theorist uses a polynomial degree of 2, while the second uses a polynomial degree of 1. The polynomial degree specifies the complexity of the learning rule that the method can discover. A higher polynomial degree allows for more complex learning rules but also increases the risk of overfitting.

We wrapp both of these theorists in an `@on_state()` decorator to add to the ``models`` and ``models_additional`` fields in the AutoRA ``state``.

```python
@on_state()
def theorist_on_state(experiment_data):
    x = experiment_data['reward-trajectory']
    y = experiment_data['choice-trajectory']
    return Delta(models=[theorist.fit(x, y)])


@on_state()
def theorist_additional_on_state(experiment_data):
    x = experiment_data['reward-trajectory']
    y = experiment_data['choice-trajectory']
    return Delta(models_additional=[theorist_additional.fit(x, y)])
```

The ``RNNSindy`` theorists take as input the reward trajectories ``x`` and the choice trajectories ``y`` and return a model that best explains the choice behavior of the participants.


## AutoRA Workflow

Finally, we specify the AutoRA workflow based on the components defined above.

```python
for c in range(1, CYCLES + 1):

    if len(state.models) > 0:
        state = pool_on_state(state, num_samples=20)
        state = model_disagreement_on_state(state, num_samples=SAMPLES_PER_CYCLE)
    else:
        state = pool_on_state(state, num_samples=SAMPLES_PER_CYCLE)

    if RUNNER_TYPE == 'synthetic':
        state = runner_on_state_synthetic(state)
    elif RUNNER_TYPE == 'firebase':
        state = runner_on_state_firebase(state)
    elif RUNNER_TYPE == 'prolific':
        state = runner_on_state_prolific(state)

    state = theorist_on_state(state)
    state = theorist_additional_on_state(state)
```

Here, we repeat the following steps for each cycle:
1. If we don't have any models yet, we will sample the initial experiment conditions using the `pool_on_state` function. Otherwise, we will use the `model_disagreement_on_state` function to generate new experiment conditions that best distinguish between the two best models discovered so far.
2. We will then run the experiment using the specified runner. 
3. Finally, we will fit the choice behavior of the participants with the recurrent neural network and extract the learning rule using the SINDy method, for each of the two theorists. 

The final lines of the code simply print the discovered learning rules, stored in the ``models`` and ``models_additional`` fields of the AutoRA ``state``.:

```python
    model = state.models[-1]
    model_additional = state.models_additional[-1]
    
    equations_model = parse_equation(model)
    equation_model_additional = parse_equation(model_additional)

    print('# MODEL DEGREE = 2#')
    print(f'chosen: {equations_model[0]}')
    print(f'non chosen: {equations_model[1]}')

    print('# MODEL DEGREE = 1#')
    print(f'chosen: {equation_model_additional[0]}')
    print(f'non chosen: {equation_model_additional[1]}')
```


## Next Steps

[Next: Test your workflow on Firebase.](testing.md)
