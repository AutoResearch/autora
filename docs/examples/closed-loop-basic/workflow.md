# Updating the AutoRA Workflow

Now that we configured all the pieces, let's integrate them into a single workflow. The AutoRA workflow will generate the experiment, upload the experiment conditions to the Firestore database, and download the results of the experiment. 

## Workflow Overview

You can learn more about AutoRA workflows in the [AutoRA Tutorial](../../tutorials/basic/Tutorial III Functional Workflow.ipynb). The figure below illustrates the workflow for our closed-loop example.

![Setup](img/workflow_overview.png)

The workflow involves the following steps:
1. Collect data from an experiment runner that communicates with our web-based experiment. Here, we will use the ``experiment-runnner-firebase-prolific``.
2. Identify models that characterizes the data using an AutoRA theorist. Here, we will use the Bayesian Machine Scientist (BMS) from the package ``autora-theorist-bms``.
3. We will compare the best performing models to identify novel experiment conditions that differentiate betweent hese models. For this part, we will use the ``autora-experimentalist-model-disagreement`` package. 
4. After identifying new conditions, we may continue with Step 1.

AutoRA passes around a ``state`` object that contains all relevant data for the workflow. This object is updated at each step of the workflow (see the Figure above). 


## Updating the Workflow

### Import Statements
First, we need to import the functions we prepared in the [previous step](experiment.md):

```python
from trial_sequence import trial_sequences
from stimulus_sequence import stimulus_sequence
```

### Study Variables

Next, we define the independent and dependent variables of our study. Here, we have two independent variables, one representing the number of dots presnted on the left side of the screen, and another representing the number of dots presented on the right side of the screen. In addition, we define a dependent variable called accuracy.

```python
variables = VariableCollection(
    independent_variables=[
        Variable(name="dots left", allowed_values=np.linspace(1, 100, 100)),
        Variable(name="dots right", allowed_values=np.linspace(1, 100, 100)),
        ],
    dependent_variables=[Variable(name="accuracy", value_range=(0, 1))])
```

Note that the independent variables are defined as a range of values from 1 to 100. The accuracy can range between 0 and 1. 

### Experiment Runner

Then, we update the experiment runner:

```python
@on_state()
def runner_on_state(conditions):
    res = []
    for idx, c in conditions.iterrows():
        i_1 = c['S1']
        i_2 = c['S2']
        # get a timeline via sweetPea
        timeline = trial_sequences(i_1, i_2, 10)[0]
        # get js code via sweetBeaan
        js_code = stimulus_sequence(timeline, i_1, i_2)
        res.append(js_code)
    
    conditions_to_send = conditions.copy()
    conditions_to_send['experiment_code'] = res
    # upload and run the experiment:
    data_raw = experiment_runner(conditions_to_send)

    # process the experiment data
    experiment_data = pd.DataFrame()
    for item in data_raw:
        _lst = json.loads(item)['trials']
        _df = trial_list_to_experiment_data(_lst)
        experiment_data = pd.concat([experiment_data, _df], axis=0)
    return Delta(experiment_data=experiment_data)
```

Now, all that is left is to implement the function "trial_list_to_experiment". Here, we average rt over intensity_1 and intensity_2 combinations and filter:

```python
def trial_list_to_experiment_data(trial_sequence):
    """
    Parse a trial sequence (from jsPsych) into dependent and independent variables
    independent: S1, S2
    dependent: rt
    """
    res_dict = {
        'S1': [],
        'S2': [],
        'rt': []
    }
    for trial in trial_sequence:
        # Filter trials that are not ROK (instructions, fixation, ...)
        if trial['trial_type'] != 'rok':
            continue
        # Filter trials without rt
        if 'rt' not in trial or trial['rt'] is None:
            continue
        # the intensity is equivalent to the number of oobs (set in sweetBean script)
        # rt is a default value of every trial
        s1 = trial['number_of_oobs'][0]
        s2 = trial['number_of_oobs'][1]
        rt = trial['rt']
        
        res_dict['S1'].append(int(s1))
        res_dict['S2'].append(int(s2))
        res_dict['rt'].append(float(rt))
    
    dataframe_raw = pd.DataFrame(res_dict)
    
    # Calculate the mean rt for each S1/S2 combination
    grouped = dataframe_raw.groupby(['S1', 'S2']).mean().reset_index()

    return grouped
```



