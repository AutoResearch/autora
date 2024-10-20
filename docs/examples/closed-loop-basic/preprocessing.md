# Adding Preprocessing Script

After retrieving the data from our experiment, we need to preprocess it for further analysis. In this example, we will simply extract the key presses from each participant's data to compute whether their responses were correct or not. 

In principle, we can implement all kinds of fancy preprocessing analyses, such as averaging responses across trial blocks or participants, filtering outliers, etc. Here, for simplicity, we will then simply collect every single response in a single data frame. 

- Navigate to the ``researcher_hub`` directory.
- Add the following code to the file ``preprocessing.py``:

```python
import pandas as pd

def trial_list_to_experiment_data(trial_sequence):
    """
    Parse a trial sequence (from jsPsych) into dependent and independent variables
    
    independent variables: dots_left, dots_right
    dependent: accuracy
    """
    
    # define dictionary to store the results
    results_dict = {
        'dots_left': [],
        'dots_right': [],
        'accuracy': []
    }
    for trial in trial_sequence:
        # Filter experiment events that are not displaying the dots
        if trial['trial_type'] != 'rok':
            continue
            
        # Filter trials without reaction time
        if 'rt' not in trial or trial['rt'] is None: # key_response
            continue
            
        # the number of dots is equivalent to the number of oobs (oriented objects) as set in the SweetBean script
        dots_left = trial['number_of_oobs'][0] # oriented objects
        dots_right = trial['number_of_oobs'][1]
        choice = trial['key_press']
        
        # compute accuracy
        if dots_left == dots_right and choice == 'y' or dots_left != dots_right and choice == 'n':
            accuracy = 1
        else:
            accuracy = 0

        # add results to dictionary
        results_dict['dots_left'].append(int(dots_left))
        results_dict['dots_right'].append(int(dots_right))
        results_dict['accuracy'].append(float(accuracy))
    
    # convert dictionary to pandas dataframe
    experiment_data = pd.DataFrame(results_dict)

    return experiment_data
    
```

Below, we explain relevant parts of the code:

### Explanation

First, note that we are looping through the trial sequence and filtering out events that are not relevant to our analysis. We are only interested in trials where the participant is asked to compare the number of dots on the left and right sides of the screen, which are represented by the 'rok' trial type. Furthermore, we are interested in events that have a reaction time (rt) and a corresponding key press (key_response).

```python
    for trial in trial_sequence:
        # Filter experiment events that are not displaying the dots
        if trial['trial_type'] != 'rok':
            continue
            
        # Filter trials without reaction time
        if 'rt' not in trial or trial['rt'] is None: # key_response
            continue
```

Next, we extract the number of dots on the left and right sides of the screen, as well as the participant's response. 

```python
        # the number of dots is equivalent to the number of oobs (oriented objects) as set in the SweetBean script
        dots_left = trial['number_of_oobs'][0] # oriented objects
        dots_right = trial['number_of_oobs'][1]
        choice = trial['key_press']
```

We then calculate the accuracy of the participant's response based on the number of dots on each side of the screen. If the participant correctly identified whether the number of dots was equal or not, we assign a value of 1; otherwise, we assign a value of 0.

```python
        # compute accuracy
        if dots_left == dots_right and choice == 'y' or dots_left != dots_right and choice == 'n':
            accuracy = 1
        else:
            accuracy = 0
```
Finally, we store the results in a dictionary and convert it to a pandas DataFrame.

```python
        # add results to dictionary
        results_dict['dots_left'].append(int(dots_left))
        results_dict['dots_right'].append(int(dots_right))
        results_dict['accuracy'].append(float(accuracy))
    
    # convert dictionary to pandas dataframe
    experiment_data = pd.DataFrame(results_dict)
```

Note that this is a simple example of preprocessing. Depending on the complexity of your experiment and the analyses you wish to perform, you may need to implement more sophisticated preprocessing steps, such as averaging the accuracy across trials:

            
```python
# Calculate the mean rt for each S1/S2 combination
    experiment_data = experiment_data.groupby(['dots_left', 'dots_right']).mean().reset_index()
```

## Next Steps

[Next: Update the AutoRA workflow to use the experiment and preprocessing functions.](workflow.md)

