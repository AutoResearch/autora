# Connect Project With Prolific

Once you have your closed-loop workflow set up, it is fairly easy to connect it to [Prolific](https://www.prolific.co/), 
a recruiting platform for web-based experiments. By connecting your project with Prolific via the `firebase-prolific-runner`, you can automatically recruit participants for your study and collect data from them. 

![system_overview](../img/system_overview.png)

!!! hint:
The `firebase-prolific-runner` will automatically set up a study on Prolific and recruit participants. It is highly recommended to test the experiment before recruiting participants, to have approval from an ethics committee, and to adhere to the ethical guidelines.

## Prerequisites

- You have a [Prolific](https://www.prolific.co/) account.
- Your behavioral study is approved by an ethics committee or institutional review board (IRB).
- You have a corresponding consent form for your study.


## Add Consent Form to Experiment

Before you can connect your project with Prolific, you will likely need to add a consent form to your experiment. The consent form should be displayed to participants before they start the experiment.

This will require you to modify the jsPsych code of your experiment which can be found in ``testing_zone/src/design.main.js``. 


## Update AutoRA Workflow to Use Prolific

- Navigate to the ``autora_workflow.py`` file in the ``researcher_hub`` folder

- All we need to do is to change the experiment runner. Replace ``RUNNER_TYPE = 'firebase' `` in the beginning of the `autora_workflow.py` file with

```python
RUNNER_TYPE = 'prolific'
```

- Next, you need to fill in the relevant data from Prolific and Firebase in the respective part ofh the ``autora_workflow.py`` file.

```python
# time between checks
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

## Update .env in testing_zone (Optional)

The ``firebase_prolific_runner`` optimally allocates slots for the experiments you submit to Prolific. If you are done with testing, and are ready for data collection you may want to update the ``.env`` file in the ``testing_zone`` folder.

- Navigate to the ``testing_zone`` folder.
- Open the ``.env`` file.
- Set the ``REACT_APP_useProlificId`` variable to ``True``.
```shell
REACT_APP_useProlificId="True"
```

## Summary

- **This is it!** Running the ``autora_workflow.py`` in the ``researcher_hub`` should now result in closed-loop reinforcement learning study that recruits human participants from Prolific to participate in your web-based experiment hosted on Firebase.

[Next: Customize your experiment.](customization.md)
