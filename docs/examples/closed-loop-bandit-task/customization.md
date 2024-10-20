# Customization

The code you used in this example may serve as a template for future projects. The most important files are:

- **Autora Workflow** ``researcher_hub/autora_workflow.py``: Here, you can customize the AutoRA workflow.

Adjusting the workflow may involve:
- Using other *experimentalists* to change the selection of experiment-conditions. 
- Adjusting *experiment runners* for alternative ways of collecting data. In this example, we use jsPsych and javascript for the experiments, but depending on the experiment, you can interface with other devices or use other means to collect data.
- Using other *theorists*. The theorist that is used in the example is custom build for recovering reinforcement learning rules from behavior measured in 2-armed bandit tasks. Depending on your experiment, you will use other theorists.


- **Behavioral experiment** ``testng_zone/src/design/main.js``: Here, you can customzie the web-based experiment that is shown to the participant. You can find great tutorials on how to build jsPsych experiments on their [website](https://www.jspsych.org/latest/). Keep in mind, that you should build your website in a way that the conditions are used to customize the experiment. A good way to do so is by creating trial-sequences in ``autora_workflow.py`` and using them as ``timeline_variables``.

