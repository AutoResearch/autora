The code you have created can serve as a template for future projects. The most important files are:

(1) Autora Workflow: research_hub/autora_workflow.py Here, you can customize the AutoRA workflow. Which includes:

Using other experimentalists to change the selection of experiment-conditions.
Adjusting runners: In this example, we use jsPsych and javascript for the experiments, but depending on the experiment, you can interface with other devices or use other means to collect data.
Using other theorists. The theorist that is used in the example is custom build for RL tasks. Depending on your experiment, you will use other theorists.
(2) Website: testng_zone/src/design/main.js Here, you can customize the website that is shown to the participant. You can find good tutorials on how to build jsPsych experiments on their website Keep in mind, that you should build your website in a way that the conditions are used to customize the experiment. A good way to do so is by creating trial-sequences in autora_workflow.py and using them as timeline_variables.
