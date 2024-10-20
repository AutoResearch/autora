# Basic Closed-Loop Two-Armed Bandit Study 

In this example, we will guide you through setting up a closed-loop computational discovery study for a human reinforcement learning task. In this behavioral study, participants will interact with a two-armed bandit task, where they must choose between two options to maximize their reward. Using AutoRA, you will build dynamic research workflow that iterates between computational model discovery, experimental design, and behavioral data collection. The ultimate goal is to make AutoRA iteratively uncover a learning rule that characterizes human participants' behavior in a two-armed bandit task.

The code builds in a method for automating the discovery of reinforcement learning rules:

[Weinhardt, W. Eckstein, M., & Musslick, S. (2024). Computational discovery of human reinforcement learning dynamics from choice behavior. *NeurIPS 2024 Workshop on Behavioral ML*.](https://openreview.net/forum?id=x2WDZrpgmB)

This example provides a **hands-on approach to understanding closed-loop computational discovery** of human behavior using the AutoRA framework. 

It may also serve as a **starting point** for developing your own computational discovery project.


## What You’ll Learn:
- **Set up a closed-loop AutoRA workflow**: Learn how to create an automated discovery process, iterating between computational model discovery, experimental design, and data collection.
- **Automate experimental design with [SweetPea](https://sites.google.com/view/sweetpea-ai)**: Use SweetPea to generate experimental designs that adapt as the study progresses.
- **Interfacing with web experiments**: Use AutoRA to update the parameterization of an existing web-based experiment written in jsPsych.
- **Host experiments using [Google Firebase](https://firebase.google.com/)**: Set up a server for hosting your behavioral experiments, making them accessible to participants.
- **Store experimental data with [Google Firestore](https://firebase.google.com/)**: Efficiently manage and store participant data collected from your experiment.
- **Automate the discovery of reinforcement learning rules**: Use AutoRA to discover reinforcement learning rules that explain participant behavior.
- **Collect data from real participants with [Prolific](https://www.prolific.com/)**: Recruit and manage participants through Prolific, ensuring high-quality behavioral data.

!!! hint
    This example set up most of your workflow automatically, so it will not cover how to write the code for the two-armed bandit task, or how to implement
    a method for discovering reinforcement learning rules from behavior. Instead, we will leverage existing templates and packages. 

## Prerequisites:
- **Basic Python knowledge**: While most of the workflow is Python-based, only a basic level of understanding is needed to follow along.
- **Minimal JavaScript knowledge**: Since the behavioral experiments are implemented in JavaScript (via jsPsych), a minimal understanding of JavaScript is required.
- **A Google account**: You will need a Google account to use Google Firebase and Firestore.

## Study Overview

The goal of this closed-loop study is to iteratively discover learning rules that explain participant's choice behavior in a simple two-armed bandit task.

### Experiment

In this experiment, participants are tasked to maximize the reward they obtain by selecting between differently colored boxes. On each trial, the participant is presented with a red box and a blue box:

![stimulus.png](img/stimulus.png)

Selecting one of the boxes will either result in a reward (```+1``` point) or not (``0`` points). The reward probability of each box changes over time, and participants must learn to select the box that provides the most reward.

### Computational Model Discovery

Our goal is to automatically discover a learning rule that best explains the choice behavior of all participants. We will accomplish this using a reinforcement learning model discovery method described in 

[Weinhardt, W. Eckstein, M., & Musslick, S. (2024). Computational discovery of human reinforcement learning dynamics from choice behavior. *NeurIPS 2024 Workshop on Behavioral ML*.](https://openreview.net/forum?id=x2WDZrpgmB)

![rnn_theorist.png](img/rnn_theorist.png)

This method, available in the ``autora[theorist-rnn-sindy-rl]`` package, fits the choice behavior of participants with a recurrent neural network, which computes values for both of the two options (the red box and the blue box), and then chooses between the options based on their relative value. Once the neural network is fit, the method applies an equation discovery technique called "Sparse Identification of Non-Linear Systems" ([SINDy](https://pysindy.readthedocs.io/en/latest/examples/2_introduction_to_sindy/example.html)) to extract the learning rule that the network has discovered. 

### Experimental Sampling

The AutoRA workflow will generate new experiment conditions based on best two learning rules that were discovered. Specifically, the workflow will generate reward trajectories that best distinguish between the two learning rules using the ``autora[experimentalist-model-disagreement]`` package.

These reward trajectories will be uploaded to the Firebase project, where the next batch of participants can interact with the web-based experiment. Data collected from the next batch will be used to update the model and discover new learning rules.

## System Overview

Our closed-loop system consists of a bunch of interacting components. Here is a high-level overview of the system:
![System Overview](../img/system_overview.png)

Our closed-loop system will have two projects talking to each other. The **Firebase project** will host and run the web experiment that participants interact with. Our **local AutoRA project** will host the code that runs the AutoRA workflow, which will generate new experiment conditions, collect data from the web experiment, and update the model based on the collected data. 

### Firebase Project
To run an online experiment, we need to host it as a **web app**. We will leverage **Google Firebase** to host our web app. Participants from **Prolific** can then interact with the web app to complete the experiment.  

Our experiment is configured by experiment conditions, which are stored in a **Google Firestore** database. In addition, we will use this database to store collected behavioral data from the web experiment. 

### Our Local AutoRA Project

The local project will consist of two folders. The ``testing_zone`` folder will contain the web app that participants interact with. 

The ``researcher_hub`` folder will contain the AutoRA workflow. 

## Next Steps 

Each step in the example will lead guide you to set up each component of the closed-loop system. 

By the end of this example, you’ll be able to run a closed-loop computational discovery study.

[Next: Set up the project.](setup.md)
