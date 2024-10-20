# Basic Closed-Loop Psychophysics Study 

In this example, we will guide you through setting up a closed-loop behavioral study for a psychophysics experiment. By leveraging AutoRA, you’ll build a dynamic research workflow that iterates between model discovery, experimental design, and behavioral data collection, all within the context of a psychophysics experiment. The ultimate goal is to make AutoRA iteratively uncover an equation that characterizes human participants' ability to distinguish between various visual stimuli.

This example provides a hands-on approach to understanding closed-loop behavioral research in the context of the AutoRA framework. 

## What You’ll Learn:
- **Set up a closed-loop AutoRA workflow**: Learn how to create an automated discovery process, iterating between hypothesis generation and data collection.
- **Automate experimental design with [SweetPea](https://sites.google.com/view/sweetpea-ai)**: Use SweetPea to generate experimental designs that adapt as the study progresses.
- **Generate behavioral experiments with [SweetBean](https://autoresearch.github.io/sweetbean/)**: Automate the creation of simple behavioral experiments, minimizing the need for manual coding.
- **Host experiments using [Google Firebase](https://firebase.google.com/)**: Set up a server for hosting your behavioral experiments, making them accessible to participants.
- **Store experimental data with [Google Firestore](https://firebase.google.com/)**: Efficiently manage and store participant data collected from your experiment.
- **Collect data from real participants with [Prolific](https://www.prolific.com/)**: Recruit and manage participants through Prolific, ensuring high-quality behavioral data.

## Prerequisites:
- **Basic Python knowledge**: While most of the workflow is Python-based, only a basic level of understanding is needed to follow along.
- **Minimal JavaScript knowledge**: Since the behavioral experiments are implemented in JavaScript (via jsPsych), SweetBean will handle much of the complexity for you. The code is generated in Python and converted into JavaScript, so only a minimal understanding of JavaScript is required.
- **A Google account**: You will need a Google account to use Google Firebase and Firestore.

## Study Overview

In this example study, we are interested in quantifying participant's ability to differentiate between two visual stimuli. Specifically, we will ask participants to indicate whether the number of dots in a left stimulus is the same as the number of dots in a right stimulus.

![stimulus.png](img/stimulus.png)

Our goal is to predict the participant's response based on the number of dots in the left and right stimuli. We will use two methods of predicting the response:
- a simple logistic regression model
- an equation discovery algorithm ([Bayesian Machine Scientist](https://autoresearch.github.io/autora/user-guide/theorists/bms/))

After each data collection phase, we will fit the logistic regression model and the Bayesian Machine Scientist from the  ``autora[theorist-bms]`` package to the collected data. We will then use both models to determine the next set of experimental conditions worth testing. Specifically, we will identify experimental conditions for which the [models disagree the most](https://autoresearch.github.io/autora/user-guide/experimentalists/model-disagreement/), using the ``autora[experimentalist-model-disagreement]`` package.

Critically, we will leverage AutoRA to embed the entire research process into a closed-loop system. This system will automatically generate new experimental conditions, collect data from the web experiment, and update the models based on the collected data.


## System Overview

Our closed-loop system consists of a bunch of interacting components. Here is a high-level overview of the system:
![System Overview](../img/system_overview.png)

Our closed-loop system will have two projects talking to each other. The **Firebase project** will host and run the web experiment that participants interact with. Our **local AutoRA project** will host the code that runs the AutoRA workflow, which will generate new experiment conditions, collect data from the web experiment, and update the model based on the collected data. 

### Firebase Project
To run an online experiment, we need to host it as a **web app**. We will leverage **Google Firebase** to host our web app. Participants from **Prolific** can then interact with the web app to complete the experiment.  

Our experiment is configured by experiment conditions, which are stored in a **Google Firestore** database. In addition, we will use this database to store collected behavioral data from the web experiment. 

### Our Local AutoRA Project

The local project will consist of two folders. The ``testing_zone```` folder will contain the web app that participants interact with. 

The ``researcher_hub`` folder will contain the AutoRA workflow. 

## Next Steps 

Each step in the example will lead guide you to set up each component of the closed-loop system. 

By the end of this example, you’ll be able to create a fully automated behavioral closed-loop study that adapts based on the collected participant data.

[Next: Set up the local project.](setup.md)

