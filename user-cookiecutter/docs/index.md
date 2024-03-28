# Closed Loop Online Experimet

To establish am online closed-loop for AutoRA, there are two key components that need to be configured:

1. AutoRA Workflow
    - This workflow can be executed locally, on a server, or using `Cylc`. It must have the ability to communicate with a website, allowing for writing of new conditions and reading of observation data.
    - The AutoRA workflow can be customized by adding or removing AutoRA functions, such as AutoRA *experimentalists* or AutoRA *theorists*. It relies on an AutoRA Prolific Firebase *erunner* to collect data from a online experiment hosted via firebase and recruit participants via prolific.

2. Website To Conduct Experiment:
    - The website serves as a platform for conducting experiments and needs to be compatible with the AutoRA workflow.
    - In this setup, we use a `Firbase` to hosted on website.

To simplify the setup process, we provide a `cookiecutter` template that generates a project folder containing the following two directories:

1. Researcher Hub:
    - This directory includes a basic example of an AutoRA workflow.

2. Testing Zone:
    - This directory provides a basic example of a website served with Firebase, ensuring compatibility with the AutoRA workflow.

The following steps outline how to set up the project:

## Set Up The Project On The Firebase Website

To serve a website via Firebase and use the Firestore Database, it is necessary to set up a Firebase project. Follow the steps below to get started:

### Google Account
You'll need a [Google account](https://www.google.com/account/about/) to use Firebase.

### Firebase Project
While logged in into your Google account head over to the [Firebase website](https://firebase.google.com/). Then create a new project:

- Click on `Get started`.
- Click on the plus sign with `add project`.
- Name your project and click on `continue`.
- For now, we don't use Google Analytics (you can leave it enabled if you want to use it in the future).
- Click `Create project`.

### Adding A Webapp To Your Project
Now, we add a webapp to the project. Navigate to the project and follow these steps:

- Click on ```<\>```.
- Name the app (can be the same as your project) and check the box `Also set up Firebase Hosting`. Click on `Register app`.
- We will use `npm`. We will use the configuration details later, but for now, click on `Next`.
- We will install firebase tools later, for now, click on `Next`.
- We will login and deploy our website later, for now, click on `Continue to console`.

### Adding Firestore To Your Project
For the online closed loop system, we will use a Firestore Database to communicate between the AutoRA workflow and the website conducting the experiment. We will upload experiment conditions to the database and store experiment data in the database. To build a Firestore Database, follow theses steps:

- In the left-hand menu of your project console, click on `Build` and select `Firestore Database`
- Click on `Create database`.
- Leave `Start in production mode` selected and click on `Next`.
- Select a Firestore location and click on `Enable`.
- To check if the database is set up correctly, click on the gear symbol next to the `Project overview` in the left-hand menu and select `Project settings`.
- Under `Default GCP resource location` you should see the Firestore location that you selected.
  - If you don't see the location, select one now (click on the `pencil-symbol` and then on `Done` in the pop-up window).

## Set Up The Project On Your System

After setting up the project on Firebase, we will setup the project on our system. Here, we will use `cookiecutter` to setup an example.

### Prerequisite

To set up an online AutoRA closed-loop you need both `Python` and `Node`.

You should also consider using an IDE. We recommend: 

- PyCharm. This is a `Python`-specific integrated development environment which comes with useful tools 
  for changing the structure of `Python` code, running tests, etc. 
- Visual Studio Code. This is a powerful general text editor with plugins to support `Python` development.

#### Install `Python` and `Node`

- You can install python using the instructions at [python.org](https://www.python.org)
- You can find information about how to install on the [official Node website](https://nodejs.org/en)

#### Create A Virtual Environment

!!! success
    We recommend setting up your virtual environment using a manager like `venv`, which creates isolated `Python`  environments. Other environment managers, such as 
    [virtualenv](https://virtualenv.pypa.io/en/latest/),
    [pipenv](https://pipenv.pypa.io/en/latest/),
    [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), 
    [hatch](https://hatch.pypa.io/latest/), 
    [poetry](https://python-poetry.org), 
    
#### Prolific

...

### Run Cookiecutter

After we have installed `Python` and `Node` and set up a virtual environment, we use the `cookiecutter`. First install it:

```shell
pip install cookiecutter
```

Then, run `cookiecutter` and select the `basic` option. 

!!! hint
    If you select advanced, there are more features, but the instructions here focus on the basic template.

```shell
cookiecutter https://github.com/AutoResearch/autora-user-cookiecutter
```

This command will result in two directories, `researcher_hub` and `testing_zone`, which are described next.

## Researcher Hub: AutoRA Workflow
 The `researcher_hub` contains a basic template for an AutoRA workflow. 

To install the necessary dependencies, move to directory to the  and install the requirements.

Move to the `researcher_hub` directory:
```shell
cd researcher_hub
```

Install the requirements:
```shell
pip install -r requirements.txt
```

You can find documentation for all parts of the AutoRA workflow in the [User Guide](https://autoresearch.github.io/autora/user-guide/)

## Testing Zone: Firebase Website

The `testing_zone` contains a basic template for a website that is compatible with the [AutoRA Experimentation Manager for Firebase](https://autoresearch.github.io/autora/user-guide/experiment-runners/experimentation-managers/firebase/) and the [AutoRA Recruitment Manager for Prolific](https://autoresearch.github.io/autora/user-guide/experiment-runners/recruitment-managers/prolific/).

You can find documentation on how to connect the website to an AutoRA workflow, as well as how to build and deploy it in the documentation for [Firebase Integration](https://autoresearch.github.io/autora/online-experiments/firebase/)




