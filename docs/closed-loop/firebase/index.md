# Web based behavioral closed loop

This is a tutorial on running web based behavioral experiments with firebase and autora.

There are to environments to set up for the closed loop:

# Researcher Environment - Autora

This is the python scripts that will be run on a server to run the closed loop (this typically consists of an
autora-experimentalist, autora-runner and autora-theorist)

## Create an autora-workflow

### Setting up an virtual environment

Install this in an environment using your chosen package manager. In this example we are using virtualenv

Install:

- python (3.8 or greater): https://www.python.org/downloads/
- virtualenv: https://virtualenv.pypa.io/en/latest/installation.html

Install the Prolific Recruitment Manager as part of the autora package:

create a new folder esearcher_environment and change to the directory. Here, we define the autora workflow

```shell
mkdir researcher_environment && cd researcher_environment
```

### Create a virtual environment

```shell
viratualenv venv
```

### Install dependencies

Install the runner:

```shell
pip install autora-runner-firebase-prolific
```

Install workflows

```shell
pip install autora-workflow
```

Create your workflow.
Here are examples are here:

- [Stroop Workflow](examples/stroop/researcher_environment)
- [Super Experiment Workflow](examples/super_experiment/researcher_environment)
- [SweetBean Experiment Upload Workflow](examples/autora_upload_full_experiment/researcher_environment)

run the script to populate the database and start the loop

# Test Subject Environment - Firebase

This is the website that is served to the participant. We use Firebase to host the website and Firestore as a database.
The database gets populated with conditions from the autora-runner and stores observations when participants attend the
website. The autora-runner will read the observations and pass them to the theorist.
## Create a firebase project in the browser

### Google account

You'll need a google account to use firebase. You can create one here:
https://www.google.com/account/about/

### Firebase Project

While logged in into your google account head over to:
https://firebase.google.com/

- Click on `Get started`
- Click on the plus sign with `add project`
- name your project and click on `continue`
- For now, we don't use google analytics (you can leave it enabled if you want to use it in the future)
- Click 'Create project'

### Adding a webapp to your project

in your project console (in the firebase project), we now want to add an app to our project

- Click on `</>`
- name the app (can be the same as your project) and check `Also set up Firebase Hosting`
- Click on `Register app`
- Click on `Next`
- Click on `Next`
- Click on `Continue to console`

### Adding Firestore to your project

in your project console in the left hand menu click on build and select Firestore Database

- Click on `Create database`
- Leave `Start in production mode` selected and click on `Next`
- Select a Firestore location and click on `Enable`
- To see if everything is set up correctly, in the menu click on the gear symbol next to the Project overview and
  select `Project settings`
- Under `Default GCP recource location` you should see the Firestore location, you've selected.
    - If you don't see the location, select one now (click on the `pencil-symbol` and then on `Done` in the pop-up
      window)

### Set up node

On your command line run:

```shell
node -v
```

If an error appears or the version number is bellow 16.0, install node. You can download the and install the newest
version here:
https://nodejs.org/

### Set up firebase

We also need firebase installed to initialize and deploy our projects. To check if firebase is already installed, run:

```shell
firebase
```

If you are getting an error, install firebase globally via:

```shell
npm install -g firebase-tools
```

### Get the template

Now we can get the template from npm run the command. This will create a new folder called test_subject_environment

```shell
npx create-react-app test_subject_environment  --template autora-firebase
```

### Add the firebase configuration

In the firebase console in your browser (https://console.firebase.google.com/project/), click on the gear symbol next
to `Project Overview`. Scroll down in the general settings,till you see a code-snippet with the firebaseConfig. Copy the
firebaseConfig values into the .env file in the test_subject_environment/.env file

### Link the local project to the firebase project

Change the directory to the newly created folder

```shell
cd test_subject_environment
```

Login to firebase

```shell
firebase login
```

Initialize the project:

```shell
firebase init
```

In your command line a dialog should appear.

- Choose (my selecting and pressing the space bar):
    - Firestore: Configure security rules and indexes files for Firestore
    - Hosting: Configure files for Firebase Hosting and (optionally) set up GitHub Action deploys
    - Press `Enter`
- Use an existing project -> `Enter`
- Select the project you created when creating the project in the browser -> `Enter`
- For Firestore Rules, leave the default -> `Enter`
- For Firestore indexes, leave the default -> `Enter`
- ATTENTION: For the public directory, type in `build` and press `Enter`
- Configure as single page app, type `y` and press `Enter`
- No automatic builds and deploys with GitHub, type `n` and press `Enter`

### write your own code

Write your own code in the src/design/main.js folder.
Here is a list of examples:
[Stroop Firebase main funciton](examples/stroop/test_subject_environment)
[Super Experiment Firebase main funciton](examples/super_experiment/test_subject_environment)

### test your experiment

you can test the experiment locally via

```shell
npm start
```

#### Attention: Testing without database

When running the experiment locally, by default the condition and id in your main function is set to 0. If you populated
your database already (for example with an autora-firebase-runner), you can use the conditions in your database by
setting the REACT_APP_devNoDb to False in the .env file.

### Build and deploy the experiment

#### Build

To build and deploy the experiment run

```shell
npm run build
```

This will create a new build folder in the test_subject_environment directory.

#### Deploy

To deploy the experiment, run

```shell
firebase deploy
```

This will deploy the experiment to firebase and you will get a link where participants can access it.
