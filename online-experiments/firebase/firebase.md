# Firebase Integration

On this page, you can find information on how to set up a [Firebase](https://firebase.google.com/) website to collect observations for an AutoRA workflow. You can find information on how to connect such a website to AutoRA and how to automatically recruit participants via [Prolific](https://www.prolific.co/) at the following pages, respectively: [AutoRA Firebase Experimentation manager](../../user-guide/experiment-runners/experimentation-managers/firebase), [AutoRA Prolific Recruitment Manager](../../user-guide/experiment-runners/recruitment-managers/prolific/).

For setting up the online experiment, we recommend using either the [user cookiecutter template](https://github.com/AutoResearch/autora-user-cookiecutter) or the [create-react-app template](https://github.com/AutoResearch/cra-template-autora-firebase).

!!! hint
    The cookiecutter template also provides a template for the AutoRA workflow used in online experiments.

## Installation

To make sure that `node` is installed on your system, run the following command. 
```shell
node -v
```
If you don't see a version number or the version number is below 16.0, install or update `node` following the instruction on the [node.js website](https://nodejs.org/en/).

When `node` is available on your system, you can use ***create-react-app*** by running the following command.
```shell
npx create-react-app path/to/react/pp --template autora-firebase
```
If you want to use ***cookiecutter***, run the following command and follow the instructions.
```shell
cookiecutter https://github.com/AutoResearch/autora-user-cookiecutter
```
This creates your ***project folder***. Before writing code for your website, you also need to set up a Firebase project.
 
## Firebase Project Setup

### Initialize Firebase Account And Project

- Create and log in to a Firebase account on the [Firebase website](https://firebase.google.com/).
- Create a Firebase project by clicking add project and enter a project name.
- You can choose to disable google analytics in the next page if you are not planning on using it for your project.

### Copy Web App Credentials

- Navigate to the [Firebase console](https://console.firebase.google.com/) and select the project
- To create a new web app, click on `Add App` or the `<>` symbol and follow the prompts
- Enter a name for the Firebase app (could be the same as the project name)
- Check `Also set up Firebase Hosting for this app`
- Click `Register App`. This auto-generates a script with several values that you need to copy for the next step.
- Copy the auto-generated values from the Firebase console to the corresponding variables in the `.env` file in the project folder that was created on your system using create-react-app or cookiecutter
```dotenv
REACT_APP_apiKey=
REACT_APP_authDomain=
REACT_APP_projectId=
REACT_APP_storageBucket=
REACT_APP_messagingSenderId=
REACT_APP_appId=
REACT_APP_devNoDb="True"
REACT_APP_useProlificId="False"
```
- Click on `Next`
- You will not need to run the command that is displayed after first clicking `Next`, so click `Next` again
- Click `Continue to console`

### Firestore Setup
AutoRA includes cloud storage for task data using Firestore. Follow these steps to initialize Firestore:

- Navigate to the current project in the developer console and select `Firestore Database` from the sidebar under `Build`.
- Click `Create Database`
- Select production mode and click `Next`
- Choose the current location and click `Enable`

### Configure Your Project For Firebase
In the project folder, enter the following commands in your terminal:
First log in to your Firebase account using
```shell
firebase login
```
Then initialize the Firebase project in this folder by running:
```shell
firebase init
```
An interactive initialization process will now run in your command line. For the first question, select these options:

- Firestore: Configure security rules and indexes files for Firestore
- Hosting: Configure files for Firebase Hosting and (optionally) set up GitHub Action deploys
- For a Firebase project, use the one you created earlier
- Use the default options for the Firestore rules and the Firestore indexes.
- ***!!! IMPORTANT !!!*** Use the build directory instead of the public directory here.
- When asked for the directory, write `build` and press `Enter`.
- Configure as a single-page app; don't set up automatic builds and deploys with GitHub. 
- Don't overwrite the index.html file if the question pops up.

## Write Code For Your Experiment
To write code for your experiment, use the `main.js` file in the `src/design` folder. For example, you can use [jsPsych](https://www.jspsych.org/7.3/) and install packages using `npm`. The main function should return an observation (the data created by a participant).

You can test the experiment locally using
```shell
npm start
```
During development, the Firestore database will not be used. If you want to load conditions from the database, you need to upload them first (for example using the [AutoRA Firebase Experimentation Manager](../../user-guide/experiment-runners/experimentation-managers/firebase/)) and set `REACT_APP_devNoDb="False"` in the `.env` file.

### Using Prolific Id's
If you want to recruit participants via Prolific (for example using the [AutoRA Prolific Recruitment Manager](../../user-guide/experiment-runners/recruitment-managers/prolific/)), we ***highly recommend*** setting `REACT_APP_useProlificId="True"`. This will speed up the recruitment of participants.

## Build And Deploy To Firebase 
To serve the website on the internet, you must build and deploy it to Firebase.
To build the project, run
```shell
npm run build
```
To deploy to Firebase, run
```shell
firebase deploy
```
This will make the website available on the web. You can find the URL of the website in the command line or on the Firebase console of your project under `Hosting`.


