# Connect AutoRA WorkFlow to Firebase

After setting up a mechanism to deploy your experiments online, you can now connect the AutoRA workflow to Firestore database. This will allow us to update experiment conditions and to download observations collected from the experiment.

![Setup](../img/system_researcherhub.png)

The workflow will manage the entire research process, from generating novel experiments to collecting data and modeling the results. 

Note that the workflow will talk to the Firebase project by uploading and downloading data to the Firestore database. We will upload new experiment conditions to the database and download the results of the experiment from it. 

The AutoRA workflow can be found in the `researcher_hub` folder, which contains a template for an AutoRA workflow for a two-armed bandit task using a model discovery process (AutoRA-theorist) described here: [rnn-sindy-rl](https://github.com/AutoResearch/autora-theorist-rnn-sindy-rl).


1. Move into the `researcher_hub` directory, where the template for the workflow is stored.

```shell
cd researcher_hub
```

2. Then install the Python packages required for the workflow using `pip`:

```shell
pip install -r requirements.txt
```

3. There are three different experiment runners:
   - `autora-synthetic-runner`: This runner is used to run the AutoRA workflow with simulated data.
   - `autora-firebase-runner`: This runner is used to run the AutoRA workflow with Firebase. It will upload new experiment conditions to the Firestore database and download the results of the experiment from it. This can be used to test the online experiment.
   - `autora-prolific-runner`: This runner is used to run the AutoRA workflow with Prolific. It will recruit participants via Prolific and upload the results of the experiment to Prolific.

## Add Firebase Credentials

The AutoRA workflow (specifically the `autora-firebase-runner` and `autora-prolific-runner`) will need access to your firebase project. Therefore, we need the corresponding credentials. 

1. To obtain the credentials, go to the [Firebase console](https://console.firebase.google.com/).
2. Navigate to the project.
3. Click on the little gear on the left and then select ``Project settings``. 
4. Click on ``Service accounts``.
![service_account.png](../img/service_account.png)
5. Having ``Node.js`` selected, generate a new private key. This should generate a json file that you can download.
6. Open the file `autora_workflow.py` in the `research_hub`-folder and navigate to the part of the code that contains a placeholder for the credentials. It should look like this
```python
firebase_credentials = {
    "type": "type",
    "project_id": "project_id",
    "private_key_id": "private_key_id",
    "private_key": "private_key",
    "client_email": "client_email",
    "client_id": "client_id",
    "auth_uri": "auth_uri",
    "token_uri": "token_uri",
    "auth_provider_x509_cert_url": "auth_provider_x509_cert_url",
    "client_x509_cert_url": "client_x509_cert_url"
}
```
7. Replace the placeholders with the credentials from the json file you downloaded.

## Try out the Workflow

First try the synthetic runner to see if everything works correctly. then you can set the experiment to run with firebase and run `autora_workflow.py`. Then head over to your website to test your first online experiment.

This will be the experiment as participants will see it. You can test it and check the output of the loop.

To check the database, go to the Firebase console and select your project. On the left menu, navigate to ``Firestore Database``. If everything worked, you should see database fields called ``autora_in`` and ``autora_out``. The former contains the experiment conditions which are used to configure the experiment. The latter will contain the results of the experiment.

![firestore_data.png](../img/firestore_data.png)

[next](customization.md)

