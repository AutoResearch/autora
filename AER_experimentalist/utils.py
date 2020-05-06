import os


def trial_to_list(trial=None, IVList = None, DVList = None):

    messages = list()

    if trial is not None:
        messages.append("--- TRIAL " + trial + " ---")

    if IVList is not None:
        for IV in IVList:
            messages.append("IV " + IV[0] + str(IV[1]))

    if IVList is not None:
        messages.append("--- MEASUREMENT: ")
        for DV in DVList:
            messages.append("DV " + DV[0] + str(DV[1]))


def get_experiment_files(path):

    experiment_files = list()

    for file in os.listdir(path):
        if file.endswith(".exp"):
            experiment_files.append(str(file)) # os.path.join(path, file)

    return experiment_files
