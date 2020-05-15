import os


def trial_to_list(trial=None, IVList = None, DVList = None):

    messages = list()

    if trial is not None:
        messages.append("--- Step " + str(trial) + " ---")

    if IVList is not None:
        for IV in IVList:
            messages.append("IV " + str(IV[0]) + " = " + str(IV[1]))

    if DVList is not None:
        messages.append("--- Measurement: ")
        for DV in DVList:
            messages.append("DV " + str(DV[0]) + " = " + str(round(DV[1],4)))

    return messages

def get_experiment_files(path):

    experiment_files = list()

    for file in os.listdir(path):
        if file.endswith(".exp"):
            experiment_files.append(str(file)) # os.path.join(path, file)

    return experiment_files
