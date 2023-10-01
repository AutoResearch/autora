import pickle
import os

noises = ['0.01', '0.025', '0.05', '0.1', '0.25', '0.5', '1.0']

gts = ['stevens_power_law',
       'weber_fechner',
       'shepard_luce_choice',
       'exp_learning']

prefix = 'data_noise_'

for noise in noises:
    path = prefix+noise+'/'
    for file in os.listdir(path):
        if 'full_theory_log' in file or 'df_validation' in file:
            continue
        for gt in gts:
            if gt in file:
                seed = int(file[int(len(gt))+1:-7])
                if seed >= 20:
                    print(noise)
                    print(gt)
                    print(file)

