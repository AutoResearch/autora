import pandas as pd
import pickle
import os
import numpy as np

path = 'data_noise_1.0/'

for file in os.listdir(path):
    if "full_theory_log" in file:
        with open(path + file, "rb") as f:
            dic = pickle.load(f)
        # print(dic)
    # with open(path + 'stevens_power_law_3.pickle', "rb") as f:
    #     dic = pickle.load(f)
    # print(dic)
with open(path + 'full_theory_log.pickle', "rb") as f:
    dic = pickle.load(f)
for k in dic.keys():
    eqs = dic[k]
    print('\n'+k+' - ' + str(len(eqs)) + '\n')
    for i in range(len(eqs)):
        print(eqs[i])

if __name__ == '__main__':
    ...

