import pandas as pd
import pickle
import os
import numpy as np

path = 'data_noise_0.1/'

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
    print('\n'+k+'\n')
    for i in range(15):
        print(eqs[i])

if __name__ == '__main__':
    ...

