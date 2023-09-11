import pandas as pd
import pickle
import os
import numpy as np

path = 'prior_0.25/'

theorists = ['BMS UniformNew',
    'BMS Prior Williams2023SUPERCognitivePsychologyNew',
    'BMS Prior Williams2023SUPERCognitiveScienceNew',
    'BMS Prior Williams2023SUPERMaterialsScienceNew',
    'BMS Prior Williams2023SUPERNeuroscienceNew',
    'BMS Average'
             ]

# df = pd.read_csv(path+'theory_log.csv')

# for file in os.listdir(path):
#     if file.startswith('full_theory_log'):
#         with open(path + file, "rb") as f:
#             dic = pickle.load(f)
#         print(dic.keys())
with open(path + 'theory_log.pickle', "rb") as f:
    dic = pickle.load(f)
for key in dic.keys():
    print(len(dic[key]))

for gt in dic.keys():
    print(gt)
    data = dic[gt]
    for i, theorist in enumerate(theorists):
        print(f'--------{theorist}----------')
        for j in range(int(np.floor(len(data)/6))):
            print(data[i+j*6])
        print('\n')
    print('\n\n\n')

if __name__ == '__main__':
    ...
