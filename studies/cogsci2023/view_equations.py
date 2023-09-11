import pandas as pd
import pickle
import os

path = 'prior_0.25/'

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
    for val in data:
        print(val)
    print('\n\n')

if __name__ == '__main__':
    ...
