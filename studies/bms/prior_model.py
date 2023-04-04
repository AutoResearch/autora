from autora.skl.bms import BMSRegressor
import os
import numpy
import pandas as pd
import pickle
import matplotlib.pyplot as plt

op_counts = {
    'sinh': 5,
    'cos': 65,
    'log': 132,
    'tanh': 6,
    'pow2': 547,
    '-': 520,
    'abs': 27,
    'sqrt': 130,
    'cosh': 4,
    'fac': 7,
    '+': 1271,
    '**': 652,
    'exp': 129,
    'pow3': 38,
    '*': 2774,
    '/': 1146,
    'sin': 39,
    'tan': 4
}
FREQUENCY = 4080


def read_prior_par(filename):
    with open(filename) as inf:
        lines = inf.readlines()
    par = dict(list(zip(lines[0].strip().split()[1:],
                    [float(x) for x in lines[-1].strip().split()[1:]])))
    return par


def extract_info(file_name):
    nv_start = file_name.index('nv')
    nv_stop = file_name.index('.', nv_start)
    np_start = file_name.index('np')
    np_stop = file_name.index('.', np_start)
    nv = int(file_name[nv_start+2:nv_stop])
    np = int(file_name[np_start + 2:np_stop])
    return nv, np


prior_dicts = []
for file in os.listdir('Prior/'):
    # print(file)
    if file.endswith('.dat'):
        [num_var, num_param] = extract_info(file)
        if num_var < 25 and num_param < 25:
            prior = read_prior_par('Prior/' + file)
            prior.update({'nv': num_var, 'np': num_param})
            prior_dicts.append(prior)

df = pd.DataFrame(index=range(len(prior_dicts)), columns=prior_dicts[0].keys())
df['nv'] = 0
df['np'] = 0

for idx, dic in enumerate(prior_dicts):
    for key in dic.keys():
        df.loc[df.index == idx, key] = dic[key]

base_dict = {}
for dic in prior_dicts:
    if dic['np'] == 1 and dic['nv'] == 1:
        base_dict = dic


map_numpy_1 = numpy.zeros((1974, 4))
map_numpy_2 = numpy.zeros((564, 4))
idx_1 = 0
idx_2 = 0
for dic in prior_dicts:
    nv, np = dic['nv'], dic['np']
    for op_name in op_counts.keys():
        if op_name in ['/', '+', '*', '**']:
            map_numpy_2[idx_2] = [nv, np, op_counts[op_name] / FREQUENCY, (dic['Nopi_' + op_name])]
            idx_2 += 1
        else:
            map_numpy_1[idx_1] = [nv, np, op_counts[op_name] / FREQUENCY, (dic['Nopi_' + op_name])]
            idx_1 += 1
print(idx_1)
print(idx_2)
map_df_1 = pd.DataFrame(data=map_numpy_1, columns=['nv', 'np', 'frequency', 'prior'])
map_df_2 = pd.DataFrame(data=map_numpy_2, columns=['nv', 'np', 'frequency', 'prior'])

print(map_df_1)
print(map_df_2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(numpy.log(map_df_1['frequency']), map_df_1['np'], map_df_1['prior'], c=map_df_1['nv'])
# ax.scatter(map_df_1['frequency'], map_df_1['nv'], map_df_1['np'], c=map_df_1['prior'])
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(numpy.log(map_df_2['frequency']), numpy.log(map_df_2['np']+map_df_2['nv']), map_df_2['prior'], c=map_df_2['np']-map_df_2['nv'])
# ax.scatter(map_df_2['frequency'], map_df_2['nv'], map_df_2['np'], c=map_df_2['prior'])
plt.show()

plt.scatter(map_df_2['frequency'], map_df_2['prior'])
plt.show()
plt.scatter(map_df_2['frequency'], map_df_2['prior']-numpy.log(map_df_2['np']+map_df_2['nv']))
plt.show()
plt.scatter(map_df_1['frequency'], map_df_1['prior'])
plt.show()

for col in ['**', '+', '-', '/', '*']:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    col_key = ('Nopi_'+str(col))
    df['model'] = numpy.log(df['nv'] + df['np']) + numpy.log(op_counts[col])/2
    # print(numpy.log(op_counts[col]))
    ax.scatter(df['nv'], df['np'], df[col_key])
    # ax.scatter(df_numpy[:, 0], df_numpy[:, 1], df_numpy[:, 2])
    ax.scatter(df['nv'], df['np'], df['model'])
    ax.set_xlabel('Number of variables')
    ax.set_ylabel('Number of parameters')
    ax.set_zlabel('Prior value')
    plt.show()

bms = BMSRegressor(epochs=30, prior_par={'Nopi_log':5.0,'Nopi_-':5.0,'Nopi_+':5.0,'Nopi_*':5.0})
models = []


for col in ['Nopi_**', 'Nopi_*']:
    if col in ['Nopi_**', 'Nopi_pow3', 'Nopi_sqrt', 'Nopi_*', 'Nopi_cos']:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # df[col] = (df[col] - df[col].min()) * 100
        # df[col] = df[col]/df[col].max()
        # df['nv'] = df['nv'] / df['nv'].max()
        # df['np'] = df['np'] / df['np'].max()
        n_copies = 100
        length = len(df[col])
        df_numpy = numpy.zeros((length*n_copies, 3))
        noise = numpy.random.normal(scale=0.05, size=(df_numpy.shape[0],))
        # print(noise)
        for n in range(n_copies):
            df_numpy[n * length:(n + 1) * length, 0] = df['nv']
            df_numpy[n * length:(n + 1) * length, 1] = df['np']
            df_numpy[n * length:(n + 1) * length, 2] = df[col]
        width = 3
        for i in range(width):
            noise = numpy.random.normal(scale=0.05, size=(df_numpy.shape[0],))
            df_numpy[:, i] += noise * df_numpy[:, i].std()
        # df['model'] = numpy.log(df['nv'])*0.5 + numpy.log(df['np'])*0.7 + 4.8
        df['model'] = numpy.log(df['nv'] + df['np']) + 4.0
        # df['model'] = 0.1*numpy.log(df['np']/(df['nv']+0*df['np'])) + 6.3
        # df['model'] = 0.1*numpy.log(df['np'])+0.05*numpy.log(10/df['nv'])*df['np'] + 6.3
        # df['model'] = numpy.sqrt(df['nv'])*0.7 + numpy.sqrt(df['np'])*0.7 + 4
        # bms.fit(df[['nv', 'np']], df[col], num_param=1)
        bms.fit(df_numpy[:, 0:1], df_numpy[:, 2], num_param=4)
        print(bms.models_)
        ax.scatter(df['nv'], df['np'], df[col])
        # ax.scatter(df_numpy[:, 0], df_numpy[:, 1], df_numpy[:, 2])
        ax.scatter(df['nv'], df['np'], df['model'])
        ax.set_xlabel('Number of variables')
        ax.set_ylabel('Number of parameters')
        ax.set_zlabel('Prior value')
        # ax.scatter(df_numpy[:, 0], df_numpy[:, 1], bms.predict(df_numpy[:, 0:1]))
        # ax.scatter(df['nv'], df['np'], bms.predict(df[['nv', 'np']]))
        plt.show()
        models.append(bms.model_)
        # print(col)

print(models)
# with open('Prior/models.pickle', 'wb') as f:
#     pickle.dump(models, f)
# 'Nopi_pow3', 'Nopi_sqrt', 'Nopi_**', 'Nopi_*', 'Nopi_cos'


if __name__ == '__main__':
    ...


'''
sinh 5
cos 65
log 132 5.14672766533
tanh 6
pow2 547
- 520
abs 27
sqrt 130
cosh 4
fac 7
+ 1271
** 652 4.77084154806
exp 129
pow3 38
* 2774
/ 1146
sin 39
tan 4
--Total: 4080 Eqs
1 2611
2 632
3 433
4 224
5 88
6 47
7 24
8 7
9 4
10 5
11 2
12 1
13 1
15 1
'''

'''
Nopi_pow3 6.2104013616
Nopi_tan 7.94257184292
Nopi2_tanh 0.666018734986
Nopi2_pow3 0
Nopi2_pow2 0
Nopi2_log 0
Nopi_log 5.14672766533
Nopi_sqrt 5.07591705874
Nopi_abs 6.77999567491
Nopi2_cosh 0.904811347048
Nopi_tanh 7.64966232054
Nopi2_sin 0.0288725467089
Nopi_** 4.77084154806
Nopi2_* 0.09643483331
Nopi_sinh 8.03302469518
Nopi2_abs 0
Nopi2_+ 0
Nopi2_fac 0
Nopi2_- 0
Nopi2_/ 0
Nopi2_sqrt 0
Nopi2_exp 0
Nopi_/ 3.50954114522
Nopi_cosh  7.83569877759
Nopi_- 3.37782149777
Nopi_+ 3.99554235253
Nopi_* 2.4269631897
Nopi2_tan 0.757841235995
Nopi_fac 8.16411359356
Nopi_sin 6.36065317999
Nopi2_sinh 0.429467981038
Nopi_cos 5.85249044701
Nopi_pow2 3.35863851091
Nopi2_cos 0
Nopi_exp 5.14991229547
Nopi2_** 0

'''
