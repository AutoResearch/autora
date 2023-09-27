import pandas as pd

noises = ['0.01', '0.025', '0.05', '0.1', '0.25', '0.5']
df_ref = pd.read_csv('data_noise_0.01/df_validation.csv')
columns = df_ref.columns.append(pd.Index(['Noise']))
theorists = df_ref['Theorist'].unique()
print(theorists)
gts = df_ref['Ground Truth'].unique()
df_all = pd.DataFrame(columns=columns)

for noise in noises:
    df_noise = pd.read_csv(f'data_noise_{noise}/df_validation.csv')
    df_noise['Noise'] = noise
    rows = [False for _ in range(df_noise.shape[0])]
    df_trim = df_noise.loc[rows]
    for idx in range(df_noise.shape[0]):
        row_theorist = df_noise['Theorist'][idx]
        row_gt = df_noise['Ground Truth'][idx]
        if len(df_trim.loc[(df_trim['Theorist'] == row_theorist)
                          & (df_trim['Ground Truth'] == row_gt)]) < 15 and row_theorist in theorists:
            rows[idx] = True
        else:
            rows[idx] = False
        df_trim = df_noise.loc[rows]
    print(df_trim.shape)
    df_all = pd.concat([df_all, df_trim])
    print(df_all.shape)
del df_all['Entry']
df_all.to_csv('df_trim.csv')
