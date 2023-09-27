import pickle

# path = 'studies/cogsci2023/priors/'
path = 'priors/'

priors_list = ['priors_SUPERCognitiveScience.pkl','priors_SUPERCognitivePsychology.pkl','priors_SUPERMaterialsScience.pkl','priors_SUPERNeuroscience.pkl']
dfs = []
for prior in priors_list:
    with open(path+prior, "rb") as f:
        dfs.append(pickle.load(f))
ops = set()
total_equations = 0
for df in dfs:
    ops = ops.union(set(df['priors']['operators_and_functions'].keys()))
ops = dict({op: 0 for op in ops})
print(ops)
for df in dfs:
    prs = df['priors']['operators_and_functions']
    print('\nNext Prior\n')
    for pr in prs.items():
        print(f"{pr[0]}: {pr[1]/df['metadata']['number_of_equations']}")
        total_equations += df['metadata']['number_of_equations']
        ops.update({pr[0]: ops[pr[0]]+pr[1]})
print('\n')
for op in ops.items():
    print(f"'{op[0]}': {op[1]/total_equations},")
print('\n')
all_ops = sum(ops.values())
for op in ops.items():
    print(f"'{op[0]}': {all_ops/(total_equations*len(ops.items()))},")

print(len(ops))
for df in dfs:
    prs = df['priors']['operators_and_functions']
    print(len(prs.items()))
