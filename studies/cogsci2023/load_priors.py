#Load priors
import pickle
import os

path = 'priors/'
prior_list = []
prior_counts = []
priors = None
for file in os.listdir(path):
    print(file)
    with open(path+file, 'rb') as f:
        priors_dict = pickle.load(f)
    number_of_equations = priors_dict['metadata']['number_of_equations']
    priors = priors_dict['priors']['operators_and_functions']
    frequencies = dict()
    frequency_sum = 0
    for key in priors.keys():
        frequencies.update({key: priors[key]/number_of_equations})
        frequency_sum += priors[key]/number_of_equations
    print(frequency_sum/len(priors.keys()))
    print(frequencies)
    prior_list.append(frequencies)
    prior_counts.append(number_of_equations)

avg_prior = dict()
for key in priors.keys():
    op_freq_avg = 0
    for idx, prior in enumerate(prior_list):
        try:
            op_freq_avg += prior[key]/len(prior_list)*prior_counts[idx]/sum(prior_counts)
        except KeyError:
            op_freq_avg += 0
    avg_prior.update({key: op_freq_avg})
print('Average Prior')
print(avg_prior)


if __name__ == "__main__":
    ...
