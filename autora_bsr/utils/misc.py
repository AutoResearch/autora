from typing import Dict


def normalize_prior_dict(prior_dict: Dict[str, float]):
    prior_sum = 0
    for k in prior_dict:
        prior_sum += prior_dict[k]
    if prior_sum > 0:
        for k in prior_dict:
            prior_dict[k] = prior_dict[k] / prior_sum
    else:
        for k in prior_dict:
            prior_dict[k] = 1 / len(prior_dict)
