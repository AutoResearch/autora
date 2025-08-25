"""
Mixture Experimentalist Sampler
"""


import numpy as np
from typing import Optional, Union
import pandas as pd


def adjust_distribution(p, temperature):
    # temperature cannot be 0
    assert temperature != 0, 'Temperature cannot be 0'

    # If the temperature is very low (close to 0), then the sampling will become almost deterministic, picking the event with the highest probability.
    # If the temperature is very high, then the sampling will be closer to uniform, with all events having roughly equal probability.
    p_ = np.array(p)
    p_ = p_ / np.sum(np.abs(p_))  # Normalizing the initial distribution
    p_ = np.exp(p_ / temperature)
    final_p = p_ / np.sum(p_)  # Normalizing the final distribution
    return final_p


def sample(conditions: Union[pd.DataFrame, np.ndarray], temperature: float,
                   samplers: list, params: dict,
                   num_samples: Optional[int] = None) -> pd.DataFrame:
    """

    Args:
        conditions: pool of experimental conditions to evaluate: pd.Dataframe
        temperature: how random is selection of conditions (cannot be 0; (0:1) - the choices are more deterministic than the choices made wrt
        samplers: tuple containing sampler functions, their names, and weights
        for sampler functions that return both positive and negative scores, user can provide a list with two weights: the first one will be applied to positive scores, the second one -- to the negative
        params: nested dictionary. keys correspond to the sampler function names (same as provided in samplers),
        values correspond to the dictionaries of function arguments (argument name: its value)
        num_samples: number of experimental conditions to select

    Returns:
        Sampled pool of experimental conditions with the scores attached to them
    """

    condition_pool = pd.DataFrame(conditions)

    rankings = pd.DataFrame()
    mixture_scores = np.zeros(len(condition_pool))
    ## getting rankings and weighted scores from each function
    for (function, name, weight) in samplers:

        try:
            sampler_params = params[name]
            pd_ranking = function(conditions=condition_pool, **sampler_params)
        except:
            pd_ranking = function(conditions=condition_pool)
        # sorting by index
        pd_ranking = pd_ranking.sort_index()

        if len(weight) == 1:
            weight = weight[0]

        # if only one weight is provided, use it for both negative and positive dimensions
        if isinstance(weight, float) or isinstance(weight, int):
            pd_ranking["score"] = pd_ranking["score"] * weight
        else:
            if len(pd_ranking["score"] < 0) > 0 and len(pd_ranking["score"] > 0) > 0:  # there are both positive and negative values

                pd_ranking.loc[pd_ranking["score"] > 0]["score"] = pd_ranking.loc[pd_ranking["score"] > 0]["score"] * weight[0]  # positive dimension gets the first weight
                pd_ranking.loc[pd_ranking["score"] < 0]["score"] = pd_ranking.loc[pd_ranking["score"] < 0]["score"] * weight[1]  # negative dimension gets the second weight
            else:
                pd_ranking["score"] = pd_ranking["score"] * weight[0]

        pd_ranking.rename(columns={"score": f"{name}_score"}, inplace=True)
        # sum_scores are arranged based on the original conditions_ indices
        mixture_scores = mixture_scores + pd_ranking[f"{name}_score"]

        rankings = pd.merge(rankings, pd_ranking, left_index=True, right_index=True, how="outer")

    # adjust mixture scores wrt temperature
    weighted_mixture_scores_adjusted = adjust_distribution(mixture_scores, temperature)

    if num_samples is None:
        num_samples = condition_pool.shape[0]

    condition_indices = np.random.choice(np.arange(len(condition_pool)), num_samples,
                                         p=weighted_mixture_scores_adjusted, replace=False)
    conditions_ = condition_pool.iloc[condition_indices]
    conditions_["score"] = mixture_scores

    return conditions_


mixture_sample = sample
mixture_sample.__doc__ = """Alias for `sample`"""
