import numpy as np
import pandas as pd


def plus_1(df: pd.DataFrame):
    x = df[["x"]]
    y = x + 1
    df["y"] = y
    return df


def ten_random_samples(random_state: int = 180):
    rng = np.random.default_rng(random_state)
    values = rng.uniform(0, 10, 10)
    df = pd.DataFrame({"x": values})
    return df
