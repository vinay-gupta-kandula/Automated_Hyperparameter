import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_and_split_data(test_size=0.2, random_state=42):
    df = pd.read_csv("data/california_housing.csv")

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
