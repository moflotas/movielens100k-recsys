import pandas as pd
import numpy as np
from recommenders.datasets.python_splitters import python_stratified_split
import os

np.random.seed(42)

data = pd.read_csv(
    "../data/raw/ml-100k/u.data",
    sep="\t",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
)

# Convert the float precision to 32-bit in order to reduce memory consumption
data["rating"] = data["rating"].astype(np.float32)

train, test = python_stratified_split(
    data, ratio=0.75, col_user="user_id", col_item="item_id", seed=42
)

if not os.path.exists("../data/interim/ml-100k"):
    os.makedirs("../data/interim/ml-100k")

train.to_csv("../data/interim/ml-100k/train.csv", index=False)
test.to_csv("../data/interim/ml-100k/test.csv", index=False)
