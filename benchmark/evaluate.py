from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.utils.timer import Timer

import pickle
from pathlib import Path
import pandas as pd

# Load the model and the test set
TOP_K = 10
TEST_DATASET_PATH = Path("../data/interim/ml-100k/test.csv")
MODEL_PATH = Path("../models/sar-best.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

test = pd.read_csv(TEST_DATASET_PATH)

with Timer() as test_time:
    top_k = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

print("Took {} seconds for prediction.".format(test_time.interval))

top_k = model.recommend_k_items(test, remove_seen=True)


# Ranking metrics
eval_map = map_at_k(
    test, top_k, col_user="user_id", col_item="item_id", col_rating="rating", k=TOP_K
)
eval_ndcg = ndcg_at_k(
    test, top_k, col_user="user_id", col_item="item_id", col_rating="rating", k=TOP_K
)
eval_precision = precision_at_k(
    test, top_k, col_user="user_id", col_item="item_id", col_rating="rating", k=TOP_K
)
eval_recall = recall_at_k(
    test, top_k, col_user="user_id", col_item="item_id", col_rating="rating", k=TOP_K
)

print(
    "Model:\t",
    "Top K:\t\t%d" % TOP_K,
    "MAP:\t\t%f" % eval_map,
    "NDCG:\t\t%f" % eval_ndcg,
    "Precision@K:\t%f" % eval_precision,
    "Recall@K:\t%f" % eval_recall,
    sep="\n",
)
