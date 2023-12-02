# Data managing manual

## Make dataset directly

To make dataset directly, run `make_dataset.sh` file using

```bash
bash make_dataset.sh
```

> Note: This script will download raw dataset and preprocess it. If you want to download raw dataset or preprocess it separately, please follow the instructions below.

## Download raw dataset

To download raw dataset, run `download.sh` file using

```bash
bash download.sh
```

## Preprocess raw dataset

To preprocess raw dataset, run `preprocess.py` file using

I split the full dataset into a `train` and `test` dataset to evaluate performance of the algorithms against a held-out set not seen during training. Because I am using SAR to generate recommendations, all users that are in the test set must also exist in the training set. For this case, I am using `python_stratified_split` function which holds out a percentage (in this case 25%) of items from each user, but ensures all users are in both `train` and `test` datasets

```bash
python preprocess.py
```

## References

- [microsoft-recommenders official example](https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/sar_movielens.ipynb)
