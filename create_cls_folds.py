import argparse
import os

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def main(root_dir: str, num_folds: int = 8, seed: int = 777):
    df = pd.read_csv(os.path.join(root_dir, "train.csv"))
    df["fold"] = 0
    kfold = StratifiedKFold(num_folds, shuffle=True, random_state=777)
    y = df.patient_overall
    for fold, (train_idx, test_idx) in enumerate(kfold.split(df, y)):
        df.loc[test_idx, "fold"] = fold
    df.to_csv("folds.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Folds")
    arg = parser.add_argument
    arg('--root_dir', type=str, default="/home/selim/datasets/rsna/")
    args = parser.parse_args()
    main(args.root_dir)