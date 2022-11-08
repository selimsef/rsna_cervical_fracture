import argparse
import os

import pandas as pd
from sklearn.model_selection import KFold


def main(root_dir: str, num_folds: int = 8, seed: int = 777):
    seg_dir = os.path.join(root_dir, "segmentations")
    seg_ids = list(os.listdir(seg_dir))
    seg_ids.sort()
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    data = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(seg_ids)):
        for idx in test_idx:
            data.append([seg_ids[idx], fold])
    pd.DataFrame(data, columns=["cube_id", "fold"]).to_csv("seg_folds.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Folds")
    arg = parser.add_argument
    arg('--root_dir', type=str, default="/home/selim/datasets/rsna/")
    args = parser.parse_args()
    main(args.root_dir)