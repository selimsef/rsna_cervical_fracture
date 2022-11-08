import argparse
import json
from collections import OrderedDict
from os.path import join as pjoin
from typing import List, Tuple

import torch


def load_metrics(json_path, metric_name: str) -> List[Tuple[int, float]]:
    with open(json_path) as f:
        lines = f.readlines()
    metrics_per_epoch = {}
    for line in lines:
        epoch_metrics = json.loads(line)
        epoch = epoch_metrics["epoch"]
        metric_value = epoch_metrics["metrics"][metric_name]
        metrics_per_epoch[epoch] = metric_value
    epoch_metrics = [(e, m) for e, m in metrics_per_epoch.items()]
    epoch_metrics.sort(key=lambda x: x[0])
    return epoch_metrics


def swa_and_save(
        path: str,
        exp_name: str,
        num_best: int,
        min_epoch: int,
        maximize: bool,
        metric_name: str = "comp_metric",
        output_path: str = ".",
):
    epoch_metrics = load_metrics(pjoin(path, exp_name + ".json"), metric_name)
    epoch_metrics = [(e, m) for e, m in epoch_metrics if e >= min_epoch]

    swa_checkpoint_name = f"swa_{num_best}_best_{exp_name}.pth"
    swa_chkp = OrderedDict({"state_dict": None})
    # sort by metric value
    epoch_metrics.sort(key=lambda x: x[1], reverse=maximize)
    best_epochs_metrics = epoch_metrics[:num_best]
    print(f"Best epochs {[e for e, _ in best_epochs_metrics]}")
    print(f"Best {metric_name} {[m for _, m in best_epochs_metrics]}")
    for best_epoch_metric in best_epochs_metrics:
        epoch, metric_value = best_epoch_metric
        temp_chkp = torch.load(pjoin(path, f"{exp_name}_{epoch}.pth"), map_location="cpu")

        if swa_chkp['state_dict'] is None:
            swa_chkp['state_dict'] = temp_chkp['state_dict']
        else:
            for k in swa_chkp['state_dict'].keys():
                if isinstance(swa_chkp['state_dict'][k], torch.FloatTensor):
                    swa_chkp['state_dict'][k] += temp_chkp['state_dict'][k]

    for k in swa_chkp['state_dict'].keys():
        if isinstance(swa_chkp['state_dict'][k], torch.FloatTensor):
            swa_chkp['state_dict'][k] /= len(best_epochs_metrics)
    swa_chkp["epoch"] = 0
    torch.save(swa_chkp, pjoin(output_path, swa_checkpoint_name))


if __name__ == '__main__':
    """
    A simple tool to use stochastic weight averaging for trained checkpoints
    """
    parser = argparse.ArgumentParser("SWA for trained checkpoints of one training cycle")
    parser.add_argument('--path', required=True, help="Path with checkpoints")
    parser.add_argument('--output_path', default=".", help="Directory to save swa checkpoint")
    parser.add_argument('--exp_name', type=str, required=True, help="Checkpoint prefix")
    parser.add_argument('--num_best', type=int, required=True, help="number of best checkpoints to be averaged")
    parser.add_argument('--min_epoch', type=int, required=True, help="min start epoch to select checkpoints")
    parser.add_argument('--metric_name', type=str, default="comp_metric", help="metric to be selected")
    parser.add_argument('--maximize', action='store_true', default=False)
    args = parser.parse_args()
    swa_and_save(args.path, args.exp_name,
                 num_best=args.num_best,
                 maximize=args.maximize,
                 min_epoch=args.min_epoch,
                 metric_name=args.metric_name,
                 output_path=args.output_path)
