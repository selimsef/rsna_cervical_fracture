import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.metrics import classification_report
from torch.nn import BCELoss

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import warnings

from training import augmentation_sets, dataset_cls
from training.augmentation_sets import FractureAugmentations

from training.utils import all_gather

import torch

from tqdm import tqdm

import torch.distributed as dist

import numpy as np

from training.config import load_config

warnings.filterwarnings("ignore")
import argparse
from typing import Dict

from training.trainer import TrainConfiguration, PytorchTrainer, Evaluator

from torch.utils.data import DataLoader
import torch.distributed


class SegFractureEvaluator(Evaluator):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        self.competition_weights = {
            'negative': torch.from_numpy(np.array([7, 1, 1, 1, 1, 1, 1, 1])).float(),
            'positive': torch.from_numpy(np.array([14, 2, 2, 2, 2, 2, 2, 2])).float()
        }

    def init_metrics(self) -> Dict:
        return {"loss": 10}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:
        os.makedirs(self.args.pred_dir, exist_ok=True)
        all_preds = []
        all_targets = []
        for sample in tqdm(dataloader):
            imgs = sample["image"].cuda().float()[0]
            label = sample["label"].cpu().float()[0]
            all_targets.append(label.numpy())
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model(imgs)["cls"]
                preds = torch.sigmoid(output.float())
                preds = np.max(preds.cpu().numpy().astype(np.float32), axis=0)
                preds = np.clip(preds, 0.01, 0.99)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        if distributed:
            all_preds = all_gather(all_preds)
            all_preds = np.concatenate(all_preds, axis=0)

            all_targets = all_gather(all_targets)
            all_targets = np.concatenate(all_targets, axis=0)
        result = 10
        if local_rank == 0:
            y = torch.from_numpy(all_targets).float()
            loss = BCELoss(reduction='none')(torch.from_numpy(all_preds).float(), y)
            weights = self.competition_weights['positive'] * y + \
                      self.competition_weights['negative'] * (1 - y)
            loss = (loss * weights).sum(axis=1)
            loss = loss / weights.sum(axis=1)
            result = (loss.sum() / y.shape[0]).item()
            print(classification_report(all_targets > 0.5, all_preds > 0.5))

        if distributed:
            dist.barrier()
        torch.cuda.empty_cache()
        return {"loss": result}

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        best_loss = prev_metrics["loss"]
        if current_metrics["loss"] < prev_metrics["loss"]:
            print("Loss improved from {:.4f} to {:.4f}".format(prev_metrics["loss"], current_metrics["loss"]))
            improved["loss"] = current_metrics["loss"]
            best_loss = current_metrics["loss"]
        print("Best Loss {:.4f} current {:.4f}".format(best_loss, current_metrics["loss"]))
        return improved


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/vgg512.json")
    arg('--workers', type=int, default=6, help='number of cpu threads to use')
    arg('--gpu', type=str, default='1', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='')
    arg('--data-dir', type=str, default="/home/selim/datasets/rsna/")
    arg('--folds-csv', type=str, default="folds.csv")
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--fp16', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world-size", default=1, type=int)
    arg("--test_every", type=int, default=1)
    arg('--freeze-epochs', type=int, default=0)
    arg('--pred-dir', type=str, default="../oof")
    arg("--val", action='store_true', default=False)
    arg("--freeze-bn", action='store_true', default=False)

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    conf = load_config(args.config)
    slice_size = conf.get("slice_size", 32)
    val_slice_size = conf.get("val_slice_size", slice_size)
    crop_size = conf.get("crop_size", 160)

    augmentations = augmentation_sets.__dict__[conf["augmentations"]]()  # type: FractureAugmentations
    dataset_type = dataset_cls.__dict__[conf["dataset"]["type"]]
    params = conf["dataset"].get("params", {})
    print(f"Using augmentations: {augmentations.__class__.__name__} with Dataset: {dataset_type.__name__}")
    train_dataset = dataset_type(mode="train",
                                 dataset_dir=args.data_dir,
                                 fold=args.fold,
                                 crop_size=crop_size,
                                 folds_csv=args.folds_csv,
                                 transforms=augmentations.get_train_augmentations(conf),
                                 slice_size=slice_size,
                                 multiplier=conf.get("multiplier", 1), **params)
    val_dataset = dataset_type(mode="val", dataset_dir=args.data_dir, fold=args.fold,
                               folds_csv=args.folds_csv,
                               crop_size=crop_size,
                               slice_size=val_slice_size,
                               transforms=augmentations.get_val_augmentations(conf), **params)
    return train_dataset, val_dataset


def main():
    args = parse_args()
    trainer_config = TrainConfiguration(
        config_path=args.config,
        gpu=args.gpu,
        resume_checkpoint=args.resume,
        prefix=args.prefix,
        world_size=args.world_size,
        test_every=args.test_every,
        local_rank=args.local_rank,
        distributed=args.distributed,
        freeze_epochs=args.freeze_epochs,
        log_dir=args.logdir,
        output_dir=args.output_dir,
        workers=args.workers,
        from_zero=args.from_zero,
        zero_score=args.zero_score,
        fp16=args.fp16,
        freeze_bn=args.freeze_bn
    )

    data_train, data_val = create_data_datasets(args)
    seg_evaluator = SegFractureEvaluator(args)
    trainer = PytorchTrainer(train_config=trainer_config, evaluator=seg_evaluator, fold=args.fold,
                             train_data=data_train, val_data=data_val)
    if args.val:
        trainer.validate()
        return
    trainer.fit()


if __name__ == '__main__':
    main()
