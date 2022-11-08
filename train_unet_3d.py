import collections
import os

from batch_slicer import get_slices
from training.losses import dice_round

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import warnings

from training import augmentation_sets, dataset_seq
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

    def init_metrics(self) -> Dict:
        return {"dice": 0}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:

        os.makedirs(self.args.pred_dir, exist_ok=True)
        class_names = ["0"] + [str(cls + 1) for cls in range(7)]
        count_per_class = np.zeros((len(class_names),), dtype=np.int64)
        dices = np.zeros((len(class_names),))
        data = collections.defaultdict(list)
        for sample in tqdm(dataloader):

            imgs = sample["image"].cpu().float()
            masks = sample["mask"].cpu().long()[0]

            case_preds = np.zeros((8, *masks.shape))
            case_targets = np.zeros(masks.shape)

            with torch.no_grad():
                slices = get_slices(imgs, dim=2, window=256, overlap=64)
                for slice in slices:
                    batch = imgs[:, :, slice.i_from:slice.i_to].cuda().float()
                    with torch.cuda.amp.autocast():
                        preds = torch.softmax(model(batch)["mask"], dim=1)[0]
                    preds = preds.cpu().numpy().astype(np.float32)
                    for pred_idx in range(slice.i_start, preds.shape[1]):
                        idx = slice.i_from + pred_idx
                        y_msk = masks[idx].cpu().numpy()
                        y_pred = preds[:, pred_idx] > 0.5
                        case_preds[:, idx] = y_pred
                        case_targets[idx] = y_msk
                    torch.cuda.empty_cache()
                for cls in range(1, 8):
                    non_zero_cnt = np.count_nonzero(masks == cls)
                    if non_zero_cnt > 10:
                        d = dice_round(torch.from_numpy(case_preds[cls]).float(), torch.from_numpy(case_targets == cls).float(), t=0.5)
                        dices[cls] += d
                        count_per_class[cls] += 1
        if distributed:
            dices = all_gather(dices)
            dices = np.sum(dices, axis=0)

            count_per_class = all_gather(count_per_class)
            count_per_class = np.sum(count_per_class, axis=0)
        result = 0
        if local_rank == 0:
            non_empty_dices = []
            for cls_idx in range(1, len(class_names)):
                class_name = class_names[cls_idx]

                cls_scans_count = count_per_class[cls_idx]
                cls_dice = dices[cls_idx] / cls_scans_count if cls_scans_count > 0 else -1
                print(f"class: {class_name} dice: {cls_dice:.4f} scans: {cls_scans_count}")
                if cls_scans_count > 0:
                    non_empty_dices.append(cls_dice)
            result = np.mean(non_empty_dices)
        if distributed:
            dist.barrier()
        torch.cuda.empty_cache()
        return {"dice": result}

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        best_dice = prev_metrics["dice"]
        if current_metrics["dice"] > prev_metrics["dice"]:
            print("Dice improved from {:.4f} to {:.4f}".format(prev_metrics["dice"], current_metrics["dice"]))
            improved["dice"] = current_metrics["dice"]
            best_dice = current_metrics["dice"]
        print("Best Dice {:.4f} current {:.4f}".format(best_dice, current_metrics["dice"]))
        return improved


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/vgg512.json")
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='1', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='')
    arg('--data-dir', type=str, default="/media/selim/d860719e-95b8-4a1a-b518-11791115e55b/rsna/")
    arg('--folds-csv', type=str, default="seg_folds.csv")
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

    augmentations = augmentation_sets.__dict__[conf["augmentations"]]()  # type: FractureAugmentations
    dataset_type = dataset_seq.__dict__[conf["dataset"]["type"]]
    params = conf["dataset"].get("params", {})
    print(f"Using augmentations: {augmentations.__class__.__name__} with Dataset: {dataset_type.__name__}")
    train_dataset = dataset_type(mode="train",
                                 dataset_dir=args.data_dir,
                                 fold=args.fold,
                                 folds_csv=args.folds_csv,
                                 transforms=augmentations.get_train_augmentations(conf),
                                 slice_size=slice_size,
                                 multiplier=conf.get("multiplier", 1), **params)
    val_dataset = dataset_type(mode="val", dataset_dir=args.data_dir, fold=args.fold,
                               folds_csv=args.folds_csv,
                               slice_size=slice_size,
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
