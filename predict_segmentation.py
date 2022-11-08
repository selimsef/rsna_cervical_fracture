import warnings
from typing import List

import tifffile
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import zoo
from batch_slicer import get_slices
from training.augmentation_sets import AsIs
from training.config import load_config
from training.dataset_seq import DatasetSeqStrideVal
from training.utils import load_checkpoint

warnings.filterwarnings("ignore")
import argparse
import os

import torch.distributed
import numpy as np


def process_distributed(models: List[torch.nn.Module], args):
    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    test_dataset_dir = args.data_dir
    augs = AsIs()
    test_dataset = DatasetSeqStrideVal(dataset_dir=test_dataset_dir, transforms=augs.get_val_augmentations({}) )
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    oof_loader = DataLoader(
        test_dataset, batch_size=1, sampler=sampler, shuffle=False, num_workers=1, pin_memory=False
    )
    pred_dir = args.out_dir
    os.makedirs(pred_dir, exist_ok=True)
    for sample in tqdm(oof_loader):
        image = sample["image"]
        h = int(sample["h"][0])
        cube_id = sample["cube_id"][0]
        imgs = image.cpu().float()
        case_preds = np.zeros((imgs.shape[2], 256, 256), dtype=np.float32)

        with torch.no_grad():
            slices = get_slices(imgs, dim=2, window=384, overlap=128)
            for slice in slices:
                batch = imgs[:, :, slice.i_from:slice.i_to].cuda().float()
                with torch.cuda.amp.autocast(enabled=True):
                    preds = None
                    for model in models:
                        if preds is None:
                            preds = torch.softmax(model(batch)["mask"], dim=1)[0]
                        else:
                            preds += torch.softmax(model(batch)["mask"], dim=1)[0]
                    preds = torch.argmax(preds, dim=0)
                preds = preds.cpu().numpy()

                for pred_idx in range(slice.i_start, preds.shape[0]):
                    idx = slice.i_from + pred_idx
                    y_pred = preds[pred_idx]
                    case_preds[idx] = y_pred[:, :]
                torch.cuda.empty_cache()
        case_preds = np.array(case_preds)[:h]
        case_preds = case_preds.astype(np.uint8)
        tifffile.imwrite(os.path.join(pred_dir, f"{cube_id}.tif"), case_preds)

def load_model(args, config_path, checkpoint):
    conf = load_config(config_path)
    model = zoo.__dict__[conf['network']](**conf["encoder_params"])
    model = model.cuda()
    load_checkpoint(model, checkpoint)
    channels_last = conf["encoder_params"].get("channels_last", False)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                        find_unused_parameters=True)
    else:
        model = DataParallel(model)
    return model.eval()


def main():
    args = parse_args()
    init_gpu(args)
    config_path = os.path.join("configs", f"{args.config}.json")
    models = []
    for checkpoint in args.checkpoint.split(","):
        checkpoint_path = os.path.join(args.weights_path, checkpoint)
        model = load_model(args, config_path, checkpoint_path)
        models.append(model)
    process_distributed(models, args)


def init_gpu(args):
    if args.distributed:
        dist.init_process_group(backend="nccl",
                                rank=args.local_rank,
                                world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', type=str)
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='1', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--checkpoint', type=str, required=True)
    arg('--weights-path', type=str, default="weights")
    arg('--data-dir', type=str, default="/mnt/md0/datasets/rsna/")
    arg('--out-dir', type=str, default="/mnt/md0/datasets/rsna/seg_preds")
    arg('--fp16', action='store_true', default=False)
    arg('--fold', type=int, default=1)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world-size", default=1, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
