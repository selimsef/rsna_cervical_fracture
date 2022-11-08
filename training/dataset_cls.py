import json
import os.path
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from albumentations import ReplayCompose
from torch.utils.data import Dataset


class ClassifierDatasetCropsFullRes(Dataset):
    def __init__(
            self,
            mode: str,
            dataset_dir: str,
            fold: int,
            transforms: A.Compose,
            slice_size: int,
            crop_size: int,
            multiplier: int = 1,
            folds_csv="folds.csv",
            metadata="meta.json",
    ):
        df = pd.read_csv(os.path.join(folds_csv))
        if mode == "train":
            self.df = df[df.fold != fold]
        else:
            self.df = df[(df.fold == fold)]
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.slice_size = slice_size
        self.crop_size = crop_size

        if self.mode == "train":
            self.df = pd.concat([self.df] * multiplier)
        if self.is_train:
            assert len(self.df[self.df.fold == fold]) == 0
        else:
            assert len(self.df[self.df.fold != fold]) == 0
        with open(metadata, "r") as f:
            self.metadata = json.load(f)
        self.cache = {}

    def __getitem__(self, i):
        return self.getitem(i)


    def getitem(self, i):
        row = self.df.iloc[i]
        cube_id = row.cube_id

        mask_cube = tifffile.memmap(os.path.join(self.dataset_dir, f"seg_preds", f"{cube_id}.tif"), mode="r")
        image_cube = tifffile.memmap(os.path.join(self.dataset_dir, f"image_cubes", f"{cube_id}.tif"), mode="r")

        meta = self.metadata[cube_id]
        image_mean = meta["image_mean"]
        image_std = meta["image_std"]
        sums = meta["sums"]
        boxes = meta["boxes"]
        boxes = {int(k):v for k, v in boxes.items()}
        slice_size = self.slice_size
        if self.is_train:
            fractured = []
            non_fractured = []
            for li in range(1, 8):
                if li in boxes and boxes[li][1] > 512:
                    if row[f"C{li}"] > 0:
                        fractured.append(li)
                    else:
                        non_fractured.append(li)

            if fractured and (not self.is_train or random.random() < 0.7):
                li = random.choice(fractured)
                bbox, area = boxes[li]
            else:
                li = random.choice(non_fractured)
                bbox, area = boxes[li]
            z1, z2 = bbox[0], bbox[3]
            if mask_cube.shape[0] > slice_size * 2 and random.random() < 0.0:
                #downsample
                image_cube = image_cube[::4]
                mask_cube = mask_cube[::2]
                z1 = z1//2
                z2 = z2//2
            elif random.random() < 0.0:
                #upsample masks back, use original cube
                mask_cube_tpm = []
                for msk in mask_cube:
                    mask_cube_tpm.append(msk)
                    mask_cube_tpm.append(msk)
                mask_cube = np.array(mask_cube_tpm)
                z1 = z1 * 2
                z2 = z2 * 2
            else:
                image_cube = image_cube[::2]
            y1, y2 = max(bbox[1] - 16, 0), min(bbox[4] + 16, 256)
            x1, x2 = max(bbox[2] - 16, 0), min(bbox[5] + 16, 256)
            if z2 - z1 < slice_size:
                z1 = random.randint(max(z2 - slice_size, 0), z1)
                z2 = z1 + slice_size
            images = image_cube[z1:z2, y1*2:y2*2, x1*2:x2*2].copy()
            masks = mask_cube[z1:z2, y1:y2, x1:x2].copy()


            replay = None
            image_crops = []
            mask_crops = []
            for i in range(images.shape[0]):
                image = images[i]
                mask = masks[i]
                h, w, = mask.shape
                mask = cv2.resize(mask, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
                if replay is None:
                    sample = self.transforms(image=image, mask=mask)
                    replay = sample["replay"]
                else:
                    sample = ReplayCompose.replay(replay, image=image, mask=mask)
                image_ = sample["image"]
                image_crops.append(image_)
                mask_crops.append(sample["mask"])
            images = np.array(image_crops).astype(np.float32)
            masks = np.array(mask_crops).astype(np.float32)
            images = np.expand_dims(images, -1)
            masks = np.expand_dims(masks, -1)
            labels = np.array([row.patient_overall, *[row[f"C{c}"] for c in range(1, 8)]]).astype(np.float32)

            for mi in range(1, 8):
                area_threshold = 0.3
                if labels[mi] > 0 and (masks == mi).sum() / (sums[mi] + 100) < area_threshold:
                    labels[mi] = 0
            labels[0] = labels[1:].sum() > 0

            images = (images - image_mean) / image_std

            images = np.concatenate([images, images, masks], axis=-1)
            h = images.shape[0]
            if h > slice_size:
                start = random.randint(0, h - slice_size)
                images = images[start: start + slice_size]

            elif h != slice_size:
                tmp = np.zeros((slice_size, *images.shape[1:]))
                tmp[:h] = images
                images = tmp
            images = np.moveaxis(images, -1, 0)
            sample = {}

            sample['image'] = torch.from_numpy(images).float()
            sample['label'] = torch.from_numpy(labels).float()
            sample['cube_id'] = cube_id
            return sample
        else:
            image_cube = image_cube[::2]
            all_images = []
            labels = np.zeros((8,))
            for li in range(1, 8):
                if row[f"C{li}"] > 0:
                    labels[0] = 1
                    labels[li] = 1
                if li not in boxes:
                    print("missing")
                    all_images.append(np.zeros((3, self.slice_size, self.crop_size, self.crop_size)))
                else:
                    bbox, area = boxes[li]
                    z1, z2 = bbox[0], bbox[3]
                    y1, y2 = max(bbox[1] - 16, 0), min(bbox[4] + 16, 256)
                    x1, x2 = max(bbox[2] - 16, 0), min(bbox[5] + 16, 256)
                    # if z2 - z1 < slice_size:
                    #     z1 = random.randint(max(z2 - slice_size, 0), z1)
                    #     z2 = z1 + slice_size
                    # todo: verify
                    if z2 - z1 < slice_size:
                        diff = (slice_size - z2 + z1) // 2
                        z1 = max(0, z1 - diff)
                        z2 = z1 + slice_size
                    images = image_cube[z1:z2, y1 * 2:y2 * 2, x1 * 2:x2 * 2].copy()
                    masks = mask_cube[z1:z2, y1:y2, x1:x2].copy()
                    slice_size = self.slice_size

                    replay = None
                    image_crops = []
                    mask_crops = []
                    for i in range(images.shape[0]):
                        image = images[i]
                        mask = masks[i]

                        h, w, = mask.shape
                        mask = cv2.resize(mask, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
                        if replay is None:
                            sample = self.transforms(image=image, mask=mask)
                            replay = sample["replay"]
                        else:
                            sample = ReplayCompose.replay(replay, image=image, mask=mask)
                        image_ = sample["image"]
                        image_crops.append(image_)
                        mask_crops.append(sample["mask"])
                    images = np.array(image_crops).astype(np.float32)
                    masks = np.array(mask_crops).astype(np.float32)
                    images = np.expand_dims(images, -1)
                    masks = np.expand_dims(masks, -1)
                    images = (images - image_mean) / image_std

                    images = np.concatenate([images, images, masks], axis=-1)
                    h = images.shape[0]
                    if h > slice_size:
                        images = images[: slice_size]
                        all_images.append(np.moveaxis(images, -1, 0))
                        images = images[-slice_size:]
                        all_images.append(np.moveaxis(images, -1, 0))
                    else:
                        if h != slice_size:
                            tmp = np.zeros((slice_size, *images.shape[1:]))
                            tmp[:h] = images
                            images = tmp
                        all_images.append(np.moveaxis(images, -1, 0))

            sample = {}
            sample['image'] = torch.from_numpy(np.array(all_images)).float()
            sample['label'] = torch.from_numpy(labels).float()
            sample['cube_id'] = cube_id
            return sample

    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return len(self.df)

    def get_weights(self):
        overall = self.df.patient_overall.values
        weights = np.zeros((len(overall),))
        weights[overall == 0] = (len(overall) / (len(overall) - overall.sum()))
        weights[overall == 1] = (len(overall) / overall.sum())
        return weights