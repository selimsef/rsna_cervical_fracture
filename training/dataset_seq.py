import os.path
import random

import albumentations as A
import numpy as np
import pandas as pd
import tifffile
import torch
from albumentations import ReplayCompose
from torch.utils.data import Dataset


class DatasetSeq(Dataset):
    def __init__(
            self,
            mode: str,
            dataset_dir: str,
            fold: int,
            transforms: A.Compose,
            slice_size: int,
            multiplier: int = 1,
            folds_csv="folds.csv",
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

        if self.mode == "train":
            self.df = pd.concat([self.df] * multiplier)
        if self.is_train:
            assert len(self.df[self.df.fold == fold]) == 0
        else:
            assert len(self.df[self.df.fold != fold]) == 0

        self.cache = {}

    def __getitem__(self, i):
        return self.getitem(i)

    def getitem(self, i):
        row = self.df.iloc[i]
        cube_id = row.cube_id

        mask_cube = tifffile.memmap(os.path.join(self.dataset_dir, f"resized_mask_cubes", f"{cube_id}.tif"), mode="r")
        #mask_cube = tifffile.memmap(os.path.join(self.dataset_dir, f"seg_preds", f"{cube_id}.tif"), mode="r")
        image_cube = tifffile.memmap(os.path.join(self.dataset_dir, f"resized_image_cubes", f"{cube_id}.tif"), mode="r")
        # todo: make configurable
        mask_cube = mask_cube[::2]
        image_cube = image_cube[::2]

        # mask_cube = np.moveaxis(mask_cube, -1, 0)
        # image_cube = np.moveaxis(image_cube, -1, 0)
        if cube_id not in self.cache:
            image_mean = image_cube.mean()
            image_std = image_cube.std()
            image_max = image_cube.max()
            sums = [0]
            for li in range(1, 8):
                sums.append((mask_cube == li).sum())

        else:
            image_mean, image_std, image_max, sums = self.cache[cube_id]
        self.cache[cube_id] = (image_mean, image_std, image_max, sums)


        slice_size = self.slice_size
        cube_num_slices = mask_cube.shape[0]
        start = 0

        if self.is_train:
            if slice_size < cube_num_slices:
                if image_cube.shape[0] <= slice_size:
                    start = 0
                else:
                    start = random.randint(0, image_cube.shape[0] - slice_size - 1)
                images = image_cube[start:start + slice_size]
                masks = mask_cube[start:start + slice_size]
            else:
                images = np.zeros(shape=(slice_size, *image_cube.shape[1:]), dtype=image_cube.dtype)
                masks = np.zeros(shape=(slice_size, *mask_cube.shape[1:]), dtype=mask_cube.dtype)
                images[:image_cube.shape[0]] = image_cube
                masks[:cube_num_slices] = mask_cube
            images = images.copy()
            masks = masks.copy()
        else:
            images = image_cube
            masks = mask_cube

        replay = None
        image_crops = []
        mask_crops = []
        for i in range(images.shape[0]):
            image = images[i]
            mask = masks[i]
            if replay is None:
                sample = self.transforms(image=image, mask=mask)
                replay = sample["replay"]
            else:
                sample = ReplayCompose.replay(replay, image=image, mask=mask)
            image_ = sample["image"]
            image_crops.append(image_)
            mask_crops.append(sample["mask"])
        images = np.array(image_crops)
        masks = np.array(mask_crops)
        h = masks.shape[0]
        if h % 16 > 0:
            tmp = np.zeros(((h//16 + 1) * 16, 256, 256))
            tmp[:h] = images
            images = tmp

            tmp = np.zeros(((h//16 + 1) * 16, 256, 256))
            tmp[:h] = masks
            masks = tmp

        images = (images - image_mean) / image_std
        images = np.expand_dims(images, 0)
        sample = {}
        masks[masks > 7] = 0
        sample['mask'] = torch.from_numpy(masks).long()
        sample['image'] = torch.from_numpy(images).float()
        sample['cube_id'] = cube_id
        return sample

    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return len(self.df)


class DatasetSeqStrideVal(Dataset):

    def __init__(
            self,
            dataset_dir: str,
            transforms: A.Compose,
            folds_csv="folds.csv",
    ):
        df = pd.read_csv(os.path.join(folds_csv))
        self.df = df
        self.dataset_dir = dataset_dir
        self.transforms = transforms

    def __getitem__(self, i):
        return self.getitem(i)

    def getitem(self, i):
        row = self.df.iloc[i]
        cube_id = row.cube_id
        image_cube = tifffile.memmap(os.path.join(self.dataset_dir, f"resized_image_cubes", f"{cube_id}.tif"), mode="r")
        image_cube = image_cube[::2]
        image_mean = image_cube.mean()
        image_std = image_cube.std()
        h = image_cube.shape[0]

        images = image_cube
        replay = None
        image_crops = []
        for i in range(images.shape[0]):
            image = images[i]
            if replay is None:
                sample = self.transforms(image=image)
                replay = sample["replay"]
            else:
                sample = ReplayCompose.replay(replay, image=image)
            image_ = sample["image"]
            image_crops.append(image_)
        images = np.array(image_crops)
        if h % 32 > 0:
            tmp = np.zeros(((h//32 + 1) * 32, 256, 256))
            tmp[:h] = images
            images = tmp
        images = (images - image_mean) / image_std
        images = np.expand_dims(images, 0)
        sample = {}
        sample['image'] = torch.from_numpy(images).float()
        sample['cube_id'] = cube_id
        sample['h'] = h
        return sample

    def __len__(self):
        return len(self.df)



class DatasetSeqDamage(Dataset):
    def __init__(
            self,
            mode: str,
            dataset_dir: str,
            fold: int,
            transforms: A.Compose,
            slice_size: int,
            multiplier: int = 1,
            folds_csv="folds.csv",
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

        if self.mode == "train":
            self.df = pd.concat([self.df] * multiplier)
        if self.is_train:
            assert len(self.df[self.df.fold == fold]) == 0
        else:
            assert len(self.df[self.df.fold != fold]) == 0

        self.cache = {}

    def __getitem__(self, i):
        return self.getitem(i)

    def getitem(self, i):
        row = self.df.iloc[i]
        cube_id = row.cube_id

        mask_cube = tifffile.memmap(os.path.join(self.dataset_dir, f"seg_preds", f"{cube_id}.tif"), mode="r")
        image_cube = tifffile.memmap(os.path.join(self.dataset_dir, f"resized_image_cubes", f"{cube_id}.tif"), mode="r")
        # todo: make configurable
        image_cube = image_cube[::2]

        if cube_id not in self.cache:
            image_mean = image_cube.mean()
            image_std = image_cube.std()
            image_max = image_cube.max()
            sums = [0]
            for li in range(1, 8):
                sums.append((mask_cube == li).sum())

        else:
            image_mean, image_std, image_max, sums = self.cache[cube_id]
        self.cache[cube_id] = (image_mean, image_std, image_max, sums)


        slice_size = self.slice_size
        cube_num_slices = mask_cube.shape[0]
        start = 0

        if self.is_train:
            if slice_size < cube_num_slices:
                if image_cube.shape[0] <= slice_size:
                    start = 0
                else:
                    start = random.randint(0, image_cube.shape[0] - slice_size - 1)
                images = image_cube[start:start + slice_size]
                masks = mask_cube[start:start + slice_size]
            else:
                images = np.zeros(shape=(slice_size, *image_cube.shape[1:]), dtype=image_cube.dtype)
                masks = np.zeros(shape=(slice_size, *mask_cube.shape[1:]), dtype=mask_cube.dtype)
                images[:image_cube.shape[0]] = image_cube
                masks[:cube_num_slices] = mask_cube
            images = images.copy()
            masks = masks.copy()
        else:
            images = image_cube
            masks = mask_cube

        replay = None
        image_crops = []
        mask_crops = []
        for i in range(images.shape[0]):
            image = images[i]
            mask = masks[i]
            if replay is None:
                sample = self.transforms(image=image, mask=mask)
                replay = sample["replay"]
            else:
                sample = ReplayCompose.replay(replay, image=image, mask=mask)
            image_ = sample["image"]
            image_crops.append(image_)
            mask_crops.append(sample["mask"])
        images = np.array(image_crops)
        masks = np.array(mask_crops)

        labels = np.zeros((8,))
        for li in range(1, 8):
            if row[f"C{li}"] > 0 and (masks == li).sum()/(sums[li] + 32) > 0.3:
                labels[li] = 1
                labels[0] = 1

        h = masks.shape[0]
        if h % 16 > 0:
            tmp = np.zeros(((h//16 + 1) * 16, 256, 256))
            tmp[:h] = images
            images = tmp

            tmp = np.zeros(((h//16 + 1) * 16, 256, 256))
            tmp[:h] = masks
            masks = tmp

        images = (images - image_mean) / image_std
        images = np.expand_dims(images, 0)
        sample = {}
        masks[masks > 7] = 0
        sample['mask'] = torch.from_numpy(masks).long()
        sample['image'] = torch.from_numpy(images).float()
        sample['label'] = torch.from_numpy(labels).float()
        sample['cube_id'] = cube_id
        return sample

    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return len(self.df)