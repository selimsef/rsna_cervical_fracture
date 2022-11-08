from abc import ABC, abstractmethod
from typing import Dict

import albumentations as A
import cv2


class FractureAugmentations(ABC):
    @abstractmethod
    def get_train_augmentations(self, config: Dict) -> A.Compose:
        pass

    @abstractmethod
    def get_val_augmentations(self, config: Dict) -> A.Compose:
        pass


class SameResAugsAlbu(FractureAugmentations):

    def get_train_augmentations(self, config: Dict) -> A.Compose:
        return A.ReplayCompose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT,
                               p=0.5),
            A.HorizontalFlip(),
            A.OneOf([A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                     A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT)], p=0.2),

        ])

    def get_val_augmentations(self, config: Dict) -> A.Compose:
        return A.ReplayCompose([

        ])


class CropAugs(FractureAugmentations):

    def get_train_augmentations(self, config: Dict) -> A.Compose:
        crop_size = config.get("crop_size", 160)
        return A.ReplayCompose([
            A.LongestMaxSize(crop_size),
            A.PadIfNeeded(crop_size, crop_size, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                               p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.OneOf([A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                     A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT)], p=0.2),

        ])

    def get_val_augmentations(self, config: Dict) -> A.Compose:
        crop_size = config.get("crop_size", 160)
        return A.ReplayCompose([
            A.LongestMaxSize(crop_size),
            A.PadIfNeeded(crop_size, crop_size, border_mode=cv2.BORDER_CONSTANT),
        ])


class AsIs(FractureAugmentations):

    def get_train_augmentations(self, config: Dict) -> A.Compose:
        return A.ReplayCompose([

        ])

    def get_val_augmentations(self, config: Dict) -> A.Compose:
        return A.ReplayCompose([

        ])


class CropAugsFullRes(FractureAugmentations):

    def get_train_augmentations(self, config: Dict) -> A.Compose:
        crop_size = config.get("crop_size", 256)
        return A.ReplayCompose([
            A.LongestMaxSize(crop_size),
            A.PadIfNeeded(crop_size, crop_size, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                               p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ])

    def get_val_augmentations(self, config: Dict) -> A.Compose:
        crop_size = config.get("crop_size", 256)
        return A.ReplayCompose([
            A.LongestMaxSize(crop_size),
            A.PadIfNeeded(crop_size, crop_size, border_mode=cv2.BORDER_CONSTANT),
        ])
