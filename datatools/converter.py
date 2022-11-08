import os

import cv2
import nibabel
import numpy as np
import pydicom
from pydicom.pixel_data_handlers import apply_voi_lut


def combine_scan(scan_dir: str, voi_lut: bool = False, fix_monochrome: bool = True) -> np.ndarray:
    num_files = len(os.listdir(scan_dir))
    images = []
    offset = 0
    first = None
    last = None
    for i in range(num_files):
        dpath = os.path.join(scan_dir, f"{i + offset}.dcm")
        if i == 0:
            while not os.path.exists(dpath):
                offset += 1
                dpath = os.path.join(scan_dir, f"{i + offset}.dcm")

        ds = pydicom.dcmread(dpath)
        if not first:
            first = ds
        last = ds
        if voi_lut:
            data = apply_voi_lut(ds.pixel_array, ds)
        else:
            data = ds.pixel_array
        data = cv2.resize(data, (256, 256))
        # depending on this value, X-ray may look inverted - fix that:
        if fix_monochrome and ds.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        images.append(data)
    if last.ImagePositionPatient[2] >  first.ImagePositionPatient[2]:
        images = images[::-1]
    return np.array(images)


def convert_nifti_to_plsmask(nifti_path: str) -> np.ndarray:
    mask = np.array(nibabel.load(nifti_path).dataobj)
    mask = np.flip(mask, axis=-1)
    mask = np.moveaxis(mask, 0, 1)
    mask = np.flip(mask, axis=0)
    mask = np.moveaxis(mask, -1, 0).astype(np.uint8)
    masks = []
    for img in mask:
        masks.append(cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST))
    return np.asarray(masks)

