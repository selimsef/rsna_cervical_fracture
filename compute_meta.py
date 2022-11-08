import argparse
import json
import os


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from functools import partial
from multiprocessing import Pool
from skimage import measure

import numpy as np
import tifffile
from tqdm import tqdm



def collect_slices_write_tif(cube_id: str, root_dir: str):
    mask_dir = os.path.join(root_dir, "seg_preds")
    os.makedirs(mask_dir, exist_ok=True)
    tiff_path = os.path.join(mask_dir, f"{cube_id}.tif")

    mask = tifffile.imread(tiff_path)
    image = tifffile.imread(os.path.join(root_dir, "resized_image_cubes", f"{cube_id}.tif"), )
    metadata = {}
    sums = [0]
    for li in range(1, 8):
        sums.append(int((mask == li).sum()))
    boxes = {}
    for rprop in measure.regionprops(mask):
        boxes[int(rprop.label)] = rprop.bbox, int(rprop.area)
    metadata["sums"] = sums
    metadata["boxes"] = boxes
    metadata["image_mean"] = float(image.mean())
    metadata["image_std"] = float(image.std())
    return {cube_id: metadata}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocessing")
    arg = parser.add_argument
    arg('--root_dir', type=str, default="/home/selim/datasets/rsna/")
    args = parser.parse_args()

    seg_dir = os.path.join(args.root_dir, "seg_preds")

    scan_ids = [os.path.splitext(f)[0] for f in os.listdir(seg_dir)]

    all_meta = {}
    with Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=len(scan_ids), desc="Process images/mask cubes and save meta") as pbar:
            for _, cube_meta in enumerate(pool.imap_unordered(partial(collect_slices_write_tif, root_dir=args.root_dir), scan_ids)):
                all_meta.update(cube_meta)
                pbar.update()
    with open("meta.json", "w") as f:
        json.dump(all_meta, f)