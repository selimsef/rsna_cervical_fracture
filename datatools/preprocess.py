import argparse
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from functools import partial
from multiprocessing import Pool

import tifffile
from tqdm import tqdm

from datatools.converter import combine_scan, convert_nifti_to_plsmask


def collect_slices_write_tif(cube_id: str, root_dir: str):

    image_dir = os.path.join(root_dir, "resized_image_cubes")
    mask_dir = os.path.join(root_dir, "resized_mask_cubes")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    tiff_path = os.path.join(image_dir, f"{cube_id}.tif")

    img_cube = combine_scan(os.path.join(root_dir, "train_images", cube_id))
    tifffile.imwrite(tiff_path, img_cube)
    nifti_path = os.path.join(root_dir, "segmentations", f"{cube_id}.nii")
    if os.path.exists(nifti_path):
        mask_cube = convert_nifti_to_plsmask(nifti_path)
        tifffile.imwrite(os.path.join(mask_dir, f"{cube_id}.tif"), mask_cube)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocessing")
    arg = parser.add_argument
    arg('--root_dir', type=str, default="/media/selim/d860719e-95b8-4a1a-b518-11791115e55b/rsna/")
    args = parser.parse_args()

    train_dir = os.path.join(args.root_dir, "train_images")
    seg_dir = os.path.join(args.root_dir, "segmentations")
    scan_ids = os.listdir(train_dir)


    with Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=len(scan_ids), desc="Process dicom/nifti and write tif images/mask cubes") as pbar:
            for _ in enumerate(pool.imap_unordered(partial(collect_slices_write_tif, root_dir=args.root_dir), scan_ids)):
                pbar.update()
