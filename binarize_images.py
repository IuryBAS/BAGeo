import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from glob import glob
from src.common import utils
import PIL.Image as Image
import rasterio as rio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="data/processed/images")
parser.add_argument("-o", "--output", type=str, default="data/processed/segmented")

def calculate_exg(img_array, threshold_method='mean'):
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    exg = 2 * g - r - b
    if threshold_method == 'mean': 
        threshold_for_binarization = ski.filters.threshold_mean(exg)
    else:
        threshold_for_binarization = ski.filters.threshold_local(exg, block_size=223, offset=0)
    exg = np.where(exg > threshold_for_binarization, 0, 1)
    return exg.astype(np.uint8) #* 255

def calculate_gli(img_array):
    """
    Calculate the Green Leaf Index (GLI) from an image array.

    Params
    ----------
    img_array :An image array with shape (height, width, channels) and type uint8.

    Returns
    -------
    gli_norm : A normalized image array with shape (height, width) and type uint8, where
        pixels with a value of 1 indicate vegetation and pixels with a value of 0 indicate
        no vegetation.
    """
    img_array = img_array.astype(np.float32)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    numerator = 2 * g - r - b
    denominator = (2 * g + r + b)
    # Using divide to avoid zero division
    gli = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    gli = np.where(gli > 0, 1, 0)
    gli_norm = gli.astype(np.uint8)
    return gli_norm

if __name__ == "__main__":
    args = parser.parse_args()
    args = parser.parse_args()
    input_path = args.input

    utils.check_dir_exists(input_path)
    patched_images = glob(f'{input_path}/*.tif')

    try:
        if patched_images == []:
            raise FileNotFoundError(f'There is no images in input directory {input_path}')
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    output_path = args.output
    utils.makedir(output_path)
    pbar = tqdm(total=len(patched_images), desc='Binarizing images')
    for patched_img in patched_images:
        img_name = patched_img.split('/')[-1]
        with rio.open(patched_img) as src:
            img = src.read()
            img_rgb_reshaped = np.transpose(img[:3, :, :], (1, 2, 0)) # (H, W, C)
            binary_mask = calculate_gli(img_rgb_reshaped)#.astype(np.uint8)
            out_meta = src.meta.copy()
            out_meta.update({
                'count': 1,
                'dtype': np.uint8
            })
            out_tif = f'{output_path}/{img_name[:-4]}_bin.tif'
            with rio.open(out_tif, 'w', **out_meta) as dst:
                dst.write(binary_mask, 1)
            out_put = f'{output_path}/{img_name[:-4]}_bin.png'
            # Multiply by 255 to visualize. This PNG is not used for training
            Image.fromarray(binary_mask*255).save(out_put)
            pbar.update(1)

    pbar.close()
    print(f'Finished binarizing {len(patched_images)} images')