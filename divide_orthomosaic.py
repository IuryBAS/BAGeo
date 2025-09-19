import os
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import argparse
import matplotlib.pyplot as plt
import src.common.utils as utils
from src.common.utils import image_to_patches
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="data/raw/")
parser.add_argument("-o", "--output", type=str, default="data/processed/images/")



if __name__ == "__main__":
    args = parser.parse_args()
    utils.makedir(args.output)
    input_path = args.input

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'File {input_path} does not exist')
    
    patch_size = 224
    with rio.open(f'{args.input}') as src:
        img_width = src.width
        img_height = src.height
        n_cols = math.ceil(img_width / patch_size)
        n_rows = math.ceil(img_height / patch_size)
        
        total_patches = n_cols * n_rows
        pbar = tqdm(total=total_patches, desc='Dividing orthomosaic into patches')
        for i in range(n_rows):
            for j in range(n_cols):
                x = j * patch_size
                y = i * patch_size
                w = min(patch_size, img_width - x)
                h = min(patch_size, img_height - y)
                windows = Window(x, y, w, h)

                transform = src.window_transform(windows)
                patch = src.read(window=windows)
                # Add pad to patch if its is in the border of the image
                if (h < patch_size) or (w < patch_size):
                    pad_height = patch_size - h
                    pad_width = patch_size - w
                    patch = np.pad(patch,
                                   ((0, 0), (0, pad_height), (0, pad_width)),
                                   mode='constant', constant_values=0
                                   )
                    h, w = patch_size, patch_size
                
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": h,
                    "width": w,
                    "transform": transform
                })

                out_path = f'{args.output}orthomosaic_{i}_{j}.tif'
                with rio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(patch)
                plt.imsave(f'{args.output}orthomosaic_{i}_{j}.png', patch.transpose(1, 2, 0))
                pbar.update(1)
        pbar.close()
        print(f'Finisehd. Divided orthomosaic into {total_patches} patches')
