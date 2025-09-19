import math
import os
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import torch
import torchvision
import torchvision.transforms.functional as TF
import argparse
from tqdm import tqdm
from rasterio.features import shapes
from src.common.utils import makedir


parser = argparse.ArgumentParser()
parser.add_argument('--rgb', required=True)
parser.add_argument('--modelpath', required=True)
parser.add_argument('--output', required=True, help='Output folder (for mask.tif and polygons.geojson)')
parser.add_argument('--patch_size', type=int, default=224)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

device = torch.device(args.device)
patch_size = args.patch_size

def divide_orthomosaic(image_path, patch_size=224):
    """
    Divide an orthomosaic image into patches of size patch_size x patch_size.

    Args:
        image_path: Path to the orthomosaic image.
        patch_size: Size of each patch. Defaults to 224.

    Returns:
        tuple: A tuple containing a list of patches, the full image transform, the image CRS, the image width, and the image height.
    """
    patches = []
    with rio.open(image_path) as src:
        img_width = src.width
        img_height = src.height
        transform_full = src.transform
        crs = src.crs
        n_cols = math.ceil(img_width / patch_size)
        n_rows = math.ceil(img_height / patch_size)

        for i in range(n_rows):
            for j in range(n_cols):
                x = j * patch_size
                y = i * patch_size
                w = min(patch_size, img_width - x)
                h = min(patch_size, img_height - y)
                win = Window(x, y, w, h)
                transform = src.window_transform(win)
                patch = src.read(window=win)  # shape (bands, h, w)

                padded = False
                if (h < patch_size) or (w < patch_size):
                    pad_height = patch_size - h
                    pad_width = patch_size - w
                    patch = np.pad(patch,
                                   ((0, 0), (0, pad_height), (0, pad_width)),
                                   mode='constant', constant_values=0
                                   )
                    padded = True

                patches.append({
                    'patch': patch,
                    'window': win,
                    'x': x, 'y': y,
                    'w': w, 'h': h,
                    'transform': transform,
                    'padded': padded
                })
    return patches, transform_full, crs, img_width, img_height

if __name__ == '__main__':
    image_path = args.rgb
    model_path = args.modelpath
    output_path = args.output

    if not os.path.exists(image_path):
        raise FileNotFoundError(f'File {image_path} does not exist')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'File {model_path} does not exist')
    
    # Create output mask folder if it does not exist
    makedir('/'.join(output_path.split('/')[:-1]))
    
    # Load model
    pretrain_weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=pretrain_weights, progress=True)
    # Replace last layer to replicate the training model
    num_classes = 2
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.to(device)
    model.eval()

    state_dict = torch.load(model_path, map_location=device)
    # Remove 'module.' in case that the model was saved with DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # Set the normalization 
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    # Divide image into patches to inference
    patches, transform_full, crs, img_w, img_h = divide_orthomosaic(image_path, patch_size=patch_size)

    # Prepare full mask array
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for idx, item in enumerate(tqdm(patches, desc='Patches')):
        patch = item['patch']
        # keep first 3 channels if more bands exist
        if patch.shape[0] >= 3:
            img_patch = patch[:3, :, :].astype(np.float32) / 255.0
        #else:
            # replicate channel if single band
        #    img_patch = np.repeat(patch[0:1, :, :], 3, axis=0).astype(np.float32) / 255.0

        # to tensor shape (B, C, H, W)
        img_patch = torch.from_numpy(img_patch).to(device)
        img_patch = img_patch.unsqueeze(0)

        img_patch = normalize(img_patch[0]).unsqueeze(0)  # normalize expects channels in same order

        with torch.no_grad():
            out = model(img_patch)['out']  # (1, num_classes, H, W)
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W)

        # Crop padding if necessary
        h_orig = item['h']
        w_orig = item['w']
        pred_cropped = pred[:h_orig, :w_orig]

        # Place into full mask at (y:y+h, x:x+w)
        y = int(item['y'])
        x = int(item['x'])
        full_mask[y:y + h_orig, x:x + w_orig] = pred_cropped

    # Save stitched mask GeoTIFF
    out_meta = {
        'driver': 'GTiff',
        'height': img_h,
        'width': img_w,
        'count': 1,
        'dtype': 'uint8',
        'crs': crs,
        'transform': transform_full
    }
    mask_tif = f'{output_path}'
    with rio.open(mask_tif, 'w', **out_meta) as dst:
        dst.write(full_mask, 1)

    print(f'Mask saved to {mask_tif}')
    print(f'Finished inference')
