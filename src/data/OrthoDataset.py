import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, PILToTensor
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import rasterio as rio


class OrthoDataset(Dataset):
    def __init__(self, rgb_dataset, input_path, gt_path, transform=None):
        self.images = rgb_dataset
        self.masks = [img.replace(input_path, gt_path).replace('.tif', '_bin.tif') for img in self.images]
        self.transform = transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with rio.open(self.images[index]) as src:
            image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0))
        
        with rio.open(self.masks[index]) as src:
            mask = src.read(1)

        
        image = to_pil_image(image.astype(np.uint8)) #Image.open(self.images[index]).convert('RGB')
        mask = Image.fromarray(mask.astype(np.uint8), mode='L')

        if self.transform:
            image, mask = self.transform(image, mask)
            
        image = ToTensor()(image)
        image = self.normalize(image)
        mask = PILToTensor()(mask)
        mask = mask.squeeze(0).long()
        return image, mask