import sys
import numpy as np
import pandas as pd
import torch
import src.common.utils as utils
from src.common.utils import reset_metrics, update_metrics, compute_metrics, log_metrics
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
from src.data.OrthoDataset import OrthoDataset
import matplotlib.pyplot as plt
from src.model.early_stopping import EarlyStopping
from src.model.dice_loss import MulticlassDiceLoss
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex
from torchmetrics.segmentation import DiceScore
from torch.utils.tensorboard import SummaryWriter
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--rgb", type=str, default="data/processed/images/")
parser.add_argument("-m", "--groundtruth", type=str, default="data/processed/segmented/")
parser.add_argument("-o", "--modelpath", type=str, default="models/")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, optimizer, criterion, epoch, metrics, device):
    
    """
    Train the model for one epoch.

    Parameters:
    model: model to be trained
    train_loader: data loader for the training set
    optimizer: optimizer to update the model parameters
    criterion: loss function to be used
    epoch: current epoch
    metrics: dictionary to store the metrics for the current epoch
    device: device to be used for training

    Returns:
    tuple: (average loss for the current epoch, metrics for the current epoch)
    """
    model.train()
    total_loss = 0
    with torch.enable_grad():
        for batch_idx, (image, mask) in enumerate(train_loader):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(image)['out']
            loss = criterion(output, mask)
            total_loss += loss
            loss.backward()
            optimizer.step()
            # Method to update the values for all metrics used
            update_metrics(output.argmax(dim=1), mask, metrics, 'train')
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        # Dict with the metrics computed for the current epoch
        metrics_dict = compute_metrics(metrics, 'train')
        reset_metrics(metrics)
        return total_loss / len(train_loader), metrics_dict
            

                    
def eval(model, val_loader, criterion, epoch, metrics,  device):
    """
    Evaluate the model on the validation set for one epoch.

    Parameters:
    model: model to be evaluated
    val_loader: data loader for the validation set
    criterion: loss function to be used
    epoch: current epoch
    metrics: dictionary to store the metrics for the current epoch
    device: device to be used for evaluation

    Returns:
    tuple: (average loss for the current epoch, metrics for the current epoch)
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (image, mask) in enumerate(val_loader):
            image, mask = image.to(device), mask.to(device)
            output = model(image)['out']
            if batch_idx == 0:
                preds = torch.argmax(output, dim=1)
                img = image[0].cpu().numpy().transpose((1, 2, 0))
                pred_mask = preds.cpu().numpy().transpose((1, 2, 0))
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                img_norm = utils.normalize_rgb_image(img)
                ax.imshow(img_norm, vmin=0, vmax=255)
                ax.imshow(pred_mask, alpha=0.4, cmap='gray')
                ax.axis('off')
                plt.savefig(f'reports/val_example_mask/epoch_{epoch}.png')
            loss = criterion(output, mask)
            total_loss += loss
            # Method to update the values for all metrics used
            update_metrics(output.argmax(dim=1), mask, metrics, 'val')
            if batch_idx % 10 == 0:
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(val_loader.dataset),
                    100. * batch_idx / len(val_loader), loss.item()))
        # Dict with the metrics computed for the current epoch
        metrics_dict = compute_metrics(metrics, 'val')
        reset_metrics(metrics)
        return total_loss / len(val_loader), metrics_dict


if __name__ == '__main__':
    # Tensorboard to log the metrics and loss
    date_time = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
    writer = SummaryWriter(log_dir=f"runs/geoproc_{date_time}")
    # Set all the seed and torch deterministic mode
    utils.set_seed(42)
    args = parser.parse_args()
    input_path = args.rgb
    gt_path = args.groundtruth
    model_path = args.modelpath
    
    utils.check_dir_exists(input_path)
    utils.check_dir_exists(gt_path)
    utils.makedir('/'.join(model_path.split('/')[:-1]))
    # Get all the images patches and groundtruth masks
    patched_images = glob(f'{input_path}/*.tif')
    gt_masks = glob(f'{gt_path}/*.tif')

    # Check if there are any images
    try:
        if patched_images == []:
            raise FileNotFoundError(f'There is no images in input directory {input_path}')
        if gt_masks == []:
            raise FileNotFoundError(f'There is no images masks in input directory {gt_path}')
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    # Check if for each image there is a ground truth mask
    for img in patched_images:
        img_name = img.split('/')[-1]
        gt_search_name = f'{gt_path}/{img_name[:-4]}_bin.tif'
        if gt_search_name not in gt_masks:
            raise FileNotFoundError(f'There is no mask for image {img_name}')
    
    # Split the dataset into training and validation
    train_split = 0.8
    val_split = abs(1 - train_split)
    assert train_split + val_split == 1, 'Train and val split must sum to 1'
    train_size = int(len(patched_images) * train_split)
    val_size = len(patched_images) - train_size
    print(f'Train size: {train_size}, Val size: {val_size}')
    assert train_size + val_size == len(patched_images), 'Train and val sets contains a different number of images than the total number of images'
    
    train_dataset = np.random.choice(patched_images, train_size, replace=False)
    val_dataset = np.asarray(list(set(patched_images).difference(set(train_dataset))))
    assert set(train_dataset).isdisjoint(set(val_dataset)), 'Train and val sets contains the same images'
    

    # Defining the transforms for the training dataset with random augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        
    ])

    # Setting the OrthoDataset and loaders
    batch_size = 12
    train_d = OrthoDataset(train_dataset, input_path, gt_path, transform=train_transform)
    val_d = OrthoDataset(val_dataset, input_path, gt_path, transform=None)
    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_d, batch_size=1, shuffle=False)


    # Defining the model. Using the pretrained DeepLabV3 model
    # Removing the last layer and adding a new one with the number of classes
    pretrain_weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model_deeplabv3 = models.segmentation.deeplabv3_resnet50(weights=pretrain_weights, progress=True)
    num_classes = 2 # 0: background, 1: foreground
    model_deeplabv3.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model_deeplabv3.to(device)
    model_deeplabv3 = nn.DataParallel(model_deeplabv3, device_ids=[0])
    
    # Setting the optimizer, criterion loss and sheduler
    optimizer = Adam(model_deeplabv3.parameters(), lr=1e-4)
    # MulticlassDiceLoss give a better performance
    #criterion = nn.CrossEntropyLoss()
    criterion = MulticlassDiceLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    early_stopping = EarlyStopping(patience=10, delta=0.01)
    
    # Defining all the metrics as a list to be computed in training and validation
    metrics = [MulticlassF1Score(num_classes=num_classes, average='macro').to(device), 
               MulticlassPrecision(num_classes=num_classes, average='macro').to(device), 
               MulticlassRecall(num_classes=num_classes, average='macro').to(device),
               MulticlassJaccardIndex(num_classes=num_classes, average='macro').to(device),
               DiceScore(num_classes=num_classes, average='macro').to(device)]

    # Training and evaluating the model on the validation set
    epochs = 20
    df_train_metrics = pd.DataFrame()
    df_val_metrics = pd.DataFrame()
    for epoch in range(epochs):
        train_loss, train_metrics = train(model_deeplabv3, train_loader, optimizer, criterion, epoch, metrics, device)
        #scheduler.step()
        val_loss, val_metrics = eval(model_deeplabv3, val_loader, criterion, epoch, metrics,  device)
        train_metrics.update({'epoch': epoch,
                              'train_loss': train_loss.item()})
        val_metrics.update({'epoch': epoch,
                            'val_loss': val_loss.item()})
        
        # Logging metrics to Tensorboard
        log_metrics(writer, train_metrics, epoch, prefix='train/')
        log_metrics(writer, val_metrics, epoch, prefix='val/')
        df_train_metrics = pd.concat([df_train_metrics, pd.DataFrame(train_metrics, index=[epoch])])
        df_val_metrics = pd.concat([df_val_metrics, pd.DataFrame(val_metrics, index=[epoch])])
        early_stopping(val_loss, model_deeplabv3)
        if early_stopping.early_stop:
            print("Train finished due to early stopping")
            break

        print(f'Epoch: {epoch + 1}, Train Metrics: {train_metrics},\n Val Metrics: {val_metrics}')
    
    print('Logging metrics')
    writer.flush()
    writer.close()
    df_train_metrics.to_csv(f'reports/train_metrics.csv')
    df_val_metrics.to_csv(f'reports/val_metrics.csv')
    # Saving the best model
    early_stopping.load_best_model(model_deeplabv3)
    print(f'Best loss: {early_stopping.min_val:.4f}')
    print('Training finished')
    print(f'Saving model in {model_path}')
    torch.save(model_deeplabv3.state_dict(), f'{model_path}')
    print(f'Model saved in {model_path}')