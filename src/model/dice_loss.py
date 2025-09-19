import torch
import torch.nn.functional as F
class MulticlassDiceLoss:
    def __init__(self, smooth=1):
        self.smooth = smooth
    
    def __call__(self, pred, targets):

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        dice = 0
        if pred.shape != targets.shape:
            targets_onehot = F.one_hot(targets.to(torch.long), num_classes)  # [N, H, W, C]
            targets = targets_onehot.permute(0, 3, 1, 2).float()  # [N, C, H, W]

        for c in range(num_classes):
            pred_class = pred[:, c]
            target_class = targets[:, c]
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
            dice += (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean() / num_classes