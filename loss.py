# loss.py

import torch

def dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Dice coefficient between prediction and target tensors.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (N, ...).
        target (torch.Tensor): Ground truth tensor of the same shape as pred.

    Returns:
        torch.Tensor: Dice coefficient averaged over the batch.
    """
    smooth = 1.0
    num = pred.size(0)

    # Flatten each sample in the batch
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)

    intersection = (m1 * m2).sum(1)
    denominator = (m1 * m1).sum(1) + (m2 * m2).sum(1)

    dice_score = (2. * intersection + smooth) / (denominator + smooth)
    return dice_score.mean()


def my_loss(pred_y: torch.Tensor, true_y: torch.Tensor) -> torch.Tensor:
    """
    Dice loss based on the Dice coefficient.

    Args:
        pred_y (torch.Tensor): Predicted tensor of shape (N, ...).
        true_y (torch.Tensor): Ground truth tensor of the same shape.

    Returns:
        torch.Tensor: Dice loss value.
    """
    return 1.0 - dice(pred_y, true_y)
