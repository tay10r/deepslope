import torch
import torch.nn.functional as F


def gradient_loss(pred, target):
    """
    Computes the gradient loss between the predicted and target images.
    This loss is calculated as the mean absolute difference between the
    gradients of the two images in the x and y directions.

    Args:
        pred (torch.Tensor): Predicted tensor of shape [B, C, H, W]
        target (torch.Tensor): Ground truth tensor of shape [B, C, H, W]

    Returns:
        torch.Tensor: The computed gradient loss.
    """
    # Define simple gradient kernels for x and y directions
    # Note: The kernel values are chosen to compute finite differences.
    grad_kernel_x = torch.tensor(
        [[-1.0, 1.0]], device=pred.device).view(1, 1, 1, 2)
    grad_kernel_y = torch.tensor(
        [[-1.0], [1.0]], device=pred.device).view(1, 1, 2, 1)

    # Compute gradients for prediction (using same padding to keep dimensions)
    grad_pred_x = F.conv2d(pred, grad_kernel_x, padding=(0, 1))
    grad_pred_y = F.conv2d(pred, grad_kernel_y, padding=(1, 0))

    # Compute gradients for target
    grad_target_x = F.conv2d(target, grad_kernel_x, padding=(0, 1))
    grad_target_y = F.conv2d(target, grad_kernel_y, padding=(1, 0))

    # Compute mean absolute difference between gradients
    loss_x = torch.abs(grad_pred_x - grad_target_x).mean()
    loss_y = torch.abs(grad_pred_y - grad_target_y).mean()

    # Total gradient loss is the sum (or average) of x and y losses
    loss = loss_x + loss_y

    return loss
