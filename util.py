import numpy as np
import torch
from torch._C import dtype
from typing import Dict


DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features

def to_coordinates_and_features_and_time(img, t, k):
    """Converts an image to a set of coordinates (with time step) and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
        t (int): Time step (1 <= t <= k).
        k (int): Total number of time steps.
    """
    # Generate original spatial coordinates
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    
    # Normalize spatial coordinates to [-1, 1]
    coordinates = coordinates / (torch.tensor(img.shape[1:]) - 1).unsqueeze(0)  # Handle division per dimension
    coordinates = coordinates - 0.5
    coordinates *= 2

    # Normalize time step to [-1, 1]
    if k == 1:
        t_normalized = 0.0  # Handle single timestep case
    else:
        t_normalized = ((t - 1) / (k - 1) - 0.5) * 2
    
    # Add time dimension to coordinates
    time_column = torch.full((coordinates.shape[0], 1), t_normalized)
    coordinates = torch.cat([coordinates, time_column], dim=1)

    # Convert image to features
    features = img.reshape(img.shape[0], -1).T
    
    return coordinates, features

def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(img, clamp_image(img_recon))


def mean(list_):
    return np.mean(list_)

def to_patch_coordinates_and_features(img, k):
    """Converts an image to a set of patch coordinates and features.
    
    Args:
        img (torch.Tensor): Shape (channels, height, width).
        k (int): Size of the square patch (k x k).
    
    Returns:
        coordinates (torch.Tensor): Shape (num_patches, 2) with coordinates in [-1, 1].
        features (torch.Tensor): Shape (num_patches, channels * k * k) with patch features.
    """
    C, H, W = img.shape
    H_patch = H - k + 1
    W_patch = W - k + 1
    
    # Generate top-left coordinates of all patches
    y_indices = torch.arange(H_patch)
    x_indices = torch.arange(W_patch)
    
    # Compute center coordinates of each patch
    y_centers = y_indices.float() + (k - 1) / 2.0
    x_centers = x_indices.float() + (k - 1) / 2.0
    
    # Create grid of center coordinates
    grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing='ij')
    coordinates = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
    
    # Normalize coordinates to [-1, 1] as in the original function
    coordinates[:, 0] = (coordinates[:, 0] / (H - 1) - 0.5) * 2  # Normalize y
    coordinates[:, 1] = (coordinates[:, 1] / (W - 1) - 0.5) * 2  # Normalize x
    
    # Extract k x k patches from the image
    patches = img.unfold(1, k, 1).unfold(2, k, 1)  # Shape: (C, H_patch, W_patch, k, k)
    
    # Permute and reshape to (num_patches, C*k*k)
    features = patches.permute(1, 2, 0, 3, 4).reshape(-1, C * k * k)
    
    return coordinates, features

def reconstruct_from_patches(features, original_shape, k):
    """Reconstruct an image from patch features.
    
    Args:
        features (torch.Tensor): Shape (num_patches, channels * k * k).
        original_shape (tuple): (channels, height, width) of the target image.
        k (int): Patch size (k x k).
    
    Returns:
        img_recon (torch.Tensor): Reconstructed image of shape `original_shape`.
    """
    C, H, W = original_shape
    H_patch = H - k + 1
    W_patch = W - k + 1

    # Reshape features to (H_patch, W_patch, C, k, k)
    patches = features.reshape(H_patch, W_patch, C, k, k)
    
    # Fold patches back into the image
    # Step 1: Permute to (C, k, k, H_patch, W_patch) for compatibility with fold
    patches = patches.permute(2, 3, 4, 0, 1)  # (C, k, k, H_patch, W_patch)
    
    # Step 2: Fold patches into the image shape (C, H, W)
    img_recon = torch.nn.functional.fold(
        patches.reshape(1, C * k * k, -1),  # (batch=1, C*k*k, num_patches)
        output_size=(H, W),
        kernel_size=k,
        stride=1  # Same stride used in patch extraction
    ).squeeze(0)  # Remove batch dimension
    
    # Step 3: Normalize overlapping regions
    # Create a mask to count overlaps (for averaging)
    ones = torch.ones_like(patches)
    norm_mask = torch.nn.functional.fold(
        ones.reshape(1, C * k * k, -1),
        output_size=(H, W),
        kernel_size=k,
        stride=1
    ).squeeze(0)
    
    img_recon = img_recon / norm_mask  # Average overlapping contributions
    
    return img_recon