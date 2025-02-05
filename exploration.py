import matplotlib.pyplot as plt
import numpy as np
import cifar10loader
import matplotlib
import util
import torch

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
train_loader,test_loader,train_dataset,test_dataset = cifar10loader.loadcifar10()
img = cifar10loader.loadImageI(1,test_dataset)  
co,f = util.to_coordinates_and_features(img)
print(f"coordinates {co.shape}\n")
print(f"features {f.shape}\n")

pco,pf = to_patch_coordinates_and_features(img , 3)
print(f"pcoordinates {pco.shape}\n")
print(f"pfeatures {pf.shape}\n")