"""Centre crops of Kodak images, no resampling, so every pixel keeps its detail."""
import imageio.v2 as imageio
import torch
from torchvision import transforms

import util


def load_kodak(image_id, root="kodak-dataset"):
    img = imageio.imread(f"{root}/kodim{str(image_id).zfill(2)}.png")
    return transforms.ToTensor()(img).float()  # (3, H, W) in [0, 1]


def centre_crop(img, scale):
    """Crop to (H / scale, W / scale) around the image centre. scale 1 = full image."""
    if scale == 1:
        return img
    _, h, w = img.shape
    ch, cw = h // scale, w // scale
    top, left = (h - ch) // 2, (w - cw) // 2
    return img[:, top:top + ch, left:left + cw].contiguous()


def load_crop(image_id, scale, device, dtype=torch.float32):
    """Returns (img, coordinates, features) on `device`, using COIN's own coordinate convention."""
    img = centre_crop(load_kodak(image_id), scale).to(device, dtype)
    coordinates, features = util.to_coordinates_and_features(img)
    return img, coordinates.to(device, dtype), features.to(device, dtype)
