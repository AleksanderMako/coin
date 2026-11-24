"""Centre crops of Kodak and DIV2K images, no resampling, so every pixel keeps its detail."""
import os

import imageio.v2 as imageio
import torch
from torchvision import transforms

import util

DIV2K_ROOT = os.path.join("div2k", "DIV2K_valid_HR")
DIV2K_SIZE = (512, 768)  # (height, width) of the centre crop taken from every DIV2K image


def load_kodak(image_id, root="kodak-dataset"):
    img = imageio.imread(f"{root}/kodim{str(image_id).zfill(2)}.png")
    return transforms.ToTensor()(img).float()  # (3, H, W) in [0, 1]


def load_div2k(image_id, root=DIV2K_ROOT):
    """DIV2K image `image_id` (801 -> 0801.png), centre-cropped to 768 wide by 512 high, no resampling."""
    path = os.path.join(root, f"{int(image_id):04d}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DIV2K image {image_id} not found at {path} (unpack DIV2K_valid_HR.zip into div2k/)")
    img = imageio.imread(path)
    if img.ndim == 2:
        img = img[:, :, None].repeat(3, axis=2)
    img = transforms.ToTensor()(img[:, :, :3]).float()
    _, h, w = img.shape
    ch, cw = DIV2K_SIZE
    if h < ch or w < cw:
        raise ValueError(f"DIV2K image {image_id} ({path}) is {w} x {h} pixels, smaller than the "
                         f"{cw} x {ch} centre crop; skip it")
    top, left = (h - ch) // 2, (w - cw) // 2
    return img[:, top:top + ch, left:left + cw].contiguous()


def centre_crop(img, scale):
    """Crop to (H / scale, W / scale) around the image centre. scale 1 = full image."""
    if scale == 1:
        return img
    _, h, w = img.shape
    ch, cw = h // scale, w // scale
    top, left = (h - ch) // 2, (w - cw) // 2
    return img[:, top:top + ch, left:left + cw].contiguous()


def load_image(dataset, index, crop, device, dtype=torch.float32, root=None):
    """Returns (img, coordinates, features) on `device`, using COIN's own coordinate convention.
    dataset "kodak": kodim<index>.png; "div2k": <index>.png cropped to 768 x 512 first. `crop` then
    takes the centre 1/crop of that (1 = all of it)."""
    if dataset == "kodak":
        full = load_kodak(index) if root is None else load_kodak(index, root)
    elif dataset == "div2k":
        full = load_div2k(index) if root is None else load_div2k(index, root)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'kodak' or 'div2k'")
    img = centre_crop(full, crop).to(device, dtype)
    coordinates, features = util.to_coordinates_and_features(img)
    return img, coordinates.to(device, dtype), features.to(device, dtype)


def load_crop(image_id, scale, device, dtype=torch.float32):
    """Kodak only; kept for the existing scripts."""
    return load_image("kodak", image_id, scale, device, dtype)
