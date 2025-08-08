import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def gaussian_kernel2d(kernel_size: int = 5, sigma: float = 1.0, channels: int = 1) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for anti-aliasing.

    Args:
        kernel_size: size of square kernel
        sigma: standard deviation
        channels: number of image channels
    Returns:
        Tensor of shape (channels, 1, k, k)
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    g1d = torch.exp(-0.5 * (coords**2) / (sigma**2))
    g1d /= g1d.sum()
    g2d = g1d[:, None] @ g1d[None, :]
    g2d /= g2d.sum()
    kernel = g2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


class LaplacianPyramid(nn.Module):
    """
    Builds both a Gaussian and Laplacian pyramid of an image batch.

    forward(x) -> (laplacian_pyr, gaussian_pyr)
      - gaussian_pyr: [G0, G1, ..., G_S] where G0 is the original image
      - laplacian_pyr: [L0, L1, ..., L_{S-1}, G_S] where each L_i = G_i - upsample(G_{i+1})
    """
    def __init__(self, max_levels: int = 3, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('_kernel', None)

    def _get_kernel(self, channels: int, device: torch.device) -> torch.Tensor:
        if self._kernel is None or self._kernel.shape[0] != channels:
            kern = gaussian_kernel2d(self.kernel_size, self.sigma, channels)
            self.register_buffer('_kernel', kern)
        return self._kernel.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute pyramids for input x of shape (B, C, H, W).

        Returns:
          laplacian_pyr: List of length max_levels+1: L0...L_{S-1}, and coarsest residual G_S
          gaussian_pyr: List of length max_levels+1: G0 (original), down to G_S
        """
        gauss_pyr: List[torch.Tensor] = [x]
        lap_pyr: List[torch.Tensor] = []
        current = x
        for _ in range(self.max_levels):
            B, C, H, W = current.shape
            kernel = self._get_kernel(C, current.device)
            # blur + downsample
            blurred = F.conv2d(current, kernel, padding=self.kernel_size // 2, groups=C)
            down = blurred[:, :, ::2, ::2]
            gauss_pyr.append(down)
            # upsample + blur
            up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
            up_blurred = F.conv2d(up, kernel, padding=self.kernel_size // 2, groups=C)
            # laplacian band
            lap = current - up_blurred
            lap_pyr.append(lap)
            current = down
        # Append coarsest gaussian as the final laplacian-level residual
        lap_pyr.append(current)
        return lap_pyr, gauss_pyr
