import argparse
import getpass
import imageio
import json
import os
import random
import torch
import util
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
import cifar10loader 
from scipy.fft import dctn, idctn
import numpy as np
import matplotlib.pyplot as plt
import tqdm

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
print("device is "+device.type)
if not os.path.exists('dct2-trainer'):
    os.makedirs('dct2-trainer')
def apply_dct2(img_tensor):
    """
    Applies 2D DCT-II (orthonormal normalization) to each channel of an image tensor.
    
    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (C, H, W)
        
    Returns:
        torch.Tensor: DCT-II coefficients tensor of shape (C, H, W)
    """
    # Detach, move to CPU, and convert to numpy
    img_np = img_tensor.detach().cpu().numpy()
    
    # Transpose to (H, W, C) for channel-wise processing
    img_np = img_np.transpose(1, 2, 0)
    dct_np = np.zeros_like(img_np)
    
    # Apply DCT-II to each channel
    for c in range(img_np.shape[2]):
        dct_np[:, :, c] = dctn(img_np[:, :, c], type=2, norm='ortho', axes=(0, 1))
    
    # Convert back to PyTorch tensor with original device/dtype
    return torch.from_numpy(dct_np.transpose(2, 0, 1)).to(
        device=img_tensor.device,
        dtype=img_tensor.dtype
    )

def apply_idct2(dct_coeffs):
    """
    Applies inverse 2D DCT-II (orthonormal normalization) to recover the original image from DCT coefficients.
    
    Args:
        dct_coeffs (torch.Tensor): DCT coefficients tensor of shape (C, H, W)
        
    Returns:
        torch.Tensor: Reconstructed image tensor of shape (C, H, W)
    """
    # Detach, move to CPU, and convert to numpy
    dct_np = dct_coeffs.detach().cpu().numpy()
    
    # Transpose to (H, W, C) for channel-wise processing
    dct_np = dct_np.transpose(1, 2, 0)
    img_np = np.zeros_like(dct_np)
    
    # Apply inverse DCT-II to each channel
    for c in range(dct_np.shape[2]):
        img_np[:, :, c] = idctn(dct_np[:, :, c], type=2, norm='ortho', axes=(0, 1))
    
    # Convert back to PyTorch tensor with original device/dtype
    return torch.from_numpy(img_np.transpose(2, 0, 1)).to(
        device=dct_coeffs.device,
        dtype=dct_coeffs.dtype
    )

def plot(img):
        img_for_plot = img.clone()
        img_for_plot = img_for_plot.cpu()

        # 2) Now that unnormalization is done, permute *once* from [C,H,W] -> [H,W,C]
        img_for_plot = img_for_plot.permute(1, 2, 0)

        # 3) Convert it to a NumPy array
        img_for_plot = img_for_plot.numpy()

        # 4) Show the image
        plt.imshow(img_for_plot)
        plt.title(f"Label: image {3}")
        plt.axis('off')
        plt.savefig("dct2_image.png") 
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}
img = imageio.imread(f"kodak-dataset/kodim{str(3).zfill(2)}.png")
img = transforms.ToTensor()(img).float().to(device, dtype)
coefficients = apply_dct2(img)
# inverseImg = apply_idct2(coefficients)
func_rep = Siren(
        dim_in=2,
        dim_hidden=100,
        dim_out=3,
        num_layers=5,
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,
        w0=30.0
    ).to(device)
trainer = Trainer(func_rep, lr=2e-4)
coordinates, features = util.to_coordinates_and_features(coefficients)
coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
    # Calculate model size. Divide by 8000 to go from bits to kB
model_size = util.model_size_in_bits(func_rep) / 8000.
print(f'Model size: {model_size:.1f}kB')
fp_bpp = util.bpp(model=func_rep, image=img)
print(f'Full precision bpp: {fp_bpp:.2f}')

#     # Train model in full precision
# trainer.train(coordinates, features, num_iters=1000)

#     # Log full precision results
# results['fp_bpp'].append(fp_bpp)
# results['fp_psnr'].append(trainer.best_vals['psnr'])

#     # Save best model
# torch.save(trainer.best_model, 'dct2-trainer' + f'/best_model_{3}.pt')
# func_rep.load_state_dict(trainer.best_model)
optimizer = torch.optim.Adam(func_rep.parameters(), lr=2e-4)
loss_func = torch.nn.MSELoss()
with tqdm.trange(1000, ncols=100) as t:
     for i in t:
        optimizer.zero_grad()
        predicted = func_rep(coordinates)
        #  predicted= predicted.reshape(coefficients.shape[1], coefficients.shape[2], 3).permute(2, 0, 1)
        loss = loss_func(predicted,features)
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print(f'loss is {loss.item()}')


with torch.no_grad():
    dct2Coefficients = func_rep(coordinates).reshape(coefficients.shape[1], coefficients.shape[2], 3).permute(2, 0, 1)
    img_recon = apply_idct2(dct2Coefficients)
    save_image(torch.clamp(img_recon, 0, 1).to('cpu'), 'dct2-trainer' + f'/fp_reconstruction_{3}.png')