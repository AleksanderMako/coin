import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
image_path = "kodak-dataset/kodim03.png"
num_epochs = 5000
print_interval = 100

# Load and prepare image
img = imageio.imread(image_path)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
img_tensor = img_tensor.to(device, dtype)
C, H, W = img_tensor.shape


x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
xy_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (H*W, 2)


def apply_dct2(img_tensor):
    img_np = img_tensor.cpu().numpy()
    dct_np = np.zeros_like(img_np)
    for c in range(C):
        dct_np[c] = dctn(img_np[c], type=2, norm='ortho')
    return torch.from_numpy(dct_np).to(device=device, dtype=dtype)

dct_target = apply_dct2(img_tensor).reshape(C, H*W).permute(1, 0)  # (H*W, C)


class HybridDCTNet(nn.Module):
    def __init__(self, num_freq=10):
        super().__init__()
        self.num_freq = num_freq
        
       
        self.frequencies = 2**torch.arange(num_freq, device=device, dtype=dtype)
        
       
        self.net = nn.Sequential(
            nn.Linear(2 + 2*2*num_freq, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, C),
        )
        
    def positional_encoding(self, xy):
        encodings = [xy]
        for freq in self.frequencies:
            encodings.append(torch.sin(freq * xy))
            encodings.append(torch.cos(freq * xy))
        return torch.cat(encodings, dim=-1)
    
    def forward(self, xy):
        encoded = self.positional_encoding(xy)
        return self.net(encoded)


model = HybridDCTNet(num_freq=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
criterion = nn.MSELoss()

# Frequency-weighted loss
y_freq, x_freq = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
freq_weights = 1 + (x_freq + y_freq).float().to(device)  # Higher weights for higher frequencies
freq_weights = freq_weights.reshape(-1, 1).expand(-1, C)  # (H*W, C)


loss_history = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    
    pred = model(xy_coords)
    
    
    loss = torch.mean(freq_weights * (pred - dct_target)**2)
    
    
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    
    
    loss_history.append(loss.item())
    
    
    if (epoch+1) % print_interval == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4e}, LR: {optimizer.param_groups[0]['lr']:.1e}")


def reconstruct(model, H, W):
    with torch.no_grad():
        pred_dct = model(xy_coords).permute(1, 0).reshape(C, H, W)
        
    
    img_np = pred_dct.cpu().numpy()
    reconstructed = np.zeros_like(img_np)
    for c in range(C):
        reconstructed[c] = idctn(img_np[c], type=2, norm='ortho')
    return torch.from_numpy(reconstructed).to(device=device, dtype=dtype)


def psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed)**2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


reconstructed_img = reconstruct(model, H, W)
final_psnr = psnr(img_tensor, reconstructed_img)
print(f"\nFinal PSNR: {final_psnr:.2f} dB")


plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(131)
plt.imshow(img_tensor.cpu().permute(1, 2, 0).numpy())
plt.title("Original Image")

# Reconstructed Image
plt.subplot(132)
plt.imshow(reconstructed_img.cpu().permute(1, 2, 0).numpy())
plt.title(f"Reconstructed (PSNR: {final_psnr:.2f}dB)")

# Training Curve
plt.subplot(133)
plt.plot(loss_history)
plt.yscale('log')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (log scale)")

plt.tight_layout()
plt.show()
plt.savefig("losses.png")