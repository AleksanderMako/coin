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
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load image
i = 3 
img = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
img = transforms.ToTensor()(img).float().to(device, dtype)

model = Siren(
        dim_in=2,
        dim_hidden=28,
        dim_out=3,
        num_layers=10,
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,
        w0=30
    ).to(device)

coordinates, features = util.to_coordinates_and_features(img)
coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
state_dict = torch.load("./logs_dir/best_model_3.pt",map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

for name,layer in model.net.named_modules():
    if isinstance(layer, torch.nn.Linear):
        # Get weights and biases
        weights = layer.weight
        biases = layer.bias

        # Print shapes
        print(f"Layer: {name}")
        print(f"  Weight shape: {weights.shape}")
        print(f"  Bias shape: {biases.shape}\n")