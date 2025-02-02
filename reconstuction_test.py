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
import weights_collector

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
print("device is "+device.type)

parser = argparse.ArgumentParser()
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
args = parser.parse_args()

_,_,train_dataset,test_dataset = cifar10loader.loadcifar10()
img = cifar10loader.loadImageI(args.image_id,test_dataset).to(device, dtype)
models = weights_collector.load_models()
model = models[args.image_id]
func_rep = Siren(
        dim_in=2,
        dim_hidden=20,
        dim_out=3,
        num_layers=6,
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,
        w0=30.0
    ).to(device)
func_rep.load_state_dict(model)
coordinates, features = util.to_coordinates_and_features(img)


if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
    
# full precision reconstruction
img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/fp_reconstruction_{args.image_id}.png')

#half precision reconstuction 
func_rep = func_rep.half().to('cuda')
coordinates = coordinates.half().to('cuda')
img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/hp_reconstruction_{args.image_id}.png')


