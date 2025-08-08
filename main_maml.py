import argparse
import getpass
import imageio
import json
import os
import random
import torch
import util
from siren import Siren
from siren import SirenWithCorrectorNet
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
import time
from training import maml_train
from training import train_from_starting_point,maml_adaptation,normalize_log2_scale

#python main_maml.py -ni 1500 -lss 28 -nl 10 -iid 14 -ld maml_training_with_scale/outer1k_tasks345_lss28_nl10_iid14 -mosteps 1000 -mt 3,4,5 -misteps 3
parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=50000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
parser.add_argument("-misteps", "--meta_inner_steps", help="maml inner loop steps.", type=int, default=5)
parser.add_argument("-mosteps", "--meta_outer_steps", help="maml outer loop steps.", type=int, default=500)
parser.add_argument("-mt", "--meta_tasks", help="maml tasks.", type=str, default="1,2,3")
parser.add_argument("-rs", "--refine_steps", help="number of steps to take during high res refinenemnt.", type=int, default=10)

args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

img_ids = [int (i) for i in args.meta_tasks.split(",")]



if args.full_dataset:
    min_id, max_id = 1, 24  # Kodak dataset runs from kodim01.png to kodim24.png
else:
    min_id, max_id = args.image_id, args.image_id

# Dictionary to register mean values (both full precision and half precision)
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# Fit images
for i in range(min_id, max_id + 1):
    print(f'Image {i}')

    maml_images = []
    for iid in img_ids:
        task_img = imageio.imread(f"kodak-dataset/kodim{str(iid).zfill(2)}.png")
        task_img = transforms.ToTensor()(task_img).float().to(device, dtype)
        maml_images.append(task_img)

    # Load image
    img = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
    img = transforms.ToTensor()(img).float().to(device, dtype)


    # transform = transforms.Resize((256, 384))
    # img = transform(img)
    print(img.shape)
    base_model = Siren(
        dim_in=3,
        dim_hidden=args.layer_size,
        dim_out=3,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)

    best_model = maml_train(base_model,maml_images,torch.nn.MSELoss(),1e-2,args.meta_inner_steps,1e-3,args.meta_outer_steps)
    model_id = f"model_tasts-{args.meta_tasks}_innersteps-{args.meta_inner_steps}_outersteps-{args.meta_outer_steps}_innerlr-{1e-2}_outerlr-{1e-3}"
    torch.save(best_model, args.logdir + f'/best_model_{model_id}.pt')
    

    # Setup model
    func_rep = Siren(
        dim_in=3,
        dim_hidden=args.layer_size,
        dim_out=3,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)

    # Set up training

    ##load starting point 
    func_rep.load_state_dict(best_model)
    trainer = Trainer(func_rep, lr=args.learning_rate)
    coordinates, features = util.to_coordinates_and_features(img)
    scale = normalize_log2_scale(1.0)
    scale = scale.expand(coordinates.shape[0],1)
    coordinates = torch.cat([coordinates,scale],dim=-1)

    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

    # Calculate model size. Divide by 8000 to go from bits to kB
    model_size = util.model_size_in_bits(func_rep) / 8000.
    print(f'Model size: {model_size:.1f}kB')
    fp_bpp = util.bpp(model=func_rep, image=img)
    print(f'Full precision bpp: {fp_bpp:.2f}')

    # Train model in full precision
    
    start_time = time.time()
    # trainer.train(coordinates, features, num_iters=args.num_iters,img=img)
    weights,pred = maml_adaptation(func_rep,img,1.0,torch.nn.MSELoss(),1e-2,args.meta_inner_steps)
    func_rep.load_state_dict(weights)
  

   
    # print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
    end_time = time.time() 
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60.0
    print("Execution time for starting point learning: {:.2f} minutes".format(elapsed_minutes))

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Save best model
    # torch.save(trainer.best_model, args.logdir + f'/best_model_{i}.pt')

    # Update current model to be best model
    # func_rep.load_state_dict(trainer.best_model)

    # Save full precision image reconstruction
    with torch.no_grad():
        img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/fp_reconstruction_{i}.png')
        fp_psnr = util.get_clamped_psnr(img_recon, img)
        print(f'Full precision psnr: {fp_psnr:.2f}')

    # Convert model and coordinates to half precision. Note that half precision
    # torch.sin is only implemented on GPU, so must use cuda
    if torch.cuda.is_available():
        func_rep = func_rep.half().to('cuda')
        coordinates = coordinates.half().to('cuda')

        # Calculate model size in half precision
        hp_bpp = util.bpp(model=func_rep, image=img)
        results['hp_bpp'].append(hp_bpp)
        print(f'Half precision bpp: {hp_bpp:.2f}')

        # Compute image reconstruction and PSNR
        with torch.no_grad():
            img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
            hp_psnr = util.get_clamped_psnr(img_recon, img)
            save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/hp_reconstruction_{i}.png')
            print(f'Half precision psnr: {hp_psnr:.2f}')
            results['hp_psnr'].append(hp_psnr)
    else:
        results['hp_bpp'].append(fp_bpp)
        results['hp_psnr'].append(0.0)

    # Save logs for individual image
    with open(args.logdir + f'/logs{i}.json', 'w') as f:
        json.dump(trainer.logs, f)

    print('\n')

print('Full results:')
print(results)
with open(args.logdir + f'/results.json', 'w') as f:
    json.dump(results, f)

# Compute and save aggregated results
results_mean = {key: util.mean(results[key]) for key in results}
with open(args.logdir + f'/results_mean.json', 'w') as f:
    json.dump(results_mean, f)

print('Aggregate results:')
print(f'Full precision, bpp: {results_mean["fp_bpp"]:.2f}, psnr: {results_mean["fp_psnr"]:.2f}')
print(f'Half precision, bpp: {results_mean["hp_bpp"]:.2f}, psnr: {results_mean["hp_psnr"]:.2f}')
