"""Meta-learn one starting point for the network on a set of images (Reptile).

Each epoch: for every training image, fit a few hundred steps from the shared start,
then move the shared start a fraction of the way toward the mean of the fitted weights.
The inner fit is Adam (first order) or L-BFGS (the solver that will use the start).
Saves the start after every epoch so a reclaimed pod loses at most one epoch.
"""
import argparse
import json
import os
import time
from collections import OrderedDict

import torch

import util
from crops import load_crop
from siren import Siren

p = argparse.ArgumentParser()
p.add_argument("--images", type=int, nargs="+", required=True)
p.add_argument("--crop", type=int, default=1)
p.add_argument("--inner", choices=["adam", "lbfgs"], default="adam")
p.add_argument("--inner_steps", type=int, default=500, help="Adam steps or L-BFGS iterations per image")
p.add_argument("--inner_lr", type=float, default=2e-4)
p.add_argument("--history", type=int, default=100)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--meta_lr", type=float, default=0.5, help="fraction of the way to move toward the mean fit")
p.add_argument("--layer_size", type=int, default=28)
p.add_argument("--num_layers", type=int, default=10)
p.add_argument("--w0", type=float, default=30.0)
p.add_argument("--w0_initial", type=float, default=30.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--init", default="", help="resume from this start")
p.add_argument("--out", required=True, help="path of the saved start, e.g. runs/reptile_adam.pt")
p.add_argument("--snapshot_every", type=int, default=10)
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(args.seed)


def build():
    return Siren(dim_in=2, dim_hidden=args.layer_size, dim_out=3, num_layers=args.num_layers,
                 final_activation=torch.nn.Identity(), w0_initial=args.w0_initial, w0=args.w0).to(device)


model = build()
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=device))
theta = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())

data = {i: load_crop(i, args.crop, device)[1:] for i in args.images}
print(f"reptile: {len(args.images)} images, inner {args.inner} x {args.inner_steps}, "
      f"{args.epochs} epochs, meta_lr {args.meta_lr}, device {device}")


def inner_fit(coords, feats):
    model.load_state_dict(theta)
    if args.inner == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.inner_lr)
        for _ in range(args.inner_steps):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(coords), feats)
            loss.backward()
            opt.step()
    else:
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.inner_steps, history_size=args.history,
                                line_search_fn="strong_wolfe", tolerance_grad=0.0, tolerance_change=0.0)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(coords), feats)
            loss.backward()
            return loss

        opt.step(closure)
    with torch.no_grad():
        psnr = util.get_clamped_psnr(feats, model(coords))
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items()), psnr


history = {"config": vars(args), "epoch_mean_psnr": [], "epoch_seconds": []}
start = time.perf_counter()
for epoch in range(args.epochs):
    t0 = time.perf_counter()
    mean_delta = OrderedDict((k, torch.zeros_like(v)) for k, v in theta.items())
    psnrs = []
    for i in args.images:
        coords, feats = data[i]
        phi, psnr = inner_fit(coords, feats)
        psnrs.append(psnr)
        for k in theta:
            mean_delta[k] += (phi[k] - theta[k]) / len(args.images)
    for k in theta:
        theta[k] += args.meta_lr * mean_delta[k]
    seconds = time.perf_counter() - t0
    history["epoch_mean_psnr"].append(sum(psnrs) / len(psnrs))
    history["epoch_seconds"].append(seconds)
    print(f"epoch {epoch + 1}/{args.epochs}: mean psnr after inner fit {history['epoch_mean_psnr'][-1]:.2f} dB, "
          f"{seconds:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(theta, args.out)
    if args.snapshot_every and (epoch + 1) % args.snapshot_every == 0:
        torch.save(theta, args.out.replace(".pt", f"_epoch{epoch + 1:03d}.pt"))
    with open(args.out.replace(".pt", ".json"), "w") as f:
        json.dump(history, f)
print(f"done in {time.perf_counter() - start:.0f}s, start saved to {args.out}")
