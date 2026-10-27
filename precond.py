"""Learn a diagonal preconditioner for L-BFGS from training images.

At a starting point (random or learned), estimate the diagonal of the Hessian of each training
image's loss with Rademacher probes, average over images, and turn it into per-weight scales
1 / sqrt(curvature): the solver then works in coordinates where every direction has unit curvature.
Writes two scale files: per weight, and per tensor (the mean curvature of each layer), which is
the robust version if the per-weight estimate is too noisy.
Usage: python precond.py --images 2 3 ... --init runs/reptile_lbfgs.pt --probes 64 --out runs/precond_lbfgs
"""
import argparse
import json
from collections import OrderedDict

import torch

from crops import load_crop
from sharpness import _hvp, _loss_and_grads
from siren import Siren

p = argparse.ArgumentParser()
p.add_argument("--images", type=int, nargs="+", required=True)
p.add_argument("--crop", type=int, default=1)
p.add_argument("--init", default="", help="starting point whose curvature is measured (empty = random, seed 0)")
p.add_argument("--probes", type=int, default=64)
p.add_argument("--layer_size", type=int, default=28)
p.add_argument("--num_layers", type=int, default=10)
p.add_argument("--w0", type=float, default=30.0)
p.add_argument("--w0_initial", type=float, default=30.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True, help="prefix; writes <out>_elem.pt and <out>_tensor.pt")
p.add_argument("--tag", default="")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
model = Siren(dim_in=2, dim_hidden=args.layer_size, dim_out=3, num_layers=args.num_layers,
              final_activation=torch.nn.Identity(), w0_initial=args.w0_initial, w0=args.w0).to(device)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=device))
names = [n for n, _ in model.named_parameters()]
diag = OrderedDict((n, torch.zeros_like(p)) for n, p in model.named_parameters())
g = torch.Generator(device=device).manual_seed(1234)
for im in args.images:
    _, coords, feats = load_crop(im, args.crop, device)
    params, grads = _loss_and_grads(model, coords, feats)
    for _ in range(args.probes):
        vs = [torch.randint(0, 2, q.shape, device=device, generator=g).float() * 2 - 1 for q in params]
        hv = _hvp(params, grads, vs)
        for n, h, v in zip(names, hv, vs):
            diag[n] += h * v / (args.probes * len(args.images))  # E[v * Hv] = diag(H) for Rademacher v
    print(f"image {im}: mean diagonal {sum(d.sum() for d in diag.values()).item() * len(args.images) / sum(d.numel() for d in diag.values()):.3g}")

floor = 1e-3 * sum(d.abs().sum() for d in diag.values()).item() / sum(d.numel() for d in diag.values())
elem = OrderedDict((n, 1.0 / torch.sqrt(d.abs() + floor)) for n, d in diag.items())
tensor = OrderedDict((n, torch.full_like(d, 1.0 / (d.abs().mean() + floor).sqrt().item())) for n, d in diag.items())
torch.save(elem, args.out + "_elem.pt")
torch.save(tensor, args.out + "_tensor.pt")
with open(args.out + ".json", "w") as f:
    json.dump({"config": vars(args), "mean_curvature_per_tensor": {n: d.abs().mean().item() for n, d in diag.items()},
               "scale_range_per_tensor": {n: [e.min().item(), e.max().item()] for n, e in elem.items()}}, f)
print("per-tensor mean curvature:", {n: round(d.abs().mean().item(), 2) for n, d in diag.items()})
