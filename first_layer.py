"""How far each layer moves from its random start over a fit, and what happens if the first layer is
put back to its start. Runs on the CPU from saved fits. Usage: python first_layer.py out.png runs/f_adam_im1 runs/f_lbfgs_im1 ..."""
import json
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import util
from crops import load_crop
from siren import Siren


def build(cfg, seed=0):
    torch.manual_seed(seed)
    return Siren(2, cfg["layer_size"], 3, cfg["num_layers"], final_activation=torch.nn.Identity(),
                 w0_initial=cfg["w0_initial"], w0=cfg["w0"])


out = sys.argv[1]
rows = []
scatter = []
for tag in sys.argv[2:]:
    cfg = json.load(open(tag + ".json"))["config"]
    fitted = torch.load(tag + ".pt", map_location="cpu")
    init = build(cfg, cfg["seed"]).state_dict()  # the same random start the fit began from
    img, coords, feats = load_crop(cfg["image"], cfg["crop"], torch.device("cpu"))

    def psnr_of(state):
        m = build(cfg)
        m.load_state_dict(state)
        with torch.no_grad():
            return util.psnr(feats, m(coords))

    move = OrderedDict()
    for k in fitted:
        move[k] = ((fitted[k] - init[k]).norm() / init[k].norm()).item()
    first = [k for k in fitted if k.startswith("net.0.")]
    swapped = OrderedDict((k, init[k] if k in first else v) for k, v in fitted.items())
    base = psnr_of(fitted)
    rows.append((tag.split("/")[-1], cfg["image"], cfg["optimizer"], move["net.0.linear.weight"], move["net.1.linear.weight"],
                 np.mean([move[k] for k in fitted if ".linear.weight" in k and not k.startswith("net.0.") and not k.startswith("last")]),
                 move["last_layer.linear.weight"], base, psnr_of(swapped)))
    scatter.append((cfg["image"], init["net.0.linear.weight"].flatten().numpy(), fitted["net.0.linear.weight"].flatten().numpy()))

print(f"{'run':20} {'img':>3} {'opt':5} {'first %':>8} {'hidden1 %':>9} {'hidden mean %':>13} {'last %':>7} {'psnr':>6} {'first layer reset to start':>26}")
for r in rows:
    print(f"{r[0]:20} {r[1]:3d} {r[2]:5} {100 * r[3]:8.1f} {100 * r[4]:9.1f} {100 * r[5]:13.1f} {100 * r[6]:7.1f} {r[7]:6.2f} {r[8]:26.2f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
for im, a, b in scatter:
    ax.scatter(a, b, s=10, alpha=0.6, label=f"image {im}")
lim = max(abs(np.concatenate([s[1] for s in scatter])).max(), abs(np.concatenate([s[2] for s in scatter])).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.8)
ax.set_xlabel("first-layer weight at the random start")
ax.set_ylabel("same weight after the fit")
ax.set_title("the first layer stays where it started", fontsize=9)
ax.legend(fontsize=6)
ax = axes[1]
labels = ["first", "hidden 1", "hidden mean", "last"]
for i, r in enumerate(rows):
    ax.plot(labels, [100 * r[3], 100 * r[4], 100 * r[5], 100 * r[6]], marker="o", markersize=3, linewidth=0.8,
            color="C0" if r[2] == "adam" else "C3", alpha=0.7, label=r[2] if i < 2 else None)
ax.set_ylabel("movement from the start, % of the start's size")
ax.set_title("how far each layer moves", fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax = axes[2]
x = np.arange(len(rows))
ax.bar(x - 0.2, [r[7] for r in rows], 0.4, label="fitted first layer")
ax.bar(x + 0.2, [r[8] for r in rows], 0.4, label="first layer reset to its start")
ax.set_xticks(x)
ax.set_xticklabels([f"{r[1]} {r[2][:2]}" for r in rows], fontsize=7)
ax.set_ylabel("quality (dB)")
ax.set_title("what survives if the decoder regenerates the first layer", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=130)
