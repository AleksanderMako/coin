"""The proposal's figures from the run JSONs of the overnight pods.
Usage: python figures.py runs/night1 out_dir
  fig1  sharpness along the fit, one line per learning rate, with the two-over-learning-rate lines
  fig2  eight-bit rounding loss against final sharpness, every optimiser and learning rate on one axis
  fig3  quality after rounding against bit depth (Adam, warm L-BFGS, two seeds) and the finish trade-off
  fig4  quality against seconds from the learned start against the random start
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
runs = {}
for f in glob.glob(os.path.join(src, "*.json")):
    d = json.load(open(f))
    if "config" in d:
        runs[d["config"]["tag"]] = d
print(f"{len(runs)} runs")
TEST = [1, 5, 8, 14, 20, 23]
BITS = [16, 12, 10, 8]


def exploded(d):
    """A fit that blew up without producing a not-a-number: useless weights, do not plot as a flat fit."""
    along = [x["lambda_max"] for x in d["log"].get("sharpness", []) if x["lambda_max"] is not None]
    return (d["diverged"] or d["final_psnr"] < 10 or (d["lambda_max"] is not None and d["lambda_max"] > 1e6)
            or (along and max(along) > 1e6) or d["log"]["psnr"][-1] < d["best_psnr"] - 5)


def psnr_at(d, t):
    log = d["log"]
    i = np.searchsorted(log["t"], t, side="right")
    return max(log["psnr"][:i]) if i else np.nan


# ---- figure 1: sharpness along the fit ---------------------------------------------------
sweep = [d for t, d in runs.items() if t.startswith("s_")]
if sweep:
    images = sorted({d["config"]["image"] for d in sweep})
    fig, axes = plt.subplots(2, len(images), figsize=(3.2 * len(images), 6), sharex=True, squeeze=False)
    for col, im in enumerate(images):
        for row, opt in enumerate(("gd", "adam")):
            ax = axes[row, col]
            ds = sorted([d for d in sweep if d["config"]["image"] == im and d["config"]["optimizer"] == opt],
                        key=lambda d: d["config"]["lr"])
            for k, d in enumerate(ds):
                lr = d["config"]["lr"]
                s = d["log"]["sharpness"]
                color = plt.cm.viridis(k / max(len(ds) - 1, 1))
                ax.plot([x["step"] for x in s], [x["lambda_max"] for x in s], color=color,
                        label=f"lr {lr:g}" + (" (diverged)" if exploded(d) else ""), linewidth=1.2)
                if opt == "gd":
                    ax.axhline(2 / lr, color=color, linestyle=":", linewidth=0.8)
            ax.set_yscale("log")
            ax.set_title(f"image {im}, {'gradient descent' if opt == 'gd' else 'Adam'}", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
            if col == 0:
                ax.set_ylabel("top Hessian eigenvalue")
            if row == 1:
                ax.set_xlabel("step")
    fig.suptitle("Sharpness along the fit; dotted lines are two over the learning rate", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_sharpness_along_fit.png"), dpi=130)

    # ---- figure 2: rounding loss against final sharpness --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, title in ((axes[0], "uniform8", "per-tensor, all layers at 8 bits"),
                           (axes[1], "channel8_first16", "per-channel, first layer at 16 bits")):
        for opt, marker in (("gd", "o"), ("adam", "s")):
            ds = [d for d in sweep if d["config"]["optimizer"] == opt and not exploded(d)]
            lrs = sorted({d["config"]["lr"] for d in ds})
            for k, lr in enumerate(lrs):
                pts = [(d["lambda_max"], d["psnr_after_rounding"]["fp32"] - d["psnr_after_rounding"][key])
                       for d in ds if d["config"]["lr"] == lr]
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=marker, s=28,
                           color=plt.cm.viridis(k / max(len(lrs) - 1, 1)) if opt == "gd" else plt.cm.autumn(k / max(len(lrs) - 1, 1)),
                           label=f"{'GD' if opt == 'gd' else 'Adam'} lr {lr:g}")
        ax.set_xscale("log")
        ax.set_xlabel("top Hessian eigenvalue at the end of the fit")
        ax.set_ylabel("decibels lost at 8 bits")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Rounding loss against sharpness, half crops of six Kodak images", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_rounding_vs_sharpness.png"), dpi=130)

# ---- figure 3: quality after rounding against bit depth, and the finish trade-off ----------
seed0 = {"adam": {1: "f_adam_im1", 8: "f_adam_im8", 23: "f_adam_im23", 5: "g_adam_im5", 14: "g_adam_im14", 20: "g_adam_im20"},
         "lbfgs": {1: "f_lbfgs_im1", 8: "f_lbfgs_im8", 23: "f_lbfgs_im23", 5: "g_lbfgs_cold_im5", 14: "g_lbfgs_cold_im14", 20: "g_lbfgs_cold_im20"}}
old = {}
for f in glob.glob("runs/[fg]_*.json"):
    d = json.load(open(f))
    old[os.path.basename(f)[:-5]] = d
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, key, title in ((axes[0], "uniform", "per-tensor rounding"), (axes[1], "channel", "per-channel, first layer at 16 bits")):
    for opt, color in (("adam", "C0"), ("lbfgs", "C3")):
        for seed, style in ((0, "-"), (1, "--")):
            curves = []
            for im in TEST:
                d = old.get(seed0[opt][im]) if seed == 0 else runs.get(f"k_{opt}_s1_im{im}")
                if d is None:
                    continue
                q = d["psnr_after_rounding"]
                if seed == 0 and key == "channel":
                    continue  # the seed-zero files predate the per-channel columns
                curves.append([q["fp32"]] + [q[f"{key}{b}" + ("_first16" if key == "channel" else "")] - q["fp32"] + q["fp32"] for b in BITS])
            if curves:
                m = np.mean(curves, axis=0)
                ax.plot([32] + BITS, m, style, color=color, marker="o", markersize=3,
                        label=f"{'Adam' if opt == 'adam' else 'warm L-BFGS'}, seed {seed} ({len(curves)} images)")
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("mean quality after rounding (dB)")
    ax.set_title(title, fontsize=9)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
ax = axes[2]
fins = [d for t, d in runs.items() if t.startswith("fin_")]
for key, marker, label in (("uniform8", "o", "per-tensor 8 bits"), ("channel8_first16", "s", "per-channel, first at 16")):
    xs, ys = [], []
    for d in fins:
        q = d["psnr_after_rounding"]
        xs.append(d["lambda_max"])
        ys.append(q[key] - d["psnr_before_finish"])
    ax.scatter(xs, ys, s=14, marker=marker, alpha=0.7, label=label)
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xscale("log")
ax.set_xlabel("top eigenvalue after the finish")
ax.set_ylabel("8-bit quality minus the unfinished fit's quality (dB)")
ax.set_title("flat-seeking finish: every radius and finishing rate", fontsize=9)
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(out, "fig3_rounding_and_finish.png"), dpi=130)

# ---- figure 4: quality against seconds from the learned start -----------------------------
series = [("base_lbfgs_cold_im{}", "cold L-BFGS, random start", "grey", "-"),
          ("k_lbfgs_s1_im{}", "warm L-BFGS, random start", "C3", "-"),
          ("k_adam_s1_im{}", "Adam, random start", "C0", "-"),
          ("ladam_lbfgs_cold_im{}", "L-BFGS from Adam-trained start", "C2", "--"),
          ("llbfgs_lbfgs_cold_im{}", "L-BFGS from L-BFGS-trained start", "C2", "-"),
          ("llbfgs_adam_im{}", "Adam from L-BFGS-trained start", "C1", "-"),
          ("llbfgsmore_lbfgs_cold_im{}", "L-BFGS from longer-trained start", "C4", "-")]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for ax, im in zip(axes.flat, TEST):
    for pat, label, color, style in series:
        d = runs.get(pat.format(im))
        if d is None:
            continue
        ax.plot(np.maximum(d["log"]["t"], 0.5), d["log"]["psnr"], style, color=color, linewidth=1.1, label=label)
    ax.set_xscale("log")
    ax.set_title(f"image {im}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 700)
    if im in (1, 14):
        ax.set_ylabel("quality (dB)")
    if im in (14, 20, 23):
        ax.set_xlabel("seconds (A40)")
axes[0, 0].legend(fontsize=6)
fig.suptitle("Quality against seconds: learned starts against random starts, same card", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, "fig4_learned_start.png"), dpi=130)
print("figures written to", out)
