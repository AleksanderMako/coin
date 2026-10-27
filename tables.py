"""The proposal's tables from the run JSONs. Usage: python tables.py runs/night1
Prints markdown tables: the learning-rate sweep, the quantiser, the finish, and the learned start.
Missing runs print as '-' so the script can run while lists are still finishing."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

src = sys.argv[1]
runs = {}
for f in glob.glob(os.path.join(src, "*.json")):
    d = json.load(open(f))
    if "config" in d:
        runs[d["config"]["tag"]] = d
TEST = [1, 5, 8, 14, 20, 23]


def exploded(d):
    along = [x["lambda_max"] for x in d["log"].get("sharpness", []) if x["lambda_max"] is not None]
    return (d["diverged"] or d["final_psnr"] < 10 or (d["lambda_max"] or 0) > 1e6
            or (along and max(along) > 1e6) or d["log"]["psnr"][-1] < d["best_psnr"] - 5)


def at(d, t):
    ps = [p for tt, p in zip(d["log"]["t"], d["log"]["psnr"]) if tt <= t]
    return max(ps) if ps else np.nan


def to(d, th):
    return next((t for t, p in zip(d["log"]["t"], d["log"]["psnr"]) if p >= th), None)


def loss(d, key):
    q = d["psnr_after_rounding"]
    return q["fp32"] - q[key]


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---- sweep ----------------------------------------------------------------------------------
print("## Learning-rate sweep, half crops, means over the images that did not explode\n")
print("| optimiser | rate | 2 / rate | images | sharpness | quality fp32 | loss at 8 bits, per tensor | loss at 8 bits, per channel + first layer 16 |")
print("|---|---|---|---|---|---|---|---|")
g = defaultdict(list)
for t, d in runs.items():
    if t.startswith("s_"):
        g[(d["config"]["optimizer"], d["config"]["lr"])].append(d)
for (opt, lr), ds in sorted(g.items()):
    ok = [d for d in ds if not exploded(d)]
    name = "gradient descent" if opt == "gd" else "Adam"
    if not ok:
        print(f"| {name} | {lr:g} | {2 / lr:.0f} | 0 of {len(ds)} | exploded | | | |")
        continue
    print(f"| {name} | {lr:g} | {2 / lr:.0f} | {len(ok)} of {len(ds)} | {np.mean([d['lambda_max'] for d in ok]):.0f} | "
          f"{np.mean([d['final_psnr'] for d in ok]):.1f} | {np.mean([loss(d, 'uniform8') for d in ok]):.2f} | "
          f"{np.mean([loss(d, 'channel8_first16') for d in ok]):.2f} |")

# ---- quantiser ------------------------------------------------------------------------------
print("\n## Quantiser, full images, mean loss in decibels over six images\n")
print("| fit | seed | per tensor 8 | per channel 8 | per tensor 8, first 16 | per channel 8, first 16 | per tensor 12 | per channel 12, first 16 |")
print("|---|---|---|---|---|---|---|---|")
for opt in ("adam", "lbfgs"):
    for seed, tags in ((0, [t for t in runs if t.startswith("fin_") and t.endswith("_plain") and (("lbfgs" in t) == (opt == "lbfgs"))]),
                       (1, [f"k_{opt}_s1_im{im}" for im in TEST if f"k_{opt}_s1_im{im}" in runs])):
        ds = [runs[t] for t in tags]
        if not ds:
            continue
        cells = [np.mean([loss(d, k) for d in ds]) for k in ("uniform8", "channel8", "uniform8_first16", "channel8_first16", "uniform12", "channel12_first16")]
        print(f"| {'Adam' if opt == 'adam' else 'warm L-BFGS'} ({len(ds)} images) | {seed} | " + " | ".join(fmt(c) for c in cells) + " |")

# ---- finish ---------------------------------------------------------------------------------
print("\n## Flat-seeking finish, full images, means over the twelve saved fits\n")
print("| radius | finishing rate | sharpness after / before | quality change | 8-bit per tensor, net vs plain | 8-bit per channel + first 16, net vs plain |")
print("|---|---|---|---|---|---|")
fin = defaultdict(list)
for t, d in runs.items():
    if t.startswith("fin_"):
        c = d["config"]
        fin[(c["sam_rho"] if c["sam_steps"] else 0.0, c["sam_lr"] or c["lr"] if c["sam_steps"] else 0.0)].append(d)
plain = {d["config"]["init"]: d for d in fin.get((0.0, 0.0), [])}
for (rho, lr), ds in sorted(fin.items()):
    if rho == 0.0:
        continue
    ratio, dq, net_u, net_c = [], [], [], []
    for d in ds:
        p = plain.get(d["config"]["init"])
        if p is None:
            continue
        ratio.append(d["lambda_max"] / p["lambda_max"])
        dq.append(d["final_psnr"] - d["psnr_before_finish"])
        net_u.append(d["psnr_after_rounding"]["uniform8"] - p["psnr_after_rounding"]["uniform8"])
        net_c.append(d["psnr_after_rounding"]["channel8_first16"] - p["psnr_after_rounding"]["channel8_first16"])
    if ratio:
        print(f"| {rho:g} | {lr:g} | {np.mean(ratio):.2f} | {np.mean(dq):+.2f} | {np.mean(net_u):+.2f} | {np.mean(net_c):+.2f} | ({len(ratio)} fits)")

# ---- learned start --------------------------------------------------------------------------
print("\n## Learned start, full images, one A40, quality in decibels at 10, 60 and 360 seconds; seconds to the quality warm L-BFGS from random reaches at 360 s\n")
cols = [("k_adam_s1_im{}", "Adam, random"), ("k_lbfgs_s1_im{}", "warm L-BFGS, random"), ("base_lbfgs_cold_im{}", "cold L-BFGS, random"),
        ("ladam_lbfgs_cold_im{}", "L-BFGS, Adam-trained start"), ("llbfgs_lbfgs_cold_im{}", "L-BFGS, L-BFGS-trained start"),
        ("llbfgs_lbfgs_warm_im{}", "warm L-BFGS, L-BFGS-trained start"), ("llbfgs_adam_im{}", "Adam, L-BFGS-trained start"),
        ("ladammore_lbfgs_cold_im{}", "L-BFGS, Adam start +30 epochs"), ("llbfgsmore_lbfgs_cold_im{}", "L-BFGS, L-BFGS start +60 epochs"),
        ("pre_learned_tensor_lbfgs_cold_im{}", "L-BFGS, learned start, per-layer preconditioner"),
        ("pre_learned_elem_lbfgs_cold_im{}", "L-BFGS, learned start, per-weight preconditioner"),
        ("pre_random_tensor_lbfgs_cold_im{}", "L-BFGS, random start, per-layer preconditioner"),
        ("pre_random_elem_lbfgs_cold_im{}", "L-BFGS, random start, per-weight preconditioner")]
print("| method | images | at 10 s | at 60 s | at 360 s | seconds to warm-L-BFGS-at-360 quality (mean where reached, count) | sharpness | 8-bit loss, good quantiser |")
print("|---|---|---|---|---|---|---|---|")
for pat, label in cols:
    ds = {im: runs[pat.format(im)] for im in TEST if pat.format(im) in runs}
    if not ds:
        continue
    tts, n_reached = [], 0
    for im, d in ds.items():
        ref = runs.get(f"k_lbfgs_s1_im{im}")
        if ref is None:
            continue
        t = to(d, at(ref, 360))
        if t is not None:
            tts.append(t); n_reached += 1
    print(f"| {label} | {len(ds)} | {np.mean([at(d, 10) for d in ds.values()]):.2f} | {np.mean([at(d, 60) for d in ds.values()]):.2f} | "
          f"{np.mean([at(d, 360) for d in ds.values()]):.2f} | {fmt(np.mean(tts) if tts else None, 0)} s ({n_reached} of {len(ds)}) | "
          f"{np.mean([d['lambda_max'] for d in ds.values()]):.0f} | {np.mean([loss(d, 'channel8_first16') for d in ds.values()]):.2f} |")
for which in ("adam", "lbfgs", "adam_more", "lbfgs_more"):
    h = runs.get(f"reptile_{which}") or (json.load(open(os.path.join(src, f"reptile_{which}.json"))) if os.path.exists(os.path.join(src, f"reptile_{which}.json")) else None)
    if h and "epoch_mean_psnr" in h:
        e = h["epoch_mean_psnr"]
        print(f"\nReptile {which}: {len(e)} epochs, mean quality after the inner fit {e[0]:.2f} -> {e[-1]:.2f} dB")
