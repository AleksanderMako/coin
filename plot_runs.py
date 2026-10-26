"""Plot quality against wall-clock seconds for a set of run JSONs, plus a rounding table."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

paths = sys.argv[2:]
out = sys.argv[1]
fig, ax = plt.subplots(figsize=(8, 5))
rows = []
for p in paths:
    d = json.load(open(p))
    log, cfg = d["log"], d["config"]
    label = cfg["optimizer"] + (f" (warm {cfg.get('warmup_iters', 0)})" if cfg.get("warmup_iters") else "")
    ax.plot(log["t"], log["psnr"], label=label, linewidth=1.2)
    if "lm_start_step" in log:
        i = log["lm_start_step"]
        if i < len(log["t"]):
            ax.axvline(log["t"][i], color="grey", linestyle=":", linewidth=0.8)
    q = d["psnr_after_rounding"]
    rows.append((label, d["steps"], d["seconds"], d["final_psnr"], q["fp16_cast"] - q["fp32"],
                 q["uniform12"] - q["fp32"], q["uniform8"] - q["fp32"], d["hessian_trace"]))
ax.set_xlabel("seconds")
ax.set_ylabel("peak signal-to-noise ratio (dB)")
ax.set_xscale("log")
ax.grid(alpha=0.3)
ax.legend()
first = json.load(open(paths[0]))
ax.set_title(f"image {first['config']['image']}, crop 1/{first['config']['crop']}, "
             f"{first['params']} parameters, {first['pixels']} pixels")
fig.tight_layout()
fig.savefig(out, dpi=130)

print(f"{'run':22} {'steps':>6} {'sec':>7} {'psnr':>6} {'fp16':>6} {'12bit':>6} {'8bit':>6} {'H trace':>10}")
for r in rows:
    tr = f"{r[7]:.3g}" if r[7] is not None else "-"
    print(f"{r[0]:22} {r[1]:6d} {r[2]:7.1f} {r[3]:6.2f} {r[4]:+6.2f} {r[5]:+6.2f} {r[6]:+6.2f} {tr:>10}")
