"""Quantisation-aware training from a saved fit ("RQAT-style", section 1A of the 2026-12-01 plan).

The latent weights stay in full precision. Every forward pass rounds them with rounding.round_state
(the same grid rules as the rounding table) and passes the gradient straight through the rounding.
Rules: good = per-row abs-max scales recomputed from the latent weights each step, biases per
tensor, first layer at 16 bits; crude = one abs-max scale per tensor, every layer at --bits.
Loss = MSE(image) + lam * MSE(teacher), the teacher being the fit's own full-precision output,
computed once before training. Fresh Adam state at --lr, with a linear warm-up (--warmup) and a
cosine decay to zero (--schedule cosine); --warmup 0 --schedule constant is plain Adam at --lr.

Usage: python qat.py --fit runs/fits/k_lbfgs_s1_im23 --rule good --bits 8 --lam 0.1 --iters 15000
                     --lr 2e-4 --probes 8 --power_iters 20 --tag T --out runs
Saves T.pt (latent weights), T_q.pt (the rounded state, dequantised) and T.json.
"""
import argparse
import json
import math
import os
import time
from collections import OrderedDict

import torch
from torch.func import functional_call

import util
from repair import build_model, load_run, run_paths
from rounding import round_state, rounding_table
from sharpness import hessian_trace, top_eigenvalue

parser = argparse.ArgumentParser()
parser.add_argument("--fit", required=True, help="saved run, path without extension")
parser.add_argument("--rule", choices=["good", "crude"], default="good")
parser.add_argument("--bits", type=int, default=8)
parser.add_argument("--lam", type=float, default=0.1, help="weight of the distillation term")
parser.add_argument("--iters", type=int, default=15000)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--warmup", type=int, default=1000,
                    help="linear learning-rate warm-up steps (0 = none). Fresh Adam's first steps are sign steps of size lr; "
                         "from a converged fit at 2e-4 that drops the PSNR by 20 dB in one step")
parser.add_argument("--schedule", choices=["cosine", "constant"], default="cosine",
                    help="after the warm-up: cosine decay to zero at --iters, or constant (the final state is then a noisy one)")
parser.add_argument("--probes", type=int, default=8, help="Hutchinson probes for the Hessian trace")
parser.add_argument("--power_iters", type=int, default=20, help="power iterations for the top eigenvalue")
parser.add_argument("--sharpness_every", type=int, default=0, help="log the latent weights' sharpness every N steps (0 = end only)")
parser.add_argument("--log_every", type=int, default=100, help="record the quantised PSNR every N steps")
parser.add_argument("--tag", default="")
parser.add_argument("--out", default="runs")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fit_run, _, _ = run_paths(args.fit)
fit_summary, fit_cfg, start_state, img, coords, feats = load_run(fit_run, device)
model = build_model(fit_cfg, device)
model.load_state_dict(start_state)
mse_loss = torch.nn.functional.mse_loss


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def quantised(state):
    """The rounded state under the chosen rule (a plain function of the given tensors, no gradient)."""
    if args.rule == "good":
        return round_state(state, args.bits, per_channel=True, first_bits=16)
    return round_state(state, args.bits)


def psnr_with(state):
    m = build_model(fit_cfg, device)
    m.load_state_dict(state)
    with torch.no_grad():
        return util.get_clamped_psnr(feats, m(coords))


def latent_state():
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


params = OrderedDict(model.named_parameters())
psnr_start = psnr_with(start_state)
psnr_start_quantised = psnr_with(quantised(start_state))

# the fit's own sharpness, same settings and card as the latent weights' at the end
trace_start = hessian_trace(model, coords, feats, args.probes)
lam_max_start = top_eigenvalue(model, coords, feats, args.power_iters)

# the teacher: the fit's full-precision output, computed exactly once
with torch.no_grad():
    teacher = model(coords).detach()

opt = torch.optim.Adam(params.values(), lr=args.lr)


def lr_factor(i):
    """Multiplier of --lr at step i (0-based)."""
    if args.warmup and i < args.warmup:
        return (i + 1) / args.warmup
    if args.schedule == "cosine" and args.iters > args.warmup:
        return 0.5 * (1 + math.cos(math.pi * (i - args.warmup) / (args.iters - args.warmup)))
    return 1.0


log = {"step": [], "t": [], "loss": [], "psnr_quantised": [], "sharpness": []}
paused = 0.0


def measure_sharpness(step):
    global paused
    sync()
    t0 = time.perf_counter()
    lam_max = top_eigenvalue(model, coords, feats, args.power_iters)
    tr = hessian_trace(model, coords, feats, args.probes)
    sync()
    paused += time.perf_counter() - t0
    log["sharpness"].append({"step": step, "lambda_max": lam_max, "trace": tr})


sync()
start = time.perf_counter()
for i in range(args.iters):
    if args.sharpness_every and i % args.sharpness_every == 0:
        measure_sharpness(i)
    for g in opt.param_groups:
        g["lr"] = args.lr * lr_factor(i)
    opt.zero_grad()
    with torch.no_grad():
        q = quantised(OrderedDict((k, p.detach()) for k, p in params.items()))
    # straight-through: the forward pass sees the rounded weights, the gradient reaches the latent ones unchanged
    fake = OrderedDict((k, p + (q[k] - p).detach()) for k, p in params.items())
    pred = functional_call(model, fake, (coords,))
    loss = mse_loss(pred, feats)
    if args.lam:
        loss = loss + args.lam * mse_loss(pred, teacher)
    loss.backward()
    opt.step()
    if i % args.log_every == 0 or i == args.iters - 1:
        sync()
        log["step"].append(i)
        log["t"].append(time.perf_counter() - start - paused)
        log["loss"].append(loss.item())
        log["psnr_quantised"].append(util.get_clamped_psnr(feats, pred.detach()))  # of the weights before this step
        print(f"step {i}: loss {log['loss'][-1]:.3e}, psnr quantised {log['psnr_quantised'][-1]:.2f}", flush=True)
sync()
seconds = time.perf_counter() - start - paused

latent = latent_state()
q_state = quantised(latent)
psnr_latent = psnr_with(latent)
psnr_q = psnr_with(q_state)
trace = hessian_trace(model, coords, feats, args.probes)
lam_max = top_eigenvalue(model, coords, feats, args.power_iters)
table = rounding_table(latent, psnr_with)

summary = {
    "config": vars(args), "fit_config": fit_cfg, "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    "psnr_start": psnr_start, "psnr_start_quantised": psnr_start_quantised,
    "psnr_latent_fp32": psnr_latent, "psnr_quantised": psnr_q,
    "lambda_max": lam_max, "hessian_trace": trace,
    "lambda_max_start": lam_max_start, "hessian_trace_start": trace_start,
    "fit_lambda_max": fit_summary.get("lambda_max"), "fit_hessian_trace": fit_summary.get("hessian_trace"),
    "seconds": seconds, "sharpness_seconds": paused, "iters": args.iters,
    "ms_per_iter": 1000 * seconds / args.iters if args.iters else None,
    "rounding_table": table, "log": log,
}
os.makedirs(args.out, exist_ok=True)
tag = args.tag or f"qat_{args.rule}_b{args.bits}_lam{args.lam:g}_{os.path.basename(fit_run)}"
with open(os.path.join(args.out, tag + ".json"), "w") as f:
    json.dump(summary, f)
torch.save(latent, os.path.join(args.out, tag + ".pt"))
torch.save(q_state, os.path.join(args.out, tag + "_q.pt"))
print(f"\n{tag}: {args.iters} iters in {seconds:.1f}s ({summary['ms_per_iter'] or 0:.1f} ms/iter); "
      f"fit {psnr_start:.3f} dB, rounded before {psnr_start_quantised:.3f}, after {psnr_q:.3f}; latent fp32 {psnr_latent:.3f}; "
      f"lambda_max {lam_max:.4g} (start {lam_max_start:.4g}), trace {trace:.4g} (start {trace_start:.4g})")
