"""Fit one Kodak crop with one optimiser, log quality against wall-clock seconds,
then measure quality after rounding the weights. Writes one JSON per run.

Optional pieces: a learned or saved starting point (--init), sharpness logged along the way
(--sharpness_every), and a flat-seeking finishing phase after the main fit (--sam_steps).
"""
import argparse
import json
import os
import time
from collections import OrderedDict

import torch
import tqdm

import util
from crops import load_crop
from lm import LevenbergMarquardt
from rounding import rounding_table
from sharpness import hessian_trace, top_eigenvalue
from siren import Siren

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=int, default=1)
parser.add_argument("--crop", type=int, default=2, help="1 = full image, 2 = half crop, 4 = quarter crop")
parser.add_argument("--optimizer", choices=["adam", "gd", "lm", "lbfgs"], default="adam")
parser.add_argument("--history", type=int, default=20, help="L-BFGS history size")
parser.add_argument("--num_iters", type=int, default=50000)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--layer_size", type=int, default=28)
parser.add_argument("--num_layers", type=int, default=10)
parser.add_argument("--w0", type=float, default=30.0)
parser.add_argument("--w0_initial", type=float, default=30.0)
parser.add_argument("--target_psnr", type=float, default=0.0, help="stop once reached (0 = never)")
parser.add_argument("--max_seconds", type=float, default=0.0, help="stop after this many seconds (0 = never)")
parser.add_argument("--lm_damping", type=float, default=1e-3)
parser.add_argument("--lm_schedule", choices=["nielsen", "simple"], default="nielsen")
parser.add_argument("--warmup_iters", type=int, default=0, help="Adam steps before L-BFGS or Levenberg–Marquardt")
parser.add_argument("--lm_scaling", choices=["marquardt", "identity"], default="marquardt")
parser.add_argument("--chunk", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--probes", type=int, default=8, help="Hutchinson probes for the Hessian trace")
parser.add_argument("--power_iters", type=int, default=20, help="power iterations for the top eigenvalue")
parser.add_argument("--sharpness_every", type=int, default=0, help="log sharpness every N steps (0 = end only)")
parser.add_argument("--init", default="", help="start from these saved weights instead of the random init")
parser.add_argument("--sam_steps", type=int, default=0, help="flat-seeking finishing steps after the main fit")
parser.add_argument("--sam_rho", type=float, default=0.05, help="perturbation radius as a fraction of each tensor's norm")
parser.add_argument("--sam_lr", type=float, default=0.0, help="Adam learning rate for the finish (0 = --lr)")
parser.add_argument("--precond", default="", help="L-BFGS only: per-weight scales (a state dict); the solver works on weight / scale")
parser.add_argument("--out", default="runs")
parser.add_argument("--tag", default="")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(args.seed)

img, coords, feats = load_crop(args.image, args.crop, device)


def build():
    return Siren(dim_in=2, dim_hidden=args.layer_size, dim_out=3, num_layers=args.num_layers,
                 final_activation=torch.nn.Identity(), w0_initial=args.w0_initial, w0=args.w0).to(device)


model = build()
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=device))
num_params = sum(p.numel() for p in model.parameters())
bpp = util.bpp(model=model, image=img)
print(f"image {args.image} crop 1/{args.crop}: {tuple(img.shape)} pixels={coords.shape[0]} "
      f"params={num_params} bpp(fp32)={bpp:.3f} bpp(fp16)={bpp / 2:.3f} device={device} init={args.init or 'random'}")

mse_loss = torch.nn.functional.mse_loss


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(m):
    pred = m(coords)
    return mse_loss(pred, feats).item(), util.get_clamped_psnr(feats, pred)


def snapshot():
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


log = {"t": [], "loss": [], "psnr": [], "sharpness": []}
best = {"psnr": -1.0, "state": None}
paused = 0.0  # seconds spent measuring sharpness, not charged to the fit


def clock():
    return time.perf_counter() - start - paused


def record(t, mse, psnr):
    log["t"].append(t)
    log["loss"].append(mse)
    log["psnr"].append(psnr)
    if psnr > best["psnr"]:
        best["psnr"] = psnr
        best["state"] = snapshot()


def should_stop(t, psnr):
    return (args.target_psnr and psnr >= args.target_psnr) or (args.max_seconds and t >= args.max_seconds)


def measure_sharpness(step):
    global paused
    sync()
    t0 = time.perf_counter()
    lam = top_eigenvalue(model, coords, feats, args.power_iters)
    tr = hessian_trace(model, coords, feats, args.probes)
    sync()
    paused += time.perf_counter() - t0
    log["sharpness"].append({"step": step, "t": clock(), "lambda_max": lam, "trace": tr})


def maybe_measure(i):
    if args.sharpness_every and i % args.sharpness_every == 0:
        measure_sharpness(len(log["t"]))


def first_order_loop(opt, n, label):
    """Adam or plain gradient descent. Returns True if the loss stopped being finite."""
    with tqdm.trange(n, ncols=100, desc=label) as bar:
        for i in bar:
            maybe_measure(i)
            opt.zero_grad()
            pred = model(coords)
            loss = mse_loss(pred, feats)
            loss.backward()
            opt.step()
            mse = loss.item()  # synchronises the device
            if not (mse == mse) or mse == float("inf"):
                print(f"{label}: loss is not finite at step {i}, stopping")
                return True
            psnr = util.get_clamped_psnr(feats, pred.detach())
            t = clock()
            record(t, mse, psnr)
            if i % 50 == 0:
                bar.set_postfix(loss=f"{mse:.2e}", psnr=f"{psnr:.2f}", best=f"{best['psnr']:.2f}")
            if should_stop(t, psnr):
                break
    return False


def adam_warmup(n):
    """A few Adam steps so a second-order method starts inside a sensible basin."""
    if n > 0:
        first_order_loop(torch.optim.Adam(model.parameters(), lr=args.lr), n, "warm-up")


sync()
start = time.perf_counter()
mse0, psnr0 = evaluate(model)
record(0.0, mse0, psnr0)  # the starting point, so a finish-only run (num_iters 0) still has a state
diverged = False
if args.optimizer in ("adam", "gd"):
    opt = (torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam"
           else torch.optim.SGD(model.parameters(), lr=args.lr))
    diverged = first_order_loop(opt, args.num_iters, args.optimizer)
elif args.optimizer == "lbfgs":
    adam_warmup(args.warmup_iters)
    log["lm_start_step"] = len(log["t"])
    # One L-BFGS iteration per call so every iteration gets a timestamp; the
    # history and line-search state carry over between calls.
    params = list(model.parameters())
    if args.precond:
        # Diagonal preconditioning by reparameterisation: the solver sees phi = weight / scale, so a
        # direction with large scale is a large move in weight space. Scales come from a learned file.
        scales = torch.load(args.precond, map_location=device)
        scale = [scales[n].to(device).expand_as(p).contiguous() for n, p in model.named_parameters()]
        phi = [(p.detach() / s).clone().requires_grad_(True) for p, s in zip(params, scale)]
        opt_params = phi
    else:
        opt_params = params
    opt = torch.optim.LBFGS(opt_params, lr=1.0, max_iter=1, history_size=args.history,
                            line_search_fn="strong_wolfe", tolerance_grad=0.0, tolerance_change=0.0)

    def sync_weights():
        if args.precond:
            with torch.no_grad():
                for p, f, s in zip(params, phi, scale):
                    p.copy_(f * s)

    def closure():
        opt.zero_grad()
        model.zero_grad()  # with --precond the solver holds phi, so the weights' own grads must be cleared too
        sync_weights()
        loss = mse_loss(model(coords), feats)
        loss.backward()
        if args.precond:
            for p, f, s in zip(params, phi, scale):
                f.grad = p.grad * s
        return loss

    with tqdm.trange(args.num_iters, ncols=100, desc="lbfgs") as bar:
        for i in bar:
            maybe_measure(i)
            opt.step(closure)
            sync_weights()
            mse, psnr = evaluate(model)
            t = clock()
            if not (mse == mse):
                print("loss is not a number, stopping")
                diverged = True
                break
            record(t, mse, psnr)
            if i % 20 == 0:
                bar.set_postfix(loss=f"{mse:.2e}", psnr=f"{psnr:.2f}", best=f"{best['psnr']:.2f}", t=f"{t:.0f}s")
            if should_stop(t, psnr):
                break
else:
    adam_warmup(args.warmup_iters)
    log["lm_start_step"] = len(log["t"])
    opt = LevenbergMarquardt(model, damping=args.lm_damping, chunk_size=args.chunk,
                             scaling=args.lm_scaling, schedule=args.lm_schedule)
    log.update(damping=[], accepted=[], retries=[])
    with tqdm.trange(args.num_iters, ncols=100, desc="lm") as bar:
        for i in bar:
            info = opt.step(coords, feats)
            sync()
            t = clock()
            mse = info["loss_after"] / feats.numel()
            _, psnr = evaluate(model)
            record(t, mse, psnr)
            log["damping"].append(info["damping"])
            log["accepted"].append(info["accepted"])
            log["retries"].append(info["retries"])
            bar.set_postfix(loss=f"{mse:.2e}", psnr=f"{psnr:.2f}", lam=f"{info['damping']:.1e}",
                            ok=int(info["accepted"]), t=f"{t:.0f}s")
            if should_stop(t, psnr):
                break
            if not info["accepted"] and opt.damping >= opt.max_damping:
                print("damping saturated, stopping")
                break
fit_seconds = clock()
psnr_before_finish = best["psnr"]

# --- flat-seeking finish: gradient taken at the worst nearby point, applied at the original weights ---
if args.sam_steps > 0:
    model.load_state_dict(best["state"])
    params = list(model.parameters())
    finish = torch.optim.Adam(params, lr=args.sam_lr or args.lr)
    log["sam_start_step"] = len(log["t"])
    with tqdm.trange(args.sam_steps, ncols=100, desc="finish") as bar:
        for i in bar:
            maybe_measure(i)
            finish.zero_grad()
            pred = model(coords)
            loss = mse_loss(pred, feats)
            loss.backward()
            eps = []
            with torch.no_grad():
                for p in params:
                    # radius scaled by the tensor's own size, so the large first layer is not under-perturbed
                    e = p.grad * (args.sam_rho * p.norm() / (p.grad.norm() + 1e-12))
                    p.add_(e)
                    eps.append(e)
            finish.zero_grad()
            mse_loss(model(coords), feats).backward()
            with torch.no_grad():
                for p, e in zip(params, eps):
                    p.sub_(e)
            finish.step()
            mse = loss.item()
            psnr = util.get_clamped_psnr(feats, pred.detach())
            log["t"].append(clock())
            log["loss"].append(mse)
            log["psnr"].append(psnr)
            if i % 20 == 0:
                bar.set_postfix(loss=f"{mse:.2e}", psnr=f"{psnr:.2f}")
    # the finished point is the one we evaluate, even if it sits a little below the best fit
    best["state"] = snapshot()
    best["psnr"] = evaluate(model)[1]
total_time = clock()

# --- quality after rounding the weights, and the curvature at the evaluated point ---
model.load_state_dict(best["state"])
final_mse, final_psnr = evaluate(model)


def psnr_with(state):
    m = build()
    m.load_state_dict(state)
    return evaluate(m)[1]


quant = rounding_table(best["state"], psnr_with)
trace = hessian_trace(model, coords, feats, args.probes)
lam_max = top_eigenvalue(model, coords, feats, args.power_iters)


def time_to(threshold):
    for t, p in zip(log["t"], log["psnr"]):
        if p >= threshold:
            return t
    return None


summary = {
    "config": vars(args), "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    "pixels": coords.shape[0], "params": num_params,
    "bpp_fp32": bpp, "bpp_fp16": bpp / 2, "steps": len(log["t"]), "seconds": total_time,
    "fit_seconds": fit_seconds, "sharpness_seconds": paused, "diverged": diverged,
    "final_psnr": final_psnr, "final_mse": final_mse, "best_psnr": best["psnr"],
    "psnr_before_finish": psnr_before_finish,
    "time_to_psnr": {str(th): time_to(th) for th in (22, 24, 26, 28, 30, 32, 34, 36)},
    "psnr_after_rounding": quant, "hessian_trace": trace, "lambda_max": lam_max, "log": log,
}
os.makedirs(args.out, exist_ok=True)
tag = args.tag or f"im{args.image:02d}_crop{args.crop}_{args.optimizer}_L{args.num_layers}x{args.layer_size}"
with open(os.path.join(args.out, tag + ".json"), "w") as f:
    json.dump(summary, f)
torch.save(best["state"], os.path.join(args.out, tag + ".pt"))
print(f"\n{tag}: {len(log['t'])} steps in {total_time:.1f}s, psnr {final_psnr:.2f}"
      + (f" (before finish {psnr_before_finish:.2f})" if args.sam_steps else "") + (" DIVERGED" if diverged else ""))
print("time to psnr:", {k: (f"{v:.1f}s" if v is not None else "-") for k, v in summary["time_to_psnr"].items()})
print("after rounding:", {k: f"{v:.2f}" for k, v in quant.items() if "first16" not in k})
print(f"hessian trace: {trace}, top eigenvalue: {lam_max}")
