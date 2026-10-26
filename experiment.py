"""Fit one Kodak crop with one optimiser, log quality against wall-clock seconds,
then measure quality after rounding the weights. Writes one JSON per run."""
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
from siren import Siren

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=int, default=1)
parser.add_argument("--crop", type=int, default=2, help="1 = full image, 2 = half crop, 4 = quarter crop")
parser.add_argument("--optimizer", choices=["adam", "lm"], default="adam")
parser.add_argument("--num_iters", type=int, default=50000)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--layer_size", type=int, default=28)
parser.add_argument("--num_layers", type=int, default=10)
parser.add_argument("--w0", type=float, default=30.0)
parser.add_argument("--w0_initial", type=float, default=30.0)
parser.add_argument("--target_psnr", type=float, default=0.0, help="stop once reached (0 = never)")
parser.add_argument("--max_seconds", type=float, default=0.0, help="stop after this many seconds (0 = never)")
parser.add_argument("--lm_damping", type=float, default=1e-3)
parser.add_argument("--lm_scaling", choices=["marquardt", "identity"], default="marquardt")
parser.add_argument("--chunk", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--probes", type=int, default=8, help="Hutchinson probes for the Hessian trace")
parser.add_argument("--out", default="runs")
parser.add_argument("--tag", default="")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(args.seed)

img, coords, feats = load_crop(args.image, args.crop, device)
model = Siren(dim_in=2, dim_hidden=args.layer_size, dim_out=3, num_layers=args.num_layers,
              final_activation=torch.nn.Identity(), w0_initial=args.w0_initial, w0=args.w0).to(device)
num_params = sum(p.numel() for p in model.parameters())
bpp = util.bpp(model=model, image=img)
print(f"image {args.image} crop 1/{args.crop}: {tuple(img.shape)} pixels={coords.shape[0]} "
      f"params={num_params} bpp(fp32)={bpp:.3f} bpp(fp16)={bpp / 2:.3f} device={device}")


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(m):
    pred = m(coords)
    mse = torch.nn.functional.mse_loss(pred, feats).item()
    return mse, util.get_clamped_psnr(feats, pred)


log = {"t": [], "loss": [], "psnr": []}
best = {"psnr": -1.0, "state": None}


def record(t, mse, psnr):
    log["t"].append(t)
    log["loss"].append(mse)
    log["psnr"].append(psnr)
    if psnr > best["psnr"]:
        best["psnr"] = psnr
        best["state"] = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


def should_stop(t, psnr):
    return (args.target_psnr and psnr >= args.target_psnr) or (args.max_seconds and t >= args.max_seconds)


sync()
start = time.perf_counter()
if args.optimizer == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    with tqdm.trange(args.num_iters, ncols=100) as bar:
        for i in bar:
            opt.zero_grad()
            pred = model(coords)
            loss = loss_fn(pred, feats)
            loss.backward()
            opt.step()
            mse = loss.item()  # synchronises the device
            psnr = util.get_clamped_psnr(feats, pred.detach())
            t = time.perf_counter() - start
            record(t, mse, psnr)
            if i % 50 == 0:
                bar.set_postfix(loss=f"{mse:.2e}", psnr=f"{psnr:.2f}", best=f"{best['psnr']:.2f}")
            if should_stop(t, psnr):
                break
else:
    opt = LevenbergMarquardt(model, damping=args.lm_damping, chunk_size=args.chunk, scaling=args.lm_scaling)
    log.update(damping=[], accepted=[], retries=[])
    mse, psnr = evaluate(model)
    record(0.0, mse, psnr)
    with tqdm.trange(args.num_iters, ncols=100) as bar:
        for i in bar:
            info = opt.step(coords, feats)
            sync()
            t = time.perf_counter() - start
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
total_time = time.perf_counter() - start

# --- quality after rounding the weights ---------------------------------------
model.load_state_dict(best["state"])
final_mse, final_psnr = evaluate(model)


def quantise_uniform(state, bits):
    levels = 2 ** (bits - 1) - 1
    out = OrderedDict()
    for k, w in state.items():
        scale = w.abs().max().clamp_min(1e-12)
        out[k] = torch.round(w / scale * levels) * scale / levels
    return out


def psnr_with(state):
    m = Siren(dim_in=2, dim_hidden=args.layer_size, dim_out=3, num_layers=args.num_layers,
              final_activation=torch.nn.Identity(), w0_initial=args.w0_initial, w0=args.w0).to(device)
    m.load_state_dict(state)
    return evaluate(m)[1]


quant = {"fp32": final_psnr,
         "fp16_cast": psnr_with(OrderedDict((k, v.half().float()) for k, v in best["state"].items()))}
for bits in (16, 12, 10, 8):
    quant[f"uniform{bits}"] = psnr_with(quantise_uniform(best["state"], bits))


# --- sharpness: Hessian trace of the mean squared error by Hutchinson probing ---
def hessian_trace(probes):
    params = [p for p in model.parameters()]
    loss = torch.nn.functional.mse_loss(model(coords), feats)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    total = 0.0
    g = torch.Generator(device=device).manual_seed(1234)
    for _ in range(probes):
        vs = [torch.randint(0, 2, p.shape, device=device, generator=g).float() * 2 - 1 for p in params]
        hv = torch.autograd.grad(grads, params, grad_outputs=vs, retain_graph=True)
        total += sum((h * v).sum().item() for h, v in zip(hv, vs))
    return total / probes


trace = hessian_trace(args.probes) if args.probes > 0 else None


def time_to(threshold):
    for t, p in zip(log["t"], log["psnr"]):
        if p >= threshold:
            return t
    return None


summary = {
    "config": vars(args), "device": str(device), "pixels": coords.shape[0], "params": num_params,
    "bpp_fp32": bpp, "bpp_fp16": bpp / 2, "steps": len(log["t"]), "seconds": total_time,
    "final_psnr": final_psnr, "final_mse": final_mse, "best_psnr": best["psnr"],
    "time_to_psnr": {str(th): time_to(th) for th in (22, 24, 26, 28, 30, 32, 34, 36)},
    "psnr_after_rounding": quant, "hessian_trace": trace, "log": log,
}
os.makedirs(args.out, exist_ok=True)
tag = args.tag or f"im{args.image:02d}_crop{args.crop}_{args.optimizer}_L{args.num_layers}x{args.layer_size}"
with open(os.path.join(args.out, tag + ".json"), "w") as f:
    json.dump(summary, f)
torch.save(best["state"], os.path.join(args.out, tag + ".pt"))
print(f"\n{tag}: {len(log['t'])} steps in {total_time:.1f}s, psnr {final_psnr:.2f}")
print("time to psnr:", {k: (f"{v:.1f}s" if v is not None else "-") for k, v in summary["time_to_psnr"].items()})
print("after rounding:", {k: f"{v:.2f}" for k, v in quant.items()})
print(f"hessian trace: {trace}")
