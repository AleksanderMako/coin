"""Layer-wise repair after rounding: each layer is re-solved against what the exact network feeds
the next one, then rounded column by column with the error pushed onto the columns not yet rounded.

Layers 1 to 9 are re-solved (ridge towards their own weights) against the exact network's
pre-activations, given the already-rounded network's inputs; the output layer against the picture.
Rounding is per row for weights, per tensor for biases, column-wise with error compensation through
the inverse Hessian of the layer's inputs. The first layer is left at full precision unless
first_bits is given, in which case it is rounded per row at that depth before the loop.

With first_bits=None this is numerically the recovered script link4_data.py (2026-11-11 figures).

Usage: python repair.py --bits 8 6 [--first_bits 16] <run> ...
prints, per run and bit depth, dB lost (unclamped and clamped) under crude, good, compensation only
and full repair, and the repair's seconds. A run is a path without extension (<run>.json, <run>.pt).
"""
import argparse
import json
import time
from collections import OrderedDict

import torch

import util
from crops import load_image
from rounding import quantise, round_state
from siren import Siren


# --- loading saved runs (shared with qat.py and drift.py) ---

def run_paths(run):
    for ext in (".json", ".pt"):
        if run.endswith(ext):
            run = run[: -len(ext)]
    return run, run + ".json", run + ".pt"


def build_model(cfg, device="cpu"):
    """The Siren of a run's config (experiment.py defaults for anything missing)."""
    return Siren(dim_in=2, dim_hidden=cfg.get("layer_size", 28), dim_out=3, num_layers=cfg.get("num_layers", 10),
                 final_activation=torch.nn.Identity(), w0_initial=cfg.get("w0_initial", 30.0),
                 w0=cfg.get("w0", 30.0)).to(device)


def load_run(run, device="cpu"):
    """Returns (summary json, config, state, img, coords, feats) of a saved run."""
    _, jpath, ppath = run_paths(run)
    with open(jpath) as f:
        summary = json.load(f)
    cfg = summary.get("config", {})
    state = torch.load(ppath, map_location=device)
    img, coords, feats = load_image(cfg.get("dataset", "kodak"), cfg.get("image", 1), cfg.get("crop", 1), device)
    return summary, cfg, state, img, coords, feats


def model_from_state(state, w0=30.0, w0_initial=30.0):
    num_layers = sum(1 for k in state if k.startswith("net.") and k.endswith(".linear.weight"))
    w = state["net.0.linear.weight"]
    m = Siren(w.shape[1], w.shape[0], state["last_layer.linear.weight"].shape[0], num_layers,
              final_activation=torch.nn.Identity(), w0_initial=w0_initial, w0=w0).to(w.device)
    m.load_state_dict(state)
    return m


def psnr_of(state, coords, feats, clamped=False, w0=30.0, w0_initial=30.0):
    m = model_from_state(state, w0, w0_initial)
    with torch.no_grad():
        pred = m(coords)
    return util.get_clamped_psnr(feats, pred) if clamped else util.psnr(feats, pred)


# --- the repair ---

def preacts(state, coords, w0=30.0, w0_initial=30.0):
    """zs[k]: pre-activation of layer k (k = 0 .. L-1 hidden, L output); hs[k]: its input."""
    m = model_from_state(state, w0, w0_initial)
    zs, hs = [], []
    with torch.no_grad():
        h = coords
        for layer in m.net:
            hs.append(h)
            z = layer.linear(h)
            zs.append(z)
            h = layer.activation(z)
        hs.append(h)
        zs.append(m.last_layer.linear(h))
    return zs, hs


def _with_ones(X):
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], 1)


def gptq_round(W, b, X, bits, damp=0.01):
    """Round [W | b] column by column (weights per row, bias per tensor), compensating each column's error
    on the remaining columns through the inverse Hessian of the inputs X (plus the constant)."""
    A = _with_ones(X)
    H = A.T @ A / A.shape[0]
    H += damp * H.diagonal().mean() * torch.eye(H.shape[0], device=X.device, dtype=X.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Wb = torch.cat([W, b[:, None]], 1).clone()
    levels = 2 ** (bits - 1) - 1
    sw = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / levels
    sb = b.abs().max().clamp_min(1e-12) / levels
    Q = torch.zeros_like(Wb)
    n = Wb.shape[1]
    for j in range(n):
        s = sb if j == n - 1 else sw
        q = torch.round(Wb[:, j:j + 1] / s) * s
        Q[:, j:j + 1] = q
        Wb[:, j:] -= ((Wb[:, j:j + 1] - q) / U[j, j]) @ U[j:j + 1, j:]
    return Q[:, :-1].contiguous(), Q[:, -1].contiguous()


def ridge_resolve(X, target, W0, b0, lam):
    """argmin ||[X 1] w - target||^2 / N + lam ||w - w0||^2 for every output column."""
    A = _with_ones(X)
    N = A.shape[0]
    H = A.T @ A + lam * N * torch.eye(A.shape[1], device=X.device, dtype=X.dtype)
    w0 = torch.cat([W0, b0[:, None]], 1).T
    sol = torch.linalg.solve(H, A.T @ target + lam * N * w0)
    return sol[:-1].T.contiguous(), sol[-1].contiguous()


def repair(state, coords, feats, bits, resolve=True, lam=1e-2, damp=1e-2, first_bits=None, w0=30.0, w0_initial=30.0):
    """Returns the repaired, rounded state (an OrderedDict with the same keys)."""
    z_exact, _ = preacts(state, coords, w0, w0_initial)
    st = OrderedDict(state)
    if first_bits is not None:
        for k in ("net.0.linear.weight", "net.0.linear.bias"):
            st[k] = quantise(state[k], first_bits, per_channel=True)
    n_hidden = len(z_exact) - 1
    for k in range(1, n_hidden + 1):
        _, hs = preacts(st, coords, w0, w0_initial)
        X = hs[k]
        name = f"net.{k}.linear" if k < n_hidden else "last_layer.linear"
        W0, b0 = state[name + ".weight"], state[name + ".bias"]
        target = z_exact[k] if k < n_hidden else feats
        W, b = ridge_resolve(X, target, W0, b0, lam) if resolve else (W0, b0)
        st[name + ".weight"], st[name + ".bias"] = gptq_round(W, b, X, bits, damp)
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*")
    p.add_argument("--bits", nargs="+", default=["8"], help="bit depths; runs may follow directly")
    p.add_argument("--first_bits", type=int, default=None, help="round the first layer per row at this depth (default: full precision)")
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--damp", type=float, default=1e-2)
    p.add_argument("--json", default="", help="also write the numbers to this file")
    args = p.parse_args()
    # "--bits 8 6 run ..." : argparse gives the runs to --bits, so split them off here
    args.runs = [b for b in args.bits if not b.isdigit()] + args.runs
    args.bits = [int(b) for b in args.bits if b.isdigit()]
    if not args.runs or not args.bits:
        p.error("need at least one bit depth and one run")
    device = torch.device("cpu")
    results = []
    print(f"dB lost (unclamped / clamped); first layer {'full precision' if args.first_bits is None else f'{args.first_bits} bits per row'}")
    for run in args.runs:
        _, cfg, state, _, coords, feats = load_run(run, device)
        w0, w0i = cfg.get("w0", 30.0), cfg.get("w0_initial", 30.0)
        base = {c: psnr_of(state, coords, feats, c, w0, w0i) for c in (False, True)}
        for bits in args.bits:
            row = {"run": run, "bits": bits, "first_bits": args.first_bits, "fp32": base[False], "fp32_clamped": base[True]}
            states = {"crude": round_state(state, bits), "good": round_state(state, bits, per_channel=True, first_bits=16)}
            t0 = time.perf_counter()
            states["compensate"] = repair(state, coords, feats, bits, resolve=False, lam=args.lam, damp=args.damp,
                                          first_bits=args.first_bits, w0=w0, w0_initial=w0i)
            row["compensate_seconds"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            states["repair"] = repair(state, coords, feats, bits, resolve=True, lam=args.lam, damp=args.damp,
                                      first_bits=args.first_bits, w0=w0, w0_initial=w0i)
            row["repair_seconds"] = time.perf_counter() - t0
            for name, st in states.items():
                row[name] = base[False] - psnr_of(st, coords, feats, False, w0, w0i)
                row[name + "_clamped"] = base[True] - psnr_of(st, coords, feats, True, w0, w0i)
            results.append(row)
            print(f"{run} {bits} bits: fp32 {base[False]:.3f} / {base[True]:.3f} dB | "
                  + " | ".join(f"{n} {row[n]:.3f} / {row[n + '_clamped']:.3f}" for n in states)
                  + f" | repair {row['repair_seconds']:.1f}s", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
