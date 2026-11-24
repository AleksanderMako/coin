"""How far each tensor moved from where its fit began, and what that buys a delta-coded rounding.

For each run the start is rebuilt: the saved file named by the config's `init`, or else the random
initialisation of the config's seed (experiment.py seeds torch, then builds the Siren on the CPU).
Per tensor: rel_drift = ||theta - theta0|| / ||theta0|| and max_delta_over_max_weight =
max|theta - theta0| / max|theta|, the factor by which a per-tensor delta grid is finer than the
weight grid. Per run (repeated on each row): clamped PSNR at fp32, under the good rule at 8 bits
(channel8_first16), with the first layer as a per-row delta at 8 bits (delta8_first), and with every
tensor as a per-row delta (delta8_all).

Usage: python drift.py --out drift.csv <run> ...    (a run is a path without extension)
"""
import argparse
import csv
import os

import torch

import util
from repair import build_model, load_run, run_paths
from rounding import delta_first_state, round_state

COLUMNS = ["run", "dataset", "image", "optimizer", "seed", "init", "tensor", "numel", "rel_drift",
           "max_delta_over_max_weight", "psnr_fp32", "psnr_good8", "psnr_delta8_first", "psnr_delta8_all"]


def find_init(path, run, init_dirs=()):
    """The init file as recorded, or the same file name next to the run, in runs/night1, or in init_dirs."""
    candidates = [path, os.path.join(os.path.dirname(run), os.path.basename(path)),
                  os.path.join("runs", "night1", os.path.basename(path))]
    candidates += [os.path.join(d, os.path.basename(path)) for d in init_dirs]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"start {path!r} of run {run} not found (tried {candidates})")


def rebuild_start(cfg, run, device="cpu", init_dirs=()):
    if cfg.get("init"):
        return torch.load(find_init(cfg["init"], run, init_dirs), map_location=device)
    torch.manual_seed(cfg.get("seed", 0))
    return {k: v.to(device) for k, v in build_model(cfg, "cpu").state_dict().items()}


def tensor_drift(w, w0):
    d = (w - w0).double()
    n0 = w0.double().norm().item()
    rel = d.norm().item() / n0 if n0 > 0 else (0.0 if d.norm().item() == 0 else float("inf"))
    mw = w.double().abs().max().item()
    md = d.abs().max().item()
    ratio = md / mw if mw > 0 else (0.0 if md == 0 else float("inf"))
    return rel, ratio


def drift_rows(run, device="cpu", init_dirs=()):
    run, _, _ = run_paths(run)
    _, cfg, state, _, coords, feats = load_run(run, device)
    start = rebuild_start(cfg, run, device, init_dirs)

    def psnr(st):
        m = build_model(cfg, device)
        m.load_state_dict(st)
        with torch.no_grad():
            return util.get_clamped_psnr(feats, m(coords))

    run_cols = {
        "psnr_fp32": psnr(state),
        "psnr_good8": psnr(round_state(state, 8, per_channel=True, first_bits=16)),
        "psnr_delta8_first": psnr(delta_first_state(state, start, 8)),
        "psnr_delta8_all": psnr(round_state(state, 8, per_channel=True, delta_from=start)),
    }
    rows = []
    for k, w in state.items():
        rel, ratio = tensor_drift(w, start[k])
        rows.append({"run": run, "dataset": cfg.get("dataset", "kodak"), "image": cfg.get("image"),
                     "optimizer": cfg.get("optimizer"), "seed": cfg.get("seed", 0), "init": cfg.get("init", ""),
                     "tensor": k, "numel": w.numel(), "rel_drift": rel, "max_delta_over_max_weight": ratio, **run_cols})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="drift.csv")
    p.add_argument("--init_dir", nargs="*", default=[], help="extra folders to look for learned starts in")
    args = p.parse_args()
    device = torch.device("cpu")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for run in args.runs:
            rows = drift_rows(run, device, args.init_dir)
            w.writerows(rows)
            f.flush()
            first = [r for r in rows if r["tensor"].startswith("net.0.")]
            r0 = rows[0]
            ratio = max(r["max_delta_over_max_weight"] for r in first)
            print(f"{run}: first layer drift {max(r['rel_drift'] for r in first):.3f}, delta grid "
                  f"{f'{1 / ratio:.1f}x finer' if ratio > 0 else 'exact (no move)'} | fp32 {r0['psnr_fp32']:.2f}, "
                  f"good8 {r0['psnr_good8']:.2f}, delta8_first {r0['psnr_delta8_first']:.2f}, delta8_all {r0['psnr_delta8_all']:.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
