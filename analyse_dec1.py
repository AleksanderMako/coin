"""Figures and tables of section 4 of the 2026-12-01 plan (A1-A4, B1-B3, C1-C4), from whatever has been pulled.

Usage (from the repository root):
    .venv/bin/python analyse_dec1.py [--src runs/dec1] [--out figures/dec1] [--jobs jobs] [--no-drift]

Inputs
  runs/dec1/<pod>/runs/d1_*.json|.pt   fits (d1_<arm>_s<seed>_im<image>) and QAT runs (d1_qat_<rule>_lam<l>_<fit>)
  runs/dec1/repair_first_fp.json, repair_first16.json   the CPU repair column (36 fits each)
  the 36 saved fits named in those files (their JSONs; runs/night1/fin_<tag>_plain.json for the seed-0 f_/g_ ones)
Outputs (all rewritten on every call, so the script is idempotent)
  figures/dec1/A1.png ... C4.png, figures/dec1/tables.md, figures/dec1/captions_dec1.md
  runs/dec1/drift_b.csv, runs/dec1/drift_c4.csv   drift.py rows, cached; only runs not yet in a cache are computed

A figure whose inputs are missing is skipped with a printed reason (and any stale PNG of it is removed).
Conventions: PSNR is clamped (util.get_clamped_psnr, as written by experiment.py and qat.py) unless a caption says
otherwise. "dB lost" is fp32 PSNR minus the PSNR after rounding. Sharpness entries are joined to PSNR by the rule of
section 0: an entry recorded at `step` pairs with log["psnr"][step - 1] (the start is log index 0).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import textwrap
from collections import OrderedDict, defaultdict

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------------------------
B_ARMS = ["adam50k", "adamsol", "lbfgswarm", "lbfgssol", "adaminitlbfgs", "adaminitadam", "lbfgsinitlbfgs", "lbfgsinitadam"]
RANDOM_ARMS = B_ARMS[:4]
LEARNED_ARMS = B_ARMS[4:]
C3_PAIRS = [("adamfrz", "adam50k", "Adam"), ("lbfgsfrz", "lbfgswarm", "warm L-BFGS")]
LABEL = {
    "adam50k": "Adam random, 50k steps", "adamsol": "Adam solved", "lbfgswarm": "warm L-BFGS random",
    "lbfgssol": "cold L-BFGS solved", "adaminitlbfgs": "Adam, learned start (L)", "adaminitadam": "Adam, learned start (A)",
    "lbfgsinitlbfgs": "cold L-BFGS, learned start (L)", "lbfgsinitadam": "cold L-BFGS, learned start (A)",
    "adamfrz": "Adam, first layer frozen", "lbfgsfrz": "warm L-BFGS, first layer frozen", "c4lbfgs": "DIV2K warm L-BFGS",
}
SHORT = {"adam50k": "Adam\n50k", "adamsol": "Adam\nsolved", "lbfgswarm": "L-BFGS\nwarm", "lbfgssol": "L-BFGS\nsolved",
         "adaminitlbfgs": "Adam\nlearn L", "adaminitadam": "Adam\nlearn A", "lbfgsinitlbfgs": "L-BFGS\nlearn L",
         "lbfgsinitadam": "L-BFGS\nlearn A"}
COLOR = {"adam50k": "tab:blue", "adamsol": "tab:cyan", "lbfgswarm": "tab:red", "lbfgssol": "tab:orange",
         "adaminitlbfgs": "tab:green", "adaminitadam": "tab:olive", "lbfgsinitlbfgs": "tab:purple",
         "lbfgsinitadam": "tab:pink", "c4lbfgs": "tab:brown"}
HELD_OUT = [1, 5, 8, 14, 20, 23]
START_TYPES = ["Adam random", "L-BFGS random", "learned-start Adam", "learned-start L-BFGS"]
START_COLOR = dict(zip(START_TYPES, ["tab:blue", "tab:red", "tab:green", "tab:purple"]))
QAT_ARMS = [("good", 0.0, "QAT good λ0"), ("good", 0.1, "QAT good λ0.1"), ("crude", 0.1, "RQAT-style crude λ0.1")]
QAT_MARK = {("good", 0.0): "o", ("good", 0.1): "s", ("crude", 0.1): "^"}
QAT_NOTE = ("QAT ran 15,000 iterations from the fit with Adam at a peak learning rate of 2e-4 after a 1,000-step linear "
            "warm-up and a cosine decay to zero (qat.py defaults; a deviation from the plan's plain constant-rate Adam, "
            "made because a fresh Adam step at 2e-4 from a converged fit costs about 20 dB)")
SHARP_GRID = np.arange(16.0, 50.0, 0.25)
B1_MIN_PSNR = 16.0
TIMES = [1, 10, 60, 360, 600]

FIT_RE = re.compile(r"^d1_(?P<arm>[a-z0-9]+)_s(?P<seed>\d+)_im(?P<image>\d+)$")
QAT_RE = re.compile(r"^d1_qat_(?P<rule>good|crude)_lam(?P<lam>[0-9.]+)_(?P<fit>.+)$")


# ---------------------------------------------------------------------------------------------------------------
# log helpers (unit-checked in selftest)
# ---------------------------------------------------------------------------------------------------------------
def joined_sharpness(d):
    """[(psnr, lambda_max, trace, step)] for each sharpness entry, psnr = log["psnr"][step - 1] (section 0)."""
    log = d["log"]
    ps = log["psnr"]
    out = []
    for e in log.get("sharpness", []) or []:
        s = e.get("step")
        if s is None or not (1 <= s <= len(ps)):
            continue
        out.append((ps[s - 1], e.get("lambda_max"), e.get("trace"), s))
    return out


def envelope(pairs):
    """Keep (psnr, value) entries, in time order, where the PSNR is a new strict maximum and the value is a positive
    number: the sharpness at the moment the fit first reached each PSNR. Returns increasing x, y arrays."""
    xs, ys, best = [], [], -math.inf
    for p, v in pairs:
        if v is None or p is None or not np.isfinite(v) or v <= 0 or not np.isfinite(p):
            continue
        if p > best:
            xs.append(p)
            ys.append(v)
            best = p
    return np.array(xs, float), np.array(ys, float)


def log_interp(xs, ys, grid):
    """log10(y) interpolated linearly in x on grid; NaN outside [xs[0], xs[-1]] or with fewer than two points."""
    out = np.full(len(grid), np.nan)
    if len(xs) < 2:
        return out
    inside = (grid >= xs[0]) & (grid <= xs[-1])
    out[inside] = np.interp(grid[inside], xs, np.log10(ys))
    return out


def budget(d):
    ms = d["config"].get("max_seconds") or 0
    return float(ms) if ms > 0 else None


def time_to(d, threshold):
    """(seconds, censored). First log time with PSNR >= threshold; if never, (budget, True), the budget being
    max_seconds or, for an unbudgeted run, its last log time. A censored value is a lower bound, not a time."""
    log = d["log"]
    for t, p in zip(log["t"], log["psnr"]):
        if p >= threshold:
            return float(t), False
    b = budget(d)
    return (b if b is not None else float(log["t"][-1])), True


def psnr_at(d, T):
    """Best logged PSNR up to fit time T (the state experiment.py keeps is the best so far). NaN if T lies beyond the
    run's budget, or before the first log entry."""
    b = budget(d)
    if b is not None and T > b + 1e-6:
        return float("nan")
    t = d["log"]["t"]
    i = int(np.searchsorted(t, T, side="right"))
    return float(max(d["log"]["psnr"][:i])) if i else float("nan")


def censored_median(values):
    """values: [(x, censored)], censored meaning the true value is > x. Returns (median, is_lower_bound)."""
    if not values:
        return float("nan"), False
    keyed = sorted(values, key=lambda v: (math.inf if v[1] else v[0]))
    n = len(keyed)
    mid = keyed[(n - 1) // 2] if n % 2 else None
    if n % 2:
        return mid[0], mid[1]
    a, b = keyed[n // 2 - 1], keyed[n // 2]
    if a[1] or b[1]:
        return (a[0] if a[1] else b[0]), True
    return (a[0] + b[0]) / 2, False


def selftest():
    d = {"config": {"max_seconds": 600}, "log": {
        "t": [0.0, 1.0, 2.0, 3.0, 4.0], "psnr": [5.0, 10.0, 12.0, 11.0, 15.0],
        "sharpness": [{"step": 1, "lambda_max": 1.0, "trace": 10.0}, {"step": 3, "lambda_max": 3.0, "trace": 30.0},
                      {"step": 4, "lambda_max": 4.0, "trace": 40.0}, {"step": 5, "lambda_max": 5.0, "trace": 50.0},
                      {"step": 0, "lambda_max": 9.0, "trace": 9.0}, {"step": 6, "lambda_max": 9.0, "trace": 9.0}]}}
    j = joined_sharpness(d)
    assert [x[0] for x in j] == [5.0, 12.0, 11.0, 15.0], j  # step s -> psnr[s - 1]; steps 0 and 6 dropped
    assert [x[1] for x in j] == [1.0, 3.0, 4.0, 5.0]
    xs, ys = envelope([(x[0], x[1]) for x in j])
    assert list(xs) == [5.0, 12.0, 15.0] and list(ys) == [1.0, 3.0, 5.0]  # 11 dB is not a new maximum
    g = log_interp(np.array([10.0, 20.0]), np.array([10.0, 1000.0]), np.array([5.0, 10.0, 15.0, 20.0, 25.0]))
    assert np.isnan(g[0]) and np.isnan(g[4]) and np.allclose(g[1:4], [1.0, 2.0, 3.0])
    assert time_to(d, 12.0) == (2.0, False)
    assert time_to(d, 11.5) == (2.0, False)
    assert time_to(d, 20.0) == (600.0, True)  # censored at the budget, flagged
    d2 = {"config": {"max_seconds": 0}, "log": d["log"]}
    assert time_to(d2, 20.0) == (4.0, True)  # unbudgeted: censored at its last time
    assert psnr_at(d, 2.5) == 12.0 and psnr_at(d, 3.5) == 12.0 and psnr_at(d, 100) == 15.0
    assert math.isnan(psnr_at(d, 700)) and math.isnan(psnr_at(d, -1))
    assert psnr_at(d2, 700) == 15.0
    assert censored_median([(1, False), (2, False), (600, True)]) == (2, False)
    assert censored_median([(1, False), (600, True), (600, True)]) == (600, True)
    assert censored_median([(1, False), (3, False)]) == (2, False)
    assert censored_median([(1, False), (600, True)]) == (600, True)
    print("selftest: join rule, envelope, interpolation, censoring and PSNR-at-time checks pass")


# ---------------------------------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"  skip {path}: {e}")
        return None


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def stats(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    a = np.array(xs, float)
    return {"n": len(a), "mean": a.mean(), "median": float(np.median(a)), "min": a.min(), "max": a.max()}


def image_of(name):
    m = re.search(r"im(\d+)", name)
    return int(m.group(1)) if m else None


def cards(ds):
    s = sorted({d.get("gpu") or "card not recorded" for d in ds})
    return ", ".join(s) if s else "no card yet"


def images_str(ims):
    ims = sorted(set(ims))
    if not ims:
        return "no images"
    return f"{len(ims)} image{'s' if len(ims) != 1 else ''} ({', '.join(str(i) for i in ims)})" if len(ims) <= 12 \
        else f"{len(ims)} images ({ims[0]}–{ims[-1]}, {len(ims)} of them)"


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, src, jobs_dir):
        self.src = src
        self.fits = {}  # (arm, seed, image) -> {"run", "d", "pod"}
        self.qat = {}  # (rule, lam, fit tag) -> {"run", "d", "pod"}
        self.ref = None
        self.expected = self._expected(jobs_dir)
        seen = {}
        for jpath in sorted(glob.glob(os.path.join(src, "*", "runs", "d1_*.json"))):
            tag = os.path.basename(jpath)[:-5]
            pod = jpath.split(os.sep)[-3]
            d = load_json(jpath)
            if d is None:
                continue
            rec = {"run": jpath[:-5], "d": d, "pod": pod, "tag": tag}
            if tag in seen:
                print(f"  duplicate {tag} on {seen[tag]} and {pod}: keeping the newer file")
                if os.path.getmtime(jpath) < os.path.getmtime(os.path.join(src, seen[tag], "runs", tag + ".json")):
                    continue
            seen[tag] = pod
            if tag == "d1_qat_ref":
                self.ref = rec
                continue
            m = QAT_RE.match(tag)
            if m:
                self.qat[(m["rule"], float(m["lam"]), m["fit"])] = rec
                continue
            m = FIT_RE.match(tag)
            if m and "log" in d:
                self.fits[(m["arm"], int(m["seed"]), int(m["image"]))] = rec
            else:
                print(f"  unrecognised run {jpath}")
        self.rep_fp = self._repair("repair_first_fp.json")
        self.rep16 = self._repair("repair_first16.json")

    @staticmethod
    def _expected(jobs_dir):
        tags = defaultdict(set)
        for f in sorted(glob.glob(os.path.join(jobs_dir, "dec1_*.txt"))):
            kind = re.sub(r"_\d+$", "", os.path.basename(f)[5:-4])  # a, b, c3, c4, qat_ref
            for line in open(f):
                m = re.search(r"--tag (\S+)", line)
                if m:
                    tags[kind].add(m.group(1))
        return tags

    def _repair(self, name):
        rows = load_json(os.path.join(self.src, name)) or []
        return {os.path.basename(r["run"]): r for r in rows}

    def arm(self, arm, seed=0):
        """{image: record} of one arm and seed."""
        return {im: r for (a, s, im), r in self.fits.items() if a == arm and s == seed}

    def status(self):
        done = set(r["tag"] for r in self.fits.values()) | set(r["tag"] for r in self.qat.values())
        if self.ref:
            done.add("d1_qat_ref")
        rows = [[k, len(v & done), len(v)] for k, v in sorted(self.expected.items())]
        return md_table(["job list", "pulled", "expected"], rows)


# ---------------------------------------------------------------------------------------------------------------
# figure output
# ---------------------------------------------------------------------------------------------------------------
class Out:
    def __init__(self, out):
        self.dir = out
        os.makedirs(out, exist_ok=True)
        self.captions = OrderedDict()
        self.tables = []
        self.rendered, self.skipped = [], []

    def save(self, fig, fid, title, caption):
        width_in, height_in = fig.get_size_inches()
        lines = textwrap.wrap(caption, width=int(width_in * 17))
        cap_h = (len(lines) * 0.135 + 0.15) / height_in
        fig.suptitle(f"{fid}. {title}", fontsize=10)
        fig.tight_layout(rect=[0, cap_h, 1, 0.96])
        fig.text(0.01, cap_h - 0.02 / height_in, "\n".join(lines), fontsize=6.5, va="top", ha="left", color="0.25")
        path = os.path.join(self.dir, fid + ".png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        self.captions[fid] = f"**{fid}. {title}.** {caption}"
        self.rendered.append(fid)
        print(f"{fid}: wrote {path}\n  caption: {caption}")

    def skip(self, fid, reason):
        path = os.path.join(self.dir, fid + ".png")
        if os.path.exists(path):
            os.remove(path)
        self.captions[fid] = f"**{fid}.** skipped: {reason}"
        self.skipped.append((fid, reason))
        print(f"{fid}: skipped, {reason}")

    def table(self, title, body, note=""):
        self.tables.append(f"## {title}\n\n" + (note + "\n\n" if note else "") + body + "\n")


# ---------------------------------------------------------------------------------------------------------------
# A: QAT on the 36 saved fits
# ---------------------------------------------------------------------------------------------------------------
def start_type(tag):
    if tag.startswith("llbfgs_adam"):
        return "learned-start Adam"
    if tag.startswith("llbfgs_lbfgs"):
        return "learned-start L-BFGS"
    return "Adam random" if "adam" in tag else "L-BFGS random"


def fit_summary(tag, run):
    """The saved fit's JSON; for the seed-0 f_/g_ fits (no sharpness recorded) the fin_<tag>_plain JSON whose .pt
    is identical supplies the sharpness."""
    d = load_json(run + ".json") or {}
    if d.get("lambda_max") is None:
        alt = load_json(os.path.join("runs", "night1", f"fin_{tag}_plain.json"))
        if alt:
            d = dict(d, lambda_max=alt.get("lambda_max"), hessian_trace=alt.get("hessian_trace"),
                     sharpness_source=f"fin_{tag}_plain")
    return d


def a_rows(D):
    rows = []
    for tag, rfp in D.rep_fp.items():
        r16 = D.rep16.get(tag)
        row = {"tag": tag, "run": rfp["run"], "type": start_type(tag), "image": image_of(tag),
               "seed": 1 if "_s1_" in tag else (None if tag.startswith("llbfgs") else 0),
               "good": rfp["good_clamped"], "repair_fp": rfp["repair_clamped"],
               "repair16": r16["repair_clamped"] if r16 else None,
               "repair_s": (r16 or rfp)["repair_seconds"]}
        for rule, lam, _ in QAT_ARMS:
            q = D.qat.get((rule, lam, tag))
            row[(rule, lam)] = q["d"] if q else None
        rows.append(row)
    return rows


def qat_loss(q):
    return q["psnr_start"] - q["psnr_quantised"]


def figure_a1(D, O, rows):
    fid = "A1"
    if not rows:
        return O.skip(fid, "runs/dec1/repair_first_fp.json missing")
    cols = [("good", "good rule"), ("repair_fp", "repair,\nfirst fp32"), ("repair16", "repair,\nfirst 16 b")] + \
           [((r, l), n.replace("RQAT-style ", "RQAT ").replace(" λ", "\nλ")) for r, l, n in QAT_ARMS]

    def val(row, key):
        v = row[key]
        if isinstance(v, dict):
            return qat_loss(v)
        return v

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    table_rows = []
    nq = sum(1 for r in rows for k in [(a, b) for a, b, _ in QAT_ARMS] if r[k])
    for ax, st in zip(axes.flat, START_TYPES):
        rs = [r for r in rows if r["type"] == st]
        for r in rs:
            ys = [val(r, k) for k, _ in cols]
            xs = [i for i, y in enumerate(ys) if y is not None]
            ax.plot(xs, [ys[i] for i in xs], "-", color="0.6", alpha=0.5, linewidth=0.8)
            ax.scatter(xs, [ys[i] for i in xs], s=10, color=START_COLOR[st], alpha=0.6, zorder=3,
                       marker="o" if r["seed"] != 1 else "D")
        for i, (k, name) in enumerate(cols):
            s = stats([val(r, k) for r in rs])
            if s["n"]:
                ax.scatter([i], [s["mean"]], s=70, marker="_", color="k", linewidths=2.5, zorder=4)
                ax.annotate(f"{s['mean']:.2f}\nn={s['n']}", (i, s["mean"]), xytext=(6, 0), textcoords="offset points",
                            fontsize=6, va="center")
            table_rows.append([st, name.replace("\n", " "), s["n"], fmt(s["mean"], 3), fmt(s["median"], 3),
                               fmt(s["min"], 3), fmt(s["max"], 3)])
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([n for _, n in cols], fontsize=7)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title(f"{st}: {len(rs)} fits, images {', '.join(str(i) for i in sorted({r['image'] for r in rs}))}",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylabel("dB lost at 8 bits (clamped PSNR)")
    seeds = "Seeds 0 (circles) and 1 (diamonds) pooled for the random starts; the learned starts are the single " \
            "L-BFGS-trained Reptile start (reptile_lbfgs.pt), no seed"
    cap = (f"Decibels lost at 8 bits by each fix, per fit, lines join the same fit (paired), black bar = mean. {seeds}; "
           f"Kodak images 1, 5, 8, 14, 20, 23 at full size. Good rule and both repair variants on the CPU (repair "
           f"variants: first layer left at full precision as in the recovered 2026-11-11 script, and first layer "
           f"rounded at 16 bits per row). QAT runs pulled so far: {nq} of 108, on {cards([q['d'] for q in D.qat.values()])}"
           f", budget 15,000 iterations. {QAT_NOTE}. All PSNRs clamped; loss is relative to the fit's own fp32 PSNR. "
           f"Symlog axis, linear below 0.05 dB.")
    O.save(fig, fid, "The 8-bit ladder with QAT, by start type", cap)
    O.table("A1: dB lost at 8 bits, per start type", md_table(
        ["start type", "method", "n", "mean", "median", "min", "max"], table_rows),
        "Clamped PSNR, paired per fit, seeds pooled. Repair variants from runs/dec1/repair_first_fp.json and "
        "repair_first16.json; QAT loss = psnr_start - psnr_quantised.")


def figure_a2(D, O, rows):
    fid = "A2"
    pts = [(r, rule, lam, r[(rule, lam)]) for r in rows for rule, lam, _ in QAT_ARMS if r[(rule, lam)]]
    if not pts:
        return O.skip(fid, "no QAT runs pulled yet (d1_qat_* on pods d1-a1, d1-a2)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, key, skey, name in ((axes[0], "lambda_max", "lambda_max_start", "top Hessian eigenvalue"),
                                (axes[1], "hessian_trace", "hessian_trace_start", "Hessian trace")):
        allv = []
        for r, rule, lam, q in pts:
            x, y = q.get(skey), q.get(key)
            if x and y and x > 0 and y > 0:
                ax.scatter(x, y, marker=QAT_MARK[(rule, lam)], color=START_COLOR[r["type"]], s=22, alpha=0.8)
                allv += [x, y]
        if allv:
            lo, hi = min(allv) / 1.5, max(allv) * 1.5
            ax.plot([lo, hi], [lo, hi], "k:", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{name} of the fit (measured by qat.py before training)")
        ax.set_ylabel(f"{name} of the QAT latent weights")
        ax.grid(alpha=0.3, which="both")
    for st in START_TYPES:
        axes[0].scatter([], [], color=START_COLOR[st], label=st, s=22)
    for rule, lam, n in QAT_ARMS:
        axes[0].scatter([], [], color="0.4", marker=QAT_MARK[(rule, lam)], label=n, s=22)
    axes[0].legend(fontsize=6)
    qs = [p[3] for p in pts]
    cap = (f"Sharpness of each fit against that of its QAT latent weights, both measured by qat.py on the same card "
           f"({cards(qs)}) on the plain full-precision forward pass with 20 power iterations and 8 Hutchinson probes; "
           f"dotted line y = x, points below it are flatter after QAT (paired by construction). {len(qs)} QAT runs of "
           f"108: random starts seeds 0 and 1 pooled, learned starts from reptile_lbfgs.pt; Kodak images 1, 5, 8, 14, "
           f"20, 23; budget 15,000 iterations. {QAT_NOTE}.")
    O.save(fig, fid, "Sharpness before and after QAT", cap)


def figure_a3(D, O, rows):
    fid = "A3"
    if not any(r[(a, b)] for r in rows for a, b, _ in QAT_ARMS):
        return O.skip(fid, "no QAT runs pulled yet")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    trows, qs = [], []
    for i, st in enumerate(START_TYPES):
        for j, (rule, lam, name) in enumerate(QAT_ARMS):
            x = i + (j - 1) * 0.25
            v = [r[(rule, lam)]["psnr_latent_fp32"] - r[(rule, lam)]["psnr_start"]
                 for r in rows if r["type"] == st and r[(rule, lam)]]
            qs += [r[(rule, lam)] for r in rows if r["type"] == st and r[(rule, lam)]]
            if v:
                ax.scatter(x + np.random.default_rng(0).uniform(-0.05, 0.05, len(v)), v, s=14,
                           marker=QAT_MARK[(rule, lam)], color=START_COLOR[st], alpha=0.7)
                ax.scatter([x], [np.mean(v)], marker="_", s=120, color="k", linewidths=2.5)
            s = stats(v)
            trows.append([st, name, s["n"], fmt(s["mean"], 3), fmt(s["median"], 3), fmt(s["min"], 3), fmt(s["max"], 3)])
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xticks(range(len(START_TYPES)))
    ax.set_xticklabels(START_TYPES, fontsize=8)
    ax.set_ylabel("latent fp32 PSNR minus fit PSNR [dB, clamped]")
    for rule, lam, n in QAT_ARMS:
        ax.scatter([], [], color="0.4", marker=QAT_MARK[(rule, lam)], label=n, s=20)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    cap = (f"Full-precision PSNR of the QAT latent weights minus the fit's, per fit (paired), black bar = mean; within "
           f"each start type the three markers are the three QAT arms. {len(qs)} QAT runs on {cards(qs)}; random "
           f"starts seeds 0 and 1 pooled, learned starts from reptile_lbfgs.pt; Kodak images 1, 5, 8, 14, 20, 23; "
           f"budget 15,000 iterations; clamped PSNR. {QAT_NOTE}.")
    O.save(fig, fid, "What QAT does to the full-precision fit", cap)
    O.table("A3: latent fp32 PSNR minus fit PSNR [dB]", md_table(
        ["start type", "QAT arm", "n", "mean", "median", "min", "max"], trows))


def figure_a4(D, O, rows):
    fid = "A4"
    # QAT ran on both a 3090 pod and an A40 pod. A40 runs are used as measured; the others are rescaled by the
    # ratio of A40 to other-card ms/iteration: d1_qat_ref against its twin when both exist, else the medians of the
    # good-lambda-0.1 runs on each card (ms/iteration does not depend on the image: same pixel count).
    ref = D.ref["d"] if D.ref else None
    q01 = [r[("good", 0.1)] for r in rows if r[("good", 0.1)] and r[("good", 0.1)].get("ms_per_iter")]
    is_a40 = lambda q: "A40" in (q.get("gpu") or "")  # noqa: E731
    match = D.qat.get(("good", 0.1, "k_lbfgs_s1_im23"))
    other = [q for q in q01 if not is_a40(q)]
    a40 = [q for q in q01 if is_a40(q)]
    scale, scale_note = None, ""
    if ref and match and not is_a40(match["d"]) and match["d"].get("ms_per_iter"):
        scale = ref["ms_per_iter"] / match["d"]["ms_per_iter"]
        how = (f"d1_qat_ref ({ref['ms_per_iter']:.2f} ms/iteration on the A40) over the same job on "
               f"{match['d'].get('gpu')} ({match['d']['ms_per_iter']:.2f})")
    elif other and (ref or a40):
        a40_ms = [q["ms_per_iter"] for q in a40] + ([ref["ms_per_iter"]] if ref else [])
        scale = float(np.median(a40_ms)) / float(np.median([q["ms_per_iter"] for q in other]))
        how = (f"median A40 ms/iteration ({len(a40_ms)} runs) over median {cards(other)} ms/iteration "
               f"({len(other)} runs)")
    if not other:
        scale_note = f"QAT seconds as measured, all on {cards(q01) if q01 else 'no card yet'}"
    elif scale is not None:
        scale_note = (f"QAT seconds of the A40 runs as measured; runs on {cards(other)} rescaled to the A40 by "
                      f"{scale:.3f} = {how}")
    else:
        scale_note = f"QAT seconds as measured on {cards(q01)}, NOT rescaled to the A40 (no A40 QAT timing yet)"
    qlabel = "fit + QAT good λ0.1" + (" (A40-equivalent)" if not other or scale is not None else " (card as measured)")

    def qat_seconds(q):
        return q["seconds"] * (scale if (scale is not None and not is_a40(q)) else 1.0)

    fits = {}
    for r in rows:
        d = load_json(r["run"] + ".json") or {}
        if d.get("gpu") == "NVIDIA A40":
            fits[r["tag"]] = (r, d.get("fit_seconds") or d["seconds"])
    if not fits:
        return O.skip(fid, "no A40-timed fits among the 36")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    trows = []
    for i, st in enumerate(START_TYPES):
        rs = [(r, s) for r, s in fits.values() if r["type"] == st]
        fit_s = [s for _, s in rs]
        rep_s = [s + r["repair_s"] for r, s in rs]
        qat_s = [s + qat_seconds(r[("good", 0.1)]) for r, s in rs if r[("good", 0.1)]]
        for j, (vals, name, c) in enumerate(((fit_s, "fit (A40)", "0.6"), (rep_s, "fit + repair (repair on CPU)", "tab:green"),
                                             (qat_s, qlabel, "tab:purple"))):
            x = i + (j - 1) * 0.27
            if vals:
                ax.bar(x, np.mean(vals), width=0.25, color=c, alpha=0.6, label=name if i == 0 else None)
                ax.scatter(np.full(len(vals), x), vals, s=8, color="k", zorder=3)
            s = stats(vals)
            trows.append([st, name, s["n"], fmt(s["mean"], 1), fmt(s["min"], 1), fmt(s["max"], 1)])
    ax.set_xticks(range(len(START_TYPES)))
    ax.set_xticklabels(START_TYPES, fontsize=8)
    ax.set_ylabel("seconds")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    cap = (f"Encode seconds per fit (bars = mean, dots = fits): the fit alone on the A40 (seed-1 random starts: Adam "
           f"50k steps, warm L-BFGS 600 s; learned starts: 360 s from reptile_lbfgs.pt, Reptile training time not "
           f"counted), plus the layer-wise repair (first layer at 16 bits, CPU seconds on the analysis laptop), plus QAT "
           f"good λ0.1 (15,000 iterations). {scale_note}. The seed-0 f_/g_ fits are left out because their card is not "
           f"recorded. Kodak images 1, 5, 8, 14, 20, 23; not a PSNR figure. {QAT_NOTE}.")
    O.save(fig, fid, "Encode time: fit, fit plus repair, fit plus QAT", cap)
    O.table("A4: encode seconds", md_table(["start type", "pipeline", "n", "mean s", "min s", "max s"], trows),
            scale_note + ".")


# ---------------------------------------------------------------------------------------------------------------
# B: the solved output layer against the sharp starts
# ---------------------------------------------------------------------------------------------------------------
def b1_column(D, arms, images, key):
    """Per image: {arm: log10 sharpness on SHARP_GRID}, restricted to the overlap of all arms present."""
    present = [a for a in arms if any(im in D.arm(a) and len(joined_sharpness(D.arm(a)[im]["d"])) >= 2 for im in images)]
    per_image = {}
    for im in images:
        env = {}
        for a in present:
            rec = D.arm(a).get(im)
            if not rec:
                break
            j = joined_sharpness(rec["d"])
            xs, ys = envelope([(p, lam if key == "lambda_max" else tr) for p, lam, tr, _ in j])
            if len(xs) < 2:
                break
            env[a] = (xs, ys)
        if len(env) != len(present) or not env:
            continue  # this image lacks an arm: it cannot be matched over all arms
        lo = max(B1_MIN_PSNR, max(xs[0] for xs, _ in env.values()))
        hi = min(xs[-1] for xs, _ in env.values())
        if hi - lo < 0.5:
            continue
        grid_ok = (SHARP_GRID >= lo) & (SHARP_GRID <= hi)
        per_image[im] = {a: np.where(grid_ok, log_interp(xs, ys, SHARP_GRID), np.nan) for a, (xs, ys) in env.items()}
    return present, per_image


def b1_gaps(per_image, a, b):
    """Per image: mean |log10 sharpness of a - of b| over the matched PSNR range."""
    out = {}
    for im, curves in per_image.items():
        if a in curves and b in curves:
            diff = np.abs(curves[a] - curves[b])
            if np.isfinite(diff).any():
                out[im] = float(np.nanmean(diff))
    return out


def figure_b1(D, O, state):
    fid = "B1"
    cols = [("24 Kodak images, random and solved starts", RANDOM_ARMS, list(range(1, 25))),
            ("held-out images 1, 5, 8, 14, 20, 23, all arms", B_ARMS, HELD_OUT)]
    results = {}
    for name, arms, ims in cols:
        for key in ("lambda_max", "trace"):
            results[(name, key)] = b1_column(D, arms, ims, key)
    if not any(per for _, per in results.values()):
        return O.skip(fid, "no image has two or more arms with a sharpness log reaching above 16 dB yet "
                           "(B1 needs at least two arms per image)" if D.fits else "no B fits pulled yet")
    state["b1"] = results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex="col")
    used = []
    for c, (name, arms, ims) in enumerate(cols):
        for r, key in enumerate(("lambda_max", "trace")):
            ax = axes[r, c]
            present, per = results[(name, key)]
            n_img = len(per)
            need = max(2, math.ceil(0.5 * n_img)) if n_img > 1 else 1
            for a in present:
                stack = np.array([per[im][a] for im in per]) if per else np.empty((0, len(SHARP_GRID)))
                for row in stack:
                    ax.plot(SHARP_GRID, 10 ** row, color=COLOR[a], alpha=0.18, linewidth=0.7)
                if len(stack):
                    cnt = np.isfinite(stack).sum(0)
                    with np.errstate(all="ignore"):
                        mean = np.nanmean(stack, 0)
                    mean[cnt < need] = np.nan
                    ax.plot(SHARP_GRID, 10 ** mean, color=COLOR[a], linewidth=2.2, label=f"{LABEL[a]}")
                used += [D.arm(a)[im]["d"] for im in per]
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{name}: {n_img} matched image{'s' if n_img != 1 else ''}", fontsize=8)
            ax.set_ylabel("top Hessian eigenvalue" if key == "lambda_max" else "Hessian trace")
            if r == 1:
                ax.set_xlabel("PSNR along the fit [dB, clamped]")
            if per:
                lo = min(np.nanmin(np.where(np.isfinite(v[present[0]]), SHARP_GRID, np.nan)) for v in per.values())
                hi = max(np.nanmax(np.where(np.isfinite(v[present[0]]), SHARP_GRID, np.nan)) for v in per.values())
                ax.set_xlim(lo - 0.5, hi + 0.5)
            ax.legend(fontsize=6)
    cap = (f"Sharpness against PSNR along the fit. Each sharpness entry recorded at step s is paired with log PSNR "
           f"[s - 1]; per run only the entries where the PSNR reaches a new maximum are kept, and log sharpness is "
           f"interpolated onto a 0.25 dB grid. Per image the grid is cut to the overlap of all arms in the panel and "
           f"never below {B1_MIN_PSNR:.0f} dB (solved starts begin at 13-16 dB, random ones at 6-9), so the comparison "
           f"is paired at matched PSNR. Faint lines = images, bold = mean of log sharpness over the matched images, "
           f"drawn where at least half of them cover the PSNR. Seed 0; learned starts reptile_lbfgs.pt (L) and "
           f"reptile_adam.pt (A), only on the six held-out images. {cards(used)}; budgets: Adam random 50k steps, every "
           f"other arm 600 s of fit time (sharpness time excluded); sharpness every 500 Adam / 50 L-BFGS steps over all "
           f"parameters; Kodak full size, clamped PSNR.")
    O.save(fig, fid, "Sharpness at matched PSNR along the fit", cap)


def b2_data(D):
    """{arm: {image: (seconds, censored)}} against each image's Adam-50k final PSNR, seed 0."""
    ref = D.arm("adam50k")
    out = {}
    for a in B_ARMS:
        recs = D.arm(a)
        out[a] = {im: time_to(recs[im]["d"], ref[im]["d"]["final_psnr"]) for im in recs if im in ref}
    return out


def figure_b2(D, O, state):
    fid = "B2"
    ref = D.arm("adam50k")
    data = b2_data(D)
    state["b2"] = data
    if not ref:
        return O.skip(fid, "no d1_adam50k_s0 run pulled yet (the target PSNR of every image)")
    arms = [a for a in B_ARMS if data[a]]
    fig, ax = plt.subplots(figsize=(11, 5))
    used = []
    xpos = {}
    for i, a in enumerate(arms):
        ims = sorted(data[a])
        jit = np.linspace(-0.18, 0.18, len(ims)) if len(ims) > 1 else [0.0]
        for x, im in zip(jit, ims):
            xpos[(a, im)] = i + x
            t, cens = data[a][im]
            ax.scatter(i + x, max(t, 0.05), s=20, facecolors="none" if cens else COLOR[a], edgecolors=COLOR[a],
                       linewidths=1.0, zorder=3)
        used += [D.arm(a)[im]["d"] for im in ims]
        med, lb = censored_median(list(data[a].values()))
        if np.isfinite(med):
            ax.plot([i - 0.3, i + 0.3], [max(med, 0.05)] * 2, color="k", linewidth=2 if not lb else 1,
                    linestyle="-" if not lb else "--")
        nc = sum(c for _, c in data[a].values())
        ax.annotate(f"{len(ims) - nc}/{len(ims)} reached" + (f"\n{nc} censored" if nc else ""), (i, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, -2), textcoords="offset points", ha="center",
                    va="top", fontsize=6)
    # paired: thin lines join one image between neighbouring arms, only where both points are measured times
    for im in ref:
        for a, b in zip(arms, arms[1:]):
            if im in data[a] and im in data[b] and not data[a][im][1] and not data[b][im][1]:
                ax.plot([xpos[(a, im)], xpos[(b, im)]], [max(data[a][im][0], 0.05), max(data[b][im][0], 0.05)],
                        color="0.7", linewidth=0.4, alpha=0.5, zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([SHORT[a] for a in arms], fontsize=7)
    ax.set_ylabel("fit seconds to the image's Adam-50k PSNR")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    ax.grid(alpha=0.3, which="both", axis="y")
    cap = (f"Fit seconds (sharpness time excluded) until each run first reaches the final clamped PSNR of the same "
           f"image's Adam run at 50k steps (paired per image; grey lines join one image between neighbouring arms where both times are measured). Filled = reached; "
           f"open = not reached within the run's budget, drawn AT the budget as a lower bound (censored), never a "
           f"measured time. Black bar = censoring-aware median (dashed when the median itself is censored). Seed 0; "
           f"learned starts reptile_lbfgs.pt (L) and reptile_adam.pt (A) on the six held-out images only. Budgets: "
           f"Adam random 50k steps (about 1,250 s), all others 600 s. {cards(used)}; Kodak full size.")
    O.save(fig, fid, "Seconds to Adam-50k quality", cap)


def b_tables(D, O, state):
    data = state.get("b2") or b2_data(D)
    ref = D.arm("adam50k")
    if ref:
        arms = [a for a in B_ARMS if data[a]]
        rows = []
        for im in sorted(ref):
            row = [im, fmt(ref[im]["d"]["final_psnr"])]
            for a in arms:
                v = data[a].get(im)
                row.append("" if v is None else (f">{v[0]:.0f} (censored)" if v[1] else f"{v[0]:.1f}"))
            rows.append(row)
        med = ["median", ""]
        for a in arms:
            m, lb = censored_median(list(data[a].values()))
            med.append(f">{m:.0f}" if lb else fmt(m, 1))
        rows.append(med)
        O.table("B2: fit seconds to each image's Adam-50k PSNR (seed 0)",
                md_table(["image", "Adam-50k final PSNR"] + [LABEL[a] for a in arms], rows),
                "Censored = not reached within the budget; the number is the budget, a lower bound. Medians are "
                "censoring-aware (a '>' median means at least half the runs are censored or the median falls on one).")
    # PSNR at fixed times
    rows = []
    for seed in (0, 1):
        for a in B_ARMS:
            recs = D.arm(a, seed)
            if not recs:
                continue
            row = [f"{LABEL[a]}" + (" (seed 1)" if seed else ""), len(recs)]
            for T in TIMES:
                s = stats([psnr_at(r["d"], T) for r in recs.values()])
                row.append(f"{fmt(s['mean'])} ({s['n']})")
            rows.append(row)
    for a, b in (("adamsol", "adam50k"), ("lbfgssol", "lbfgswarm")):
        A, B = D.arm(a), D.arm(b)
        common = sorted(set(A) & set(B))
        if common:
            row = [f"paired lead {LABEL[a]} - {LABEL[b]}", len(common)]
            for T in TIMES:
                s = stats([psnr_at(A[i]["d"], T) - psnr_at(B[i]["d"], T) for i in common])
                row.append(f"{fmt(s['mean'])} ({s['n']})")
            rows.append(row)
    if rows:
        O.table("B: PSNR at 1, 10, 60, 360 and 600 s of fit time", md_table(
            ["arm", "runs"] + [f"{T} s" for T in TIMES], rows),
            "Mean over images of the best clamped PSNR logged up to that fit time (n in brackets; a run with a "
            "budget below T contributes nothing). Seed 0 unless stated; learned-start arms are the six held-out images "
            "only, so their means are not comparable with the 24-image rows.")


def figure_b3(D, O, state):
    fid = "B3"
    arms = [a for a in B_ARMS if D.arm(a)]
    if not arms:
        return O.skip(fid, "no B fits pulled yet")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    used = []
    trows = []
    for ax, key, name in ((axes[0], "lambda", "top Hessian eigenvalue at the end"),
                          (axes[1], "loss", "dB lost at 8 bits, good rule (clamped)")):
        for im in range(1, 25):
            xs, ys = [], []
            for i, a in enumerate(arms):
                rec = D.arm(a).get(im)
                if not rec:
                    continue
                d = rec["d"]
                v = d.get("lambda_max") if key == "lambda" else \
                    d["psnr_after_rounding"]["fp32"] - d["psnr_after_rounding"]["channel8_first16"]
                if v is None:
                    continue
                xs.append(i)
                ys.append(v)
                ax.scatter(i, v, s=14, color=COLOR[a], alpha=0.8, zorder=3,
                           marker="o" if im in HELD_OUT else ".")
                if key == "lambda":
                    used.append(d)
            ax.plot(xs, ys, color="0.7", linewidth=0.4, alpha=0.6)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([SHORT[a] for a in arms], fontsize=7)
        ax.set_yscale("log" if key == "lambda" else "symlog", **({} if key == "lambda" else {"linthresh": 0.05}))
        ax.set_ylabel(name)
        ax.grid(alpha=0.3, which="both", axis="y")
    for a in arms:
        recs = D.arm(a)
        lam = stats([r["d"].get("lambda_max") for r in recs.values()])
        loss = stats([r["d"]["psnr_after_rounding"]["fp32"] - r["d"]["psnr_after_rounding"]["channel8_first16"]
                      for r in recs.values()])
        trows.append([LABEL[a], lam["n"], fmt(lam["median"], 0), fmt(lam["min"], 0), fmt(lam["max"], 0),
                      fmt(loss["mean"], 3), fmt(loss["max"], 3)])
    cap = (f"End-of-fit top eigenvalue (left) and 8-bit good-rule loss (per-row scales, first layer at 16 bits; right), "
           f"per run; grey lines join one image across arms (paired), large dots = held-out images. Seed 0; learned "
           f"starts reptile_lbfgs.pt (L) and reptile_adam.pt (A) on the six held-out images only. Budgets: Adam random "
           f"50k steps, all others 600 s. {cards(used)}; Kodak full size; clamped PSNR.")
    O.save(fig, fid, "Final sharpness and 8-bit loss per arm", cap)
    O.table("B3: final sharpness and 8-bit good-rule loss (seed 0)", md_table(
        ["arm", "n", "median λmax", "min λmax", "max λmax", "mean dB lost", "max dB lost"], trows))


# ---------------------------------------------------------------------------------------------------------------
# C: drift
# ---------------------------------------------------------------------------------------------------------------
def drift_cache(path, recs, compute):
    """Rows of drift.py for every record, computing only runs not yet cached (or cached without PSNR columns when the
    image has since become available). Rewrites the cache atomically."""
    import drift  # noqa: WPS433  (torch import is slow; only when needed)

    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    changed = False
    if compute:
        for rec in recs:
            run = rec["run"]
            if not os.path.exists(run + ".pt"):
                continue
            have = by_run.get(run)
            if have and have[0]["psnr_delta8_first"] not in ("", "nan"):
                continue
            try:
                new = drift.drift_rows(run)
            except FileNotFoundError as e:
                if "DIV2K" not in str(e):
                    print(f"  drift: skip {run}: {e}")
                    continue
                if have:
                    continue  # still no image; keep the partial rows
                new = drift_rows_no_image(drift, run, rec["d"])
                print(f"  drift: {run}: image not on this machine, tensor drift only ({e})")
            except Exception as e:  # noqa: BLE001  a half-copied .pt and the like
                print(f"  drift: skip {run}: {type(e).__name__}: {e}")
                continue
            by_run[run] = [{k: ("" if v is None else v) for k, v in r.items()} for r in new]
            changed = True
            print(f"  drift: {run} done")
    if changed:
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=drift.COLUMNS)
            w.writeheader()
            for run in sorted(by_run):
                w.writerows(by_run[run])
        os.replace(tmp, path)
    out = []
    for run, rs in by_run.items():
        for r in rs:
            r = dict(r)
            for k in ("numel", "rel_drift", "max_delta_over_max_weight", "psnr_fp32", "psnr_good8", "psnr_delta8_first",
                      "psnr_delta8_all"):
                r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
            r["image"] = int(float(r["image"]))
            r["arm"] = FIT_RE.match(os.path.basename(run))["arm"] if FIT_RE.match(os.path.basename(run)) else "?"
            out.append(r)
    return out


def drift_rows_no_image(drift, run, d):
    """Tensor drift without the image (DIV2K not downloaded locally): the fp32 and good-rule PSNRs from the run's
    own rounding table, the delta PSNRs left empty until the image exists."""
    import torch

    cfg = d["config"]
    state = torch.load(run + ".pt", map_location="cpu")
    start = drift.rebuild_start(cfg, run)
    tab = d.get("psnr_after_rounding", {})
    rows = []
    for k, w in state.items():
        rel, ratio = drift.tensor_drift(w, start[k])
        rows.append({"run": run, "dataset": cfg.get("dataset", "kodak"), "image": cfg.get("image"),
                     "optimizer": cfg.get("optimizer"), "seed": cfg.get("seed", 0), "init": cfg.get("init", ""),
                     "tensor": k, "numel": w.numel(), "rel_drift": rel, "max_delta_over_max_weight": ratio,
                     "psnr_fp32": tab.get("fp32"), "psnr_good8": tab.get("channel8_first16"),
                     "psnr_delta8_first": None, "psnr_delta8_all": None})
    return rows


def budgets_c(D):
    """Budget sentence for the drift figures: B's budgets and the DIV2K iteration count actually used."""
    iters = sorted({r["d"]["config"].get("num_iters") for (a, _, _), r in D.fits.items() if a == "c4lbfgs"})
    c4 = (f"{', '.join(str(i) for i in iters)} L-BFGS iterations" if iters else "5,000 L-BFGS iterations (job list)")
    return (f"Budgets: B arms Adam random 50k steps, others 600 s; DIV2K 500 Adam warm-up steps then {c4}, "
            f"history 100, no time limit, seed 0")


def layer_index(tensor):
    if tensor.startswith("last_layer"):
        return 10
    return int(tensor.split(".")[1])


DRIFT_GROUPS = [("Kodak, random start (Adam 50k, warm L-BFGS)", ("adam50k", "lbfgswarm"), "tab:blue"),
                ("Kodak, solved start", ("adamsol", "lbfgssol"), "tab:orange"),
                ("Kodak, learned start", tuple(LEARNED_ARMS), "tab:green"),
                ("DIV2K, warm L-BFGS", ("c4lbfgs",), "tab:brown")]


def figure_c1(D, O, drows):
    fid = "C1"
    w = [r for r in drows if r["tensor"].endswith("weight")]
    if not w:
        return O.skip(fid, "no drift rows (no B or C4 fit with a .pt pulled yet)")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    groups = [(n, a, c) for n, a, c in DRIFT_GROUPS if any(r["arm"] in a for r in w)]
    width = 0.8 / max(len(groups), 1)
    for g, (name, arms, c) in enumerate(groups):
        rs = [r for r in w if r["arm"] in arms]
        data = [[r["rel_drift"] for r in rs if layer_index(r["tensor"]) == L] for L in range(11)]
        pos = [L + (g - (len(groups) - 1) / 2) * width for L in range(11)]
        bp = ax.boxplot([d if d else [np.nan] for d in data], positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=True, flierprops={"markersize": 2},
                        medianprops={"color": "k", "linewidth": 1.2})
        for b in bp["boxes"]:
            b.set_facecolor(c)
            b.set_alpha(0.5)
        nruns = len({r["run"] for r in rs})
        ax.scatter([], [], color=c, marker="s", label=f"{name}: {nruns} fits")
    for y, ls in ((0.05, ":"), (0.3, "--"), (0.7, "--"), (1.0, "-.")):
        ax.axhline(y, color="k", linestyle=ls, linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(11))
    ax.set_xticklabels(["first"] + [f"h{i}" for i in range(1, 10)] + ["output"], fontsize=8)
    ax.set_ylabel("relative drift ||W - W0|| / ||W0||")
    ax.grid(alpha=0.3, which="both", axis="y")
    ax.legend(fontsize=7)
    runs = {r["run"] for r in w}
    kod = sorted({r["image"] for r in w if r["dataset"] == "kodak"})
    div = sorted({r["image"] for r in w if r["dataset"] == "div2k"})
    cap = (f"Relative drift of each weight matrix from its start, {len(runs)} fits: Kodak {images_str(kod)} from all "
           f"B arms (seeds 0 and 1; the start is the seed's random initialisation, or the learned start file; for the "
           f"solved arms the start is the random initialisation before the output-layer solve), DIV2K "
           f"{images_str(div)} (centre crops 768 x 512, warm L-BFGS, iteration budget). Lines: 5 percent (dotted), "
           f"30 and 70 percent (dashed), 100 percent (dash-dot), the thresholds of prediction C1. {budgets_c(D)}. Not paired; no PSNR "
           f"involved. DIV2K patches carry less detail per pixel than Kodak frames, so these fits sit in the "
           f"smooth-image regime. Cards: B fits A40, DIV2K fits RTX 3090 (from the JSONs: "
           f"{cards([rec['d'] for rec in D.fits.values() if rec['d']['config'].get('dataset', 'kodak') == 'div2k']) or 'none'}).")
    O.save(fig, fid, "Per-tensor drift from the start", cap)
    trows = []
    for name, arms, _ in groups:
        rs = [r for r in w if r["arm"] in arms]
        first = stats([r["rel_drift"] for r in rs if layer_index(r["tensor"]) == 0])
        hidden = stats([r["rel_drift"] for r in rs if 1 <= layer_index(r["tensor"]) <= 9])
        outl = stats([r["rel_drift"] for r in rs if layer_index(r["tensor"]) == 10])
        trows.append([name, len({r["run"] for r in rs}), fmt(first["median"], 4), fmt(first["max"], 4),
                      fmt(hidden["min"], 3), fmt(hidden["median"], 3), fmt(hidden["max"], 3), fmt(outl["min"], 3),
                      fmt(outl["median"], 3)])
    O.table("C1: relative drift of the weight matrices", md_table(
        ["group", "fits", "first median", "first max", "hidden min", "hidden median", "hidden max", "output min",
         "output median"], trows), "Hidden = net.1 to net.9 weights; output = last_layer weight.")


def first_rows(drows):
    return [r for r in drows if r["tensor"] == "net.0.linear.weight"]


def figure_c2(D, O, drows):
    fid = "C2"
    f = [r for r in first_rows(drows) if r["max_delta_over_max_weight"] > 0]
    if not f:
        return O.skip(fid, "no drift rows yet")
    kod = [r for r in f if r["dataset"] == "kodak"]
    div = [r for r in f if r["dataset"] == "div2k"]
    panels = [(n, rs) for n, rs in (("Kodak (B fits)", kod), ("DIV2K (C4 fits)", div)) if rs]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.6), squeeze=False)
    for ax, (name, rs) in zip(axes[0], panels):
        arms = sorted({r["arm"] for r in rs}, key=lambda a: (B_ARMS + ["c4lbfgs"]).index(a) if a in B_ARMS + ["c4lbfgs"] else 99)
        for a in arms:
            pts = [(r["image"], 1 / r["max_delta_over_max_weight"]) for r in rs if r["arm"] == a]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=COLOR.get(a, "k"), label=LABEL.get(a, a),
                       alpha=0.8)
        ax.axhspan(15, 40, color="0.85", alpha=0.6, zorder=0, label="predicted 15-40x")
        ax.set_yscale("log")
        ax.set_xlabel("image")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel("first-layer max|W| / max|W - W0|  (delta grid finer by)")
        ax.set_title(f"{name}: {len(rs)} fits", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6)
    cap = (f"By how much a first-layer delta grid is finer than its weight grid, per fit: max|W| / max|W - W0| of the "
           f"first-layer weight matrix (per tensor, from drift.py). Kodak: B fits, seeds 0 and 1, all arms; DIV2K: "
           f"warm L-BFGS crops 768 x 512. {budgets_c(D)}. Not paired; no PSNR involved. Shaded band: "
           f"prediction C2. Cards: A40 (Kodak), RTX 3090 (DIV2K).")
    O.save(fig, fid, "First-layer delta grid against weight grid", cap)


def figure_c3(D, O, state):
    fid = "C3"
    T = 360.0
    pairs = {}
    for frz, free, name in C3_PAIRS:
        F, R = D.arm(frz), D.arm(free)
        common = sorted(set(F) & set(R))
        pairs[name] = [(im, psnr_at(F[im]["d"], T), psnr_at(R[im]["d"], T), F[im]["d"], R[im]["d"]) for im in common]
    state["c3"] = pairs
    if not any(pairs.values()):
        return O.skip(fid, "no image has both a frozen (d1_adamfrz / d1_lbfgsfrz) and its free B run yet")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    trows, used = [], []
    for k, (name, ps) in enumerate(pairs.items()):
        if not ps:
            continue
        c = "tab:blue" if name == "Adam" else "tab:red"
        diff = [f - r for _, f, r, _, _ in ps]
        ax.scatter([p[0] + (k - 0.5) * 0.3 for p in ps], diff, color=c, s=22, label=f"{name}: mean {np.mean(diff):+.3f} dB, n={len(ps)}")
        ax.axhline(np.mean(diff), color=c, linewidth=1, linestyle="--")
        used += [p[3] for p in ps] + [p[4] for p in ps]
        for im, f, r, _, _ in ps:
            trows.append([name, im, fmt(r, 3), fmt(f, 3), fmt(f - r, 3)])
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("Kodak image")
    ax.set_ylabel("frozen minus free PSNR at 360 s [dB]")
    ax.set_xticks(range(1, 25))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    cap = (f"PSNR at 360 s of fit time with the first layer frozen at its random initialisation minus the free fit's "
           f"PSNR at 360 s, paired per image (same seed 0, so the same random start). The free runs are B's Adam-50k "
           f"and warm-L-BFGS (500 Adam warm-up steps) logs truncated at 360 s of fit time (sharpness time is outside "
           f"the clock in both); best PSNR logged up to 360 s on both sides. {cards(used)}; Kodak full size; clamped "
           f"PSNR. PSNR only: a frozen run's sharpness is over a different parameter set.")
    O.save(fig, fid, "Freezing the first layer at the random start, 360 s", cap)
    O.table("C3: frozen against free at 360 s (paired)", md_table(
        ["optimizer", "image", "free PSNR", "frozen PSNR", "frozen - free"], trows))


def figure_c4(D, O, drows):
    fid = "C4"
    f = [r for r in first_rows(drows) if np.isfinite(r["psnr_delta8_first"])]
    if not f:
        return O.skip(fid, "no drift rows with the delta roundings yet" +
                      ("; the DIV2K rows lack them because div2k/DIV2K_valid_HR is not on this machine"
                       if first_rows(drows) else ""))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, arms, c in DRIFT_GROUPS:
        rs = [r for r in f if r["arm"] in arms]
        if not rs:
            continue
        g = [r["psnr_fp32"] - r["psnr_good8"] for r in rs]
        dl = [r["psnr_fp32"] - r["psnr_delta8_first"] for r in rs]
        axes[0].scatter(g, dl, s=14, color=c, alpha=0.8, label=f"{name}: {len(rs)}")
        axes[1].hist(np.array(dl) - np.array(g), bins=30, color=c, alpha=0.5, label=name)
    lim = max(max(r["psnr_fp32"] - r["psnr_good8"] for r in f), max(r["psnr_fp32"] - r["psnr_delta8_first"] for r in f))
    axes[0].plot([0, lim], [0, lim], "k:", linewidth=0.8)
    axes[0].set_xlabel("dB lost, first layer at 16 bits per row (good rule)")
    axes[0].set_ylabel("dB lost, first layer as 8-bit per-row delta")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)
    for x in (-0.05, 0.05):
        axes[1].axvline(x, color="k", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("delta-coded minus 16-bit first layer, dB lost")
    axes[1].set_ylabel("fits")
    axes[1].grid(alpha=0.3)
    cap = (f"Rounding at 8 bits with the first layer as a per-row 8-bit delta from its start against the good rule's "
           f"16-bit first-layer exception (all other tensors per-row 8 bits in both), per fit (paired by construction). "
           f"{len(f)} fits: Kodak B fits (A40, seeds 0 and 1, all arms) and DIV2K C4 fits (RTX 3090, centre crops "
           f"768 x 512, smooth-image regime) where the image was available to recompute the PSNR. {budgets_c(D)}. Clamped PSNR. "
           f"Dotted lines: the ±0.05 dB of prediction C3.")
    O.save(fig, fid, "Delta-coded first layer against the 16-bit exception", cap)
    trows = []
    for name, arms, _ in DRIFT_GROUPS:
        rs = [r for r in first_rows(drows) if r["arm"] in arms]
        if not rs:
            continue
        g = stats([r["psnr_fp32"] - r["psnr_good8"] for r in rs])
        d1 = stats([r["psnr_fp32"] - r["psnr_delta8_first"] for r in rs])
        da = stats([r["psnr_fp32"] - r["psnr_delta8_all"] for r in rs])
        diff = stats([(r["psnr_good8"] - r["psnr_delta8_first"]) for r in rs])
        ratio = stats([1 / r["max_delta_over_max_weight"] for r in rs if r["max_delta_over_max_weight"] > 0])
        trows.append([name, len(rs), fmt(ratio["median"], 1), fmt(ratio["min"], 1), fmt(ratio["max"], 1),
                      fmt(g["mean"], 3), fmt(d1["mean"], 3), fmt(da["mean"], 3), fmt(diff["max"], 3), d1["n"]])
    O.table("C1/C2/C4: first-layer delta grid and delta rounding at 8 bits", md_table(
        ["group", "fits", "grid ratio median", "min", "max", "dB lost good8", "dB lost delta8_first",
         "dB lost delta8_all", "max (delta8_first - good8) loss", "fits with delta PSNR"], trows),
        "grid ratio = max|W| / max|W - W0| of the first-layer weight matrix. Clamped PSNR.")


# ---------------------------------------------------------------------------------------------------------------
# predictions of section 1
# ---------------------------------------------------------------------------------------------------------------
def predictions(D, O, arows, state, drows):
    P = []
    n_drift_expected = len({t for t in D.expected.get("b", set()) if FIT_RE.match(t) and FIT_RE.match(t)["arm"] in B_ARMS}) \
        + len(D.expected.get("c4", set()))

    def add(pid, claim, measure, value, verdict):
        P.append([pid, claim, measure, value, verdict])

    # A1
    pairs = [(r, qat_loss(r[("good", 0.1)]) - r["repair16"], qat_loss(r[("good", 0.1)]) - r["repair_fp"])
             for r in arows if r[("good", 0.1)] and r["type"] != "Adam random"]
    if pairs:
        d16 = [p[1] for p in pairs]
        dfp = [p[2] for p in pairs]
        ok = max(d16) <= 0.1
        add("A1", "QAT good λ0.1 within 0.1 dB of repair on L-BFGS and learned-start fits",
            "QAT loss - repair loss (first layer 16 b; first fp32 in brackets), max and mean over fits",
            f"max {max(d16):+.3f} ({max(dfp):+.3f}), mean {np.mean(d16):+.3f} ({np.mean(dfp):+.3f}), n={len(pairs)}/18",
            ("pass" if ok else "fail") + ("" if len(pairs) == 18 else " (partial)"))
    else:
        add("A1", "QAT good λ0.1 within 0.1 dB of repair", "", "no QAT runs yet", "not yet measurable")
    # A2
    qs = [(r, rule, lam, r[(rule, lam)]) for r in arows for rule, lam, _ in QAT_ARMS if r[(rule, lam)]]
    if qs:
        below = [q["lambda_max"] < q["lambda_max_start"] for *_, q in qs]
        tb = [q["hessian_trace"] < q["hessian_trace_start"] for *_, q in qs]
        ratio = [q["lambda_max"] / q["lambda_max_start"] for *_, q in qs]
        add("A2", "latent weights' sharpness after QAT below the fit's",
            "share of QAT runs with λmax (trace) below the fit's, same card; median λ ratio",
            f"{sum(below)}/{len(qs)} ({sum(tb)}/{len(qs)}), median ratio {np.median(ratio):.2f}",
            ("pass" if all(below) else "fail") + ("" if len(qs) == 108 else " (partial)"))
        loss = [q["psnr_start"] - q["psnr_latent_fp32"] for *_, q in qs]
        s = stats(loss)
        add("A3", "latent weights lose a few tenths of a dB at full precision",
            "fit PSNR - latent fp32 PSNR, mean [min, max]; pass if mean in [0.1, 0.9] dB",
            f"{s['mean']:.3f} [{s['min']:.3f}, {s['max']:.3f}], n={s['n']}",
            ("pass" if 0.1 <= s["mean"] <= 0.9 else "fail") + ("" if len(qs) == 108 else " (partial)"))
    else:
        add("A2", "latent sharpness after QAT below the fit's", "", "no QAT runs yet", "not yet measurable")
        add("A3", "latent weights lose a few tenths of a dB", "", "no QAT runs yet", "not yet measurable")
    crude = [(r, qat_loss(r[("crude", 0.1)])) for r in arows if r[("crude", 0.1)] and r["image"] in (20, 23)]
    if crude:
        add("A4", "RQAT-style crude λ0.1 stays above 0.5 dB loss on images 20 and 23",
            "min dB lost over those fits", f"{min(c[1] for c in crude):.3f}, n={len(crude)}/12",
            ("pass" if min(c[1] for c in crude) > 0.5 else "fail") + ("" if len(crude) == 12 else " (partial)"))
    else:
        add("A4", "crude λ0.1 above 0.5 dB on images 20, 23", "", "no crude QAT runs on 20/23 yet", "not yet measurable")

    # B1: final sharpness of the solved arms between Adam random and the sharp starts, nearer Adam (log scale)
    msgs, verdicts = [], []
    for sol in ("adamsol", "lbfgssol"):
        S, A, L, Ln = D.arm(sol), D.arm("adam50k"), D.arm("lbfgswarm"), D.arm("lbfgsinitlbfgs")
        ims = sorted(set(S) & set(A) & set(L))
        if not ims:
            continue
        ok = 0
        for im in ims:
            s, a = S[im]["d"]["lambda_max"], A[im]["d"]["lambda_max"]
            sharp = [L[im]["d"]["lambda_max"]] + ([Ln[im]["d"]["lambda_max"]] if im in Ln else [])
            between = a <= s <= min(sharp)
            nearer = abs(math.log(s / a)) < abs(math.log(min(sharp) / s))
            ok += between and nearer
        msgs.append(f"{LABEL[sol]}: {ok}/{len(ims)}")
        verdicts.append(ok == len(ims))
    if msgs:
        complete = all(len(set(D.arm(s)) & set(D.arm("adam50k"))) >= 24 for s in ("adamsol", "lbfgssol"))
        add("B1", "solved start's final sharpness between Adam random and the sharp starts, nearer Adam, every image",
            "images where λ_Adam ≤ λ_solved ≤ min(λ_warm-L-BFGS, λ_learned-L-BFGS where run) and log-nearer Adam",
            "; ".join(msgs), ("pass" if all(verdicts) and len(verdicts) == 2 else "fail" if not all(verdicts) else "partial")
            + ("" if complete else " (partial)"))
    else:
        add("B1", "solved sharpness between Adam and the sharp starts", "", "needs adamsol/lbfgssol with adam50k and lbfgswarm",
            "not yet measurable")
    # B2: matched-PSNR curves
    b1 = state.get("b1")
    key = ("24 Kodak images, random and solved starts", "lambda_max")
    if b1 and key in b1 and b1[key][1]:
        per = b1[key][1]
        g_a = b1_gaps(per, "adamsol", "adam50k")
        g_l = b1_gaps(per, "lbfgssol", "lbfgswarm")
        g_x = b1_gaps(per, "adam50k", "lbfgswarm")
        if g_a and g_l and g_x:
            ma, ml, mx = np.median(list(g_a.values())), np.median(list(g_l.values())), np.median(list(g_x.values()))
            ok = ma < 0.25 * mx and ml < 0.25 * mx
            add("B2", "at matched PSNR solved Adam lies on random Adam and solved L-BFGS on random L-BFGS",
                "median over images of mean |Δlog10 λmax| on the matched grid; pass if both < 1/4 of the Adam vs L-BFGS gap",
                f"Adam sol-rand {ma:.3f}, L-BFGS sol-rand {ml:.3f}, Adam-vs-L-BFGS {mx:.3f} decades, n={len(g_x)}",
                ("pass" if ok else "fail") + ("" if len(g_x) >= 24 else " (partial)"))
        else:
            add("B2", "matched-PSNR curves coincide", "", "needs all four 24-image arms on a common image",
                "not yet measurable")
    else:
        add("B2", "matched-PSNR curves coincide", "", "no matched images yet", "not yet measurable")
    # B3a: solved Adam's lead
    A, R = D.arm("adamsol"), D.arm("adam50k")
    common = sorted(set(A) & set(R))
    if common:
        lead = {T: np.nanmean([psnr_at(A[i]["d"], T) - psnr_at(R[i]["d"], T) for i in common]) for T in TIMES}
        ok = lead[1] >= max(lead.values()) - 1e-9 and abs(lead[360]) < 0.1
        add("B3a", "solved Adam's lead over random Adam largest at 1 s and gone by 360 s",
            "mean paired lead at 1/10/60/360/600 s; pass if largest at 1 s and |lead at 360 s| < 0.1 dB",
            ", ".join(f"{fmt(lead[T])}" for T in TIMES) + f" dB, n={len(common)}",
            ("pass" if ok else "fail") + ("" if len(common) >= 24 else " (partial)"))
    else:
        add("B3a", "solved Adam's lead largest at 1 s, gone by 360 s", "", "needs adamsol and adam50k", "not yet measurable")
    data = state.get("b2") or b2_data(D)
    S, W = data["lbfgssol"], data["lbfgswarm"]
    common = sorted(set(S) & set(W))
    ratios = []
    for im in common:
        (ts, cs), (tw, cw) = S[im], W[im]
        if cs and cw:
            continue
        ratios.append(math.inf if cs else (0.0 if cw else ts / tw))
    if ratios:
        med = float(np.median(ratios))
        add("B3b", "cold L-BFGS solved reaches Adam-50k quality within 20 % of warm L-BFGS random's time",
            "median paired time ratio solved/warm (censored solved = inf, censored warm = 0, both censored dropped)",
            f"{med:.2f}, n={len(ratios)} (+{len(common) - len(ratios)} both censored)",
            ("pass" if med <= 1.2 else "fail") + ("" if len(common) >= 24 else " (partial)"))
    else:
        add("B3b", "cold solved within 20 % of warm L-BFGS", "", "needs lbfgssol and lbfgswarm with adam50k",
            "not yet measurable")
    msgs, oks, nmin = [], [], 6
    for sol, learned in (("adamsol", ("adaminitlbfgs", "adaminitadam")), ("lbfgssol", ("lbfgsinitlbfgs", "lbfgsinitadam"))):
        ms = censored_median([data[sol][i] for i in HELD_OUT if i in data[sol]])
        for l in learned:
            ml = censored_median([data[l][i] for i in HELD_OUT if i in data[l]])
            nmin = min(nmin, sum(i in data[sol] for i in HELD_OUT), sum(i in data[l] for i in HELD_OUT))
            if np.isfinite(ms[0]) and np.isfinite(ml[0]):
                slower = (ms[0] > ml[0] and not ml[1]) or (ms[1] and not ml[1])
                msgs.append(f"{SHORT[sol].replace(chr(10), ' ')} {'>' if ms[1] else ''}{ms[0]:.0f}s vs "
                            f"{SHORT[l].replace(chr(10), ' ')} {'>' if ml[1] else ''}{ml[0]:.0f}s")
                oks.append(slower)
    if msgs:
        add("B3c", "both solved arms slower to Adam-50k quality than the learned starts on the six images",
            "censoring-aware median seconds on the held-out images", "; ".join(msgs),
            ("pass" if all(oks) else "fail") + ("" if len(oks) == 4 and nmin == 6 else " (partial)"))
    else:
        add("B3c", "solved slower than learned starts", "", "needs solved and learned arms on the six", "not yet measurable")

    # C1..C4
    w = [r for r in drows if r["tensor"].endswith("weight")]
    if w:
        first = [r["rel_drift"] for r in w if layer_index(r["tensor"]) == 0]
        hidden = [r["rel_drift"] for r in w if 1 <= layer_index(r["tensor"]) <= 9]
        outl = [r["rel_drift"] for r in w if layer_index(r["tensor"]) == 10]
        inr = np.mean([(0.3 <= h <= 0.7) for h in hidden])
        ok = max(first) < 0.05 and 0.3 <= np.median(hidden) <= 0.7 and inr >= 0.8 and min(outl) > 1.0
        nruns = len({r["run"] for r in w})
        add("C1", "first-layer drift < 5 % on every fit; hidden 30-70 %; output > 100 %",
            "first max; hidden median and share in [0.3, 0.7] (pass ≥ 80 %); output min",
            f"first max {max(first):.4f}; hidden median {np.median(hidden):.3f}, {100 * inr:.0f} % in range; "
            f"output min {min(outl):.3f}; {nruns} fits", ("pass" if ok else "fail") + " (partial)" * (nruns < n_drift_expected))
        f = [1 / r["max_delta_over_max_weight"] for r in first_rows(drows) if r["max_delta_over_max_weight"] > 0]
        inr = np.mean([(15 <= x <= 40) for x in f])
        add("C2", "first-layer delta grid 15-40 times finer than the weight grid",
            "median max|W|/max|W-W0| and share in [15, 40] (pass: median in range and ≥ 80 % in range)",
            f"median {np.median(f):.1f}, range [{min(f):.1f}, {max(f):.1f}], {100 * inr:.0f} % in range, n={len(f)}",
            ("pass" if 15 <= np.median(f) <= 40 and inr >= 0.8 else "fail") + " (partial)" * (nruns < n_drift_expected))
        dd = [r["psnr_good8"] - r["psnr_delta8_first"] for r in first_rows(drows) if np.isfinite(r["psnr_delta8_first"])]
        if dd:
            add("C3", "delta-coding the first layer at 8 bits within 0.05 dB of the 16-bit exception",
                "max over fits of |loss(delta8_first) - loss(good8)|",
                f"max {max(abs(x) for x in dd):.4f} dB, mean {np.mean(dd):+.4f}, n={len(dd)}",
                ("pass" if max(abs(x) for x in dd) <= 0.05 else "fail") + " (partial)" * (len(dd) < n_drift_expected))
        else:
            add("C3", "delta first layer within 0.05 dB", "", "no delta PSNRs yet", "not yet measurable")
    else:
        for pid, claim in (("C1", "drift thresholds"), ("C2", "delta grid 15-40x finer"), ("C3", "delta first layer within 0.05 dB")):
            add(pid, claim, "", "no drift rows yet", "not yet measurable")
    pairs = state.get("c3") or {}
    msgs, oks, n = [], [], 0
    for name, lim in (("Adam", 0.3), ("warm L-BFGS", 0.1)):
        ps = pairs.get(name) or []
        if ps:
            cost = [r - f for _, f, r, _, _ in ps]
            msgs.append(f"{name}: mean {np.mean(cost):.3f}, max {max(cost):.3f} dB (limit {lim}), n={len(ps)}")
            oks.append(np.mean(cost) < lim)
            n += len(ps)
    if msgs:
        add("C4", "freezing the first layer at the random start costs < 0.3 dB (Adam), < 0.1 dB (L-BFGS) at 360 s",
            "mean paired free - frozen PSNR at 360 s", "; ".join(msgs),
            ("pass" if all(oks) and len(oks) == 2 else "fail" if not all(oks) else "partial") + " (partial)" * (n < 48))
    else:
        add("C4", "freezing costs < 0.3 / 0.1 dB", "", "no frozen/free pairs yet", "not yet measurable")
    O.table("Prediction checks (section 1)", md_table(["#", "prediction", "measure", "value", "verdict"], P),
            "'(partial)' = verdict on the runs pulled so far, not the full set; re-run when the pods finish. "
            "Thresholds that the plan states only in words are made explicit in the measure column.")
    return P


# ---------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join("runs", "dec1"))
    p.add_argument("--out", default=os.path.join("figures", "dec1"))
    p.add_argument("--jobs", default="jobs")
    p.add_argument("--no-drift", action="store_true", help="use the drift caches as they are, compute nothing new")
    args = p.parse_args()
    os.chdir(HERE)
    selftest()
    D = Data(args.src, args.jobs)
    O = Out(args.out)
    print(f"pulled: {len(D.fits)} fits, {len(D.qat)} QAT runs, reference {'yes' if D.ref else 'no'}")
    state = {}
    arows = a_rows(D)
    for f in (figure_a1, figure_a2, figure_a3, figure_a4):
        f(D, O, arows)
    figure_b1(D, O, state)
    figure_b2(D, O, state)
    b_tables(D, O, state)
    figure_b3(D, O, state)
    b_recs = [r for (a, _, _), r in sorted(D.fits.items()) if a in B_ARMS]
    c4_recs = [r for (a, _, _), r in sorted(D.fits.items()) if a == "c4lbfgs"]
    drows = []
    if b_recs or os.path.exists(os.path.join(args.src, "drift_b.csv")):
        drows += drift_cache(os.path.join(args.src, "drift_b.csv"), b_recs, not args.no_drift)
    if c4_recs or os.path.exists(os.path.join(args.src, "drift_c4.csv")):
        drows += drift_cache(os.path.join(args.src, "drift_c4.csv"), c4_recs, not args.no_drift)
    figure_c1(D, O, drows)
    figure_c2(D, O, drows)
    figure_c3(D, O, state)
    figure_c4(D, O, drows)
    P = predictions(D, O, arows, state, drows)

    head = (f"# Dec-1 analysis tables\n\nGenerated by `analyse_dec1.py` from `{args.src}`. Re-run to refresh.\n\n"
            f"## Pull status\n\n{D.status()}\n\nFigures rendered: {', '.join(O.rendered) or 'none'}. "
            f"Skipped: {'; '.join(f'{a} ({b})' for a, b in O.skipped) or 'none'}.\n")
    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write(head + "\n" + "\n".join(O.tables[:-1]) + "\n" + O.tables[-1])
    with open(os.path.join(args.out, "captions_dec1.md"), "w") as f:
        f.write("# Dec-1 figure captions\n\n" + "\n\n".join(O.captions.values()) + "\n")
    print("\n" + md_table(["#", "prediction", "measure", "value", "verdict"], P))
    print(f"\nrendered {O.rendered}; skipped {[s[0] for s in O.skipped]}; tables and captions in {args.out}")


if __name__ == "__main__":
    main()
