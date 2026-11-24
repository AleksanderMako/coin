"""Blind tests for the 2026-12-01 code (plan section 2, items 1 to 6): repair.py, qat.py, the delta
rounding in rounding.py, drift.py, load_image / --dataset in crops.py and experiment.py, and
make_jobs_dec1.py. Written from the spec only, against the interfaces of commit 719b54d.

Usage (from the repository root):
    python test_dec1.py            quick suite, CPU, about one to two minutes
    python test_dec1.py --full     also the three full-image repair acceptance numbers, the gate of the
                                   whole plan (runs/night1/k_lbfgs_s1_im23 full repair 0.53 and good rule
                                   1.63 unclamped; runs/night1/k_adam_s1_im23 good rule 0.57 clamped)
    python test_dec1.py -k drift   only the tests whose name contains "drift" (any substring)
    DEC1_FULL=1 pytest test_dec1.py   the same under pytest (the env var replaces --full)

Needs: runs/night1/{k_adam_s1_im1,k_adam_s1_im23,k_lbfgs_s1_im23}.{json,pt} (read only) and
kodak-dataset/. Every output goes to a temporary directory; nothing is written into runs/.
A synthetic DIV2K image is written to a temporary div2k/DIV2K_valid_HR/ which is used as the
working directory; if the implementation resolves div2k/ relative to the repository instead,
a temporary symlink <repo>/div2k is created for the test and removed afterwards (only if no
div2k/ exists there already).

Helper: _ref_link4.py holds the function definitions of the recovered reference script
recovered-2026-11-24-scripts/link4_data.py (os.chdir removed). The 719b54d versions of
quantise / round_state / rounding_table / load_crop are embedded below as _old_* references.

Conventions used here: PSNR is clamped (util.get_clamped_psnr) unless a name says unclamped. The CSV
header of drift.py is compared after splitting on commas and stripping spaces, so both "a,b" and
"a, b" are accepted. Exits non-zero if any test fails; skips are reported and do not fail the suite.
"""
import csv
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, OrderedDict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import util  # noqa: E402
from siren import Siren  # noqa: E402

FULL = os.environ.get("DEC1_FULL", "") not in ("", "0") or "--full" in sys.argv
CPU = torch.device("cpu")
PY = sys.executable
SIX = (1, 5, 8, 14, 20, 23)
K_ADAM_IM1 = "runs/night1/k_adam_s1_im1"
K_ADAM_IM23 = "runs/night1/k_adam_s1_im23"
K_LBFGS_IM23 = "runs/night1/k_lbfgs_s1_im23"
DRIFT_COLUMNS = ["run", "dataset", "image", "optimizer", "seed", "init", "tensor", "numel", "rel_drift",
                 "max_delta_over_max_weight", "psnr_fp32", "psnr_good8", "psnr_delta8_first", "psnr_delta8_all"]


# ----------------------------------------------------------------------------------------------
# small framework: plain script, also collectable by pytest
# ----------------------------------------------------------------------------------------------
class Skip(Exception):
    pass


def skip(msg):
    if "pytest" in sys.modules and not __name__ == "__main__":
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def need_full():
    if not FULL:
        skip("full-image acceptance test: run with --full or DEC1_FULL=1")


_TMP = []


@__import__("atexit").register
def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


def tmpdir():
    d = tempfile.mkdtemp(prefix="dec1_")
    _TMP.append(d)
    return d


def run(args, cwd=ROOT, timeout=1800):
    r = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"command failed ({r.returncode}): {' '.join(args)}\n{out[-4000:]}"
    return out


def build_from_config(cfg):
    return Siren(dim_in=2, dim_hidden=cfg.get("layer_size", 28), dim_out=3, num_layers=cfg.get("num_layers", 10),
                 final_activation=torch.nn.Identity(), w0_initial=cfg.get("w0_initial", 30.0), w0=cfg.get("w0", 30.0))


def start_from_seed(cfg):
    """experiment.py seeds, then builds the network on the CPU before anything else draws random numbers."""
    torch.manual_seed(cfg.get("seed", 0))
    return OrderedDict((k, v.clone()) for k, v in build_from_config(cfg).state_dict().items())


def psnr_fn(cfg, coords, feats, clamped=True):
    def f(state):
        m = build_from_config(cfg)
        m.load_state_dict(state)
        with torch.no_grad():
            pred = m(coords)
        return util.get_clamped_psnr(feats, pred) if clamped else util.psnr(feats, pred)
    return f


def load_run(path):
    cfg = json.load(open(path + ".json"))
    state = torch.load(path + ".pt", map_location="cpu")
    return cfg, state


def state_equal(a, b, atol=0.0):
    assert list(a.keys()) == list(b.keys()), (list(a.keys()), list(b.keys()))
    for k in a:
        if atol == 0.0:
            if not torch.equal(a[k], b[k]):
                return False, k, (a[k] - b[k]).abs().max().item()
        elif not torch.allclose(a[k], b[k], rtol=0.0, atol=atol):
            return False, k, (a[k] - b[k]).abs().max().item()
    return True, None, 0.0


# ----------------------------------------------------------------------------------------------
# references from commit 719b54d, verbatim
# ----------------------------------------------------------------------------------------------
_OLD_BIT_DEPTHS = (16, 12, 10, 8)


def _old_quantise(w, bits, per_channel=False):
    if bits >= 32:
        return w
    levels = 2 ** (bits - 1) - 1
    if per_channel and w.dim() == 2:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        scale = w.abs().max().clamp_min(1e-12)
    return torch.round(w / scale * levels) * scale / levels


def _old_round_state(state, bits, per_channel=False, first_bits=None):
    out = OrderedDict()
    for k, w in state.items():
        b = first_bits if (first_bits and k.startswith("net.0.")) else bits
        out[k] = _old_quantise(w, b, per_channel)
    return out


def _old_rounding_table(state, psnr_of_state):
    table = OrderedDict()
    table["fp32"] = psnr_of_state(state)
    table["fp16_cast"] = psnr_of_state(OrderedDict((k, v.half().float()) for k, v in state.items()))
    for bits in _OLD_BIT_DEPTHS:
        table[f"uniform{bits}"] = psnr_of_state(_old_round_state(state, bits))
        table[f"channel{bits}"] = psnr_of_state(_old_round_state(state, bits, per_channel=True))
        table[f"uniform{bits}_first16"] = psnr_of_state(_old_round_state(state, bits, first_bits=16))
        table[f"channel{bits}_first16"] = psnr_of_state(_old_round_state(state, bits, per_channel=True, first_bits=16))
    return table


def _old_load_crop(image_id, scale, device, dtype=torch.float32):
    import imageio.v2 as imageio
    from torchvision import transforms
    img = transforms.ToTensor()(imageio.imread(f"kodak-dataset/kodim{str(image_id).zfill(2)}.png")).float()
    if scale != 1:
        _, h, w = img.shape
        ch, cw = h // scale, w // scale
        top, left = (h - ch) // 2, (w - cw) // 2
        img = img[:, top:top + ch, left:left + cw].contiguous()
    img = img.to(device, dtype)
    c, f = util.to_coordinates_and_features(img)
    return img, c.to(device, dtype), f.to(device, dtype)


def _crop(image, scale):
    from crops import load_crop
    return load_crop(image, scale, CPU)


# ----------------------------------------------------------------------------------------------
# shared fixtures (built once, lazily)
# ----------------------------------------------------------------------------------------------
_CACHE = {}


def tiny_fit():
    """A small converging fit made by experiment.py: image 1, crop 8 (96 x 64), 16 x 4 SIREN."""
    if "tiny" not in _CACHE:
        d = tmpdir()
        run(["experiment.py", "--image", "1", "--crop", "8", "--layer_size", "16", "--num_layers", "4",
             "--optimizer", "lbfgs", "--warmup_iters", "20", "--history", "20", "--num_iters", "30",
             "--probes", "2", "--power_iters", "5", "--out", d, "--tag", "tiny"])
        _CACHE["tiny"] = os.path.join(d, "tiny")
    return _CACHE["tiny"]


def k_lbfgs_crop8_run():
    """The real 28 x 10 fit k_lbfgs_s1_im23 re-labelled as a crop-8 run (config crop 8), with its rounding
    table recomputed on that crop, so the repair CLI runs in seconds."""
    if "kcrop" not in _CACHE:
        d = tmpdir()
        cfg, state = load_run(K_LBFGS_IM23)
        cfg = {"config": dict(cfg["config"], crop=8, tag="kcrop")}
        _, coords, feats = _crop(23, 8)
        cfg["psnr_after_rounding"] = _old_rounding_table(state, psnr_fn(cfg["config"], coords, feats))
        p = os.path.join(d, "kcrop")
        json.dump(cfg, open(p + ".json", "w"))
        torch.save(state, p + ".pt")
        _CACHE["kcrop"] = p
    return _CACHE["kcrop"]


def import_impl(name):
    saved = sys.argv
    sys.argv = [name + ".py"]  # in case the module parses arguments at import time
    try:
        mod = __import__(name)
    finally:
        sys.argv = saved
    return mod


# ==============================================================================================
# item 3: rounding.py
# ==============================================================================================
def test_rounding_unchanged_without_new_arguments():
    """round_state and rounding_table without delta_from / start are bit for bit the 719b54d ones."""
    from rounding import quantise, round_state, rounding_table
    _, state = load_run(K_ADAM_IM1)
    torch.manual_seed(0)
    tiny = OrderedDict((k, v.clone()) for k, v in Siren(2, 8, 3, 3, final_activation=torch.nn.Identity()).state_dict().items())
    for st in (state, tiny):
        for bits in (2, 4, 6, 8, 10, 12, 16, 32):
            for pc in (False, True):
                for fb in (None, 16, 12):
                    ok, k, d = state_equal(round_state(st, bits, per_channel=pc, first_bits=fb),
                                           _old_round_state(st, bits, per_channel=pc, first_bits=fb))
                    assert ok, f"round_state(bits={bits}, per_channel={pc}, first_bits={fb}) changed on {k}: max diff {d}"
                    ok, k, d = state_equal(round_state(st, bits, pc, fb), _old_round_state(st, bits, pc, fb))
                    assert ok, f"positional round_state(bits={bits}, {pc}, {fb}) changed on {k}"
            for k, w in st.items():
                assert torch.equal(quantise(w, 8, True), _old_quantise(w, 8, True)), f"quantise changed on {k}"
    cfgj = json.load(open(K_ADAM_IM1 + ".json"))
    _, coords, feats = _crop(1, 1)
    fn = psnr_fn(cfgj["config"], coords, feats)
    new, old = rounding_table(state, fn), _old_rounding_table(state, fn)
    assert list(new.keys()) == list(old.keys()), f"keys changed without start: {list(new.keys())}"
    for k in old:
        assert new[k] == old[k], f"rounding_table[{k}] {new[k]} != 719b54d {old[k]}"
    js = cfgj["psnr_after_rounding"]
    worst = max(abs(new[k] - js[k]) for k in js)
    assert worst < 0.01, f"CPU table differs from the saved (cuda) table by {worst} dB"
    print(f"   18 entries identical to 719b54d; within {worst:.1e} dB of the saved JSON")


def test_rounding_delta_from_state_is_lossless():
    from rounding import round_state
    _, state = load_run(K_ADAM_IM1)
    for bits in (2, 3, 4, 8, 12, 16):
        for pc in (False, True):
            for fb in (None, 16):
                out = round_state(state, bits, per_channel=pc, first_bits=fb, delta_from=state)
                ok, k, d = state_equal(out, state, atol=1e-7)
                assert ok, f"delta_from=state not lossless at bits={bits} per_channel={pc} first_bits={fb}: {k} off by {d}"
    print("   delta_from=state returns the state at 2 to 16 bits, per tensor and per row")


def test_rounding_delta_from_semantics():
    """quantise(state - start) per row, then add the start back."""
    from rounding import quantise, round_state
    cfg, state = load_run(K_ADAM_IM1)
    start = start_from_seed(cfg["config"])
    for bits in (8, 4):
        out = round_state(state, bits, per_channel=True, delta_from=start)
        for k in state:
            exp = start[k] + quantise(state[k] - start[k], bits, per_channel=True)
            assert torch.allclose(out[k], exp, rtol=0, atol=1e-6), \
                f"{k} at {bits} bits: delta rounding differs by {(out[k] - exp).abs().max().item()}"
    # a delta of constant magnitude per row is exactly representable, so this is lossless
    syn = OrderedDict((k, v + 0.01 * torch.sign(v)) for k, v in start.items())
    ok, k, d = state_equal(round_state(syn, 8, per_channel=True, delta_from=start), syn, atol=1e-6)
    assert ok, f"start + c*sign(start) should round exactly as a delta: {k} off by {d}"
    print("   delta rounding = start + quantise(state - start, per row)")


def test_rounding_table_with_start():
    from rounding import quantise, rounding_table
    cfgj, state = load_run(K_ADAM_IM1)
    cfg = cfgj["config"]
    start = start_from_seed(cfg)
    _, coords, feats = _crop(1, 1)
    fn = psnr_fn(cfg, coords, feats)
    tab = rounding_table(state, fn, start=start)
    old = _old_rounding_table(state, fn)
    for k in old:
        assert k in tab and tab[k] == old[k], f"with start, existing key {k} changed: {tab.get(k)} vs {old[k]}"
    assert "delta8_first" in tab and "delta8_all" in tab, f"missing delta keys: {list(tab.keys())}"
    first = OrderedDict((k, start[k] + quantise(v - start[k], 8, True)) if k.startswith("net.0.")
                        else (k, quantise(v, 8, True)) for k, v in state.items())
    alld = OrderedDict((k, start[k] + quantise(v - start[k], 8, True)) for k, v in state.items())
    e_first, e_all = fn(first), fn(alld)
    assert abs(tab["delta8_first"] - e_first) < 1e-4, f"delta8_first {tab['delta8_first']} expected {e_first}"
    assert abs(tab["delta8_all"] - e_all) < 1e-4, f"delta8_all {tab['delta8_all']} expected {e_all}"
    print(f"   delta8_first {e_first:.3f} dB, delta8_all {e_all:.3f} dB, channel8_first16 {old['channel8_first16']:.3f} dB")


# ==============================================================================================
# item 1: repair.py
# ==============================================================================================
def _ref_repair(state, coords, feats, bits, resolve=True, lam=1e-2, damp=1e-2, first_bits=None):
    """The reference loop with damp passed through and an optional first-layer rounding before it.
    With damp=1e-2 and first_bits=None this is _ref_link4.repair verbatim."""
    import _ref_link4 as R
    from rounding import quantise
    z_exact, _ = R.preacts(state, coords)
    st = OrderedDict(state)
    if first_bits:
        for k in ("net.0.linear.weight", "net.0.linear.bias"):
            st[k] = quantise(state[k], first_bits, per_channel=True)
    for k in range(1, 11):
        _, hs = R.preacts(st, coords)
        X = hs[k]
        name = f"net.{k}.linear" if k < 10 else "last_layer.linear"
        W0, b0 = state[name + ".weight"], state[name + ".bias"]
        target = z_exact[k] if k < 10 else feats
        W, b = R.ridge_resolve(X, target, W0, b0, lam) if resolve else (W0, b0)
        st[name + ".weight"], st[name + ".bias"] = R.gptq_round(W, b, X, bits, damp)
    return st


def _compare_repair(out, ref, coords, feats, what):
    import _ref_link4 as R
    assert list(out.keys()) == list(ref.keys()), f"{what}: keys {list(out.keys())}"
    mism, total, worst = 0, 0, 0.0
    for k in ref:
        d = (out[k] - ref[k]).abs()
        worst = max(worst, d.max().item())
        mism += (d > 1e-5).sum().item()
        total += d.numel()
    p_out, p_ref = R.psnr_of(out, coords, feats), R.psnr_of(ref, coords, feats)
    assert abs(p_out - p_ref) < 1e-3, f"{what}: unclamped PSNR {p_out} vs reference {p_ref}"
    # identical code gives identical tensors; allow a stray flipped rounding decision but no more
    assert mism <= 0.01 * total, f"{what}: {mism}/{total} entries differ from the reference by > 1e-5 (max {worst})"
    return worst, p_out, p_ref


def test_repair_matches_reference_on_crop():
    import _ref_link4 as R
    rep = import_impl("repair")
    _, state = load_run(K_LBFGS_IM23)
    _, coords, feats = _crop(23, 8)
    keep = OrderedDict((k, v.clone()) for k, v in state.items())
    for kwargs in ({}, {"resolve": False}, {"lam": 1e-1}, {"damp": 1e-1}):
        for bits in (8, 6):
            out = rep.repair(state, coords, feats, bits, **kwargs)
            assert isinstance(out, dict), type(out)
            if "damp" in kwargs:
                ref = _ref_repair(state, coords, feats, bits, **kwargs)
            else:
                ref = R.repair(state, coords, feats, bits, **kwargs)
            worst, p, pr = _compare_repair(out, ref, coords, feats, f"repair(bits={bits}, {kwargs})")
            for k in ("net.0.linear.weight", "net.0.linear.bias"):
                assert torch.equal(out[k], state[k]), f"first_bits=None must leave {k} untouched"
            ok, k, _ = state_equal(state, keep)
            assert ok, f"repair modified its input state ({k})"
        print(f"   {kwargs or 'defaults'}: matches the reference (max diff {worst:.1e}, {p:.3f} vs {pr:.3f} dB at 6 bits)")
    # defaults are resolve=True, lam=1e-2, damp=1e-2, first_bits=None
    a = rep.repair(state, coords, feats, 8)
    b = rep.repair(state, coords, feats, 8, resolve=True, lam=1e-2, damp=1e-2, first_bits=None)
    assert state_equal(a, b)[0], "explicit default arguments change the result"


def test_repair_first_bits_16():
    """first layer rounded at 16 bits per row before the loop; the rest re-solved as before. Whether the
    targets come from the exact network or the one with the rounded first layer is not fixed by the spec
    (it says 'the exact network'), so only the PSNR is compared, to 0.01 dB."""
    import _ref_link4 as R
    from rounding import quantise
    rep = import_impl("repair")
    _, state = load_run(K_LBFGS_IM23)
    _, coords, feats = _crop(23, 8)
    out = rep.repair(state, coords, feats, 8, first_bits=16)
    for k in ("net.0.linear.weight", "net.0.linear.bias"):
        exp = quantise(state[k], 16, per_channel=True)
        assert torch.allclose(out[k], exp, rtol=0, atol=1e-7), f"{k} is not rounded at 16 bits per row"
    ref = _ref_repair(state, coords, feats, 8, first_bits=16)
    p, pr = R.psnr_of(out, coords, feats), R.psnr_of(ref, coords, feats)
    assert abs(p - pr) < 0.01, f"first_bits=16: {p} dB vs expected {pr} dB"
    print(f"   first layer at 16 bits per row; {p:.3f} dB vs expected {pr:.3f} dB")


def repair_cli(args_bits, runs_, first_bits=None):
    """The spec's form: python repair.py --bits 8 6 [--first_bits 16] <run> ... (note that a plain argparse
    nargs='+' on --bits swallows the run paths; the implementation must still accept this order)."""
    args = ["repair.py", "--bits"] + [str(b) for b in args_bits]
    if first_bits:
        args += ["--first_bits", str(first_bits)]
    try:
        return run(args + list(runs_))
    except AssertionError as e:
        alt = ["repair.py"] + list(runs_) + args[1:]
        r = subprocess.run([PY] + alt, cwd=ROOT, capture_output=True, text=True)
        hint = (" (the run-first order 'repair.py <run> --bits ...' does work)" if r.returncode == 0 else "")
        raise AssertionError(f"the spec's command line failed{hint}:\n{e}")


def _floats(text):
    return [float(x) for x in re.findall(r"-?\d+\.\d+", text)]


def test_repair_cli():
    import _ref_link4 as R
    from rounding import round_state
    path = k_lbfgs_crop8_run()
    _, state = load_run(path)
    _, coords, feats = _crop(23, 8)
    out = repair_cli([8, 6], [path])
    low = out.lower()
    for word in ("crude", "good", "compens", "repair"):
        assert word in low, f"repair.py output does not mention '{word}':\n{out[-2000:]}"
    assert "kcrop" in out, f"repair.py output does not name the run:\n{out[-2000:]}"
    assert re.search(r"sec|\d\s*s\b", low), f"repair.py output shows no seconds:\n{out[-2000:]}"
    base = R.psnr_of(state, coords, feats)
    nums = _floats(out)
    for bits in (8, 6):
        exp_rep = base - R.psnr_of(R.repair(state, coords, feats, bits), coords, feats)
        exp_good = base - R.psnr_of(round_state(state, bits, per_channel=True, first_bits=16), coords, feats)
        for name, v in (("full repair", exp_rep), ("good rule", exp_good)):
            assert any(abs(x - v) < 0.006 for x in nums), \
                f"{bits} bits: no printed number matches the unclamped {name} loss {v:.3f} dB:\n{out[-2000:]}"
    out16 = repair_cli([8], [path], first_bits=16)
    assert "repair" in out16.lower()
    print("   prints crude/good/compensation/repair and seconds for 8 and 6 bits; --first_bits 16 runs")


def test_full_repair_acceptance():
    """THE GATE. Full images, first_bits=None. k_lbfgs_s1_im23: full repair 0.53 +- 0.03 dB and good rule
    1.63 +- 0.03 dB unclamped; k_adam_s1_im23: good rule 0.57 +- 0.03 dB clamped (= JSON fp32 - channel8_first16)."""
    need_full()
    import _ref_link4 as R
    from rounding import round_state
    rep = import_impl("repair")
    _, coords, feats = _crop(23, 1)
    _, s_l = load_run(K_LBFGS_IM23)
    base = R.psnr_of(s_l, coords, feats)
    t = time.perf_counter()
    rep_loss = base - R.psnr_of(rep.repair(s_l, coords, feats, 8), coords, feats)
    secs = time.perf_counter() - t
    good_loss = base - R.psnr_of(round_state(s_l, 8, per_channel=True, first_bits=16), coords, feats)
    assert abs(rep_loss - 0.53) <= 0.03, f"k_lbfgs_s1_im23 full repair {rep_loss:.3f} dB, expected 0.53 +- 0.03"
    assert abs(good_loss - 1.63) <= 0.03, f"k_lbfgs_s1_im23 good rule {good_loss:.3f} dB, expected 1.63 +- 0.03"
    cj, s_a = load_run(K_ADAM_IM23)
    fn = psnr_fn(cj["config"], coords, feats)
    good_a = fn(s_a) - fn(round_state(s_a, 8, per_channel=True, first_bits=16))
    pr = cj["psnr_after_rounding"]
    assert abs(good_a - 0.57) <= 0.03, f"k_adam_s1_im23 good rule {good_a:.3f} dB clamped, expected 0.57 +- 0.03"
    assert abs(good_a - (pr["fp32"] - pr["channel8_first16"])) < 0.01, (good_a, pr["fp32"] - pr["channel8_first16"])
    out = repair_cli([8], [K_LBFGS_IM23])
    nums = _floats(out)
    assert any(abs(x - 0.53) <= 0.03 for x in nums), f"repair.py CLI prints no full-repair loss near 0.53:\n{out[-2000:]}"
    assert any(abs(x - 1.63) <= 0.03 for x in nums), f"repair.py CLI prints no good-rule loss near 1.63:\n{out[-2000:]}"
    print(f"   k_lbfgs_s1_im23: repair {rep_loss:.3f} dB ({secs:.1f} s), good {good_loss:.3f} dB unclamped; "
          f"k_adam_s1_im23: good {good_a:.3f} dB clamped")


# ==============================================================================================
# item 2: qat.py
# ==============================================================================================
QAT_KEYS = ("config", "psnr_start", "psnr_latent_fp32", "psnr_quantised", "lambda_max", "hessian_trace",
            "seconds", "iters", "ms_per_iter", "rounding_table")


def qat(fit, rule, iters, out, tag, lam="0.1", bits="8", extra=()):
    run(["qat.py", "--fit", fit, "--rule", rule, "--bits", bits, "--lam", lam, "--iters", str(iters),
         "--lr", "2e-4", "--probes", "1", "--power_iters", "2", "--tag", tag, "--out", out] + list(extra))
    p = os.path.join(out, tag)
    for suffix in (".json", ".pt", "_q.pt"):
        assert os.path.exists(p + suffix), f"qat.py did not write {tag}{suffix}"
    s = json.load(open(p + ".json"))
    missing = [k for k in QAT_KEYS if k not in s]
    assert not missing, f"{tag}.json lacks {missing}"
    assert s["iters"] == iters, f"iters {s['iters']} != {iters}"
    return s, torch.load(p + ".pt", map_location="cpu"), torch.load(p + "_q.pt", map_location="cpu")


def _rule_file(latent, rule, bits=8):
    from rounding import round_state
    return round_state(latent, bits, per_channel=True, first_bits=16) if rule == "good" else round_state(latent, bits)


def test_qat_iters0_reproduces_rounding_table():
    d = tmpdir()
    cj, fit_state = load_run(K_ADAM_IM1)
    pr = cj["psnr_after_rounding"]
    _, coords, feats = _crop(1, 1)
    fn = psnr_fn(cj["config"], coords, feats)
    for rule, key in (("good", "channel8_first16"), ("crude", "uniform8")):
        s, latent, q = qat(K_ADAM_IM1, rule, 0, d, f"z_{rule}")
        assert abs(s["psnr_quantised"] - pr[key]) < 0.01, f"{rule}: psnr_quantised {s['psnr_quantised']} vs JSON {key} {pr[key]}"
        assert abs(s["psnr_start"] - pr["fp32"]) < 0.01, f"psnr_start {s['psnr_start']} vs JSON fp32 {pr['fp32']}"
        assert abs(s["psnr_latent_fp32"] - pr["fp32"]) < 0.01, f"iters 0: latent {s['psnr_latent_fp32']} vs fp32 {pr['fp32']}"
        ok, k, dd = state_equal(latent, fit_state)
        assert ok, f"iters 0: latent weights differ from the fit on {k} by {dd}"
        ok, k, dd = state_equal(q, _rule_file(latent, rule), atol=1e-7)
        assert ok, f"{rule}: _q.pt is not the {rule}-rule rounding of the latent weights ({k}, {dd})"
        assert abs(fn(q) - s["psnr_quantised"]) < 1e-3, f"psnr_quantised {s['psnr_quantised']} vs PSNR of _q.pt {fn(q)}"
        assert abs(s["rounding_table"]["fp32"] - s["psnr_latent_fp32"]) < 1e-3
        assert abs(s["rounding_table"][key] - pr[key]) < 0.01, f"rounding_table[{key}] {s['rounding_table'][key]}"
        assert np.isfinite(s["lambda_max"]) and np.isfinite(s["hessian_trace"]), (s["lambda_max"], s["hessian_trace"])
        print(f"   {rule}: psnr_quantised {s['psnr_quantised']:.3f} vs JSON {key} {pr[key]:.3f}")


def test_qat_saved_files_are_rule_files():
    """After a few steps: _q.pt is exactly the good/crude rule applied to the latent .pt (first layer of the good
    rule = quantise(latent, 16, per_channel=True)), and the JSON PSNRs are those of the saved files."""
    from rounding import quantise
    fit = tiny_fit()
    cj, start = load_run(fit)
    _, coords, feats = _crop(1, 8)
    fn = psnr_fn(cj["config"], coords, feats)
    d = tmpdir()
    for rule in ("good", "crude"):
        s, latent, q = qat(fit, rule, 5, d, f"r_{rule}")
        ok, k, dd = state_equal(q, _rule_file(latent, rule), atol=1e-7)
        assert ok, f"{rule}: _q.pt is not the {rule}-rule file of the latent weights ({k}, max diff {dd})"
        if rule == "good":
            for k in ("net.0.linear.weight", "net.0.linear.bias"):
                assert torch.allclose(q[k], quantise(latent[k], 16, per_channel=True), rtol=0, atol=1e-7), k
        assert abs(fn(q) - s["psnr_quantised"]) < 1e-3, f"{rule}: psnr_quantised {s['psnr_quantised']} vs file {fn(q)}"
        assert abs(fn(latent) - s["psnr_latent_fp32"]) < 1e-3, f"{rule}: psnr_latent_fp32 {s['psnr_latent_fp32']} vs file {fn(latent)}"
        assert abs(fn(start) - s["psnr_start"]) < 1e-3, f"{rule}: psnr_start {s['psnr_start']} vs fit {fn(start)}"
        assert s["ms_per_iter"] and s["ms_per_iter"] > 0 and s["seconds"] > 0, (s["ms_per_iter"], s["seconds"])
        print(f"   {rule}: start {s['psnr_start']:.2f}, latent {s['psnr_latent_fp32']:.2f}, quantised {s['psnr_quantised']:.2f} dB")


def test_qat_latent_moves_after_one_iteration():
    fit = tiny_fit()
    _, start = load_run(fit)
    d = tmpdir()
    for rule in ("good", "crude"):
        _, latent, _ = qat(fit, rule, 1, d, f"one_{rule}")
        moved = sum((latent[k] - start[k]).abs().sum().item() for k in start)
        assert moved > 0, f"{rule}: latent weights equal the start after one iteration"
    print(f"   latent weights moved after one step (sum |delta| {moved:.3g})")


def test_qat_bits32_lam0_keeps_psnr():
    """--lam 0 --bits 32 --iters 200 on a quarter crop: no lower than the start by more than 0.05 dB."""
    if "q4" not in _CACHE:
        d0 = tmpdir()
        run(["experiment.py", "--image", "1", "--crop", "4", "--layer_size", "16", "--num_layers", "4",
             "--optimizer", "lbfgs", "--warmup_iters", "50", "--history", "20", "--num_iters", "100",
             "--probes", "2", "--power_iters", "5", "--out", d0, "--tag", "q4"])
        _CACHE["q4"] = os.path.join(d0, "q4")
    d = tmpdir()
    s, _, _ = qat(_CACHE["q4"], "good", 200, d, "b32", lam="0", bits="32")
    assert s["psnr_latent_fp32"] >= s["psnr_start"] - 0.05, (s["psnr_start"], s["psnr_latent_fp32"])
    assert s["psnr_quantised"] >= s["psnr_start"] - 0.05, (s["psnr_start"], s["psnr_quantised"])
    print(f"   start {s['psnr_start']:.3f} -> latent {s['psnr_latent_fp32']:.3f}, quantised {s['psnr_quantised']:.3f} dB")


_COUNTER = r'''
import os, runpy, sys
root, argv = sys.argv[1], sys.argv[2:]
sys.path.insert(0, root)
os.chdir(root)
import torch, siren
n = {"grad": 0, "nograd": 0}
orig = siren.Siren.forward
def forward(self, *a, **k):
    n["grad" if torch.is_grad_enabled() else "nograd"] += 1
    return orig(self, *a, **k)
siren.Siren.forward = forward
sys.argv = [os.path.join(root, "qat.py")] + argv
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    print("FORWARD_COUNT", n["grad"], n["nograd"], flush=True)
'''


def test_qat_teacher_computed_once():
    """ASSUMPTION: qat.py builds its network from siren.Siren and runs it through Siren.forward (directly or via
    torch.func.functional_call). Every forward call is counted by patching the class before qat.py runs.
    With lam=0.1 a per-step teacher would add one forward per iteration over lam=0; computed once, the per-step
    count is the same for both and lam=0.1 adds at most one forward in total. Skips if no forward is seen."""
    fit = tiny_fit()
    d = tmpdir()
    script = os.path.join(d, "count_forwards.py")
    open(script, "w").write(_COUNTER)
    counts = {}
    for lam in ("0", "0.1"):
        for iters in (2, 6):
            r = subprocess.run([PY, script, ROOT, "--fit", fit, "--rule", "good", "--bits", "8", "--lam", lam,
                                "--iters", str(iters), "--lr", "2e-4", "--probes", "1", "--power_iters", "1",
                                "--tag", f"c_{lam}_{iters}", "--out", d], capture_output=True, text=True, timeout=600)
            m = re.search(r"FORWARD_COUNT (\d+) (\d+)", r.stdout)
            assert r.returncode == 0 and m, f"qat.py under the forward counter failed:\n{(r.stdout + r.stderr)[-3000:]}"
            counts[(lam, iters)] = int(m.group(1)) + int(m.group(2))
    if all(v == 0 for v in counts.values()):
        skip("qat.py never calls siren.Siren.forward; the teacher count cannot be observed this way")
    slope0 = (counts[("0", 6)] - counts[("0", 2)]) / 4
    slope1 = (counts[("0.1", 6)] - counts[("0.1", 2)]) / 4
    assert slope1 >= 1, f"no training forward per iteration seen ({counts})"
    assert slope1 == slope0, f"lam=0.1 costs {slope1} forwards per step, lam=0 {slope0}: teacher recomputed per step? {counts}"
    for it in (2, 6):
        extra = counts[("0.1", it)] - counts[("0", it)]
        assert 0 <= extra <= 1, f"lam=0.1 adds {extra} forwards at {it} iterations (expected 0 or 1): {counts}"
    print(f"   forwards per step {slope1:g} for both lam=0 and lam=0.1; teacher adds "
          f"{counts[('0.1', 2)] - counts[('0', 2)]} in total")


# ==============================================================================================
# item 4: drift.py
# ==============================================================================================
def read_drift_csv(path):
    text = open(path).read()
    lines = [l for l in text.splitlines() if l.strip()]
    header = [h.strip() for h in lines[0].split(",")]
    assert header == DRIFT_COLUMNS, f"drift.csv header is\n{header}\nexpected\n{DRIFT_COLUMNS}"
    rows = []
    for rec in csv.reader(io.StringIO("\n".join(lines[1:])), skipinitialspace=True):
        assert len(rec) == len(header), f"row with {len(rec)} fields: {rec}"
        rows.append({h: v.strip() for h, v in zip(header, rec)})
    return rows


def _rows_for(rows, name):
    rs = [r for r in rows if os.path.basename(r["run"].rstrip("/")).replace(".json", "").replace(".pt", "") == name
          or r["run"].rstrip("/").endswith(name)]
    assert rs, f"no rows for run {name} (run column values: {sorted(set(r['run'] for r in rows))})"
    return rs


def _by_tensor(rs, state):
    names = [r["tensor"] for r in rs]
    assert len(rs) == len(state), f"{len(rs)} rows for a run with {len(state)} tensors"
    if set(names) == set(state.keys()):
        return {r["tensor"]: r for r in rs}
    print(f"   note: tensor column {names[:3]}... does not use state-dict keys; matching rows by order")
    return dict(zip(state.keys(), rs))


def test_drift():
    from rounding import quantise
    d = tmpdir()
    tiny_cfg = dict(layer_size=16, num_layers=4)
    # (a) weights equal the seeded start: a fit with zero iterations
    run(["experiment.py", "--image", "1", "--crop", "8", "--layer_size", "16", "--num_layers", "4", "--num_iters", "0",
         "--seed", "3", "--probes", "1", "--power_iters", "2", "--out", d, "--tag", "r_equal_seed"])
    # (b) weights equal a saved --init start
    torch.manual_seed(11)
    init = OrderedDict((k, v.clone()) for k, v in Siren(2, 16, 3, 4, final_activation=torch.nn.Identity()).state_dict().items())
    init_path = os.path.join(d, "init_start.pt")
    torch.save(init, init_path)
    run(["experiment.py", "--image", "1", "--crop", "8", "--layer_size", "16", "--num_layers", "4", "--num_iters", "0",
         "--init", init_path, "--probes", "1", "--power_iters", "2", "--out", d, "--tag", "r_equal_init"])
    # (c) start + c * sign(start), config of (a)
    cfg_a = json.load(open(os.path.join(d, "r_equal_seed.json")))
    start_a = start_from_seed(cfg_a["config"])
    _, st_a = load_run(os.path.join(d, "r_equal_seed"))
    assert state_equal(st_a, start_a)[0], "test premise: experiment.py --num_iters 0 should save the seeded start"
    c = 0.01
    syn = OrderedDict((k, v + c * torch.sign(v)) for k, v in start_a.items())
    torch.save(syn, os.path.join(d, "r_sign.pt"))
    json.dump({"config": dict(cfg_a["config"], tag="r_sign")}, open(os.path.join(d, "r_sign.json"), "w"))
    csv_path = os.path.join(d, "drift.csv")
    runs_ = [os.path.join(d, n) for n in ("r_equal_seed", "r_equal_init", "r_sign")] + [K_ADAM_IM1]
    run(["drift.py", "--out", csv_path] + runs_)
    rows = read_drift_csv(csv_path)
    assert len(rows) == 3 * 10 + 22, f"{len(rows)} rows, expected one per tensor (10 + 10 + 10 + 22)"
    _, coords, feats = _crop(1, 8)
    fn = psnr_fn(dict(tiny_cfg), coords, feats)

    for name, st in (("r_equal_seed", st_a), ("r_equal_init", init)):
        rs = _by_tensor(_rows_for(rows, name), st)
        r0 = next(iter(rs.values()))
        for k, r in rs.items():
            assert float(r["rel_drift"]) == 0.0, f"{name} {k}: rel_drift {r['rel_drift']}"
            assert float(r["max_delta_over_max_weight"]) == 0.0, f"{name} {k}: ratio {r['max_delta_over_max_weight']}"
            assert int(float(r["numel"])) == st[k].numel(), f"{name} {k}: numel {r['numel']}"
            for col in ("psnr_fp32", "psnr_good8", "psnr_delta8_first", "psnr_delta8_all"):
                assert r[col] == r0[col], f"{name}: {col} not repeated per tensor"
        fp32 = fn(st)
        good = fn(OrderedDict((k, quantise(v, 16 if k.startswith("net.0.") else 8, True)) for k, v in st.items()))
        first = fn(OrderedDict((k, v if k.startswith("net.0.") else quantise(v, 8, True)) for k, v in st.items()))
        assert abs(float(r0["psnr_fp32"]) - fp32) < 1e-3, (name, r0["psnr_fp32"], fp32)
        assert abs(float(r0["psnr_good8"]) - good) < 1e-3, (name, r0["psnr_good8"], good)
        # zero delta: the delta-coded tensors are exact; delta8_all loses nothing, delta8_first only the 8-bit rest
        assert abs(float(r0["psnr_delta8_all"]) - fp32) < 1e-4, f"{name}: delta8_all {r0['psnr_delta8_all']} != fp32 {fp32}"
        assert abs(float(r0["psnr_delta8_first"]) - first) < 1e-3, f"{name}: delta8_first {r0['psnr_delta8_first']} vs {first}"
        assert r0["dataset"] == "kodak" and int(float(r0["image"])) == 1, (r0["dataset"], r0["image"])
    ri = next(iter(_rows_for(rows, "r_equal_init")))
    assert "init_start" in ri["init"], f"init column of an --init run: {ri['init']!r}"
    print("   weights == start (seeded and --init): rel_drift 0, delta8_all lossless, PSNR columns as expected")

    rs = _by_tensor(_rows_for(rows, "r_sign"), syn)
    for k, r in rs.items():
        exp = c / (start_a[k].abs().max().item() + c)
        got = float(r["max_delta_over_max_weight"])
        assert abs(got - exp) <= 1e-4 * exp + 1e-7, f"r_sign {k}: max_delta_over_max_weight {got}, expected {exp}"
        exp_rel = (c * torch.sign(start_a[k])).norm().item() / start_a[k].norm().item()
        assert abs(float(r["rel_drift"]) - exp_rel) <= 1e-4 * exp_rel, f"r_sign {k}: rel_drift {r['rel_drift']} vs {exp_rel}"
    r0 = next(iter(rs.values()))
    assert abs(float(r0["psnr_delta8_all"]) - float(r0["psnr_fp32"])) < 1e-3, "a constant-magnitude delta rounds exactly"
    print(f"   start + {c} sign(start): ratio c/(max|start|+c) per tensor, e.g. first layer {float(rs['net.0.linear.weight']['max_delta_over_max_weight']):.4f}")

    # (d) a real saved fit, seed 1, full image: start rebuilt from the seed in its config
    cj, st = load_run(K_ADAM_IM1)
    start = start_from_seed(cj["config"])
    rs = _by_tensor(_rows_for(rows, "k_adam_s1_im1"), st)
    for k, r in rs.items():
        exp = (st[k] - start[k]).norm().item() / start[k].norm().item()
        assert abs(float(r["rel_drift"]) - exp) <= 1e-3 * exp, f"k_adam_s1_im1 {k}: rel_drift {r['rel_drift']} vs {exp}"
    r0 = next(iter(rs.values()))
    pr = cj["psnr_after_rounding"]
    assert abs(float(r0["psnr_fp32"]) - pr["fp32"]) < 0.01 and abs(float(r0["psnr_good8"]) - pr["channel8_first16"]) < 0.01, \
        (r0["psnr_fp32"], r0["psnr_good8"], pr["fp32"], pr["channel8_first16"])
    assert r0["optimizer"] == "adam" and int(float(r0["seed"])) == 1 and r0["dataset"] == "kodak", r0
    print(f"   k_adam_s1_im1: first-layer rel_drift {float(rs['net.0.linear.weight']['rel_drift']):.4f}, "
          f"good8 {float(r0['psnr_good8']):.2f} dB (JSON {pr['channel8_first16']:.2f})")


# ==============================================================================================
# item 5: crops.py and experiment.py
# ==============================================================================================
def test_load_image_kodak_equals_load_crop():
    import crops
    assert hasattr(crops, "load_crop"), "load_crop must stay as a Kodak alias"
    for image, scale in ((1, 8), (23, 4), (5, 1)):
        a = crops.load_image("kodak", image, scale, CPU)
        b = crops.load_crop(image, scale, CPU)
        o = _old_load_crop(image, scale, CPU)
        for x, y, z, what in zip(a, b, o, ("img", "coords", "feats")):
            assert torch.equal(x, y), f"load_image('kodak', {image}, {scale}) {what} != load_crop"
            assert torch.equal(y, z), f"load_crop({image}, {scale}) {what} differs from 719b54d"
    print("   load_image('kodak', ...) == load_crop == 719b54d load_crop on three crops")


def _write_png(path, w, h, seed):
    import imageio.v2 as imageio
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, arr)
    return arr


def div2k_dir():
    """A temporary directory holding div2k/DIV2K_valid_HR/{0801,0802,0803}.png: 2000 x 1300, 700 x 500 and
    exactly 768 x 512."""
    if "div2k" not in _CACHE:
        d = tmpdir()
        hr = os.path.join(d, "div2k", "DIV2K_valid_HR")
        arrs = {801: _write_png(os.path.join(hr, "0801.png"), 2000, 1300, 1),
                802: _write_png(os.path.join(hr, "0802.png"), 700, 500, 2),
                803: _write_png(os.path.join(hr, "0803.png"), 768, 512, 3)}
        _CACHE["div2k"] = (d, arrs)
    return _CACHE["div2k"]


class _Div2kRoot:
    """cwd = the temporary dir; with repo_link=True also a temporary <repo>/div2k symlink to it."""
    def __init__(self, d, repo_link=False):
        self.d, self.link, self.made = d, repo_link, False

    def __enter__(self):
        self.cwd = os.getcwd()
        os.chdir(self.d)
        target = os.path.join(ROOT, "div2k")
        if self.link and not os.path.lexists(target):
            os.symlink(os.path.join(self.d, "div2k"), target)
            self.made = True
        return self

    def __exit__(self, *a):
        os.chdir(self.cwd)
        if self.made:
            os.remove(os.path.join(ROOT, "div2k"))


def _load_div2k(index):
    import crops
    d, _ = div2k_dir()
    try:
        with _Div2kRoot(d):
            return crops.load_image("div2k", index, 1, CPU)
    except FileNotFoundError:
        if os.path.lexists(os.path.join(ROOT, "div2k")):
            raise
        print("   note: div2k/ is resolved relative to the repository, not the working directory")
        with _Div2kRoot(d, repo_link=True):
            return crops.load_image("div2k", index, 1, CPU)


def test_load_image_div2k_centre_crop():
    _, arrs = div2k_dir()
    img, coords, feats = _load_div2k(801)
    assert tuple(img.shape) == (3, 512, 768), f"DIV2K crop shape {tuple(img.shape)}, expected (3, 512, 768)"
    top, left = (1300 - 512) // 2, (2000 - 768) // 2
    exp = torch.from_numpy(arrs[801][top:top + 512, left:left + 768]).permute(2, 0, 1).float() / 255.0
    assert torch.allclose(img, exp, rtol=0, atol=1e-6), \
        f"not the centre crop without resampling (max diff {(img - exp).abs().max().item()})"
    c, f = util.to_coordinates_and_features(img)
    assert torch.equal(coords, c) and torch.equal(feats, f), "coordinates/features differ from util.to_coordinates_and_features"
    img3, _, _ = _load_div2k(803)
    assert torch.allclose(img3, torch.from_numpy(arrs[803]).permute(2, 0, 1).float() / 255.0, rtol=0, atol=1e-6)
    try:
        _load_div2k(802)
    except FileNotFoundError:
        raise AssertionError("a too-small image must raise a clear size error, not FileNotFoundError")
    except Exception as e:  # noqa: BLE001
        assert str(e).strip(), "the too-small error has no message"
        print(f"   2000x1300 -> 768x512 centre crop; 768x512 kept whole; 700x500 raises: {type(e).__name__}: {str(e)[:80]}")
        return
    raise AssertionError("load_image('div2k', 802) on a 700 x 500 image did not raise")


# recorded with the 719b54d experiment.py, torch 2.14.1, CPU (deterministic: two runs agreed bit for bit)
_EXP_BASELINE = {
    "adam": (["--image", "1", "--crop", "8", "--layer_size", "8", "--num_layers", "3", "--num_iters", "30",
              "--probes", "2", "--power_iters", "5"],
             dict(final_psnr=7.290891408920288, ch8=7.290887832641602, steps=31, lambda_max=30.97699546813965)),
    "lbfgs": (["--image", "1", "--crop", "8", "--layer_size", "8", "--num_layers", "3", "--optimizer", "lbfgs",
               "--warmup_iters", "5", "--history", "20", "--num_iters", "5", "--probes", "2", "--power_iters", "5",
               "--seed", "2"],
              dict(final_psnr=13.997658491134644, ch8=13.994536399841309, steps=11, lambda_max=106.66827392578125)),
}


def test_experiment_without_dataset_unchanged():
    d = tmpdir()
    same_torch = torch.__version__.startswith("2.14.1")
    for name, (args, exp) in _EXP_BASELINE.items():
        run(["experiment.py"] + args + ["--out", d, "--tag", name])
        s = json.load(open(os.path.join(d, name + ".json")))
        assert s["config"].get("dataset") == "kodak", f"config dataset {s['config'].get('dataset')!r}, expected 'kodak'"
        assert s["pixels"] == 96 * 64, s["pixels"]
        if same_torch:
            assert s["steps"] == exp["steps"], (name, s["steps"], exp["steps"])
            assert abs(s["final_psnr"] - exp["final_psnr"]) < 1e-4, (name, s["final_psnr"], exp["final_psnr"])
            assert abs(s["psnr_after_rounding"]["channel8_first16"] - exp["ch8"]) < 1e-4, name
            assert abs(s["lambda_max"] - exp["lambda_max"]) < 1e-3 * exp["lambda_max"], (name, s["lambda_max"])
        r2 = run(["experiment.py"] + args + ["--dataset", "kodak", "--out", d, "--tag", name + "_k"])
        s2 = json.load(open(os.path.join(d, name + "_k.json")))
        assert s2["final_psnr"] == s["final_psnr"], "--dataset kodak differs from the default"
    print("   no --dataset: config dataset 'kodak'" + (", results equal to 719b54d" if same_torch else
          " (torch version differs from the recording; numbers not compared)"))


def test_experiment_div2k():
    d, _ = div2k_dir()
    out = tmpdir()
    args = ["--dataset", "div2k", "--image", "801", "--crop", "1", "--layer_size", "8", "--num_layers", "3",
            "--num_iters", "3", "--probes", "1", "--power_iters", "2", "--out", out, "--tag", "dv"]
    script = os.path.join(ROOT, "experiment.py")
    r = subprocess.run([PY, script] + args, cwd=d, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 and not os.path.lexists(os.path.join(ROOT, "div2k")):
        with _Div2kRoot(d, repo_link=True):
            r = subprocess.run([PY, script] + args, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    s = json.load(open(os.path.join(out, "dv.json")))
    assert s["config"]["dataset"] == "div2k" and s["config"]["image"] == 801, s["config"]
    assert s["pixels"] == 768 * 512, f"{s['pixels']} pixels, expected 768 x 512"
    print("   experiment.py --dataset div2k --image 801 --crop 1 fits a 768 x 512 image; config records div2k")


# ==============================================================================================
# item 6: make_jobs_dec1.py
# ==============================================================================================
EXP_DEFAULTS = {"seed": "0", "warmup_iters": "0", "max_seconds": "0", "crop": "2", "history": "20", "lr": "2e-4",
                "dataset": "kodak", "init": "", "sharpness_every": "0"}


def parse_line(line):
    toks = shlex.split(line)
    idx = [i for i, t in enumerate(toks) if t.endswith(".py")]
    assert idx, f"no script in job line: {line}"
    script = os.path.basename(toks[idx[0]])
    flags, i, rest = {}, idx[0] + 1, toks
    while i < len(rest):
        t = rest[i]
        assert t.startswith("--"), f"stray token {t!r} in: {line}"
        if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            assert t[2:] not in flags, f"flag {t} given twice in: {line}"
            flags[t[2:]] = rest[i + 1]
            i += 2
        else:
            flags[t[2:]] = True
            i += 1
    return script, flags


def read_jobs(path):
    assert os.path.exists(path), f"missing job file {os.path.basename(path)}"
    lines = [l.strip() for l in open(path) if l.strip() and not l.strip().startswith("#")]
    return [(l,) + parse_line(l) for l in lines]


def fget(f, k):
    v = f.get(k, EXP_DEFAULTS.get(k))
    return v


def num(v):
    return float(v) if v not in (None, True, "") else 0.0


def classify_b(line, f):
    opt = f.get("optimizer")
    solve, init = f.get("solve_last") is True, fget(f, "init")
    ni, ms = num(f.get("num_iters")), num(fget(f, "max_seconds"))
    warm, hist = num(fget(f, "warmup_iters")), num(fget(f, "history"))
    if opt == "adam" and ni == 50000:
        if not solve and not init and ms == 0:
            return "adam_ref"
        if solve and not init and ms == 600:
            return "adam_solved"
        if not solve and init and ms == 600:
            return "adam_learned"
    if opt == "lbfgs" and ni == 100000 and hist == 100 and ms == 600:
        if warm == 500 and not solve and not init:
            return "lbfgs_warm"
        if warm == 0 and solve and not init:
            return "lbfgs_cold_solved"
        if warm == 0 and not solve and init:
            return "lbfgs_learned"
    raise AssertionError(f"B line matches no arm of the table in section 1B: {line}")


def check_fit_line(line, script, f, what):
    assert script == "experiment.py", f"{what} line does not call experiment.py: {line}"
    m = re.fullmatch(r"d1_.+_s(\d+)_im(\d+)", f.get("tag", ""))
    assert m, f"{what} tag does not follow d1_<arm>_s<seed>_im<image>: {line}"
    assert int(m.group(1)) == int(num(fget(f, "seed"))), f"tag seed differs from --seed: {line}"
    assert int(m.group(2)) == int(f["image"]), f"tag image differs from --image: {line}"
    assert num(fget(f, "lr")) == 2e-4, f"{what}: lr is not 2e-4: {line}"
    assert int(num(fget(f, "crop"))) == 1, f"{what}: not a full image (--crop 1 missing; experiment.py defaults to 2): {line}"
    assert "sharpness_every" in f, f"{what}: no --sharpness_every: {line}"


def make_jobs(extra=()):
    d = tmpdir()
    run(["make_jobs_dec1.py", "--out", d] + list(extra))
    return d


def test_make_jobs_dec1():
    d = make_jobs()
    files = {"b": [f"dec1_b_{k}.txt" for k in range(1, 5)], "c3": [f"dec1_c3_{k}.txt" for k in range(1, 5)],
             "a": ["dec1_a_1.txt", "dec1_a_2.txt"], "c4": ["dec1_c4_1.txt", "dec1_c4_2.txt"], "ref": ["dec1_qat_ref.txt"]}
    jobs = {g: [read_jobs(os.path.join(d, n)) for n in names] for g, names in files.items()}
    counts = {g: sum(len(x) for x in v) for g, v in jobs.items()}
    assert counts == {"b": 132, "c3": 48, "a": 108, "c4": 100, "ref": 1}, f"job counts {counts}"
    assert [len(x) for x in jobs["a"]] == [54, 54], [len(x) for x in jobs["a"]]
    assert [len(x) for x in jobs["c4"]] == [50, 50], [len(x) for x in jobs["c4"]]
    every = [j for v in jobs.values() for x in v for j in x]
    tags = [f.get("tag") for _, _, f in every]
    assert all(isinstance(t, str) and t for t in tags), "a line without --tag"
    dup = [t for t, n in Counter(tags).items() if n > 1]
    assert not dup, f"duplicate tags across files: {dup[:10]}"
    for line, script, f in every:
        assert f.get("out") == "runs", f"line without --out runs: {line}"
        if f.get("init"):
            assert int(f.get("image", 0)) in SIX, f"--init arm on image outside the six held-out ones: {line}"
            assert f.get("dataset", "kodak") == "kodak", line

    # B: the table of section 1B
    keys, img_sets = Counter(), []
    for x in jobs["b"]:
        imgs, refs = set(), 0
        for line, script, f in x:
            check_fit_line(line, script, f, "B")
            arm = classify_b(line, f)
            assert fget(f, "dataset") == "kodak" and "freeze_first" not in f, line
            want = "500" if f["optimizer"] == "adam" else "50"
            assert num(f["sharpness_every"]) == int(want), f"B {f['optimizer']} needs --sharpness_every {want}: {line}"
            im = int(f["image"])
            imgs.add(im)
            refs += arm == "adam_ref"
            if arm.endswith("learned"):
                keys[(arm, im, os.path.normpath(f["init"]))] += 1
            else:
                keys[(arm, im, int(num(fget(f, "seed"))))] += 1
        img_sets.append(imgs)
        assert refs == 6, f"Adam-50k jobs not spread evenly: {refs} in one B file"
    for a in range(4):
        for b in range(a + 1, 4):
            assert not img_sets[a] & img_sets[b], f"B files {a + 1} and {b + 1} share images {img_sets[a] & img_sets[b]}"
    assert set().union(*img_sets) == set(range(1, 25)), sorted(set().union(*img_sets))
    expect = Counter()
    for im in range(1, 25):
        for arm in ("adam_ref", "adam_solved", "lbfgs_warm", "lbfgs_cold_solved"):
            expect[(arm, im, 0)] += 1
    for im in SIX:
        expect[("adam_solved", im, 1)] += 1
        expect[("lbfgs_cold_solved", im, 1)] += 1
        for start in ("runs/night1/reptile_lbfgs.pt", "runs/night1/reptile_adam.pt"):
            expect[("adam_learned", im, start)] += 1
            expect[("lbfgs_learned", im, start)] += 1
    assert keys == expect, f"B arms differ from section 1B: missing {dict(expect - keys)}, extra {dict(keys - expect)}"

    # C3: frozen first layer, same image split as B
    c3keys = Counter()
    for k, x in enumerate(jobs["c3"]):
        imgs = set()
        for line, script, f in x:
            check_fit_line(line, script, f, "C3")
            assert f.get("freeze_first") is True and num(fget(f, "max_seconds")) == 360, f"C3 needs --freeze_first --max_seconds 360: {line}"
            assert not fget(f, "init") and "solve_last" not in f and int(num(fget(f, "seed"))) == 0, line
            opt = f["optimizer"]
            if opt == "adam":
                assert num(f["num_iters"]) == 50000 and num(f["sharpness_every"]) == 500, line
            else:
                assert opt == "lbfgs" and num(fget(f, "warmup_iters")) == 500 and num(fget(f, "history")) == 100 \
                    and num(f["num_iters"]) == 100000 and num(f["sharpness_every"]) == 50, line
            c3keys[(opt, int(f["image"]))] += 1
            imgs.add(int(f["image"]))
        assert imgs == img_sets[k], f"dec1_c3_{k + 1} images {sorted(imgs)} != dec1_b_{k + 1} images {sorted(img_sets[k])}"
    assert c3keys == Counter({(o, im): 1 for o in ("adam", "lbfgs") for im in range(1, 25)}), "C3 arms"

    # C4: DIV2K 801..900, warm L-BFGS by iteration count
    c4imgs = []
    for x in jobs["c4"]:
        for line, script, f in x:
            check_fit_line(line, script, f, "C4")
            assert f.get("dataset") == "div2k" and f.get("optimizer") == "lbfgs", line
            assert num(fget(f, "warmup_iters")) == 500 and num(fget(f, "history")) == 100, line
            assert num(f["num_iters"]) == 5000, f"C4 default --num_iters should be 5000: {line}"
            assert num(f["sharpness_every"]) == 0 and num(fget(f, "max_seconds")) == 0, line
            assert int(num(fget(f, "seed"))) == 0 and not fget(f, "init") and "freeze_first" not in f, line
            c4imgs.append(int(f["image"]))
    assert sorted(c4imgs) == list(range(801, 901)), "C4 images are not 801..900 once each"
    d2 = make_jobs(["--c4_iters", "1234"])
    for n in ("dec1_c4_1.txt", "dec1_c4_2.txt"):
        assert all(num(f["num_iters"]) == 1234 for _, _, f in read_jobs(os.path.join(d2, n))), "--c4_iters ignored"

    # A: QAT, 36 fits x {good lam0, good lam0.1, crude lam0.1}
    arms = Counter()
    fits = Counter()
    for x in jobs["a"]:
        for line, script, f in x:
            assert script == "qat.py", f"A line does not call qat.py: {line}"
            m = re.fullmatch(r"runs/fits/([^/]+)", f.get("fit", ""))
            assert m, f"A line's --fit is not runs/fits/<tag>: {line}"
            fit = m.group(1)
            rule, lam = f.get("rule"), num(f.get("lam"))
            assert (rule, lam) in (("good", 0.0), ("good", 0.1), ("crude", 0.1)), f"unexpected arm: {line}"
            assert num(f.get("bits")) == 8 and num(f.get("iters")) == 15000 and num(f.get("lr")) == 2e-4, line
            assert num(f.get("probes")) == 8 and num(f.get("power_iters")) == 20, line
            want_tag = f"d1_qat_{rule}_lam{'0' if lam == 0 else '0.1'}_{fit}"
            assert f["tag"] == want_tag, f"QAT tag {f['tag']}, expected {want_tag}"
            arms[(fit, rule, lam)] += 1
            fits[fit] += 1
    assert len(fits) == 36 and all(n == 3 for n in fits.values()), f"{len(fits)} fits, arms per fit {set(fits.values())}"
    assert all(n == 1 for n in arms.values()), "a QAT arm repeated"
    names = set(fits)
    k_names = {f"k_{o}_s1_im{i}" for o in ("adam", "lbfgs") for i in SIX}
    l_names = {f"llbfgs_adam_im{i}" for i in SIX} | {f"llbfgs_lbfgs_cold_im{i}" for i in SIX}
    assert k_names <= names, f"missing seed-1 fits {sorted(k_names - names)}"
    assert l_names <= names, f"missing learned-start fits {sorted(l_names - names)}"
    seed0 = names - k_names - l_names
    per = Counter()
    for n in seed0:
        m = re.search(r"im(\d+)(_plain)?$", n)
        assert m and int(m.group(1)) in SIX, f"seed-0 fit {n} not on one of the six images"
        per[(int(m.group(1)), "adam" if "adam" in n else "lbfgs" if "lbfgs" in n else "?")] += 1
    assert per == Counter({(i, o): 1 for i in SIX for o in ("adam", "lbfgs")}), f"seed-0 fits: {sorted(seed0)}"

    (line, script, f), = jobs["ref"][0]
    assert script == "qat.py" and os.path.basename(f.get("fit", "")) == "k_lbfgs_s1_im23", f"qat ref: {line}"
    assert f.get("rule") == "good" and num(f.get("lam")) == 0.1 and num(f.get("iters")) == 15000, f"qat ref: {line}"
    assert f.get("fit", "").startswith("runs/"), line
    print(f"   counts {counts}; B arms as in 1B; C3 split as B; C4 801..900; 36 QAT fits x 3 arms; tags unique")


# ==============================================================================================
# runner
# ==============================================================================================
TESTS = [
    test_rounding_unchanged_without_new_arguments,
    test_rounding_delta_from_state_is_lossless,
    test_rounding_delta_from_semantics,
    test_rounding_table_with_start,
    test_repair_matches_reference_on_crop,
    test_repair_first_bits_16,
    test_repair_cli,
    test_qat_iters0_reproduces_rounding_table,
    test_qat_saved_files_are_rule_files,
    test_qat_latent_moves_after_one_iteration,
    test_qat_bits32_lam0_keeps_psnr,
    test_qat_teacher_computed_once,
    test_drift,
    test_load_image_kodak_equals_load_crop,
    test_load_image_div2k_centre_crop,
    test_experiment_without_dataset_unchanged,
    test_experiment_div2k,
    test_make_jobs_dec1,
    test_full_repair_acceptance,
]


def main():
    sel = None
    if "-k" in sys.argv:
        sel = sys.argv[sys.argv.index("-k") + 1]
    print(f"test_dec1: {'full' if FULL else 'quick'} suite, torch {torch.__version__}, cwd {ROOT}")
    passed, failed, skipped = [], [], []
    t_all = time.perf_counter()
    try:
        for n, t in enumerate(TESTS, 1):
            if sel and sel not in t.__name__:
                continue
            t0 = time.perf_counter()
            print(f"{n:2d}. {t.__name__}", flush=True)
            try:
                t()
                passed.append(t.__name__)
                print(f"    ok ({time.perf_counter() - t0:.1f} s)", flush=True)
            except Skip as e:
                skipped.append((t.__name__, str(e)))
                print(f"    SKIP: {e}", flush=True)
            except Exception:  # noqa: BLE001
                failed.append(t.__name__)
                print("    FAIL\n" + "".join("      " + l for l in traceback.format_exc().splitlines(True)), flush=True)
    finally:
        for d in _TMP:
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped in {time.perf_counter() - t_all:.0f} s")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    for name in failed:
        print(f"  FAILED {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
