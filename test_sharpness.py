"""Blind checks for the new pieces on a tiny problem on the CPU: sharpness meter against the
exact Hessian, gradient descent stability, the flat-seeking finish, per-channel rounding,
and Reptile plus --init round trip through experiment.py."""
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict

import torch
from torch.func import functional_call

from crops import load_crop
from rounding import round_state
from sharpness import hessian_trace, top_eigenvalue
from siren import Siren

torch.manual_seed(0)
device = torch.device("cpu")
img, coords, feats = load_crop(1, 8, device)        # 96 x 64 crop
coords, feats = coords[:1000], feats[:1000]
mse = torch.nn.functional.mse_loss

model = Siren(2, 8, 3, 3, final_activation=torch.nn.Identity()).to(device)
# a few Adam steps so the Hessian is not the one at the random start
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(100):
    opt.zero_grad(); mse(model(coords), feats).backward(); opt.step()

# 1. power iteration against the exact top eigenvalue of the explicit Hessian; trace against the exact trace
names = [n for n, _ in model.named_parameters()]
shapes = [p.shape for _, p in model.named_parameters()]
numels = [p.numel() for _, p in model.named_parameters()]
flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def unflatten(f):
    out, i = {}, 0
    for n, s, k in zip(names, shapes, numels):
        out[n] = f[i:i + k].reshape(s); i += k
    return out


H = torch.autograd.functional.hessian(lambda f: mse(functional_call(model, unflatten(f), (coords,)), feats), flat)
eig = torch.linalg.eigvalsh(H)
exact_top = eig.abs().max().item()
exact_trace = torch.trace(H).item()
lam = top_eigenvalue(model, coords, feats, iters=60)
tr = hessian_trace(model, coords, feats, probes=256)
assert abs(abs(lam) - exact_top) / exact_top < 0.02, (lam, exact_top)
assert abs(tr - exact_trace) / abs(exact_trace) < 0.15, (tr, exact_trace)
assert torch.allclose(torch.cat([p.detach().reshape(-1) for p in model.parameters()]), flat)
print(f"1. top eigenvalue {lam:.4g} vs exact {exact_top:.4g}; trace {tr:.4g} vs exact {exact_trace:.4g}; weights untouched")

# 2. gradient descent lowers the loss below the stability limit and diverges far above it
def gd(lr, steps):
    m = Siren(2, 8, 3, 3, final_activation=torch.nn.Identity()); m.load_state_dict(model.state_dict())
    o = torch.optim.SGD(m.parameters(), lr=lr)
    l0 = mse(m(coords), feats).item()
    for _ in range(steps):
        o.zero_grad(); l = mse(m(coords), feats); l.backward(); o.step()
    return l0, mse(m(coords), feats).item()


stable = 1.0 / exact_top
l0, l1 = gd(stable, 200)
assert l1 < l0, (l0, l1)
_, l2 = gd(50.0 / exact_top, 200)
assert not (l2 < l0), l2
print(f"2. gd at 1/sharpness: {l0:.4g} -> {l1:.4g}; at 50/sharpness: not improving ({l2:.3g})")

# 3. the finish runs, keeps a finite loss, and reports its sharpness
state0 = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
def finish(rho, steps=100, lr=1e-3):
    m = Siren(2, 8, 3, 3, final_activation=torch.nn.Identity()); m.load_state_dict(state0)
    params = list(m.parameters()); o = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        o.zero_grad(); mse(m(coords), feats).backward()
        eps = []
        with torch.no_grad():
            for p in params:
                e = p.grad * (rho * p.norm() / (p.grad.norm() + 1e-12)); p.add_(e); eps.append(e)
        o.zero_grad(); mse(m(coords), feats).backward()
        with torch.no_grad():
            for p, e in zip(params, eps): p.sub_(e)
        o.step()
    return mse(m(coords), feats).item(), top_eigenvalue(m, coords, feats, 40)
l_plain, s_plain = finish(0.0)
l_flat, s_flat = finish(0.05)
assert l_flat == l_flat and l_flat < 1.0
print(f"3. finish: plain adam loss {l_plain:.4g} sharpness {s_plain:.4g}; flat-seeking loss {l_flat:.4g} sharpness {s_flat:.4g}")

# 4. per-channel rounding is never worse than per-tensor at equal bits; first-layer-at-16 never worse than all-8
def psnr_of(state):
    m = Siren(2, 8, 3, 3, final_activation=torch.nn.Identity()); m.load_state_dict(state)
    with torch.no_grad():
        return -10 * torch.log10(mse(m(coords), feats)).item()
for bits in (8, 6):
    u, c = psnr_of(round_state(state0, bits)), psnr_of(round_state(state0, bits, per_channel=True))
    f16 = psnr_of(round_state(state0, bits, first_bits=16))
    assert c >= u - 0.05, (bits, u, c)
    assert f16 >= u - 0.05, (bits, u, f16)
    print(f"4. {bits} bits: per-tensor {u:.2f} dB, per-channel {c:.2f} dB, first layer at 16 {f16:.2f} dB")

# 5. Reptile moves the start; experiment.py accepts it through --init and runs gd and the finish
with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "start.pt")
    r = subprocess.run([sys.executable, "reptile.py", "--images", "1", "2", "--crop", "8", "--inner_steps", "20",
                        "--epochs", "2", "--layer_size", "8", "--num_layers", "3", "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    theta = torch.load(out)
    torch.manual_seed(0)
    fresh = Siren(2, 8, 3, 3, final_activation=torch.nn.Identity()).state_dict()
    moved = sum((theta[k] - fresh[k]).abs().sum().item() for k in theta)
    assert moved > 0, "reptile did not move the start"
    r = subprocess.run([sys.executable, "reptile.py", "--images", "1", "--crop", "8", "--inner", "lbfgs", "--inner_steps", "5",
                        "--epochs", "1", "--layer_size", "8", "--num_layers", "3", "--out", os.path.join(d, "s2.pt")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for extra in (["--optimizer", "gd", "--lr", "1e-3", "--sharpness_every", "10"],
                  ["--optimizer", "adam", "--num_iters", "0", "--sam_steps", "10", "--sam_rho", "0.05"],
                  ["--optimizer", "lbfgs", "--warmup_iters", "5", "--num_iters", "5"]):
        r = subprocess.run([sys.executable, "experiment.py", "--image", "1", "--crop", "8", "--layer_size", "8",
                            "--num_layers", "3", "--init", out, "--num_iters", "30", "--out", d, "--tag", "t",
                            "--probes", "2", "--power_iters", "5"] + extra, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        import json
        s = json.load(open(os.path.join(d, "t.json")))
        assert s["lambda_max"] is not None and "channel8_first16" in s["psnr_after_rounding"]
        if "--sharpness_every" in extra:
            assert len(s["log"]["sharpness"]) >= 3
        os.remove(os.path.join(d, "t.json"))
    print(f"5. reptile moved the start by {moved:.3g} (sum abs); experiment.py ran gd, finish-only and lbfgs from it")
print("all checks passed")
