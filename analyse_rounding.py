"""Per-tensor sensitivity to rounding and a quadratic-model prediction of the rounding loss,
computed on the CPU from saved weights. Usage: python analyse_rounding.py runs/f_adam_im1 ..."""
import json, sys
from collections import OrderedDict
import torch
import util
from crops import load_crop
from siren import Siren

torch.manual_seed(0)

def build(cfg):
    return Siren(2, cfg["layer_size"], 3, cfg["num_layers"], final_activation=torch.nn.Identity(),
                 w0_initial=cfg["w0_initial"], w0=cfg["w0"])

def quant(w, bits):
    levels = 2 ** (bits - 1) - 1
    s = w.abs().max().clamp_min(1e-12)
    return torch.round(w / s * levels) * s / levels, s / levels  # value, step

for tag in sys.argv[1:]:
    d = json.load(open(tag + ".json")); cfg = d["config"]
    state = torch.load(tag + ".pt", map_location="cpu")
    img, coords, feats = load_crop(cfg["image"], cfg["crop"], torch.device("cpu"))
    m = build(cfg); m.load_state_dict(state)
    with torch.no_grad():
        base = torch.nn.functional.mse_loss(m(coords), feats).item()
    names = [n for n, _ in m.named_parameters()]
    params = list(m.parameters())

    # per-tensor Hessian traces from shared Rademacher probes (block-diagonal Hutchinson)
    loss = torch.nn.functional.mse_loss(m(coords), feats)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    tr = torch.zeros(len(params))
    probes = 8
    for _ in range(probes):
        vs = [torch.randint(0, 2, p.shape).float() * 2 - 1 for p in params]
        hv = torch.autograd.grad(grads, params, grad_outputs=vs, retain_graph=True)
        tr += torch.tensor([(h * v).sum().item() for h, v in zip(hv, vs)])
    tr /= probes

    # per-tensor rounding at 8 bits: round one tensor, keep the rest exact
    print(f"\n== {tag}: image {cfg['image']} {cfg['optimizer']}  base psnr {util.psnr(feats, m(coords)):.2f}  total trace {tr.sum():.0f}")
    print(f"{'tensor':28} {'n':>5} {'max|w|':>7} {'step8':>8} {'trace':>9} {'pred dMSE':>10} {'meas dMSE':>10} {'dB alone':>8}")
    pred_total = 0.0
    for i, (n, p) in enumerate(zip(names, params)):
        q, step = quant(p.data, 8)
        st = OrderedDict(state); st[n] = q
        mq = build(cfg); mq.load_state_dict(st)
        with torch.no_grad():
            mse_q = torch.nn.functional.mse_loss(mq(coords), feats).item()
        pred = 0.5 * (step.item() ** 2 / 12) * tr[i].item()
        pred_total += pred
        print(f"{n:28} {p.numel():5d} {p.abs().max():7.3f} {step.item():8.1e} {tr[i].item():9.0f} {pred:10.2e} {mse_q - base:10.2e} {10*torch.log10(torch.tensor(base/mse_q)).item():8.2f}")
    # all tensors rounded at once
    st = OrderedDict((n, quant(w, 8)[0]) for n, w in state.items())
    mq = build(cfg); mq.load_state_dict(st)
    with torch.no_grad():
        mse_all = torch.nn.functional.mse_loss(mq(coords), feats).item()
    print(f"all at 8 bits: predicted dMSE {pred_total:.2e}, measured {mse_all - base:.2e}, base MSE {base:.2e}, "
          f"measured drop {10*torch.log10(torch.tensor(base/mse_all)).item():.2f} dB, predicted drop {10*torch.log10(torch.tensor(base/(base+pred_total))).item():.2f} dB")
