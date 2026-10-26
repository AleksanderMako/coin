"""Round saved weights with different bit depths per layer and measure quality and size.
Usage: python mixed_precision.py runs/f_adam_im1 runs/f_lbfgs_im1 ..."""
import json, sys
from collections import OrderedDict
import torch
import util
from crops import load_crop
from siren import Siren

def build(cfg):
    return Siren(2, cfg["layer_size"], 3, cfg["num_layers"], final_activation=torch.nn.Identity(),
                 w0_initial=cfg["w0_initial"], w0=cfg["w0"])

def quant(w, bits):
    if bits >= 32: return w
    levels = 2 ** (bits - 1) - 1
    s = w.abs().max().clamp_min(1e-12)
    return torch.round(w / s * levels) * s / levels

def scheme(names, first, hidden_early, hidden_late, last, split=4):
    """Bits per tensor: first layer, early hidden layers (index < split), late hidden layers, last layer."""
    out = {}
    for n in names:
        if n.startswith("net.0."): out[n] = first
        elif n.startswith("last_layer"): out[n] = last
        else:
            idx = int(n.split(".")[1]); out[n] = hidden_early if idx < split else hidden_late
    return out

print(f"{'run':20} {'scheme (first/early/late/last)':32} {'bits/pixel':>10} {'psnr':>6} {'drop':>6}")
for tag in sys.argv[1:]:
    d = json.load(open(tag + ".json")); cfg = d["config"]
    state = torch.load(tag + ".pt", map_location="cpu")
    img, coords, feats = load_crop(cfg["image"], cfg["crop"], torch.device("cpu"))
    pixels = coords.shape[0]
    names = list(state.keys())
    m = build(cfg); m.load_state_dict(state)
    with torch.no_grad(): base = util.psnr(feats, m(coords))
    for label, bits in [("32 all", scheme(names, 32, 32, 32, 32)),
                        ("16 all", scheme(names, 16, 16, 16, 16)),
                        ("8 all", scheme(names, 8, 8, 8, 8)),
                        ("16 / 8 / 8 / 8", scheme(names, 16, 8, 8, 8)),
                        ("16 / 8 / 7 / 8", scheme(names, 16, 8, 7, 8)),
                        ("16 / 9 / 7 / 8", scheme(names, 16, 9, 7, 8)),
                        ("16 / 8 / 6 / 8", scheme(names, 16, 8, 6, 8)),
                        ("16 / 7 / 6 / 8", scheme(names, 16, 7, 6, 8)),
                        ("16 / 6 / 6 / 8", scheme(names, 16, 6, 6, 8)),
                        ("6 all", scheme(names, 6, 6, 6, 6))]:
        st = OrderedDict((n, quant(w, bits[n])) for n, w in state.items())
        mq = build(cfg); mq.load_state_dict(st)
        with torch.no_grad(): p = util.psnr(feats, mq(coords))
        total_bits = sum(state[n].numel() * bits[n] for n in names)
        print(f"{tag.split('/')[-1]:20} {label:32} {total_bits / pixels:10.3f} {p:6.2f} {p - base:+6.2f}")
    print()
