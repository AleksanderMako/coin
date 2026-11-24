"""Round saved weights to fewer bits and measure the quality that survives.

Per-tensor: one scale per tensor (COIN's crude baseline). Per-channel: one scale per output
row of each weight matrix, biases stay per-tensor. first_bits keeps the first layer, which
barely moves from its random start, at a higher depth than the rest. delta_from stores each
tensor as its (rounded) difference from a known start, e.g. a seed the decoder can regenerate.
"""
from collections import OrderedDict

import torch

BIT_DEPTHS = (16, 12, 10, 8)


def quantise(w, bits, per_channel=False):
    if bits >= 32:
        return w
    levels = 2 ** (bits - 1) - 1
    if per_channel and w.dim() == 2:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        scale = w.abs().max().clamp_min(1e-12)
    return torch.round(w / scale * levels) * scale / levels


def round_state(state, bits, per_channel=False, first_bits=None, delta_from=None):
    """With delta_from (a state with the same keys, e.g. the fit's start), each tensor is stored as
    its difference from delta_from: the difference is rounded (scales, per row or per tensor, come
    from the difference) and delta_from is added back."""
    out = OrderedDict()
    for k, w in state.items():
        b = first_bits if (first_bits and k.startswith("net.0.")) else bits
        if delta_from is None:
            out[k] = quantise(w, b, per_channel)
        else:
            base = delta_from[k].to(w.device, w.dtype)
            out[k] = base + quantise(w - base, b, per_channel)
    return out


def delta_first_state(state, start, bits=8):
    """First layer as a per-row delta from its start at `bits`; every other tensor per row at `bits`."""
    out = round_state(state, bits, per_channel=True)
    first = OrderedDict((k, v) for k, v in state.items() if k.startswith("net.0."))
    out.update(round_state(first, bits, per_channel=True, delta_from=start))
    return out


def rounding_table(state, psnr_of_state, start=None):
    """psnr_of_state(state) -> dB. Returns every scheme the thesis compares, keyed by name.
    With start (the weights the fit began from), also delta8_first (first layer as a per-row delta
    from its start at 8 bits, the rest per row at 8 bits) and delta8_all (every tensor as a delta)."""
    table = OrderedDict()
    table["fp32"] = psnr_of_state(state)
    table["fp16_cast"] = psnr_of_state(OrderedDict((k, v.half().float()) for k, v in state.items()))
    for bits in BIT_DEPTHS:
        table[f"uniform{bits}"] = psnr_of_state(round_state(state, bits))
        table[f"channel{bits}"] = psnr_of_state(round_state(state, bits, per_channel=True))
        table[f"uniform{bits}_first16"] = psnr_of_state(round_state(state, bits, first_bits=16))
        table[f"channel{bits}_first16"] = psnr_of_state(round_state(state, bits, per_channel=True, first_bits=16))
    if start is not None:
        table["delta8_first"] = psnr_of_state(delta_first_state(state, start, 8))
        table["delta8_all"] = psnr_of_state(round_state(state, 8, per_channel=True, delta_from=start))
    return table
