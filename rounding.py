"""Round saved weights to fewer bits and measure the quality that survives.

Per-tensor: one scale per tensor (COIN's crude baseline). Per-channel: one scale per output
row of each weight matrix, biases stay per-tensor. first_bits keeps the first layer, which
barely moves from its random start, at a higher depth than the rest.
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


def round_state(state, bits, per_channel=False, first_bits=None):
    out = OrderedDict()
    for k, w in state.items():
        b = first_bits if (first_bits and k.startswith("net.0.")) else bits
        out[k] = quantise(w, b, per_channel)
    return out


def rounding_table(state, psnr_of_state):
    """psnr_of_state(state) -> dB. Returns every scheme the thesis compares, keyed by name."""
    table = OrderedDict()
    table["fp32"] = psnr_of_state(state)
    table["fp16_cast"] = psnr_of_state(OrderedDict((k, v.half().float()) for k, v in state.items()))
    for bits in BIT_DEPTHS:
        table[f"uniform{bits}"] = psnr_of_state(round_state(state, bits))
        table[f"channel{bits}"] = psnr_of_state(round_state(state, bits, per_channel=True))
        table[f"uniform{bits}_first16"] = psnr_of_state(round_state(state, bits, first_bits=16))
        table[f"channel{bits}_first16"] = psnr_of_state(round_state(state, bits, per_channel=True, first_bits=16))
    return table
