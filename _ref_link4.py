"""Test helper for test_dec1.py: the function definitions of the recovered reference script
recovered-2026-11-24-scripts/link4_data.py, verbatim except that the os.chdir line is removed
(the module path is put on sys.path instead) and the data-collection loop after the functions is dropped.
"""
import json, sys, os, numpy as np, torch
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siren import Siren
from crops import load_crop
from rounding import round_state
import util
def build(): return Siren(2, 28, 3, 10, final_activation=torch.nn.Identity(), w0_initial=30.0, w0=30.0)
def psnr_of(st, coords, feats):
    m = build(); m.load_state_dict(st)
    with torch.no_grad(): return util.psnr(feats, m(coords))
def preacts(st, coords):
    m = build(); m.load_state_dict(st); zs, hs = [], []
    with torch.no_grad():
        h = coords
        for layer in m.net:
            hs.append(h); z = layer.linear(h); zs.append(z); h = torch.sin(30.0 * z)
        hs.append(h); zs.append(m.last_layer.linear(h))
    return zs, hs
def gptq_round(W, b, X, bits, damp=0.01):
    A = torch.cat([X, torch.ones(X.shape[0], 1)], 1); H = A.T @ A / A.shape[0]; H += damp * H.diagonal().mean() * torch.eye(H.shape[0])
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Wb = torch.cat([W, b[:, None]], 1).clone(); levels = 2 ** (bits - 1) - 1
    sw = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / levels; sb = b.abs().max().clamp_min(1e-12) / levels
    Q = torch.zeros_like(Wb); n = Wb.shape[1]
    for j in range(n):
        s = sb if j == n - 1 else sw; q = torch.round(Wb[:, j:j + 1] / s) * s; Q[:, j:j + 1] = q
        Wb[:, j:] -= ((Wb[:, j:j + 1] - q) / U[j, j]) @ U[j:j + 1, j:]
    return Q[:, :-1].contiguous(), Q[:, -1].contiguous()
def ridge_resolve(X, target, W0, b0, lam):
    A = torch.cat([X, torch.ones(X.shape[0], 1)], 1); N = A.shape[0]
    H = A.T @ A + lam * N * torch.eye(A.shape[1]); w0 = torch.cat([W0, b0[:, None]], 1).T
    sol = torch.linalg.solve(H, A.T @ target + lam * N * w0); return sol[:-1].T.contiguous(), sol[-1].contiguous()
def repair(state, coords, feats, bits, resolve=True, lam=1e-2):
    z_exact, _ = preacts(state, coords); st = OrderedDict(state)
    for k in range(1, 11):
        _, hs = preacts(st, coords); X = hs[k]; name = f'net.{k}.linear' if k < 10 else 'last_layer.linear'
        W0, b0 = state[name + '.weight'], state[name + '.bias']; target = z_exact[k] if k < 10 else feats
        W, b = ridge_resolve(X, target, W0, b0, lam) if resolve else (W0, b0)
        st[name + '.weight'], st[name + '.bias'] = gptq_round(W, b, X, bits)
    return st
