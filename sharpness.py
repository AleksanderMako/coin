"""Curvature of the mean squared error at the current weights, from Hessian-vector products only.

top_eigenvalue: power iteration, the sharpness in the edge-of-stability sense.
hessian_trace: Hutchinson estimate with Rademacher probes, the average curvature times the parameter count.
Neither touches the weights or their .grad fields.
"""
import torch


def _loss_and_grads(model, coords, feats):
    params = [p for p in model.parameters()]
    loss = torch.nn.functional.mse_loss(model(coords), feats)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return params, grads


def _hvp(params, grads, vs):
    return torch.autograd.grad(grads, params, grad_outputs=vs, retain_graph=True)


def _dot(xs, ys):
    return sum((x * y).sum() for x, y in zip(xs, ys))


def hessian_trace(model, coords, feats, probes=8, seed=1234):
    if probes <= 0:
        return None
    params, grads = _loss_and_grads(model, coords, feats)
    g = torch.Generator(device=coords.device).manual_seed(seed)
    total = 0.0
    for _ in range(probes):
        vs = [torch.randint(0, 2, p.shape, device=p.device, generator=g).float() * 2 - 1 for p in params]
        total += _dot(_hvp(params, grads, vs), vs).item()
    return total / probes


def top_eigenvalue(model, coords, feats, iters=20, seed=1234):
    """Largest-magnitude Hessian eigenvalue by power iteration (Rayleigh quotient at the last iterate)."""
    if iters <= 0:
        return None
    params, grads = _loss_and_grads(model, coords, feats)
    g = torch.Generator(device=coords.device).manual_seed(seed)
    vs = [torch.randn(p.shape, device=p.device, generator=g) for p in params]
    norm = torch.sqrt(_dot(vs, vs))
    vs = [v / norm for v in vs]
    lam = 0.0
    for _ in range(iters):
        hv = _hvp(params, grads, vs)
        lam = _dot(hv, vs).item()
        norm = torch.sqrt(_dot(hv, hv))
        if norm.item() == 0.0:
            break
        vs = [h / norm for h in hv]
    return lam
