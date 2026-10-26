"""Levenberg–Marquardt for fitting a network to one image as a least squares problem.

The loss is the sum of squared residuals r = f(x) - y over every pixel and channel.
The Gauss–Newton normal matrix J^T J is accumulated over pixel chunks (the full
Jacobian is never materialised), then (J^T J + lambda * D) delta = -J^T r is solved
in float64. A step that lowers the loss is accepted and lambda shrinks; a step
that raises it is rejected and lambda grows, reusing the same normal matrix.
"""
import torch
from torch.func import functional_call, jacrev, vmap


class LevenbergMarquardt:
    def __init__(self, model, damping=1e-3, damping_up=10.0, damping_down=0.1,
                 min_damping=1e-10, max_damping=1e10, chunk_size=8192,
                 scaling="marquardt", max_retries=15):
        self.model = model
        self.names = [n for n, _ in model.named_parameters()]
        self.shapes = [p.shape for _, p in model.named_parameters()]
        self.numels = [p.numel() for _, p in model.named_parameters()]
        self.damping = damping
        self.up, self.down = damping_up, damping_down
        self.min_damping, self.max_damping = min_damping, max_damping
        self.chunk_size = chunk_size
        self.scaling = scaling
        self.max_retries = max_retries
        self.num_params = sum(self.numels)

    # --- parameter plumbing -------------------------------------------------
    def flat_params(self):
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])

    def unflatten(self, flat):
        out, i = {}, 0
        for name, shape, n in zip(self.names, self.shapes, self.numels):
            out[name] = flat[i:i + n].reshape(shape)
            i += n
        return out

    def write_params(self, flat):
        with torch.no_grad():
            for p, t in zip(self.model.parameters(), self.unflatten(flat).values()):
                p.copy_(t)

    def _f_single(self, flat, x):
        return functional_call(self.model, self.unflatten(flat), (x.unsqueeze(0),)).squeeze(0)

    # --- least squares pieces ----------------------------------------------
    @torch.no_grad()
    def sum_squares(self, flat, coords, feats):
        params = self.unflatten(flat)
        total = torch.zeros((), device=coords.device, dtype=torch.float64)
        for s in range(0, coords.shape[0], 4 * self.chunk_size):
            pred = functional_call(self.model, params, (coords[s:s + 4 * self.chunk_size],))
            total += (pred - feats[s:s + 4 * self.chunk_size]).double().pow(2).sum()
        return total

    def normal_equations(self, flat, coords, feats):
        P = self.num_params
        JtJ = torch.zeros(P, P, device=coords.device)
        Jtr = torch.zeros(P, device=coords.device)
        sq = torch.zeros((), device=coords.device, dtype=torch.float64)
        jac_fn = vmap(jacrev(self._f_single), in_dims=(None, 0))
        params = self.unflatten(flat)
        for s in range(0, coords.shape[0], self.chunk_size):
            x, y = coords[s:s + self.chunk_size], feats[s:s + self.chunk_size]
            J = jac_fn(flat, x).reshape(-1, P)                       # (chunk*3, P)
            with torch.no_grad():
                r = (functional_call(self.model, params, (x,)) - y).reshape(-1)
            JtJ.addmm_(J.T, J)
            Jtr.addmv_(J.T, r)
            sq += r.double().pow(2).sum()
        return JtJ, Jtr, sq

    def solve(self, JtJ, Jtr, D, lam):
        # Solve in float64; on a CPU-only or Metal machine do it on the CPU.
        dev = JtJ.device if JtJ.device.type == "cuda" else torch.device("cpu")
        A = JtJ.to(dev, torch.float64) + lam * torch.diag(D.to(dev, torch.float64))
        L, info = torch.linalg.cholesky_ex(A)
        if info.item() != 0:
            return None
        delta = torch.cholesky_solve(-Jtr.to(dev, torch.float64).unsqueeze(1), L).squeeze(1)
        return delta.to(JtJ.device, torch.float32)

    # --- one iteration ------------------------------------------------------
    def step(self, coords, feats):
        flat = self.flat_params()
        JtJ, Jtr, sq = self.normal_equations(flat, coords, feats)
        if self.scaling == "marquardt":
            D = torch.diagonal(JtJ).clamp_min(1e-12)
        else:
            D = torch.ones_like(Jtr)

        info = {"loss_before": sq.item(), "accepted": False, "retries": 0}
        for attempt in range(self.max_retries):
            delta = self.solve(JtJ, Jtr, D, self.damping)
            if delta is not None and torch.isfinite(delta).all():
                new_flat = flat + delta
                new_sq = self.sum_squares(new_flat, coords, feats)
                if new_sq < sq:
                    self.write_params(new_flat)
                    info.update(accepted=True, loss_after=new_sq.item(), damping=self.damping,
                                step_norm=delta.norm().item(), retries=attempt)
                    self.damping = max(self.damping * self.down, self.min_damping)
                    return info
            self.damping = min(self.damping * self.up, self.max_damping)
        info.update(loss_after=sq.item(), damping=self.damping, retries=self.max_retries)
        return info
