"""Blind checks for the Levenberg–Marquardt pieces on a tiny problem on the CPU."""
import torch
from torch.func import functional_call

from crops import load_crop
from lm import LevenbergMarquardt
from siren import Siren

torch.manual_seed(0)
device = torch.device("cpu")
img, coords, feats = load_crop(1, 8, device)        # 96 x 64 crop, 6144 pixels
coords, feats = coords[:1000], feats[:1000]

model = Siren(2, 8, 3, 3, final_activation=torch.nn.Identity()).to(device)
opt = LevenbergMarquardt(model, chunk_size=256)
flat = opt.flat_params()

# 1. J^T r must equal the gradient of half the sum of squares; J^T J must match an explicit Jacobian.
JtJ, Jtr, sq = opt.normal_equations(flat, coords, feats)
f = flat.clone().requires_grad_(True)
pred = functional_call(model, opt.unflatten(f), (coords,))
loss = 0.5 * (pred - feats).pow(2).sum()
grad = torch.autograd.grad(loss, f)[0]
J_full = torch.autograd.functional.jacobian(
    lambda p: functional_call(model, opt.unflatten(p), (coords,)).reshape(-1), flat)
assert torch.allclose(Jtr, grad, rtol=1e-4, atol=1e-5), (Jtr - grad).abs().max()
assert torch.allclose(JtJ, J_full.T @ J_full, rtol=1e-4, atol=1e-4), (JtJ - J_full.T @ J_full).abs().max()
assert abs(sq.item() - 2 * loss.item()) < 1e-4
print("1. Jacobian pieces match autograd")

# 2. Writing flat parameters back round-trips.
opt.write_params(flat + 1.0)
assert torch.allclose(opt.flat_params(), flat + 1.0)
opt.write_params(flat)
print("2. parameter round trip ok")

# 3. The loss never goes up across steps, and the method makes real progress.
losses = [opt.sum_squares(opt.flat_params(), coords, feats).item()]
for _ in range(15):
    info = opt.step(coords, feats)
    losses.append(opt.sum_squares(opt.flat_params(), coords, feats).item())
    assert losses[-1] <= losses[-2] + 1e-9, (losses[-2], losses[-1])
print(f"3. monotone: {losses[0]:.4f} -> {losses[-1]:.4f} in 15 steps, damping now {opt.damping:.1e}")
assert losses[-1] < 0.5 * losses[0]

# 4. Same start, Adam for 300 steps against LM for 15 steps.
torch.manual_seed(0)
m2 = Siren(2, 8, 3, 3, final_activation=torch.nn.Identity()).to(device)
adam = torch.optim.Adam(m2.parameters(), lr=2e-4)
for _ in range(300):
    adam.zero_grad()
    l = (m2(coords) - feats).pow(2).sum()
    l.backward()
    adam.step()
adam_loss = (m2(coords) - feats).pow(2).sum().item()
print(f"4. adam 300 steps: {adam_loss:.4f}, lm 15 steps: {losses[-1]:.4f}")
print("all checks passed")
