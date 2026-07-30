"""Score a translated leucine fragment against its own rendered map.

Demonstrates the analytic translation anchor: under a rigid shift by t along e,
sliced W1 equals t * mean_l |u_l · e|.
"""
from __future__ import annotations

import numpy as np
import torch

from slicedot import SlicedOT, SlicedOTConfig
from slicedot.fixtures import X0, W as W_np, sigma_of

torch.set_default_dtype(torch.float64)


def render_ortho(X, sp, NG, sigma, weights):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.array(NG) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG)
    for p, w in zip(X, weights):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2 * sigma * sigma))
    return T / T.sum(), org, sp


def main():
    sig = sigma_of(2.5)
    T, org, sp = render_ortho(X0, 0.45, (48, 48, 48), sig, W_np)
    model = SlicedOT(
        torch.tensor(T),
        org,
        torch.tensor(sp),
        sig,
        SlicedOTConfig(n_dirs=32, dt=0.3, window=96.0, map_cutoff=1e-7),
    )
    w = torch.tensor(W_np)
    x = torch.tensor(X0)
    floor = model(x, w, sig).item()
    e = np.array([1.0, 0.0, 0.0])
    ex = np.abs(model.U.numpy() @ e).mean()
    print(f"floor (self-score)          {floor:.3e}")
    for t in (1.0, 3.0, 6.0, 12.0):
        val = model(torch.tensor(X0 + t * e), w, sig).item()
        print(f"W1 at t={t:4.1f} A             {val:.6f}  (anchor {t * ex:.6f})")
    dv = model.deformation(torch.tensor(X0 + np.array([5.0, 2.0, -3.0])), w, sig)
    rms = (dv + torch.tensor([5.0, 2.0, -3.0])).norm(dim=1).max().item()
    print(f"1-step deformation residual {rms:.3e} A")


if __name__ == "__main__":
    main()
