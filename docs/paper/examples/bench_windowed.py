#!/usr/bin/env python3
"""Benchmark Phase-1 (direct) vs Phase-3 (grid) WindowedSlicedOT cost.

Sweeps n_windows and N (atom count via repeated leucine copies).

Usage:
    uv run python docs/paper/examples/bench_windowed.py
"""
from __future__ import annotations

import time

import numpy as np
import torch

from slicedot import SlicedOTConfig, WindowedSlicedOT, sigma_from_resolution
from slicedot.fixtures import X0
from slicedot.fixtures import W as W_np

torch.set_default_dtype(torch.float64)

SIG = sigma_from_resolution(2.5)
OUT_ROWS = []


def render_ortho(X, w, sp, NG, sigma):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.array(NG) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG)
    for p, wi in zip(X, w):
        T += wi * np.exp(-((G - p) ** 2).sum(-1) / (2 * sigma * sigma))
    return T / T.sum(), org, sp


def make_model(n_copies: int):
    X = np.vstack([X0 + np.array([8.0 * k, 0.0, 0.0]) for k in range(n_copies)])
    w = np.tile(W_np, n_copies)
    w = w / w.sum()
    return X, w


def bench_once(backend: str, n_windows: int, n_copies: int, repeats: int = 5):
    X, w = make_model(n_copies)
    half = 0.5 * (X.max(0) - X.min(0)) + 6.0
    NG = tuple(int(2 * h / 0.5) | 1 for h in half)  # odd
    NG = tuple(max(24, n) for n in NG)
    T, org, sp = render_ortho(X, w, 0.5, NG, SIG)
    s_lo = 3.0 * SIG
    cfg = SlicedOTConfig(
        n_dirs=24, dt=0.4, window=96.0, map_cutoff=1e-7, backend=backend,
    )
    win = WindowedSlicedOT(
        torch.tensor(T), org, torch.tensor(sp), SIG,
        s_range=(s_lo, 12.0), n_windows=n_windows, n_directions=24,
        backend=backend, seed=0, config=cfg,
    )
    xt = torch.tensor(X, requires_grad=True)
    wt = torch.tensor(w)
    # warmup
    win(xt, wt, SIG).backward()
    xt.grad = None
    times = []
    for _ in range(repeats):
        xt.grad = None
        t0 = time.perf_counter()
        win(xt, wt, SIG).backward()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), X.shape[0]


def main():
    print(f"{'backend':<12} {'n_win':>6} {'N':>6} {'ms':>10}")
    for backend in ("direct", "grid"):
        for n_windows in (1, 4, 8, 16):
            for n_copies in (1, 4, 16):
                ms, N = bench_once(backend, n_windows, n_copies)
                print(f"{backend:<12} {n_windows:6d} {N:6d} {1000*ms:10.2f}")
                OUT_ROWS.append((backend, n_windows, N, ms))


if __name__ == "__main__":
    main()
