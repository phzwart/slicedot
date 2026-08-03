#!/usr/bin/env python3
"""Gabor-windowed sliced-W1 refine after global free OT (AFSSFN hexapeptide).

Loads a free-OT npz, continues with ``WindowedSlicedOT`` (soft local windows),
and reports Hungarian NN-RMSD vs the true structure.

Usage
-----
  uv run python run_afssfn_gabor_refine.py \\
      --npz out/8dts_afssfn_free_ot_3A_L32_x2_seed0.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from slicedot import SlicedOTConfig, WindowedSlicedOT, sigma_from_resolution

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def nn_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    ri, cj = linear_sum_assignment(d2)
    return float(np.sqrt(d2[ri, cj].mean()))


def render_ortho(X, sp, NG, sigma, weights):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.asarray(NG, dtype=np.float64) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG, dtype=np.float64)
    for p, w in zip(X, weights):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2.0 * sigma * sigma))
    return T / T.sum(), org, sp


class Adam:
    def __init__(self, shape, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(self, X: np.ndarray, G: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * G
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (G * G)
        mhat = self.m / (1.0 - self.beta1 ** self.t)
        vhat = self.v / (1.0 - self.beta2 ** self.t)
        return X - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz",
        type=Path,
        default=OUT / "8dts_afssfn_free_ot_3A_L32_x2_seed0.npz",
    )
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument(
        "--n-windows", type=int, default=4,
        help="MC windows per step (each touches the full map — keep small).",
    )
    ap.add_argument("--n-dirs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lambda-mass", type=float, default=0.0)
    ap.add_argument(
        "--pi-a",
        choices=("uniform", "model"),
        default="model",
        help="Window-centre proposal (model-centred is denser on the cloud).",
    )
    ap.add_argument(
        "--start",
        choices=("final", "best_nn", "named", "after_global"),
        default="final",
        help="Which pose in the npz to refine.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device", type=str, default="auto",
        help="'auto', 'cpu', or a torch device string.",
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    z = np.load(args.npz, allow_pickle=True)
    X_true = np.asarray(z["X_true"], dtype=np.float64)
    key = {
        "final": "X_final",
        "best_nn": "X_best_nn",
        "named": "X_named",
        "after_global": "X_after_global",
    }[args.start]
    if key not in z.files:
        raise SystemExit(f"{args.npz.name} missing {key}")
    X0 = np.asarray(z[key], dtype=np.float64)
    W_chem = np.asarray(z["W"], dtype=np.float64)
    origin = np.asarray(z["origin"], dtype=np.float64)
    spacing = np.asarray(z["spacing"], dtype=np.float64)
    NG = tuple(int(x) for x in z["NG"])
    sigma = float(z["sigma"])
    res = float(z["resolution"])
    sp0 = float(spacing.ravel()[0])

    # Rebuild the same ortho map (true atoms + chemical weights).
    T, org, sp = render_ortho(X_true, sp0, NG, sigma, W_chem)
    if not (np.allclose(org, origin) and np.allclose(sp, spacing)):
        print("warn: rebuilt origin/spacing differ from npz; using rebuilt map")

    n_model = X0.shape[0]
    w_free = (
        W_chem if n_model == X_true.shape[0]
        else np.full(n_model, 1.0 / n_model, dtype=np.float64)
    )

    s_floor = 3.0 * sigma
    # Anneal s_max from large → floor so early steps stay broad, late steps local.
    s_max0 = max(12.0, 8.0 * sigma)
    s_range = (s_floor, s_max0)

    cfg = SlicedOTConfig(
        n_dirs=args.n_dirs, dt=0.3, window=float(3.0 * max(NG) * sp0),
        map_cutoff=1e-7, backend="direct",
    )
    win = WindowedSlicedOT(
        torch.tensor(T, device=device),
        org,
        torch.tensor(sp, device=device),
        sigma,
        s_range=s_range,
        n_windows=args.n_windows,
        n_directions=args.n_dirs,
        pi_a=args.pi_a,
        lambda_mass=args.lambda_mass,
        backend="direct",
        seed=args.seed,
        config=cfg,
        device=device,
    )

    nn0 = nn_rmsd(X0, X_true)
    print(
        f"AFSSFN Gabor refine  res={res:g} Å  σ={sigma:.3f}  "
        f"s∈[{s_floor:.2f},{s_max0:.2f}]  "
        f"n_win={args.n_windows}  L={args.n_dirs}  pi_a={args.pi_a}  "
        f"N={n_model} (true={X_true.shape[0]})  device={device}",
        flush=True,
    )
    print(f"start={args.start}  NN-RMSD={nn0:.4f} Å", flush=True)

    X = X0.copy()
    opt = Adam(X.shape, lr=args.lr)
    poses = [X.copy()]
    nns = [nn0]
    Es = []
    best_nn = nn0
    best_X = X.copy()
    best_E = np.inf
    stagnant = 0
    t0 = time.perf_counter()

    for step in range(args.steps):
        # Mild s_max anneal: keep floor fixed, shrink upper end toward ~4σ.
        frac = step / max(args.steps - 1, 1)
        s_hi = s_max0 * (1.0 - 0.55 * frac) + (4.0 * sigma) * (0.55 * frac)
        s_hi = max(s_hi, s_floor + 1e-6)
        win.s_max = float(s_hi)

        xt = torch.tensor(X, device=device, requires_grad=True)
        wt = torch.tensor(w_free, device=device)
        E = win(xt, wt, sigma)
        E.backward()
        G = xt.grad.detach().cpu().numpy()
        e = float(E.detach().cpu())
        X = opt.step(X, G)
        nn = nn_rmsd(X, X_true)
        poses.append(X.copy())
        nns.append(nn)
        Es.append(e)
        if nn < best_nn - 1e-5:
            best_nn = nn
            best_X = X.copy()
            stagnant = 0
        else:
            stagnant += 1
        if e < best_E:
            best_E = e
        if step % 25 == 0 or step + 1 == args.steps:
            print(
                f"  [{step:4d}] E={e:.6g}  NN={nn:.4f} Å  "
                f"s_max={win.s_max:.2f}  best_NN={best_nn:.4f}  "
                f"({time.perf_counter() - t0:.1f}s, stagnant={stagnant})",
                flush=True,
            )
        if stagnant >= args.patience:
            print(f"  stop: NN plateau ({args.patience} steps)", flush=True)
            break

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{res:g}".replace(".", "p")
    out = OUT / f"8dts_afssfn_gabor_refine_{tag}A_L{args.n_dirs}_x{n_model / X_true.shape[0]:g}_seed{args.seed}.npz"
    np.savez_compressed(
        out,
        X_true=X_true,
        X0=X0,
        X_final=X,
        X_best_nn=best_X,
        poses=np.stack(poses, axis=0),
        nn_rmsds=np.asarray(nns),
        energies=np.asarray(Es),
        W=W_chem,
        W_free=w_free,
        origin=org,
        spacing=sp,
        NG=np.asarray(NG),
        resolution=np.array(res),
        sigma=np.array(sigma),
        seed=np.array(args.seed),
        device=np.array(device),
        n_windows=np.array(args.n_windows),
        n_dirs=np.array(args.n_dirs),
        pi_a=np.array(args.pi_a),
        s_min=np.array(s_floor),
        s_max0=np.array(s_max0),
        start_key=np.array(args.start),
        source_npz=np.array(args.npz.name),
        nn_start=np.array(nn0),
        nn_final=np.array(nns[-1]),
        nn_best=np.array(best_nn),
    )
    print()
    print("--- Gabor refine summary ---")
    print(f"NN start : {nn0:.4f} Å")
    print(f"NN final : {nns[-1]:.4f} Å")
    print(f"NN best  : {best_nn:.4f} Å")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
