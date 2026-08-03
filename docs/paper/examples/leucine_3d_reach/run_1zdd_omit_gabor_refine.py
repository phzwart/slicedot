#!/usr/bin/env python3
"""Gabor-windowed refine after global free OT on a 1ZDD omit Δρ land.

Compares two continuations from the same free-atom pose:
  A) more global sliced W1 steps
  B) WindowedSlicedOT (Gabor) with s_max anneal and model-centred windows

Primary metric: Hungarian NN-RMSD to the omitted true fragment (before
geometry / naming).  Optionally dumps a small panel figure.

Usage
-----
  uv run python run_1zdd_omit_gabor_refine.py \\
      --npz out/1zdd_omit_r1n12_2A_L32_x1.5_seed0.npz --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from slicedot import SlicedOT, SlicedOTConfig, WindowedSlicedOT
from slicedot.windowed import suggest_L

from run_1zdd_free_ot import Adam, OUT, nn_rmsd, vg_ot

torch.set_default_dtype(torch.float64)


def _hungarian_keep(X_free: np.ndarray, X_true: np.ndarray) -> np.ndarray:
    """If overcomplete, keep the N_true atoms closest (Hungarian) to truth."""
    if X_free.shape[0] == X_true.shape[0]:
        return X_free
    d2 = ((X_true[:, None, :] - X_free[None, :, :]) ** 2).sum(-1)
    _ri, cj = linear_sum_assignment(d2)
    return X_free[cj].copy()


def crop_diff_map(
    T: np.ndarray,
    origin: np.ndarray,
    spacing,
    X_focus: np.ndarray,
    *,
    pad: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Axis-aligned crop of Δρ around ``X_focus`` ± pad (Å)."""
    sp = np.atleast_1d(spacing).astype(np.float64) * np.ones(3)
    org = np.asarray(origin, dtype=np.float64)
    lo = X_focus.min(0) - pad
    hi = X_focus.max(0) + pad
    i0 = np.maximum(0, np.floor((lo - org) / sp).astype(int))
    i1 = np.minimum(np.asarray(T.shape) - 1, np.ceil((hi - org) / sp).astype(int))
    Tc = np.ascontiguousarray(
        T[i0[0] : i1[0] + 1, i0[1] : i1[1] + 1, i0[2] : i1[2] + 1]
    )
    org_c = org + i0 * sp
    mass = float(Tc.sum())
    if mass <= 0:
        raise RuntimeError("cropped Δρ has no mass")
    Tc = Tc / mass
    return Tc, org_c, sp, tuple(int(x) for x in Tc.shape)


def run_global(
    ot: SlicedOT,
    X0: np.ndarray,
    w: np.ndarray,
    sigma: float,
    device,
    *,
    lr: float,
    steps: int,
    patience: int,
    X_true: np.ndarray,
    label: str = "global",
) -> dict:
    X = np.asarray(X0, dtype=np.float64).copy()
    opt = Adam(X.shape, lr=lr)
    nn0 = nn_rmsd(X, X_true)
    nns = [nn0]
    Es = []
    best_nn = nn0
    best_X = X.copy()
    stagnant = 0
    t0 = time.perf_counter()
    print(
        f"\n=== A) Global continue  lr={lr:g}  steps≤{steps} ===",
        flush=True,
    )
    print(f"  start NN={nn0:.4f} Å", flush=True)
    for k in range(steps):
        E, G = vg_ot(ot, X, w, sigma, device=device)
        X = opt.step(X, G)
        nn = nn_rmsd(X, X_true)
        Es.append(E)
        nns.append(nn)
        if nn < best_nn - 1e-5:
            best_nn = nn
            best_X = X.copy()
            stagnant = 0
        else:
            stagnant += 1
        if k % 25 == 0 or k + 1 == steps or stagnant >= patience:
            print(
                f"  [{label} {k:4d}] E={E:.6g}  NN={nn:.4f}  "
                f"best={best_nn:.4f}  ({time.perf_counter() - t0:.1f}s, "
                f"stagnant={stagnant})",
                flush=True,
            )
        if stagnant >= patience:
            print(f"  stop: NN plateau ({patience})", flush=True)
            break
    return {
        "X_final": X,
        "X_best_nn": best_X,
        "nn_rmsds": np.asarray(nns),
        "energies": np.asarray(Es),
        "nn_best": float(best_nn),
        "nn_final": float(nns[-1]),
    }


def run_gabor(
    win: WindowedSlicedOT,
    X0: np.ndarray,
    w: np.ndarray,
    sigma: float,
    device,
    *,
    lr: float,
    steps: int,
    patience: int,
    X_true: np.ndarray,
    s_max0: float,
    s_floor: float,
    anneal_L: bool = False,
) -> dict:
    X = np.asarray(X0, dtype=np.float64).copy()
    opt = Adam(X.shape, lr=lr)
    nn0 = nn_rmsd(X, X_true)
    nns = [nn0]
    Es = []
    best_nn = nn0
    best_X = X.copy()
    stagnant = 0
    t0 = time.perf_counter()
    print(
        f"\n=== B) Gabor refine  lr={lr:g}  steps≤{steps}  "
        f"s∈[{s_floor:.2f},{s_max0:.2f}]→floor  L0={win.n_directions} ===",
        flush=True,
    )
    print(f"  start NN={nn0:.4f} Å", flush=True)
    for k in range(steps):
        frac = k / max(steps - 1, 1)
        # Anneal s_max: broad → local (cannot go below 3σ API floor).
        s_hi = s_max0 * (1.0 - 0.7 * frac) + s_floor * (0.7 * frac)
        s_hi = max(float(s_hi), s_floor + 1e-6)
        win.s_max = s_hi
        if anneal_L:
            d_res = float(sigma) * 2.3548
            win.n_directions = int(suggest_L(s_hi, d_res))

        xt = torch.tensor(X, device=device, requires_grad=True)
        wt = torch.tensor(w, device=device)
        E = win(xt, wt, sigma)
        E.backward()
        G = xt.grad.detach().cpu().numpy()
        e = float(E.detach().cpu())
        X = opt.step(X, G)
        nn = nn_rmsd(X, X_true)
        Es.append(e)
        nns.append(nn)
        if nn < best_nn - 1e-5:
            best_nn = nn
            best_X = X.copy()
            stagnant = 0
        else:
            stagnant += 1
        if k % 25 == 0 or k + 1 == steps or stagnant >= patience:
            print(
                f"  [gabor {k:4d}] E={e:.6g}  NN={nn:.4f}  "
                f"best={best_nn:.4f}  s_max={s_hi:.2f}  L={win.n_directions}  "
                f"({time.perf_counter() - t0:.1f}s, stagnant={stagnant})",
                flush=True,
            )
        if stagnant >= patience:
            print(f"  stop: NN plateau ({patience})", flush=True)
            break
    return {
        "X_final": X,
        "X_best_nn": best_X,
        "nn_rmsds": np.asarray(nns),
        "energies": np.asarray(Es),
        "nn_best": float(best_nn),
        "nn_final": float(nns[-1]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz",
        type=Path,
        default=OUT / "1zdd_omit_r1n12_2A_L32_x1.5_seed0.npz",
    )
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument(
        "--n-windows", type=int, default=8,
        help="MC windows/step (cheap after omit-bbox crop; 4–12 is typical).",
    )
    ap.add_argument(
        "--n-dirs",
        type=int,
        default=24,
        help="Fixed L for Gabor; 0 → suggest_L(s_max) each step.",
    )
    ap.add_argument(
        "--anneal-L",
        action="store_true",
        help="Recompute L=suggest_L(s_max) every step (expensive; off by default).",
    )
    ap.add_argument("--lambda-mass", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--skip-global",
        action="store_true",
        help="Only run Gabor arm (A skipped).",
    )
    ap.add_argument(
        "--crop-pad",
        type=float,
        default=6.0,
        help="Å pad around omit atoms when cropping Δρ for both arms.",
    )
    ap.add_argument(
        "--no-crop",
        action="store_true",
        help="Use the full-box Δρ (slow Phase-1 Gabor).",
    )
    args = ap.parse_args()

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    z = np.load(args.npz, allow_pickle=True)
    X_omit_true = np.asarray(z["X_omit_true"], dtype=np.float64)
    # Prefer post-SGD free cloud; fall back to Adam / X0.
    if "X_after_sgd" in z.files and z["X_after_sgd"].shape[0] > 0:
        X0 = np.asarray(z["X_after_sgd"], dtype=np.float64)
        start_key = "X_after_sgd"
    elif "X_after_adam" in z.files:
        X0 = np.asarray(z["X_after_adam"], dtype=np.float64)
        start_key = "X_after_adam"
    else:
        X0 = np.asarray(z["X0"], dtype=np.float64)
        start_key = "X0"

    T_full = np.asarray(z["T_diff"], dtype=np.float64)
    T_full = T_full / T_full.sum()
    org_full = np.asarray(z["origin"], dtype=np.float64)
    sp = np.asarray(z["spacing"], dtype=np.float64)
    sigma = float(z["sigma"])
    res = float(z["resolution"])

    if args.no_crop:
        T_diff, org, NG = T_full, org_full, tuple(int(x) for x in z["NG"])
        crop_note = "full box"
    else:
        T_diff, org, sp, NG = crop_diff_map(
            T_full, org_full, sp, X_omit_true, pad=float(args.crop_pad),
        )
        crop_note = f"crop pad={args.crop_pad:g} Å → {NG[0]}×{NG[1]}×{NG[2]}"

    half = 0.5 * max(NG) * float(np.atleast_1d(sp).ravel()[0])

    n_model = X0.shape[0]
    w_free = np.full(n_model, 1.0 / n_model, dtype=np.float64)
    nn_start = nn_rmsd(X0, X_omit_true)

    s_floor = 3.0 * sigma
    # Broad enough to cover ~a helical turn, then anneal to API floor.
    s_max0 = max(6.0, 5.0 * sigma)

    print(
        f"1ZDD omit Gabor refine  source={args.npz.name}  start={start_key}\n"
        f"  omit atoms={X_omit_true.shape[0]}  free={n_model}  "
        f"res={res:g} Å  σ={sigma:.3f}  device={device}\n"
        f"  Δρ: {crop_note}  start NN={nn_start:.4f} Å  "
        f"s∈[{s_floor:.2f},{s_max0:.2f}]  n_win={args.n_windows}",
        flush=True,
    )

    cfg = SlicedOTConfig(
        n_dirs=32,
        dt=0.3,
        window=float(3.0 * half),
        map_cutoff=1e-7,
        backend="direct",
    )
    Tt = torch.tensor(T_diff, device=device)
    spt = torch.tensor(sp, device=device)

    global_res = None
    if not args.skip_global:
        ot = SlicedOT(Tt, org, spt, sigma, cfg, device=device)
        global_res = run_global(
            ot, X0, w_free, sigma, device,
            lr=args.lr, steps=args.steps, patience=args.patience,
            X_true=X_omit_true,
        )

    n_dirs0 = (
        int(args.n_dirs)
        if args.n_dirs > 0
        else int(suggest_L(s_max0, sigma * 2.3548))
    )
    win = WindowedSlicedOT(
        Tt,
        org,
        spt,
        sigma,
        s_range=(s_floor, s_max0),
        n_windows=args.n_windows,
        n_directions=n_dirs0,
        pi_a="model",
        lambda_mass=args.lambda_mass,
        backend="direct",
        seed=args.seed,
        config=cfg,
        device=device,
    )
    gabor_res = run_gabor(
        win, X0, w_free, sigma, device,
        lr=args.lr, steps=args.steps, patience=args.patience,
        X_true=X_omit_true,
        s_max0=s_max0,
        s_floor=s_floor,
        anneal_L=bool(args.anneal_L),
    )

    # Summaries on Hungarian-pruned clouds (fair vs omit truth size).
    def pruned_nn(X):
        return nn_rmsd(_hungarian_keep(X, X_omit_true), X_omit_true)

    print("\n========== summary ==========")
    print(f"start NN              : {nn_start:.4f} Å")
    if global_res is not None:
        print(
            f"A global  best / final: "
            f"{global_res['nn_best']:.4f} / {global_res['nn_final']:.4f} Å  "
            f"(pruned best {pruned_nn(global_res['X_best_nn']):.4f})"
        )
    print(
        f"B Gabor   best / final: "
        f"{gabor_res['nn_best']:.4f} / {gabor_res['nn_final']:.4f} Å  "
        f"(pruned best {pruned_nn(gabor_res['X_best_nn']):.4f})"
    )
    if global_res is not None:
        delta = global_res["nn_best"] - gabor_res["nn_best"]
        print(
            f"Δ(best NN) global−gabor: {delta:+.4f} Å  "
            f"({'Gabor better' if delta > 0 else 'global better / tie'})"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{res:g}".replace(".", "p")
    stem = args.npz.stem.replace("1zdd_omit_", "1zdd_omit_gabor_")
    out = OUT / f"{stem}_refine_{tag}A_w{args.n_windows}_seed{args.seed}.npz"
    payload = dict(
        X_omit_true=X_omit_true,
        X0=X0,
        X_gabor_final=gabor_res["X_final"],
        X_gabor_best_nn=gabor_res["X_best_nn"],
        gabor_nn_rmsds=gabor_res["nn_rmsds"],
        gabor_energies=gabor_res["energies"],
        T_diff=T_diff,
        origin=org,
        spacing=sp,
        NG=np.asarray(NG),
        resolution=np.array(res),
        sigma=np.array(sigma),
        seed=np.array(args.seed),
        n_windows=np.array(args.n_windows),
        s_min=np.array(s_floor),
        s_max0=np.array(s_max0),
        nn_start=np.array(nn_start),
        gabor_nn_best=np.array(gabor_res["nn_best"]),
        gabor_nn_final=np.array(gabor_res["nn_final"]),
        source_npz=np.array(args.npz.name),
        start_key=np.array(start_key),
    )
    if global_res is not None:
        payload.update(
            X_global_final=global_res["X_final"],
            X_global_best_nn=global_res["X_best_nn"],
            global_nn_rmsds=global_res["nn_rmsds"],
            global_energies=global_res["energies"],
            global_nn_best=np.array(global_res["nn_best"]),
            global_nn_final=np.array(global_res["nn_final"]),
        )
    np.savez_compressed(out, **payload)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
