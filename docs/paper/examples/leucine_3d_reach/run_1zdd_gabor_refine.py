#!/usr/bin/env python3
"""Global vs Gabor refine on the *full* 1ZDD density (not an omit Δρ).

Starts from an existing free-OT pose (×2 cloud preferred), rebuilds ρ_obs from
the true model, then compares:
  A) more global sliced-W1 steps
  B) WindowedSlicedOT with s_max anneal (model-centred windows, detached)

Memory guardrails
-----------------
* No ``GaborTarget`` precompute (Phase-1 only).
* Default molecular bbox crop (drops empty solvent; still full-protein density).
* Lean MC: 4 windows × 16 dirs (override carefully — each window touches the map).
* Reports CUDA peak allocation after each arm.

Usage
-----
  uv run python run_1zdd_gabor_refine.py \\
      --npz out/1zdd_fnmqcqrrfyealhdpnlneeqrnakiksirddc_free_ot_2A_L32_x2_sgd0.1_seed0.npz \\
      --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from slicedot import SlicedOT, SlicedOTConfig, WindowedSlicedOT
from slicedot.windowed import suggest_L

from run_1zdd_free_ot import OUT, Adam, nn_rmsd, render_ortho, vg_ot
from run_1zdd_omit_gabor_refine import crop_diff_map, run_gabor, run_global

torch.set_default_dtype(torch.float64)


def _cuda_peak_gb(device: str) -> float | None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024 ** 3)


def _reset_cuda_peak(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz",
        type=Path,
        default=OUT
        / "1zdd_fnmqcqrrfyealhdpnlneeqrnakiksirddc_free_ot_2A_L32_x2_sgd0.1_seed0.npz",
    )
    ap.add_argument(
        "--start",
        choices=("best_nn", "final", "after_global"),
        default="best_nn",
        help="Which free-atom pose to continue from.",
    )
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument(
        "--n-windows", type=int, default=4,
        help="MC windows/step — keep ≤8 on the full 81³ box.",
    )
    ap.add_argument("--n-dirs", type=int, default=16)
    ap.add_argument("--lambda-mass", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--crop-pad", type=float, default=5.0,
        help="Å pad for molecular bbox crop of ρ_obs (memory/speed).",
    )
    ap.add_argument(
        "--no-crop",
        action="store_true",
        help="Use the full saved grid (larger V_abs footprint).",
    )
    ap.add_argument("--skip-global", action="store_true")
    ap.add_argument(
        "--mem-warn-gb", type=float, default=1.5,
        help="Abort before building WindowedSlicedOT if crop/grid looks too big.",
    )
    args = ap.parse_args()

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    z = np.load(args.npz, allow_pickle=True)
    X_true = np.asarray(z["X_true"], dtype=np.float64)
    W = np.asarray(z["W"], dtype=np.float64)
    key = {
        "best_nn": "X_best_nn",
        "final": "X_final",
        "after_global": "X_after_global",
    }[args.start]
    if key not in z.files:
        raise SystemExit(f"{args.npz.name} missing {key}; have {z.files}")
    X0 = np.asarray(z[key], dtype=np.float64)

    NG_full = tuple(int(x) for x in z["NG"])
    sigma = float(z["sigma"])
    res = float(z["resolution"])
    sp0 = float(np.asarray(z["spacing"]).ravel()[0])
    org_npz = np.asarray(z["origin"], dtype=np.float64)

    # Rebuild full-protein density (same kernel as free OT).
    T_full, org_full, sp = render_ortho(X_true, sp0, NG_full, sigma, W)
    if not np.allclose(org_full, org_npz, atol=1e-6):
        print("warn: rebuilt origin differs from npz; using rebuilt map", flush=True)

    if args.no_crop:
        T, org, NG = T_full, org_full, NG_full
        crop_note = f"full grid {NG[0]}³"
    else:
        T, org, sp, NG = crop_diff_map(
            T_full, org_full, sp, X_true, pad=float(args.crop_pad),
        )
        crop_note = f"mol crop pad={args.crop_pad:g} Å → {NG[0]}×{NG[1]}×{NG[2]}"

    # Footprint estimate for WindowedSlicedOT buffers: map + V_abs (Nvox × 3).
    nvox = int(np.prod(NG))
    est_gb = nvox * (1 + 3) * 8 / (1024 ** 3)  # float64 map_flat + V_abs
    print(
        f"1ZDD full-density Gabor refine  source={args.npz.name}\n"
        f"  start={key}  N_free={X0.shape[0]}  N_true={X_true.shape[0]}  "
        f"res={res:g} Å  σ={sigma:.3f}  device={device}\n"
        f"  ρ: {crop_note}  est. window buffers ≈{est_gb:.3f} GB",
        flush=True,
    )
    if est_gb > args.mem_warn_gb:
        raise SystemExit(
            f"estimated WindowedSlicedOT buffers {est_gb:.2f} GB exceed "
            f"--mem-warn-gb={args.mem_warn_gb:g}; tighten --crop-pad or lower grid"
        )

    n_model = X0.shape[0]
    w_free = (
        W if n_model == X_true.shape[0]
        else np.full(n_model, 1.0 / n_model, dtype=np.float64)
    )
    nn_start = nn_rmsd(X0, X_true)
    print(f"  start NN={nn_start:.4f} Å  n_win={args.n_windows}  L={args.n_dirs}", flush=True)

    s_floor = 3.0 * sigma
    s_max0 = max(6.0, 5.0 * sigma)
    half = 0.5 * max(NG) * float(np.atleast_1d(sp).ravel()[0])
    cfg = SlicedOTConfig(
        n_dirs=32, dt=0.3, window=float(3.0 * half),
        map_cutoff=1e-7, backend="direct",
    )
    Tt = torch.tensor(T, device=device)
    spt = torch.tensor(sp, device=device)

    global_res = None
    peak_g = None
    if not args.skip_global:
        _reset_cuda_peak(device)
        ot = SlicedOT(Tt, org, spt, sigma, cfg, device=device)
        global_res = run_global(
            ot, X0, w_free, sigma, device,
            lr=args.lr, steps=args.steps, patience=args.patience,
            X_true=X_true, label="global",
        )
        peak_g = _cuda_peak_gb(device)
        if peak_g is not None:
            print(f"  [mem] global arm peak CUDA alloc = {peak_g:.3f} GB", flush=True)
        del ot
        _reset_cuda_peak(device)

    n_dirs0 = (
        int(args.n_dirs) if args.n_dirs > 0
        else int(suggest_L(s_max0, sigma * 2.3548))
    )
    if args.n_windows > 8 and args.no_crop:
        print(
            f"warn: n_windows={args.n_windows} on full box is heavy; "
            "consider ≤4 or enable crop",
            flush=True,
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
        memory_warn_gb=args.mem_warn_gb,
    )
    # Drop Python refs to host map copy once module buffers own it.
    del T_full
    gabor_res = run_gabor(
        win, X0, w_free, sigma, device,
        lr=args.lr, steps=args.steps, patience=args.patience,
        X_true=X_true,
        s_max0=s_max0,
        s_floor=s_floor,
        anneal_L=False,
    )
    peak_b = _cuda_peak_gb(device)
    if peak_b is not None:
        print(f"  [mem] Gabor arm peak CUDA alloc = {peak_b:.3f} GB", flush=True)

    print("\n========== summary (full density) ==========")
    print(f"start NN              : {nn_start:.4f} Å")
    if global_res is not None:
        print(
            f"A global  best / final: "
            f"{global_res['nn_best']:.4f} / {global_res['nn_final']:.4f} Å"
        )
    print(
        f"B Gabor   best / final: "
        f"{gabor_res['nn_best']:.4f} / {gabor_res['nn_final']:.4f} Å"
    )
    if global_res is not None:
        delta = global_res["nn_best"] - gabor_res["nn_best"]
        print(
            f"Δ(best NN) global−gabor: {delta:+.4f} Å  "
            f"({'Gabor better' if delta > 0 else 'global better / tie'})"
        )
    if peak_g is not None or peak_b is not None:
        print(
            f"CUDA peak GB: global={peak_g if peak_g is not None else '—'}  "
            f"gabor={peak_b if peak_b is not None else '—'}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{res:g}".replace(".", "p")
    out = OUT / (
        f"1zdd_full_gabor_refine_{tag}A_w{args.n_windows}_L{n_dirs0}"
        f"_seed{args.seed}.npz"
    )
    payload = dict(
        X_true=X_true,
        X0=X0,
        X_gabor_final=gabor_res["X_final"],
        X_gabor_best_nn=gabor_res["X_best_nn"],
        gabor_nn_rmsds=gabor_res["nn_rmsds"],
        gabor_energies=gabor_res["energies"],
        origin=org,
        spacing=sp,
        NG=np.asarray(NG),
        resolution=np.array(res),
        sigma=np.array(sigma),
        seed=np.array(args.seed),
        n_windows=np.array(args.n_windows),
        n_dirs=np.array(n_dirs0),
        s_min=np.array(s_floor),
        s_max0=np.array(s_max0),
        nn_start=np.array(nn_start),
        gabor_nn_best=np.array(gabor_res["nn_best"]),
        gabor_nn_final=np.array(gabor_res["nn_final"]),
        source_npz=np.array(args.npz.name),
        start_key=np.array(key),
        cropped=np.array(not args.no_crop),
        cuda_peak_gabor_gb=np.array(peak_b if peak_b is not None else np.nan),
    )
    if global_res is not None:
        payload.update(
            X_global_final=global_res["X_final"],
            X_global_best_nn=global_res["X_best_nn"],
            global_nn_rmsds=global_res["nn_rmsds"],
            global_energies=global_res["energies"],
            global_nn_best=np.array(global_res["nn_best"]),
            global_nn_final=np.array(global_res["nn_final"]),
            cuda_peak_global_gb=np.array(peak_g if peak_g is not None else np.nan),
        )
    np.savez_compressed(out, **payload)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
