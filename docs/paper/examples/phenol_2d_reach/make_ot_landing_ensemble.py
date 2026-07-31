#!/usr/bin/env python3
"""Ensemble of unrestrained OT landings around a converged 3 Å pose.

1. Run Adam OT (no P_restr / L1) from the usual 90°/3R start → seed pose.
2. For each of ``n_trials``: Gaussian-perturb the seed, re-minimize.
3. Contour the cloud of final atom positions (KDE) vs the target density.

Perturbing *after* the long-range reach keeps each restart local and cheap.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

import make_figure as mf
from make_figure import (
    MISALIGN_DEG,
    N_DIRS,
    OT_LR,
    SHIFT_RADII,
    build_scene,
    rmsd,
    value_grad_fn,
)
from make_ot_unrestrained import nn_rmsd, run_ot_unrestrained
from targets2d import ConsistentSlicedW1, directions_2d

OUT_DIR = Path(__file__).resolve().parent / "out"

# Defaults tuned for a local basin probe at ~3 Å.
RESOLUTION = 3.0
N_TRIALS = 100
# Isotropic Gaussian on each atom (Å). ~σ_map/2 so we leave the basin a bit
# but stay near the density support (σ_map ≈ 3/2.35 ≈ 1.27 Å).
PERTURB_SIGMA = 0.6
SEED = 0
LR = OT_LR


def _res_tag(resolution: float) -> str:
    return f"{float(resolution):g}".replace(".", "p")


def perturb(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Independent N(0, σ²) noise on each coordinate."""
    return np.asarray(X, dtype=np.float64) + rng.normal(
        0.0, float(sigma), size=X.shape,
    )


def kde_grid(
    points: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a 2-D Gaussian KDE on a regular grid; returns xx, yy, zz."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2).T  # (2, M)
    # Slight ridge so singular collapses (all atoms on one site) still plot.
    kde = gaussian_kde(pts, bw_method="scott")
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def draw_contour(
    scene: dict,
    landings: np.ndarray,
    seed_pose: np.ndarray,
    *,
    resolution: float,
    perturb_sigma: float,
    out_stem: str,
) -> None:
    """Contour KDE of all final atom positions over the target density."""
    X_true = scene["X_true"]
    rhoT, origin, dx, shape = (
        scene["rhoT"], scene["origin"], scene["dx"], scene["shape"],
    )
    Ny, Nx = shape
    full_extent = [
        origin[0] - 0.5 * dx,
        origin[0] + (Nx - 0.5) * dx,
        origin[1] - 0.5 * dx,
        origin[1] + (Ny - 0.5) * dx,
    ]

    cloud = landings.reshape(-1, 2)
    coms = landings.mean(axis=1)  # (n_trials, 2)
    pad = 1.5
    pts = np.vstack([cloud, X_true, seed_pose, coms])
    xmin, ymin = pts.min(0) - pad
    xmax, ymax = pts.max(0) + pad
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    xx, yy, zz = kde_grid(cloud, xlim, ylim, n=180)
    # Contour levels as fractions of the KDE peak.
    zmax = float(zz.max())
    levels = zmax * np.array([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)

    # --- left: atom-landing KDE over density ---
    ax = axes[0]
    ax.imshow(
        rhoT, origin="lower", extent=full_extent, cmap="YlOrBr",
        vmin=0.0, vmax=float(rhoT.max()), interpolation="nearest",
        aspect="equal", alpha=0.85,
    )
    ax.contour(
        xx, yy, zz, levels=levels, colors="#0b5fff", linewidths=1.0,
        alpha=0.95,
    )
    ax.contourf(
        xx, yy, zz, levels=levels, cmap="Blues", alpha=0.35,
        norm=Normalize(vmin=0.0, vmax=zmax),
    )
    # subsample scatter so the panel stays readable
    rng = np.random.default_rng(1)
    idx = rng.choice(cloud.shape[0], size=min(400, cloud.shape[0]), replace=False)
    ax.plot(
        cloud[idx, 0], cloud[idx, 1],
        ".", ms=2.2, color="#0b5fff", alpha=0.35, zorder=3,
    )
    ax.plot(
        X_true[:, 0], X_true[:, 1],
        "o", ms=6.0, mfc="none", mec="0.2", mew=1.1, zorder=5,
        label="true atoms",
    )
    ax.plot(
        seed_pose[:, 0], seed_pose[:, 1],
        "x", ms=5.5, mew=1.0, color="#b33a3a", zorder=4,
        label="OT seed",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (Å)")
    ax.set_ylabel(r"$y$ (Å)")
    ax.set_title("Final atom positions (KDE contours)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # --- right: COM landings ---
    ax = axes[1]
    ax.imshow(
        rhoT, origin="lower", extent=full_extent, cmap="YlOrBr",
        vmin=0.0, vmax=float(rhoT.max()), interpolation="nearest",
        aspect="equal", alpha=0.85,
    )
    true_com = X_true.mean(0)
    seed_com = seed_pose.mean(0)
    if coms.shape[0] >= 5:
        xx_c, yy_c, zz_c = kde_grid(coms, xlim, ylim, n=160)
        zc = float(zz_c.max())
        lev_c = zc * np.array([0.15, 0.35, 0.55, 0.75, 0.9])
        ax.contour(xx_c, yy_c, zz_c, levels=lev_c, colors="#0b5fff",
                   linewidths=1.1)
        ax.contourf(xx_c, yy_c, zz_c, levels=lev_c, cmap="Blues", alpha=0.40)
    ax.plot(
        coms[:, 0], coms[:, 1],
        "o", ms=3.5, mfc="#0b5fff", mec="none", alpha=0.55, zorder=4,
        label="trial COM",
    )
    ax.plot(
        true_com[0], true_com[1],
        "o", ms=8, mfc="none", mec="0.15", mew=1.3, zorder=5,
        label="true COM",
    )
    ax.plot(
        seed_com[0], seed_com[1],
        "x", ms=7, mew=1.2, color="#b33a3a", zorder=5,
        label="seed COM",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (Å)")
    ax.set_ylabel(r"$y$ (Å)")
    ax.set_title("Final centre-of-mass landings", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.suptitle(
        f"ortho-pentyl phenol @ {resolution:g} Å · unrestrained OT ensemble\n"
        f"{landings.shape[0]} restarts from seed ± "
        f"N(0, {perturb_sigma:g}² Å) · "
        f"{MISALIGN_DEG:.0f}° / {SHIFT_RADII}$R$ reach seed",
        fontsize=10,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(
    resolution: float = RESOLUTION,
    n_trials: int = N_TRIALS,
    perturb_sigma: float = PERTURB_SIGMA,
    seed: int = SEED,
    lr: float = LR,
):
    resolution = float(resolution)
    mf.RESOLUTION = resolution
    tag = _res_tag(resolution)
    out_stem = f"phenol_ot_landing_ensemble_{tag}A"
    traj_path = OUT_DIR / f"trajectory_ot_landing_ensemble_{tag}A.npz"

    scene = build_scene()
    ot = ConsistentSlicedW1(
        scene["rhoT"], scene["V"], directions_2d(N_DIRS), nbins=320, pad=12.0,
    )
    vg = value_grad_fn("ot", ot, scene["sigma"])
    X0, w, X_true = scene["X_start"], scene["w"], scene["X_true"]

    print(
        f"Seed reach: Adam OT @ {resolution:g} Å "
        f"(lr={lr:g}, no P_restr / L1) ...",
        flush=True,
    )
    seed_cache = run_ot_unrestrained(X0, w, X_true, vg, lr=lr)
    b = int(seed_cache["best_step"])
    seed_pose = seed_cache["poses"][b].copy()
    print(
        f"  seed @ step {b}: NN-RMSD={seed_cache['nn_rmsds'][b]:.3f} Å  "
        f"label={seed_cache['label_rmsds'][b]:.3f} Å  "
        f"E={seed_cache['energies'][b]:.5g}",
        flush=True,
    )

    rng = np.random.default_rng(int(seed))
    landings = np.empty((n_trials, seed_pose.shape[0], 2), dtype=np.float64)
    starts = np.empty_like(landings)
    nn_final = np.empty(n_trials)
    label_final = np.empty(n_trials)
    E_final = np.empty(n_trials)
    n_steps = np.empty(n_trials, dtype=np.int32)

    print(
        f"Ensemble: {n_trials} restarts, perturb σ={perturb_sigma:g} Å ...",
        flush=True,
    )
    for k in range(n_trials):
        X_pert = perturb(seed_pose, perturb_sigma, rng)
        starts[k] = X_pert
        cache = run_ot_unrestrained(X_pert, w, X_true, vg, lr=lr)
        ib = int(cache["best_step"])
        landings[k] = cache["poses"][ib]
        nn_final[k] = cache["nn_rmsds"][ib]
        label_final[k] = cache["label_rmsds"][ib]
        E_final[k] = cache["energies"][ib]
        n_steps[k] = cache["n_steps"]
        if (k + 1) % 10 == 0 or k == 0:
            print(
                f"  [{k + 1:3d}/{n_trials}]  "
                f"NN={nn_final[k]:.3f} Å  label={label_final[k]:.3f} Å  "
                f"steps={n_steps[k]}  "
                f"⟨NN⟩={nn_final[:k + 1].mean():.3f}",
                flush=True,
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        traj_path,
        seed_pose=seed_pose,
        starts=starts,
        landings=landings,
        nn_final=nn_final,
        label_final=label_final,
        E_final=E_final,
        n_steps=n_steps,
        X_true=X_true,
        resolution=np.array(resolution),
        perturb_sigma=np.array(perturb_sigma),
        lr=np.array(lr),
        seed=np.array(seed),
    )

    com_disp = np.linalg.norm(landings.mean(1) - X_true.mean(0), axis=1)
    print(
        f"\nLanding stats ({n_trials} trials):\n"
        f"  NN-RMSD   median={np.median(nn_final):.3f}  "
        f"mean={nn_final.mean():.3f}  "
        f"[{nn_final.min():.3f}, {nn_final.max():.3f}] Å\n"
        f"  label     median={np.median(label_final):.3f}  "
        f"mean={label_final.mean():.3f} Å\n"
        f"  ‖COM−true‖ median={np.median(com_disp):.3f}  "
        f"mean={com_disp.mean():.3f} Å\n"
        f"  steps/trial median={int(np.median(n_steps))}  "
        f"mean={n_steps.mean():.1f}",
        flush=True,
    )
    # Spread of landings about their mean pose (permutation-invariant via NN).
    mean_landing = landings.mean(0)
    spreads = np.array([nn_rmsd(landings[k], mean_landing) for k in range(n_trials)])
    print(
        f"  NN-RMSD to ensemble-mean pose: "
        f"median={np.median(spreads):.3f}  mean={spreads.mean():.3f} Å",
        flush=True,
    )
    # Reference: seed vs true (already known) and seed vs mean landing.
    print(
        f"  seed vs true NN-RMSD={nn_rmsd(seed_pose, X_true):.3f} Å  "
        f"label={rmsd(seed_pose, X_true):.3f} Å",
        flush=True,
    )

    draw_contour(
        scene, landings, seed_pose,
        resolution=resolution, perturb_sigma=perturb_sigma, out_stem=out_stem,
    )
    print(f"\nwrote {OUT_DIR / f'{out_stem}.pdf'}")
    print(f"wrote {OUT_DIR / f'{out_stem}.png'}")
    print(f"wrote {traj_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=float, default=RESOLUTION)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument(
        "--perturb-sigma", type=float, default=PERTURB_SIGMA,
        help="Per-coordinate Gaussian σ (Å) applied to the seed pose.",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()
    main(
        resolution=args.resolution,
        n_trials=args.n_trials,
        perturb_sigma=args.perturb_sigma,
        seed=args.seed,
        lr=args.lr,
    )
