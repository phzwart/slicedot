#!/usr/bin/env python3
"""Multi-seed free-OT → name → ADMM; wire-frame overlay of cleaned poses.

Usage
-----
  PYTHONPATH=../../../src python make_ot_name_refine_ensemble.py \\
      --resolution 2 --chain zigzag --n-seeds 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import make_figure as mf
from make_figure import (
    L1_LR,
    N_DIRS,
    OT_LR,
    ROI_PAD,
    build_scene,
    rmsd,
    roi_limits,
    value_grad_fn,
)
from make_ot_name_refine import (
    CLEANUP_ANNEAL,
    CLEANUP_SLACK0,
    CLEANUP_SLACK1,
    OUT_DIR,
    _aligned_ideal_prior,
    _random_scatter,
    _res_tag,
    _run_admm_cleanup,
    _throw_away_names,
)
from make_ot_unrestrained import nn_rmsd, run_ot_unrestrained
from phenol import BONDS, NAMES, build_phenol, embed3, phenol_geometry, phenol_namer
from targets2d import ConsistentSlicedW1, L1Diff, directions_2d


def run_one(
    *,
    resolution: float,
    chain_style: str,
    seed: int,
    scene: dict,
    vg_ot,
    vg_l1,
) -> dict:
    rng = np.random.default_rng(int(seed))
    X_true = scene["X_true"]
    w = scene["w"]
    n_atoms = int(X_true.shape[0])
    half = float(scene["R_full"] + 4.0 * scene["sigma"] + scene["R"])
    X_start = _random_scatter(
        n_atoms, rng, center=X_true.mean(0), half_width=half,
    )

    free = run_ot_unrestrained(
        X_start, w, X_true, vg_ot, lr=OT_LR,
        max_steps=5000, patience=80,
    )
    ib = int(free["best_step"])
    X_free = free["poses"][ib].copy()
    Y, wY, order = _throw_away_names(X_free, w, rng)

    X_ideal, _ = build_phenol(chain_style=chain_style)
    namer = phenol_namer(X_ideal)
    X_prior = _aligned_ideal_prior(X_ideal, Y)
    asn = namer.assign(embed3(Y), embed3(X_prior), weights=None)
    named = asn.Y_named[:, :2].copy()
    nearest = [
        NAMES[int(np.argmin(np.linalg.norm(X_true - named[i], axis=1)))]
        for i in range(len(NAMES))
    ]
    n_match = sum(1 for a, b in zip(NAMES, nearest) if a == b)

    geom = phenol_geometry(X_ideal)
    admm = _run_admm_cleanup(
        "ot_name_refine", named, w, X_true, geom,
        vg_ot=vg_ot, vg_l1=vg_l1,
    )
    jb = int(admm["best_step"])
    cleaned = admm["poses"][jb].copy()
    return {
        "seed": int(seed),
        "free_nn": float(free["nn_rmsds"][ib]),
        "named_rmsd": float(rmsd(named, X_true)),
        "n_match": int(n_match),
        "restr_rms": float(asn.restraint_rms),
        "admm_rmsd": float(admm["rmsds"][jb]),
        "admm_final": float(admm["rmsds"][-1]),
        "named": named,
        "cleaned": cleaned,
        "X_start": X_start,
    }


def _draw_wire(ax, X, bonds, *, color, lw, alpha, zorder=3, ms=2.5):
    X = np.asarray(X, dtype=np.float64)
    for i, j in bonds:
        ax.plot(
            [X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]],
            "-", color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round",
        )
    ax.plot(
        X[:, 0], X[:, 1],
        "o", ms=ms, mfc=color, mec=color, alpha=alpha, zorder=zorder + 1,
    )


def draw_overlay(
    scene: dict,
    results: list[dict],
    *,
    resolution: float,
    chain_style: str,
    out_stem: str,
) -> None:
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
    vmax = float(rhoT.max())

    poses = [r["cleaned"] for r in results] + [r["named"] for r in results]
    xmin, xmax, ymin, ymax = roi_limits([X_true, *poses], ROI_PAD)
    xlim, ylim = (xmin, xmax), (ymin, ymax)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), constrained_layout=True)

    # Colour seeds by ADMM RMSD (green = good, rust = poor).
    rms = np.array([r["admm_rmsd"] for r in results], dtype=np.float64)
    r_lo, r_hi = float(rms.min()), float(max(rms.max(), rms.min() + 1e-6))
    cmap = plt.get_cmap("viridis")

    for ax, key, title in (
        (axes[0], "named", "after naming"),
        (axes[1], "cleaned", "ADMM cleaned (best)"),
    ):
        ax.imshow(
            rhoT, origin="lower", extent=full_extent, cmap="YlOrBr",
            vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal",
            zorder=0,
        )
        for r in results:
            t = (r["admm_rmsd"] - r_lo) / (r_hi - r_lo)
            col = cmap(0.15 + 0.75 * t)
            _draw_wire(
                ax, r[key], BONDS,
                color=col, lw=1.0, alpha=0.55, zorder=3, ms=2.2,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r"$x$ (Å)", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left", pad=3)

    axes[0].set_ylabel(r"$y$ (Å)", fontsize=8)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=r_lo, vmax=r_hi),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("ADMM best RMSD (Å)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    handles = [
        Line2D([0], [0], color=cmap(0.5), lw=1.0, alpha=0.7,
               label=f"seed runs (n={len(results)})"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=1, frameon=False,
        fontsize=8, bbox_to_anchor=(0.45, -0.06),
    )
    fig.suptitle(
        f"ortho-pentyl phenol ({chain_style}) @ {resolution:g} Å · "
        f"{len(results)} seeds · free OT → Namer → ADMM "
        f"(slack {CLEANUP_SLACK0:g}→{CLEANUP_SLACK1:g} Å / {CLEANUP_ANNEAL} steps)",
        fontsize=10,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(
    resolution: float = 2.0,
    chain_style: str = "zigzag",
    n_seeds: int = 10,
    seed0: int = 0,
):
    resolution = float(resolution)
    mf.RESOLUTION = resolution
    chain_style = str(chain_style).lower()
    tag = _res_tag(resolution)
    chain_tag = "zig" if chain_style.startswith("zig") else "ext"
    out_stem = f"phenol_ot_name_refine_{tag}A_{chain_tag}_n{n_seeds}"

    scene = build_scene(chain_style=chain_style)
    ot = ConsistentSlicedW1(
        scene["rhoT"], scene["V"], directions_2d(N_DIRS), nbins=320, pad=12.0,
    )
    l1 = L1Diff(scene["rhoT"], scene["V"], scene["sigma"])
    vg_ot = value_grad_fn("ot", ot, scene["sigma"])
    vg_l1 = value_grad_fn("l1", l1, scene["sigma"])

    results = []
    print(
        f"ensemble: {n_seeds} seeds @ {resolution:g} Å  chain={chain_style}  "
        f"seed0={seed0}",
        flush=True,
    )
    for k in range(int(n_seeds)):
        seed = int(seed0) + k
        print(f"\n=== seed {seed} ({k + 1}/{n_seeds}) ===", flush=True)
        r = run_one(
            resolution=resolution,
            chain_style=chain_style,
            seed=seed,
            scene=scene,
            vg_ot=vg_ot,
            vg_l1=vg_l1,
        )
        results.append(r)
        print(
            f"  free NN {r['free_nn']:.3f} Å · "
            f"named {r['named_rmsd']:.3f} Å ({r['n_match']}/12) · "
            f"ADMM best {r['admm_rmsd']:.3f} Å (final {r['admm_final']:.3f})",
            flush=True,
        )

    print("\nsummary:", flush=True)
    print(
        f"  naming match: "
        f"{np.mean([r['n_match'] for r in results]):.1f}/12  "
        f"(min {min(r['n_match'] for r in results)}, "
        f"max {max(r['n_match'] for r in results)})",
        flush=True,
    )
    print(
        f"  ADMM best RMSD: "
        f"mean {np.mean([r['admm_rmsd'] for r in results]):.3f} Å  "
        f"median {np.median([r['admm_rmsd'] for r in results]):.3f} Å  "
        f"min {min(r['admm_rmsd'] for r in results):.3f}  "
        f"max {max(r['admm_rmsd'] for r in results):.3f}",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"trajectory_{out_stem}.npz",
        seeds=np.array([r["seed"] for r in results]),
        free_nn=np.array([r["free_nn"] for r in results]),
        named_rmsd=np.array([r["named_rmsd"] for r in results]),
        n_match=np.array([r["n_match"] for r in results]),
        admm_rmsd=np.array([r["admm_rmsd"] for r in results]),
        named_poses=np.stack([r["named"] for r in results], axis=0),
        cleaned_poses=np.stack([r["cleaned"] for r in results], axis=0),
        resolution=np.array(resolution),
    )
    draw_overlay(
        scene, results,
        resolution=resolution, chain_style=chain_style, out_stem=out_stem,
    )
    print(f"\nwrote {OUT_DIR / f'{out_stem}.pdf'}")
    print(f"wrote {OUT_DIR / f'{out_stem}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--chain", choices=("extended", "zigzag"), default="zigzag")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    main(
        resolution=args.resolution,
        chain_style=args.chain,
        n_seeds=args.n_seeds,
        seed0=args.seed0,
    )
