#!/usr/bin/env python3
"""Phenol: random scatter → free-atom OT → naming → OT+L1 ADMM cleanup.

Pipeline
--------
1. Place N atoms uniformly at random in the map box (no chemistry, no pose).
   Unrestrained Adam OT until NN-RMSD plateaus; then shuffle indices so the
   cloud is unlabelled.
2. ``Namer`` recovers labels (COM-aligned ideal prior; C/N/O weights ignored).
3. Consensus ADMM (OT + L1 + P_restr) from the named pose with relaxed
   geometry slack.

Usage
-----
  uv run python make_ot_name_refine.py
  uv run python make_ot_name_refine.py --resolution 3
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
    run_admm,
    value_grad_fn,
)
from make_ot_unrestrained import nn_rmsd, run_ot_unrestrained
from phenol import (
    NAMES,
    build_phenol,
    embed3,
    phenol_geometry,
    phenol_namer,
)
from targets2d import ConsistentSlicedW1, L1Diff, directions_2d

OUT_DIR = Path(__file__).resolve().parent / "out"

# Post-naming cleanup slack: start loose enough to absorb free-atom jitter,
# but anneal to 0 over CLEANUP_ANNEAL steps — and do not plateau-stop before
# that (otherwise P_restr never bites; last run died at slack≈1.3 Å).
CLEANUP_SLACK0 = 1.0
CLEANUP_SLACK1 = 0.0
CLEANUP_ANNEAL = 40
CLEANUP_MIN_STEPS = CLEANUP_ANNEAL + 20


def _res_tag(resolution: float) -> str:
    return f"{float(resolution):g}".replace(".", "p")


def _throw_away_names(X: np.ndarray, w: np.ndarray, rng: np.random.Generator):
    """Shuffle positions (+weights) → unlabelled cloud."""
    order = rng.permutation(len(X))
    return X[order].copy(), w[order].copy(), order


def _aligned_ideal_prior(X_ideal: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Place the ideal labelled geometry on the free cloud's COM (unary prior)."""
    X_ideal = np.asarray(X_ideal, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return X_ideal - X_ideal.mean(0) + Y.mean(0)


def _random_scatter(
    n: int,
    rng: np.random.Generator,
    *,
    center: np.ndarray,
    half_width: float,
) -> np.ndarray:
    """N atoms i.i.d. uniform in a square about ``center`` (no chemistry)."""
    c = np.asarray(center, dtype=np.float64)
    hw = float(half_width)
    return c + rng.uniform(-hw, hw, size=(int(n), 2)).astype(np.float64)


def _panel(ax, rhoT, full_extent, vmax, xlim, ylim, X_true, X_cur,
           cur_color="#0b5fff", title=""):
    ax.imshow(
        rhoT, origin="lower", extent=full_extent, cmap="YlOrBr",
        vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal",
    )
    ax.plot(
        X_true[:, 0], X_true[:, 1],
        "o", ms=5.0, mfc="none", mec="0.35", mew=1.0, zorder=3,
    )
    ax.plot(
        X_cur[:, 0], X_cur[:, 1],
        "o", ms=5.5, mfc=cur_color, mec=cur_color, zorder=4,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=8, loc="left", pad=2)


def draw_figure(scene, free_cache, named_pose, admm_cache, asn, *,
                resolution: float, out_stem: str,
                chain_style: str = "extended") -> None:
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
    free_best = free_cache["poses"][free_cache["best_step"]]
    admm_best = admm_cache["poses"][admm_cache["best_step"]]
    xmin, xmax, ymin, ymax = roi_limits(
        [X_true, scene["X_start"], free_best, named_pose, admm_best], ROI_PAD,
    )
    xlim, ylim = (xmin, xmax), (ymin, ymax)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6), constrained_layout=True)

    _panel(
        axes[0, 0], rhoT, full_extent, vmax, xlim, ylim,
        X_true, scene["X_start"], cur_color="#1b1b1b",
        title=(
            f"random scatter  label-RMSD {rmsd(scene['X_start'], X_true):.2f} Å  "
            f"NN {nn_rmsd(scene['X_start'], X_true):.2f} Å"
        ),
    )
    axes[0, 0].set_ylabel("pose", fontsize=8)

    _panel(
        axes[0, 1], rhoT, full_extent, vmax, xlim, ylim,
        X_true, free_best, cur_color="#0b5fff",
        title=(
            f"free OT best ({free_cache['best_step']})  "
            f"NN-RMSD {free_cache['nn_rmsds'][free_cache['best_step']]:.3f} Å  "
            f"label {free_cache['label_rmsds'][free_cache['best_step']]:.2f} Å"
        ),
    )

    _panel(
        axes[1, 0], rhoT, full_extent, vmax, xlim, ylim,
        X_true, named_pose, cur_color="#0b5fff",
        title=(
            f"after naming  label-RMSD {rmsd(named_pose, X_true):.3f} Å  "
            f"restr {asn.restraint_rms:.3f}  repaired {asn.n_repaired}"
        ),
    )
    axes[1, 0].set_xlabel(r"$x$ (Å)", fontsize=8)
    axes[1, 0].set_ylabel("pose", fontsize=8)

    _panel(
        axes[1, 1], rhoT, full_extent, vmax, xlim, ylim,
        X_true, admm_best, cur_color="#0b5fff",
        title=(
            f"ADMM OT+L1+P_restr best ({admm_cache['best_step']})  "
            f"RMSD {admm_cache['rmsds'][admm_cache['best_step']]:.3f} Å  "
            f"({admm_cache['n_steps']} steps)"
        ),
    )
    axes[1, 1].set_xlabel(r"$x$ (Å)", fontsize=8)

    handles = [
        Line2D([0], [0], marker="o", color="0.35", mfc="none", ms=5, lw=0,
               label="true"),
        Line2D([0], [0], marker="o", color="#1b1b1b", ms=5, lw=0,
               label="start"),
        Line2D([0], [0], marker="o", color="#0b5fff", ms=5, lw=0,
               label="current"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        fontsize=8, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"ortho-pentyl phenol ({chain_style}) @ {resolution:g} Å · "
        f"random scatter · free OT → Namer → ADMM "
        f"(slack {CLEANUP_SLACK0:g}→{CLEANUP_SLACK1:g} Å)",
        fontsize=10,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_admm_cleanup(name, X0, w, X_true, geom, *, vg_ot, vg_l1):
    """ADMM with sharp slack anneal; plateau stop only after anneal finishes."""
    import math
    from phenol import project_2d

    T = int(CLEANUP_ANNEAL)
    _beta_fn = mf.admm_geom_beta  # capture before patch

    def _slack(t, T=T, s0=CLEANUP_SLACK0, s1=CLEANUP_SLACK1):
        u = float(np.clip(t / max(T, 1), 0.0, 1.0))
        return float(s0 + (s1 - s0) * u)

    def _ot_lr(t, T=T, lr0=mf.ADMM_OT_LR0, lr1=mf.ADMM_OT_LR1):
        u = float(np.clip(t / max(T, 1), 0.0, 1.0))
        ww = 0.5 * (1.0 - math.cos(math.pi * u))
        return float(lr0 + (lr1 - lr0) * ww)

    def _beta(t, T=T, lo=mf.GEOM_BETA_LOW, hi=mf.GEOM_BETA_HIGH, rng=None):
        return _beta_fn(t, T=T, lo=lo, hi=hi, rng=rng)

    mf.geom_slack = _slack
    mf.admm_ot_lr = _ot_lr
    mf.admm_geom_beta = _beta

    cache = run_admm(
        name, X0, w, X_true, geom,
        vg_ot=vg_ot, vg_l1=vg_l1,
        lr_ot0=mf.ADMM_OT_LR0, lr_ot1=mf.ADMM_OT_LR1, lr_l1=L1_LR,
        rho=mf.ADMM_RHO,
        max_steps=max(CLEANUP_MIN_STEPS, 200),
        patience=CLEANUP_MIN_STEPS,  # ≥ anneal length
        rmsd_atol=1e-3,
    )
    # Final snap onto the tight manifold (slack = 0).
    X_fin, _, _ = project_2d(
        geom, cache["poses"][-1], tol=1e-3, max_iter=200, slack=0.0,
    )
    r_fin = rmsd(X_fin, X_true)
    cache = dict(cache)
    cache["poses"] = np.concatenate([cache["poses"], X_fin[None]], axis=0)
    cache["rmsds"] = np.concatenate([cache["rmsds"], [r_fin]])
    cache["energies"] = np.concatenate([
        np.asarray(cache["energies"]), [float(cache["energies"][-1])],
    ])
    cache["grad_norms"] = np.concatenate([
        np.asarray(cache["grad_norms"]), [float(cache["grad_norms"][-1])],
    ])
    cache["n_steps"] = int(cache["n_steps"]) + 1
    cache["best_step"] = int(np.argmin(cache["rmsds"]))
    cache["final_slack0_project"] = True
    return cache


def main(
    resolution: float | None = None,
    seed: int = 0,
    chain_style: str = "extended",
):
    resolution = float(mf.RESOLUTION if resolution is None else resolution)
    mf.RESOLUTION = resolution
    chain_style = str(chain_style).lower()

    tag = _res_tag(resolution)
    chain_tag = "zig" if chain_style.startswith("zig") else "ext"
    out_stem = f"phenol_ot_name_refine_{tag}A_{chain_tag}"
    rng = np.random.default_rng(int(seed))

    scene = build_scene(chain_style=chain_style)
    ot = ConsistentSlicedW1(
        scene["rhoT"], scene["V"], directions_2d(N_DIRS), nbins=320, pad=12.0,
    )
    l1 = L1Diff(scene["rhoT"], scene["V"], scene["sigma"])
    vg_ot = value_grad_fn("ot", ot, scene["sigma"])
    vg_l1 = value_grad_fn("l1", l1, scene["sigma"])

    X_true = scene["X_true"]
    w = scene["w"]
    n_atoms = int(X_true.shape[0])
    # Fully random positions in a box covering the density support (true COM
    # ± ~molecular extent + a few σ).  No bonds, no orientation, no labels.
    half = float(scene["R_full"] + 4.0 * scene["sigma"] + scene["R"])
    X_start = _random_scatter(
        n_atoms, rng, center=X_true.mean(0), half_width=half,
    )
    scene = {**scene, "X_start": X_start}

    # ------------------------------------------------------------------ 1 free OT
    print(
        f"[1/3] free-atom Adam OT  @ {resolution:g} Å  chain={chain_style}  "
        f"from {n_atoms} random scatter  (lr={OT_LR}, no P_restr / L1) ...",
        flush=True,
    )
    print(
        f"  scatter box ±{half:.2f} Å about true COM; "
        f"NN-RMSD={nn_rmsd(X_start, X_true):.3f} Å  "
        f"label={rmsd(X_start, X_true):.3f} Å",
        flush=True,
    )
    free = run_ot_unrestrained(
        X_start, w, X_true, vg_ot, lr=OT_LR,
        max_steps=5000, patience=80,
    )
    ib = int(free["best_step"])
    X_free = free["poses"][ib].copy()
    print(
        f"  free OT: {free['n_steps']} steps ({free['stop_reason']})  "
        f"NN-RMSD {free['nn_rmsds'][0]:.3f} → {free['nn_rmsds'][ib]:.3f} Å  "
        f"label {free['label_rmsds'][ib]:.3f} Å",
        flush=True,
    )

    # Throw away names: shuffle the free cloud.
    Y, wY, order = _throw_away_names(X_free, w, rng)
    print(
        f"  shuffled names (seed={seed}); "
        f"label-RMSD of shuffled cloud {rmsd(Y, X_true):.3f} Å",
        flush=True,
    )

    # ------------------------------------------------------------------ 2 name
    # Prior: ideal labelled geometry on the free-cloud COM.  Random scatter
    # carries no label prior; C/N/O weights are not discriminating (ΔZ < 10).
    print("[2/3] Namer.assign (prior = COM-aligned ideal) ...", flush=True)
    X_ideal, _ = build_phenol(chain_style=chain_style)
    namer = phenol_namer(X_ideal)
    X_prior = _aligned_ideal_prior(X_ideal, Y)
    asn = namer.assign(embed3(Y), embed3(X_prior), weights=None)
    named = asn.Y_named[:, :2].copy()
    print(
        f"  naming: unary_rms={asn.unary_rms:.3f} Å  "
        f"restr_rms={asn.restraint_rms:.3f}  "
        f"repaired={asn.n_repaired}  flags={asn.flags}",
        flush=True,
    )
    print(
        f"  named label-RMSD={rmsd(named, X_true):.3f} Å  "
        f"NN-RMSD={nn_rmsd(named, X_true):.3f} Å",
        flush=True,
    )
    # Chemical recovery: each named atom's nearest true label.
    nearest = [
        NAMES[int(np.argmin(np.linalg.norm(X_true - named[i], axis=1)))]
        for i in range(len(NAMES))
    ]
    n_match = sum(1 for a, b in zip(NAMES, nearest) if a == b)
    print(
        f"  nearest-true label match: {n_match}/{len(NAMES)}  "
        f"O→{nearest[6] if len(nearest) > 6 else '?'}",
        flush=True,
    )

    # ------------------------------------------------------------------ 3 ADMM
    print(
        f"[3/3] ADMM OT+L1+P_restr cleanup  "
        f"(slack {CLEANUP_SLACK0:g}→{CLEANUP_SLACK1:g} Å over "
        f"{CLEANUP_ANNEAL} steps, then slack=0 project) ...",
        flush=True,
    )
    geom = phenol_geometry(X_ideal)
    admm = _run_admm_cleanup(
        "ot_name_refine", named, w, X_true, geom,
        vg_ot=vg_ot, vg_l1=vg_l1,
    )
    jb = int(admm["best_step"])
    print(
        f"  ADMM: {admm['n_steps']} steps ({admm['stop_reason']})  "
        f"RMSD {admm['rmsds'][0]:.3f} → min {admm['rmsds'][jb]:.3f} Å "
        f"(final {admm['rmsds'][-1]:.3f})",
        flush=True,
    )
    if admm.get("slack_hist") is not None and len(admm["slack_hist"]):
        sh = np.asarray(admm["slack_hist"])
        print(
            f"  geom slack: {sh[0]:.3f} → {sh[min(len(sh)-1, CLEANUP_ANNEAL-1)]:.3f} "
            f"→ final project @ 0.0 Å",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"trajectory_{out_stem}.npz",
        free_poses=free["poses"],
        free_nn_rmsds=free["nn_rmsds"],
        free_label_rmsds=free["label_rmsds"],
        shuffle_order=order,
        named_pose=named,
        named_perm=asn.perm,
        named_restraint_rms=np.array(asn.restraint_rms),
        named_unary_rms=np.array(asn.unary_rms),
        admm_poses=admm["poses"],
        admm_rmsds=admm["rmsds"],
        admm_energies=admm["energies"],
        resolution=np.array(resolution),
        cleanup_slack0=np.array(CLEANUP_SLACK0),
        cleanup_slack1=np.array(CLEANUP_SLACK1),
    )

    draw_figure(
        scene, free, named, admm, asn,
        resolution=resolution, out_stem=out_stem, chain_style=chain_style,
    )
    print(f"\nwrote {OUT_DIR / f'{out_stem}.pdf'}")
    print(f"wrote {OUT_DIR / f'{out_stem}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--chain", choices=("extended", "zigzag"), default="extended",
        help="Target / reference chain fold.",
    )
    args = ap.parse_args()
    main(resolution=args.resolution, seed=args.seed, chain_style=args.chain)
