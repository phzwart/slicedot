#!/usr/bin/env python3
"""Unrestrained OT-only phenol reach: no geometry, no L1 / DAC.

Same reach scene as ``make_figure.py`` (90° / 3R start). Pure Adam on the
sliced-W₁ fidelity. Label RMSD is reported for reference; the figure and
stopping criterion use nearest-neighbour (Hungarian) assignment RMSD, since
unrestrained OT may permute chemically similar carbons.

Usage
-----
  uv run python make_ot_unrestrained.py              # default resolution
  uv run python make_ot_unrestrained.py --resolution 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment

import make_figure as mf
from make_figure import (
    Adam,
    MISALIGN_DEG,
    N_DIRS,
    N_SHOW,
    OT_LR,
    PATIENCE,
    ROI_PAD,
    RMSD_ATOL,
    SHIFT_RADII,
    STEP_ATOL,
    build_scene,
    rmsd,
    roi_limits,
    select_frames,
    value_grad_fn,
)
from targets2d import ConsistentSlicedW1, directions_2d

OUT_DIR = Path(__file__).resolve().parent / "out"
MAX_STEPS = 2000
LR = OT_LR


def _res_tag(resolution: float) -> str:
    return f"{float(resolution):g}".replace(".", "p")


def nn_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    """RMSD after optimal bipartite matching (Hungarian on squared distances)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    # (N, N) cost: ||X_i - Y_j||²
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    ri, cj = linear_sum_assignment(d2)
    return float(np.sqrt(d2[ri, cj].mean()))


def nn_match(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_ind, col_ind) for Hungarian match of X → Y."""
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return linear_sum_assignment(d2)


def run_ot_unrestrained(
    X0: np.ndarray,
    w: np.ndarray,
    X_true: np.ndarray,
    vg,
    lr: float = LR,
    max_steps: int = MAX_STEPS,
    atol: float = STEP_ATOL,
    rmsd_atol: float = RMSD_ATOL,
    patience: int = PATIENCE,
) -> dict:
    """Adam on OT only — no P_restr, no L1."""
    X = np.asarray(X0, dtype=np.float64).copy()
    opt = Adam(X.shape, lr=float(lr))
    E0, G0 = vg(X, w)

    energies = [float(E0)]
    grad_norms = [float(np.linalg.norm(G0, axis=1).mean())]
    label_rmsds = [rmsd(X, X_true)]
    nn_rmsds = [nn_rmsd(X, X_true)]
    poses = [X.copy()]
    step_sizes = [0.0]

    best = nn_rmsds[0]
    stagnant_step = 0
    stagnant_rmsd = 0
    reason = "max_steps"

    for _ in range(max_steps):
        X_prev = X
        E, G = vg(X, w)
        X = opt.step(X, G)
        ds = float(np.linalg.norm(X - X_prev, axis=1).mean())
        r_lab = rmsd(X, X_true)
        r_nn = nn_rmsd(X, X_true)

        poses.append(X.copy())
        energies.append(float(E))
        grad_norms.append(float(np.linalg.norm(G, axis=1).mean()))
        label_rmsds.append(r_lab)
        nn_rmsds.append(r_nn)
        step_sizes.append(ds)

        if r_nn < best - rmsd_atol:
            best = r_nn
            stagnant_rmsd = 0
        else:
            stagnant_rmsd += 1
        if ds < atol:
            stagnant_step += 1
        else:
            stagnant_step = 0
        if stagnant_step >= patience:
            reason = "step_atol"
            break
        if stagnant_rmsd >= patience:
            reason = "nn_rmsd_plateau"
            break

    nn_arr = np.asarray(nn_rmsds)
    return {
        "name": "ot_free",
        "poses": np.stack(poses, axis=0),
        "energies": np.asarray(energies),
        "grad_norms": np.asarray(grad_norms),
        "rmsds": nn_arr,  # primary metric used by select_frames / titles
        "label_rmsds": np.asarray(label_rmsds),
        "nn_rmsds": nn_arr,
        "step_sizes": np.asarray(step_sizes),
        "n_steps": len(poses) - 1,
        "converged": reason != "max_steps",
        "stop_reason": reason,
        "best_step": int(np.argmin(nn_arr)),
        "used_geometry": False,
        "lr": float(lr),
    }


def _panel(ax, rhoT, full_extent, vmax, xlim, ylim, X_true, X_cur,
           G=None, quiver_scale=None, match_lines=False):
    ax.imshow(
        rhoT, origin="lower", extent=full_extent, cmap="YlOrBr",
        vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal",
    )
    ax.plot(
        X_true[:, 0], X_true[:, 1],
        "o", ms=5.0, mfc="none", mec="0.35", mew=1.0, zorder=3,
    )
    if match_lines:
        ri, cj = nn_match(X_cur, X_true)
        for i, j in zip(ri, cj):
            ax.plot(
                [X_cur[i, 0], X_true[j, 0]],
                [X_cur[i, 1], X_true[j, 1]],
                "-", color="#6a6a6a", lw=0.6, alpha=0.55, zorder=2,
            )
    ax.plot(
        X_cur[:, 0], X_cur[:, 1],
        "o", ms=5.5, mfc="#0b5fff", mec="#0b5fff", zorder=4,
    )
    if G is not None:
        ax.quiver(
            X_cur[:, 0], X_cur[:, 1], -G[:, 0], -G[:, 1],
            angles="xy", scale_units="xy", scale=quiver_scale,
            width=0.012, color="#b33a3a", zorder=5,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def draw_figure(scene, cache, show_idxs, G0, *, resolution: float,
                out_stem: str) -> None:
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
    mean_g = float(np.linalg.norm(G0, axis=1).mean())
    quiver_scale = mean_g / (0.45 * scene["R"]) if mean_g > 0 else 1.0
    vmax = float(rhoT.max())

    show_poses = [cache["poses"][i] for i in show_idxs]
    xmin, xmax, ymin, ymax = roi_limits([X_true, *show_poses], ROI_PAD)
    xlim, ylim = (xmin, xmax), (ymin, ymax)

    n = len(show_idxs)
    fig = plt.figure(figsize=(5.4, 2.05 * n + 2.0))
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1] * n + [0.85], hspace=0.32)

    for row, idx in enumerate(show_idxs):
        ax = fig.add_subplot(gs[row, 0])
        is_start = idx == 0
        is_best = idx == cache["best_step"]
        _panel(
            ax, rhoT, full_extent, vmax, xlim, ylim, X_true,
            cache["poses"][idx],
            G=G0 if is_start else None,
            quiver_scale=quiver_scale,
            match_lines=not is_start,
        )
        if is_start:
            label = "start"
        elif is_best:
            label = f"best\n({idx})"
        else:
            label = f"step {idx}"
        ax.set_ylabel(
            f"OT free\n{label}", fontsize=8, rotation=0,
            ha="right", va="center", labelpad=36,
        )
        gnote = ""
        if is_start:
            gnote = f"   $\\langle\\|g\\|\\rangle$={cache['grad_norms'][0]:.2e}"
        ax.set_title(
            f"NN-RMSD {cache['nn_rmsds'][idx]:.3f} Å"
            f"   (label {cache['label_rmsds'][idx]:.3f} Å){gnote}",
            fontsize=8, loc="left", pad=2,
        )
        if row < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$x$ (Å)", fontsize=8)

    ax = fig.add_subplot(gs[-1, 0])
    t = np.arange(len(cache["nn_rmsds"]))
    ax.plot(t, cache["nn_rmsds"], color="#0b5fff", lw=1.4, label="NN-matched")
    ax.plot(t, cache["label_rmsds"], color="#8a5a2b", lw=1.1, ls="--",
            label="label (fixed index)")
    ax.axvline(cache["best_step"], color="0.45", lw=0.8, ls=":")
    ax.set_xlabel("Adam step", fontsize=8)
    ax.set_ylabel("RMSD (Å)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_xlim(0, max(t[-1], 1))
    ax.set_ylim(bottom=0.0)

    handles = [
        Line2D([0], [0], marker="o", color="0.35", mfc="none", ms=5, lw=0,
               label="true"),
        Line2D([0], [0], marker="o", color="#0b5fff", ms=5, lw=0,
               label="current"),
        Line2D([0], [0], color="#b33a3a", lw=2,
               label=r"$-\nabla E$ at start"),
        Line2D([0], [0], color="#6a6a6a", lw=1,
               label="Hungarian match"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        fontsize=8, bbox_to_anchor=(0.55, -0.01),
    )
    fig.suptitle(
        f"ortho-pentyl phenol @ {resolution:g} Å · {MISALIGN_DEG:.0f}° · "
        f"{SHIFT_RADII}$R$ ($R$={scene['R']:.2f} Å) · "
        f"Adam OT only (no $P_{{\\mathrm{{restr}}}}$, no L1) · "
        f"lr={cache['lr']:g} · {cache['n_steps']} steps",
        fontsize=9, y=0.995,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(resolution: float | None = None):
    resolution = float(mf.RESOLUTION if resolution is None else resolution)
    mf.RESOLUTION = resolution
    tag = _res_tag(resolution)
    out_stem = f"phenol_ot_unrestrained_{tag}A"
    traj_path = OUT_DIR / f"trajectory_ot_unrestrained_{tag}A.npz"

    scene = build_scene()
    ot = ConsistentSlicedW1(
        scene["rhoT"], scene["V"], directions_2d(N_DIRS), nbins=320, pad=12.0,
    )
    vg = value_grad_fn("ot", ot, scene["sigma"])
    X0, w, X_true = scene["X_start"], scene["w"], scene["X_true"]

    print(
        f"Unrestrained Adam OT  (lr={LR}, no P_restr, no L1)  "
        f"@ {resolution:g} Å, {MISALIGN_DEG:.0f}° / {SHIFT_RADII}R",
        flush=True,
    )
    print(
        f"  start  label-RMSD={rmsd(X0, X_true):.3f} Å  "
        f"NN-RMSD={nn_rmsd(X0, X_true):.3f} Å",
        flush=True,
    )

    cache = run_ot_unrestrained(X0, w, X_true, vg, lr=LR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        traj_path,
        poses=cache["poses"],
        energies=cache["energies"],
        grad_norms=cache["grad_norms"],
        label_rmsds=cache["label_rmsds"],
        nn_rmsds=cache["nn_rmsds"],
        step_sizes=cache["step_sizes"],
        n_steps=np.array(cache["n_steps"]),
        best_step=np.array(cache["best_step"]),
        stop_reason=np.array(cache["stop_reason"]),
        lr=np.array(cache["lr"]),
        resolution=np.array(resolution),
    )

    b = cache["best_step"]
    print(
        f"  stopped: {cache['stop_reason']} after {cache['n_steps']} steps",
        flush=True,
    )
    print(
        f"  NN-RMSD:    {cache['nn_rmsds'][0]:.3f} → "
        f"min {cache['nn_rmsds'].min():.3f} (step {b}) → "
        f"final {cache['nn_rmsds'][-1]:.3f} Å",
        flush=True,
    )
    print(
        f"  label-RMSD: {cache['label_rmsds'][0]:.3f} → "
        f"min {cache['label_rmsds'].min():.3f} → "
        f"final {cache['label_rmsds'][-1]:.3f} Å",
        flush=True,
    )
    print(
        f"  E: {cache['energies'][0]:.6g} → final {cache['energies'][-1]:.6g}",
        flush=True,
    )

    for k in (1, 2, 5, 10, 20):
        if k < len(cache["nn_rmsds"]):
            print(
                f"  step {k:3d}: NN-RMSD={cache['nn_rmsds'][k]:.3f} Å  "
                f"label={cache['label_rmsds'][k]:.3f} Å  "
                f"E={cache['energies'][k]:.6g}",
                flush=True,
            )

    _, G0 = vg(X0, w)
    show_idxs = select_frames(cache, N_SHOW)
    print(f"  show frames: {show_idxs}", flush=True)
    draw_figure(scene, cache, show_idxs, G0, resolution=resolution,
                out_stem=out_stem)
    print(f"\nwrote {OUT_DIR / f'{out_stem}.pdf'}")
    print(f"wrote {OUT_DIR / f'{out_stem}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--resolution", type=float, default=None,
        help="Map resolution in Å (default: make_figure.RESOLUTION).",
    )
    args = ap.parse_args()
    main(resolution=args.resolution)
