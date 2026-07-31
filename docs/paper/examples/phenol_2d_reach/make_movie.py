#!/usr/bin/env python3
"""3×3 movie: ADMM OT | L1 | OT+L1 × three start poses.

Rows (shared density / true pose)
---------------------------------
  1. 90° + 3R reach (COM translated)
  2. 90° rotation, COM matched
  3. 180° rotation, COM matched

Writes ``out/phenol_2d_reach_movie_<res>.gif``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from phenol import BONDS, build_phenol, phenol_geometry, project_2d
from targets2d import ConsistentSlicedW1, L1Diff, directions_2d
import make_figure as mf
from make_figure import (
    ADMM_OT_LR0,
    ADMM_OT_LR1,
    ADMM_RHO,
    GEOM_BETA_HIGH,
    GEOM_BETA_LOW,
    GEOM_SLACK0,
    GEOM_SLACK1,
    GEOM_TOL,
    L1_LR,
    N_DIRS,
    OUT_DIR,
    SHIFT_RADII,
    USE_GEOMETRY,
    build_scene,
    run_admm,
    save_cache,
    start_pose,
    value_grad_fn,
)

# Columns left → right
COLS = (
    ("ot_only", "OT"),
    ("l1", "L1"),
    ("admm", "OT+L1"),
)

# Rows top → bottom: (key, short label, misalign_deg, shift_radii)
ROWS = (
    ("reach", r"90° + 3$R$", 90.0, SHIFT_RADII),
    ("rot90", r"90°, COM", 90.0, 0.0),
    ("rot180", r"180°, COM", 180.0, 0.0),
)


def _load_or_none(path: Path) -> dict | None:
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=True)
    return {
        "name": path.stem.replace("trajectory_", ""),
        "poses": z["poses"],
        "energies": z["energies"],
        "rmsds": z["rmsds"],
        "n_steps": int(z["n_steps"]),
        "best_step": int(z["best_step"]),
        "stop_reason": str(z["stop_reason"]),
    }


def _tag(resolution: float, chain_style: str = "extended") -> str:
    """Filename tag, e.g. 1.5 → '1p5A', zigzag+3.0 → 'zigzag_3A'."""
    s = f"{resolution:g}".replace(".", "p")
    style = str(chain_style).lower().replace("-", "")
    if style in ("zigzag", "zipzag"):
        return f"zigzag_{s}A"
    return f"{s}A"


def _idealize_start(geom, X_start: np.ndarray) -> np.ndarray:
    if geom is None:
        return np.asarray(X_start, dtype=np.float64)
    X0, _, _ = project_2d(geom, X_start, tol=GEOM_TOL, slack=0.0)
    return X0


def run_all(
    reuse: bool, resolution: float, chain_style: str = "extended",
) -> tuple[dict, dict[str, dict[str, dict]]]:
    """Return (scene, caches[row_key][col_key])."""
    mf.RESOLUTION = float(resolution)
    chain_style = str(chain_style)
    # Shared map / true pose from the reach scene (shift sizes the grid).
    scene = build_scene(
        misalign_deg=90.0, shift_radii=SHIFT_RADII, chain_style=chain_style,
    )
    rhoT, V, origin, dx, sig = (
        scene["rhoT"], scene["V"], scene["origin"], scene["dx"], scene["sigma"],
    )
    print(
        f"scene @ {resolution:g} Å  chain={chain_style}  "
        f"(σ={sig:.3f} Å, R_core={scene['R']:.2f} Å, R_full={scene['R_full']:.2f} Å)",
        flush=True,
    )
    ot = ConsistentSlicedW1(rhoT, V, directions_2d(N_DIRS), nbins=320, pad=12.0)
    l1 = L1Diff(rhoT, V, sig)

    # Geometry ideal matches the target chain fold.
    X_ideal = scene["X0"]
    geom = phenol_geometry(X_ideal) if USE_GEOMETRY else None
    if geom is not None:
        print(
            f"P_restr (ADMM β∈[{GEOM_BETA_LOW:g},{GEOM_BETA_HIGH:g}], "
            f"slack {GEOM_SLACK0}→{GEOM_SLACK1} Å); starts idealised",
            flush=True,
        )

    vg_ot = value_grad_fn("ot", ot, sig)
    vg_l1 = value_grad_fn("l1", l1, sig)
    vg_kwargs = {
        "ot_only": dict(vg_ot=vg_ot),
        "l1": dict(vg_l1=vg_l1),
        "admm": dict(vg_ot=vg_ot, vg_l1=vg_l1),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    caches: dict[str, dict[str, dict]] = {}
    tag = _tag(resolution, chain_style)
    w, X_true = scene["w"], scene["X_true"]
    X0, true_com, R = scene["X0"], scene["true_com"], scene["R"]

    for row_key, row_label, misalign, shift in ROWS:
        X_start = _idealize_start(
            geom, start_pose(X0, true_com, R, misalign, shift),
        )
        caches[row_key] = {}
        print(f"\n=== row {row_label}  (misalign={misalign:g}°, shift={shift:g} R) ===",
              flush=True)
        for col_key, col_label in COLS:
            path = OUT_DIR / f"trajectory_{row_key}_{col_key}_{tag}.npz"
            cache = _load_or_none(path) if reuse else None
            if cache is None:
                print(f"running ADMM {col_label}+P_restr ...", flush=True)
                cache = run_admm(
                    f"{row_key}_{col_key}", X_start, w, X_true, geom=geom,
                    lr_ot0=ADMM_OT_LR0, lr_ot1=ADMM_OT_LR1, lr_l1=L1_LR,
                    rho=ADMM_RHO, **vg_kwargs[col_key],
                )
                save_cache(cache, path)
                print(
                    f"  {path.name}: {cache['n_steps']} steps, "
                    f"best RMSD {cache['rmsds'].min():.3f} Å",
                    flush=True,
                )
            else:
                print(
                    f"reusing {path.name}: {cache['n_steps']} steps, "
                    f"best RMSD {cache['rmsds'].min():.3f} Å",
                    flush=True,
                )
            caches[row_key][col_key] = cache

    return scene, caches


def make_movie(
    scene: dict,
    caches: dict[str, dict[str, dict]],
    out_path: Path,
    fps: int = 12,
    max_frames: int = 240,
) -> Path:
    rhoT, origin, dx, shape = (
        scene["rhoT"], scene["origin"], scene["dx"], scene["shape"],
    )
    Ny, Nx = shape
    extent = [
        origin[0] - 0.5 * dx,
        origin[0] + (Nx - 0.5) * dx,
        origin[1] - 0.5 * dx,
        origin[1] + (Ny - 0.5) * dx,
    ]
    vmax = float(rhoT.max())
    X_true = scene["X_true"]

    T_max = max(
        caches[rk][ck]["n_steps"] for rk, *_ in ROWS for ck, _ in COLS
    )
    n_frames = min(T_max + 1, max_frames)
    global_steps = np.linspace(0, T_max, n_frames).round().astype(int)

    pts = [X_true]
    for rk, *_ in ROWS:
        for ck, *_ in COLS:
            pts.append(caches[rk][ck]["poses"].reshape(-1, 2))
    pts = np.concatenate(pts, axis=0)
    pad = 1.5
    xmin, ymin = pts.min(0) - pad
    xmax, ymax = pts.max(0) + pad
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    colors = {
        "ot_only": "#0b5fff",
        "l1": "#c45c26",
        "admm": "#1a7a3a",
    }

    fig, axes = plt.subplots(3, 3, figsize=(9.6, 9.2), constrained_layout=True)
    artists = []  # (row_key, col_key, bond_lines, pts_line, txt)
    for r, (row_key, row_label, _, _) in enumerate(ROWS):
        for c, (col_key, col_label) in enumerate(COLS):
            ax = axes[r, c]
            ax.imshow(
                rhoT, origin="lower", extent=extent, cmap="YlOrBr",
                vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal",
            )
            for i, j in BONDS:
                ax.plot(
                    [X_true[i, 0], X_true[j, 0]],
                    [X_true[i, 1], X_true[j, 1]],
                    "-", color="0.65", lw=0.7, zorder=1,
                )
            ax.plot(
                X_true[:, 0], X_true[:, 1],
                "o", ms=2.8, mfc="none", mec="0.4", mew=0.65, zorder=2,
            )
            bond_lines = []
            for _ in BONDS:
                (ln,) = ax.plot(
                    [], [], "-", color=colors[col_key], lw=1.2, zorder=3,
                )
                bond_lines.append(ln)
            (pts_line,) = ax.plot(
                [], [], "o", ms=3.6, color=colors[col_key], zorder=4,
            )
            txt = ax.text(
                0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                fontsize=7, color="0.15",
                bbox=dict(
                    boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75,
                ),
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(col_label, fontsize=11)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=10)
            if r == len(ROWS) - 1:
                ax.set_xlabel(r"$x$ (Å)", fontsize=8)
            else:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])
            artists.append((row_key, col_key, bond_lines, pts_line, txt))

    res = float(mf.RESOLUTION)
    chain = scene.get("chain_style", "extended")
    fig.suptitle(
        f"ortho-pentyl phenol ({chain}) @ {res:g} Å · ADMM + P_restr "
        f"(OT lr {ADMM_OT_LR0}→{ADMM_OT_LR1}, ρ={ADMM_RHO:g}, "
        f"β∈[{GEOM_BETA_LOW:g},{GEOM_BETA_HIGH:g}], "
        f"slack {GEOM_SLACK0}→{GEOM_SLACK1})",
        fontsize=11,
    )

    def update(fi: int):
        step = int(global_steps[fi])
        outs = []
        for row_key, col_key, bond_lines, pts_line, txt in artists:
            cache = caches[row_key][col_key]
            i = min(step, cache["n_steps"])
            X = cache["poses"][i]
            for ln, (a, b) in zip(bond_lines, BONDS):
                ln.set_data([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]])
                outs.append(ln)
            pts_line.set_data(X[:, 0], X[:, 1])
            outs.append(pts_line)
            held = " (hold)" if step > cache["n_steps"] else ""
            txt.set_text(f"step {i}{held}\nRMSD {cache['rmsds'][i]:.2f} Å")
            outs.append(txt)
        return outs

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"writing {out_path}  ({n_frames} frames, T_max={T_max}, fps={fps}) ...",
        flush=True,
    )
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--resolution", type=float, default=None,
        help="Map resolution in Å (default: make_figure.RESOLUTION).",
    )
    ap.add_argument(
        "--chain", choices=("extended", "zigzag"), default="extended",
        help="Target/start chain fold (default: extended).",
    )
    ap.add_argument(
        "--reuse", action="store_true",
        help="Reuse trajectory_*_<res>.npz if present (still runs missing ones).",
    )
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output GIF path (default: out/phenol_2d_reach_movie_<tag>.gif).",
    )
    args = ap.parse_args()
    resolution = float(args.resolution if args.resolution is not None else mf.RESOLUTION)
    tag = _tag(resolution, args.chain)
    out = args.output or (OUT_DIR / f"phenol_2d_reach_movie_{tag}.gif")

    scene, caches = run_all(
        reuse=args.reuse, resolution=resolution, chain_style=args.chain,
    )
    for row_key, row_label, _, _ in ROWS:
        for col_key, col_label in COLS:
            c = caches[row_key][col_key]
            print(
                f"  [{row_label:12s} | {col_label:6s}] steps={c['n_steps']:4d}  "
                f"RMSD {c['rmsds'][0]:.2f} → {c['rmsds'].min():.3f} Å "
                f"(best @{c['best_step']})"
            )
    path = make_movie(
        scene, caches, out, fps=args.fps, max_frames=args.max_frames,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
