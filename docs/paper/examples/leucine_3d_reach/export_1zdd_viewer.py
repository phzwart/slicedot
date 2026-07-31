#!/usr/bin/env python3
"""Export 1ZDD free-OT placement assets for the web viewer.

Reads ``out/1zdd_free_ot_<res>A_seed*.npz`` and writes into
``viewer_1zdd/data/``:
  * structure.json — true / start / final coords, bonds, box, NN matches
  * density.cube  — rendered 2 Å map
  * meta.json     — labels and isovalue hints
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from export_viewer_assets import write_cube, _res_info, Z_TO_ELEM, render_ortho

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DATA = ROOT / "viewer_1zdd" / "data"


def nn_pairs(X: np.ndarray, Y: np.ndarray) -> list[list[int]]:
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    ri, cj = linear_sum_assignment(d2)
    return [[int(i), int(j)] for i, j in zip(ri, cj)]


def nn_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    ri, cj = linear_sum_assignment(d2)
    return float(np.sqrt(d2[ri, cj].mean()))


def subsample_poses(poses: np.ndarray, max_frames: int = 80) -> np.ndarray:
    n = len(poses)
    if n <= max_frames:
        return poses
    idx = np.unique(np.linspace(0, n - 1, max_frames).astype(int))
    # Always keep first and last.
    idx[0], idx[-1] = 0, n - 1
    return poses[idx]


def main(npz_path: Path | None = None, seed: int = 0, resolution: float = 2.0):
    if npz_path is None:
        tag = f"{float(resolution):g}".replace(".", "p")
        npz_path = OUT / f"1zdd_free_ot_{tag}A_seed{seed}.npz"
    if not npz_path.is_file():
        raise SystemExit(
            f"missing {npz_path}; run run_1zdd_free_ot.py first"
        )

    z = np.load(npz_path, allow_pickle=False)
    X_true = np.asarray(z["X_true"], dtype=np.float64)
    X0 = np.asarray(z["X0"], dtype=np.float64)
    X_final = np.asarray(z["X_final"], dtype=np.float64)
    W = np.asarray(z["W"], dtype=np.float64)
    origin = np.asarray(z["origin"], dtype=np.float64)
    spacing = np.asarray(z["spacing"], dtype=np.float64)
    NG = tuple(int(x) for x in z["NG"])
    sigma = float(z["sigma"])
    res = float(z["resolution"])
    seq = str(z["sequence"])
    names = [str(n) for n in z["names"]]
    Z = np.asarray(z["Z"], dtype=np.float64)
    bonds = sorted({
        (int(a), int(b)) if a < b else (int(b), int(a))
        for a, b in np.asarray(z["bonds"])
    })
    nn = np.asarray(z["nn_rmsds"], dtype=np.float64)
    energies = np.asarray(z["energies"], dtype=np.float64)
    poses = np.asarray(z["poses"], dtype=np.float64) if "poses" in z.files else None

    # Rebuild density with the same Gaussian model as the OT run.
    T, org, sp = render_ortho(X_true, float(spacing.ravel()[0]), NG, sigma, W)
    # Prefer stored origin/spacing if they already match.
    if np.allclose(org, origin) and np.allclose(sp, spacing):
        pass
    else:
        # Fall back to stored geometry; re-render into that box.
        T, org, sp = render_ortho(X_true, spacing, NG, sigma, W)
        org, sp = origin, spacing

    half = 0.5 * (NG[0] - 1) * float(sp.ravel()[0])
    box = {
        "center": X_true.mean(0).tolist(),
        "half": float(half),
        "min": (X_true.mean(0) - half).tolist(),
        "max": (X_true.mean(0) + half).tolist(),
    }

    seq_tuple = tuple(seq)
    atoms = []
    for i, (xyz, name, zz) in enumerate(zip(X_true, names, Z)):
        aname, resn, resid = _res_info(str(name), seq_tuple)
        atoms.append({
            "i": i,
            "name": aname,
            "full": str(name),
            "elem": Z_TO_ELEM.get(int(zz), "C"),
            "resn": resn,
            "resi": resid,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        })

    path_coords = None
    path_nn = None
    path_E = None
    if poses is not None:
        sub = subsample_poses(poses, max_frames=80)
        # Match subsampled indices for metrics.
        if len(poses) == len(nn):
            idx = np.unique(np.linspace(0, len(poses) - 1, len(sub)).astype(int))
            idx[0], idx[-1] = 0, len(poses) - 1
            path_nn = nn[idx].tolist()
            path_E = energies[idx].tolist()
        path_coords = sub.tolist()

    structure = {
        "atoms": atoms,
        "bonds": [list(p) for p in bonds],
        "true_coords": X_true.tolist(),
        "start_coords": X0.tolist(),
        "final_coords": X_final.tolist(),
        "nn_start_pairs": nn_pairs(X0, X_true),
        "nn_final_pairs": nn_pairs(X_final, X_true),
        "box": box,
        "path_coords": path_coords,
        "path_nn_rmsd": path_nn,
        "path_energy": path_E,
    }

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "structure.json").write_text(json.dumps(structure) + "\n")
    write_cube(
        DATA / "density.cube", T, org, sp,
        title=f"1ZDD Z34C @{res:g}A",
    )

    vals = np.sort(T.ravel())[::-1]
    csum = np.cumsum(vals)
    iso = float(vals[int(np.searchsorted(csum, 0.35 * csum[-1]))])
    meta = {
        "label": "1ZDD · Z34C helix–loop–helix",
        "sequence": list(seq),
        "pdb_id": "1ZDD",
        "resolution": res,
        "sigma": sigma,
        "n_atoms": int(X_true.shape[0]),
        "n_residues": len(seq),
        "n_bonds": len(bonds),
        "grid": list(NG),
        "spacing": float(sp.ravel()[0]),
        "box_half": float(half),
        "seed": int(z["seed"]),
        "nn_rmsd_start": float(nn[0]),
        "nn_rmsd_best": float(nn.min()),
        "nn_rmsd_final": float(nn[-1]),
        "nn_rmsd_start_check": nn_rmsd(X0, X_true),
        "nn_rmsd_final_check": nn_rmsd(X_final, X_true),
        "n_steps": int(len(nn) - 1),
        "stop_reason": str(z["stop_reason"]),
        "iso_default": iso,
        "iso_high": float(iso * 2.5),
        "iso_low": float(iso * 0.4),
        "density_max": float(T.max()),
        "n_path_frames": 0 if path_coords is None else len(path_coords),
        "source_npz": npz_path.name,
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {DATA / 'structure.json'}  "
        f"({len(atoms)} atoms, {len(bonds)} bonds, "
        f"path={meta['n_path_frames']} frames)"
    )
    print(f"wrote {DATA / 'density.cube'}  shape={T.shape}")
    print(f"wrote {DATA / 'meta.json'}")
    print(
        f"NN-RMSD start={meta['nn_rmsd_start']:.3f}  "
        f"final={meta['nn_rmsd_final']:.3f} Å"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=float, default=2.0)
    args = ap.parse_args()
    main(npz_path=args.npz, seed=args.seed, resolution=args.resolution)
