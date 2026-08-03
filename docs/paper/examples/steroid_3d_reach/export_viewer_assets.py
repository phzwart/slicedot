#!/usr/bin/env python3
"""Export steroid structure + density for the web viewer.

Writes into ``viewer/data/``:
  * structure.json — reference + cleaned/named model poses
  * density.cube  — rendered map from the reference
  * meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ligand_refs import load_ligand  # noqa: E402
from slicedot import sigma_from_resolution  # noqa: E402

DATA = ROOT / "viewer" / "data"
Z_TO_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}


def render_ortho(X, sp, NG, sigma, weights):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.asarray(NG, dtype=np.float64) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG, dtype=np.float64)
    for p, w in zip(X, weights):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2.0 * sigma * sigma))
    return T / T.sum(), org, sp


def write_cube(path: Path, T: np.ndarray, origin: np.ndarray, spacing, title="density"):
    bohr = 1.8897259886
    sp = np.atleast_1d(spacing) * np.ones(3)
    nx, ny, nz = T.shape
    org_b = np.asarray(origin, dtype=np.float64) * bohr
    sp_b = sp * bohr
    lines = [
        title,
        "slicedot rendered ortho density",
        f"  1 {org_b[0]:12.6f} {org_b[1]:12.6f} {org_b[2]:12.6f}",
        f"{nx:5d} {sp_b[0]:12.6f} {0.0:12.6f} {0.0:12.6f}",
        f"{ny:5d} {0.0:12.6f} {sp_b[1]:12.6f} {0.0:12.6f}",
        f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {sp_b[2]:12.6f}",
        f"  6 {0.0:12.6f} {org_b[0]:12.6f} {org_b[1]:12.6f} {org_b[2]:12.6f}",
    ]
    flat = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                flat.append(f"{T[ix, iy, iz]:13.5E}")
                if len(flat) == 6:
                    lines.append(" ".join(flat))
                    flat = []
    if flat:
        lines.append(" ".join(flat))
    path.write_text("\n".join(lines) + "\n")


def main(slug: str = "dexamethasone", resolution: float | None = None,
         spacing: float = 0.5, seed: int = 0):
    topo = load_ligand(slug)
    out_dir = ROOT / "ligands" / slug / "out"

    # Prefer trajectory matching requested resolution; else newest.
    npz_path = None
    if resolution is not None:
        tag = f"{float(resolution):g}".replace(".", "p")
        cand = out_dir / f"trajectory_{slug}_ot_name_refine_{tag}A_n1.npz"
        if cand.is_file():
            npz_path = cand
    if npz_path is None:
        cands = sorted(out_dir.glob(f"trajectory_{slug}_ot_name_refine_*A_n*.npz"))
        if cands:
            npz_path = cands[-1]

    named = admm = cleaned = None
    res = float(resolution) if resolution is not None else 2.0
    if npz_path is not None:
        z = np.load(npz_path, allow_pickle=False)
        res = float(z["resolution"])
        # pick seed index
        seeds = np.asarray(z["seeds"])
        idx = int(np.where(seeds == seed)[0][0]) if seed in set(seeds.tolist()) else 0
        named = np.asarray(z["named_poses"][idx], dtype=np.float64)
        cleaned = np.asarray(z["cleaned_poses"][idx], dtype=np.float64)
        if "admm_poses" in z.files:
            admm = np.asarray(z["admm_poses"][idx], dtype=np.float64)
        polish_rms = (
            float(z["polish_rmsd"][idx]) if "polish_rmsd" in z.files
            else float("nan")
        )
        print(
            f"loaded {npz_path.name}  seed={seeds[idx]}  "
            f"named={float(z['named_rmsd'][idx]):.3f}  "
            f"admm={float(z['cleanup_rmsd'][idx]):.3f}  "
            f"polish={polish_rms:.3f} Å",
            flush=True,
        )
    else:
        print("no trajectory npz; exporting reference only", flush=True)

    X = np.asarray(topo["X_ref"], dtype=np.float64)
    w = topo["W"]
    sig = float(sigma_from_resolution(res))
    R = float(np.linalg.norm(X - X.mean(0), axis=1).max())
    half = R + 4.0 * sig + 3.0
    n = int(np.ceil(2.0 * half / spacing))
    if n % 2 == 0:
        n += 1
    n = min(n, 81)
    T, org, sp = render_ortho(X, spacing, (n, n, n), sig, w)

    DATA.mkdir(parents=True, exist_ok=True)
    atoms = []
    for i, (xyz, name, zz) in enumerate(zip(X, topo["names"], topo["elements"])):
        elem = Z_TO_ELEM.get(int(zz), str(topo["names"][i])[0])
        atoms.append({
            "i": i,
            "name": str(name),
            "full": str(name),
            "elem": elem if elem != "C" or str(name)[0] == "C" else str(name)[0],
            "resn": "LIG",
            "resi": 1,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        })
        # Fix element from Z properly
        atoms[-1]["elem"] = Z_TO_ELEM.get(int(zz), "C")

    bonds = sorted({
        (int(a), int(b)) if a < b else (int(b), int(a))
        for a, b in topo["bonds"]
    })

    ensemble = []
    path_coords = []
    path_stages = []
    if named is not None:
        path_coords.append(named.tolist())
        path_stages.append("named")
        ensemble.append(named.tolist())
    if admm is not None:
        path_coords.append(admm.tolist())
        path_stages.append("cleanup")
        ensemble.append(admm.tolist())
    if cleaned is not None:
        path_coords.append(cleaned.tolist())
        path_stages.append("polish")
        ensemble.append(cleaned.tolist())

    structure = {
        "atoms": atoms,
        "bonds": [list(p) for p in bonds],
        "true_coords": X.tolist(),
        "ensemble_coords": ensemble,
        "path_coords": path_coords or None,
        "path_stages": path_stages or None,
        "path_meta": {
            "seed": seed,
            "n_free": 0,
            "n_cleanup": 1 if cleaned is not None else 0,
            "source_npz": None if npz_path is None else npz_path.name,
        },
        "source": topo.get("source"),
    }
    (DATA / "structure.json").write_text(json.dumps(structure) + "\n")
    write_cube(DATA / "density.cube", T, org, sp, title=f"{slug} @{res:g}A")

    vals = np.sort(T.ravel())[::-1]
    csum = np.cumsum(vals)
    iso = float(vals[int(np.searchsorted(csum, 0.35 * csum[-1]))])
    meta = {
        "slug": slug,
        "label": topo.get("label", slug),
        "resolution": float(res),
        "sigma": sig,
        "n_atoms": int(topo["n"]),
        "n_bonds": len(bonds),
        "grid": list(T.shape),
        "spacing": float(spacing),
        "iso_default": iso,
        "iso_high": float(iso * 2.5),
        "iso_low": float(iso * 0.4),
        "n_ensemble": len(ensemble),
        "density_max": float(T.max()),
        "source_npz": None if npz_path is None else npz_path.name,
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {DATA / 'structure.json'}  "
        f"({len(atoms)} atoms, {len(bonds)} bonds, {len(ensemble)} model frames)"
    )
    print(f"wrote {DATA / 'density.cube'}  shape={T.shape} iso≈{iso:.3e}")
    print(f"wrote {DATA / 'meta.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", type=str, default="dexamethasone")
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(
        slug=args.slug,
        resolution=args.resolution,
        spacing=args.spacing,
        seed=args.seed,
    )
