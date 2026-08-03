#!/usr/bin/env python3
"""Export all steroid ligands for the ×1 / ×2 compare viewer.

Writes ``viewer_compare/data/``:
  catalog.json
  <slug>/<tag>/structure.json
  <slug>/<tag>/density.cube

Final model poses = polished ``cleaned_poses`` from each trajectory npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from export_viewer_assets import render_ortho, write_cube  # noqa: E402
from ligand_refs import list_ligands, load_ligand  # noqa: E402
from slicedot import sigma_from_resolution  # noqa: E402

DATA = ROOT / "viewer_compare" / "data"
Z_TO_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}
DEFAULT_RESOLUTIONS = (1.5, 2.0, 2.5, 3.0, 3.5)


def _tag(resolution: float) -> str:
    return f"{float(resolution):g}".replace(".", "p") + "A"


def _traj_path(slug: str, resolution: float, *, x2: bool) -> Path:
    tag = _tag(resolution)
    name = f"trajectory_{slug}_ot_name_refine_{tag}_n1"
    if x2:
        name += "_x2"
    return ROOT / "ligands" / slug / "out" / f"{name}.npz"


def _finite_or_none(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _load_final(npz_path: Path, seed: int = 0) -> dict | None:
    if not npz_path.is_file():
        return None
    z = np.load(npz_path, allow_pickle=False)
    seeds = np.asarray(z["seeds"])
    idx = int(np.where(seeds == seed)[0][0]) if seed in set(seeds.tolist()) else 0
    cleaned = np.asarray(z["cleaned_poses"][idx], dtype=np.float64)
    cleanup_rmsd = _finite_or_none(z["cleanup_rmsd"][idx])
    polish_rmsd = (
        _finite_or_none(z["polish_rmsd"][idx])
        if "polish_rmsd" in z.files else None
    )
    return {
        "coords": cleaned.tolist(),
        # Prefer polish RMSD; older npz files only stored cleanup.
        "rmsd": polish_rmsd if polish_rmsd is not None else cleanup_rmsd,
        "named_rmsd": _finite_or_none(z["named_rmsd"][idx]),
        "cleanup_rmsd": cleanup_rmsd,
        "polish_rmsd": polish_rmsd,
        "n_match": int(z["n_match"][idx]),
        "atom_factor": float(z["atom_factor"]) if "atom_factor" in z.files else (2.0 if "_x2" in npz_path.name else 1.0),
        "npz": npz_path.name,
        "available": True,
    }


def export_one(slug: str, resolution: float, *, spacing: float = 0.5, seed: int = 0) -> dict:
    topo = load_ligand(slug)
    X = np.asarray(topo["X_ref"], dtype=np.float64)
    w = topo["W"]
    sig = float(sigma_from_resolution(resolution))
    R = float(np.linalg.norm(X - X.mean(0), axis=1).max())
    half = R + 4.0 * sig + 3.0
    n = int(np.ceil(2.0 * half / spacing))
    if n % 2 == 0:
        n += 1
    n = min(n, 81)
    T, org, sp = render_ortho(X, spacing, (n, n, n), sig, w)

    atoms = []
    for xyz, name, zz in zip(X, topo["names"], topo["elements"]):
        atoms.append({
            "name": str(name),
            "elem": Z_TO_ELEM.get(int(zz), "C"),
            "resn": "LIG",
            "resi": 1,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        })
    bonds = sorted({
        (int(a), int(b)) if a < b else (int(b), int(a))
        for a, b in topo["bonds"]
    })

    x1 = _load_final(_traj_path(slug, resolution, x2=False), seed=seed)
    x2 = _load_final(_traj_path(slug, resolution, x2=True), seed=seed)

    vals = np.sort(T.ravel())[::-1]
    csum = np.cumsum(vals)
    iso = float(vals[int(np.searchsorted(csum, 0.35 * csum[-1]))])

    tag = _tag(resolution)
    out_dir = DATA / slug / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cube(out_dir / "density.cube", T, org, sp, title=f"{slug} @{resolution:g}A")

    structure = {
        "slug": slug,
        "label": topo.get("label", slug),
        "resolution": float(resolution),
        "sigma": sig,
        "n_atoms": int(topo["n"]),
        "n_bonds": len(bonds),
        "grid": list(T.shape),
        "spacing": float(spacing),
        "iso_default": iso,
        "iso_high": float(iso * 2.5),
        "iso_low": float(iso * 0.4),
        "density_max": float(T.max()),
        "atoms": atoms,
        "bonds": [list(p) for p in bonds],
        "true_coords": X.tolist(),
        "x1": x1 or {"available": False},
        "x2": x2 or {"available": False},
        "density": f"data/{slug}/{tag}/density.cube",
    }
    (out_dir / "structure.json").write_text(json.dumps(structure) + "\n")
    return {
        "slug": slug,
        "label": structure["label"],
        "resolution": float(resolution),
        "tag": tag,
        "path": f"data/{slug}/{tag}/structure.json",
        "has_x1": bool(x1),
        "has_x2": bool(x2),
        "x1_rmsd": None if not x1 else x1["rmsd"],
        "x2_rmsd": None if not x2 else x2["rmsd"],
    }


def main(
    resolutions: tuple[float, ...] = DEFAULT_RESOLUTIONS,
    spacing: float = 0.5,
    slugs: list[str] | None = None,
):
    entries = list_ligands()
    if slugs:
        want = set(slugs)
        entries = [e for e in entries if e["slug"] in want]
    DATA.mkdir(parents=True, exist_ok=True)
    catalog_ligands = []
    all_jobs = []

    for entry in entries:
        slug = entry["slug"]
        avail = []
        for res in resolutions:
            # Export if at least reference exists (always) — prefer when any traj present
            info = export_one(slug, float(res), spacing=spacing)
            avail.append({
                "resolution": float(res),
                "tag": info["tag"],
                "path": info["path"],
                "has_x1": info["has_x1"],
                "has_x2": info["has_x2"],
                "x1_rmsd": info["x1_rmsd"],
                "x2_rmsd": info["x2_rmsd"],
            })
            all_jobs.append(info)
            print(
                f"  {slug:28s} @{res:g}Å  "
                f"×1={'Y' if info['has_x1'] else '.'}  "
                f"×2={'Y' if info['has_x2'] else '.'}",
                flush=True,
            )
        catalog_ligands.append({
            "slug": slug,
            "label": entry.get("label", slug),
            "resolutions": avail,
        })

    default_slug = None
    for pref in ("dexamethasone", "prednisone"):
        if any(L["slug"] == pref for L in catalog_ligands):
            default_slug = pref
            break
    if default_slug is None and catalog_ligands:
        default_slug = catalog_ligands[0]["slug"]
    catalog = {
        "default_slug": default_slug,
        "default_resolution": float(resolutions[1] if len(resolutions) > 1 else resolutions[0]),
        "ligands": catalog_ligands,
        "n_exported": len(all_jobs),
    }
    (DATA / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"wrote {DATA / 'catalog.json'}  ({len(all_jobs)} ligand×resolution)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=str, default=None,
                    help="Comma-separated Å (default: 1.5,2,2.5,3,3.5).")
    ap.add_argument("--ligands", type=str, default=None,
                    help="Comma-separated slugs (default: all).")
    ap.add_argument("--spacing", type=float, default=0.5)
    args = ap.parse_args()
    res = (
        tuple(float(x) for x in args.resolutions.split(",") if x.strip())
        if args.resolutions else DEFAULT_RESOLUTIONS
    )
    slug_list = (
        [s.strip() for s in args.ligands.split(",") if s.strip()]
        if args.ligands else None
    )
    main(resolutions=res, spacing=args.spacing, slugs=slug_list)
