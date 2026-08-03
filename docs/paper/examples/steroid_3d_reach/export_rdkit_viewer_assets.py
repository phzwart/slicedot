#!/usr/bin/env python3
"""Export finished RDKit screen runs for the web explorer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from export_viewer_assets import render_ortho, write_cube  # noqa: E402
from ligand_refs import load_ligand  # noqa: E402
from slicedot import sigma_from_resolution  # noqa: E402

DATA = ROOT / "viewer_rdkit" / "data"
Z_TO_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}


def _tag(resolution: float) -> str:
    return f"{float(resolution):g}".replace(".", "p") + "A"


def export_job(npz_path: Path, *, spacing: float = 0.5) -> dict | None:
    z = np.load(npz_path, allow_pickle=True)
    slug = npz_path.parts[npz_path.parts.index("ligands") + 1]
    res = float(z["resolution"])
    topo = load_ligand(slug)
    X_true = np.asarray(z["X_true"], dtype=np.float64)
    w = topo["W"]
    sig = float(z["sigma"]) if "sigma" in z.files else float(sigma_from_resolution(res))

    R = float(np.linalg.norm(X_true - X_true.mean(0), axis=1).max())
    half = R + 4.0 * sig + 3.0
    n = int(np.ceil(2.0 * half / spacing))
    if n % 2 == 0:
        n += 1
    n = min(n, 81)
    T, org, sp = render_ortho(X_true, spacing, (n, n, n), sig, w)

    atoms = [
        {
            "name": str(name),
            "elem": Z_TO_ELEM.get(int(zz), "C"),
            "resn": "LIG",
            "resi": 1,
        }
        for name, zz in zip(topo["names"], topo["elements"])
    ]
    bonds = sorted({
        (int(a), int(b)) if a < b else (int(b), int(a))
        for a, b in topo["bonds"]
    })

    top_idx = np.asarray(z["top_idx"], dtype=np.int64)
    refine_poses = np.asarray(z["refine_poses"], dtype=np.float64)
    start_poses = np.asarray(z["start_poses"], dtype=np.float64)
    refine_E = np.asarray(z["refine_E"], dtype=np.float64)
    refine_rmsd = np.asarray(z["refine_rmsd"], dtype=np.float64)
    screen_E = np.asarray(z["screen_E"], dtype=np.float64)
    screen_rmsd = np.asarray(z["screen_rmsd"], dtype=np.float64)
    refine_stop = np.asarray(z["refine_stop"], dtype=object)
    refine_time = np.asarray(z["refine_time_s"], dtype=np.float64)

    order = np.argsort(refine_E)
    models = []
    for rank, j in enumerate(order):
        k = int(top_idx[j])
        models.append({
            "rank": int(rank + 1),
            "slot": int(j),
            "conf_index": k,
            "coords": refine_poses[j].tolist(),
            "start_coords": start_poses[j].tolist(),
            "refine_E": float(refine_E[j]),
            "refine_rmsd": float(refine_rmsd[j]),
            "screen_E": float(screen_E[k]),
            "screen_rmsd": float(screen_rmsd[k]),
            "stop": str(refine_stop[j]),
            "time_s": float(refine_time[j]),
        })

    vals = np.sort(T.ravel())[::-1]
    csum = np.cumsum(vals)
    iso = float(vals[int(np.searchsorted(csum, 0.35 * csum[-1]))])

    tag = _tag(res)
    out_dir = DATA / slug / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cube(out_dir / "density.cube", T, org, sp, title=f"{slug} @{res:g}A rdkit")

    structure = {
        "slug": slug,
        "label": topo.get("label", slug),
        "resolution": res,
        "sigma": sig,
        "n_atoms": int(topo["n"]),
        "n_bonds": len(bonds),
        "grid": list(T.shape),
        "iso_default": iso,
        "iso_high": float(iso * 2.5),
        "iso_low": float(iso * 0.4),
        "atoms": atoms,
        "bonds": [list(p) for p in bonds],
        "true_coords": X_true.tolist(),
        "models": models,
        "best_rank": 1,
        "density": f"data/{slug}/{tag}/density.cube",
        "t_screen_s": float(z["t_screen_s"]) if "t_screen_s" in z.files else None,
        "t_total_s": float(z["t_total_s"]) if "t_total_s" in z.files else None,
        "n_conf": int(z["confs"].shape[0]),
    }
    (out_dir / "structure.json").write_text(json.dumps(structure) + "\n")
    return {
        "slug": slug,
        "label": structure["label"],
        "resolution": res,
        "tag": tag,
        "path": f"data/{slug}/{tag}/structure.json",
        "best_rmsd": models[0]["refine_rmsd"],
        "best_E": models[0]["refine_E"],
        "n_models": len(models),
    }


def main(slugs: list[str] | None = None):
    DATA.mkdir(parents=True, exist_ok=True)
    pattern = "ligands/*/out/rdkit_screen/screen_*_l1top*.npz"
    jobs = []
    for npz_path in sorted(ROOT.glob(pattern)):
        slug = npz_path.parts[npz_path.parts.index("ligands") + 1]
        if slugs and slug not in slugs:
            continue
        info = export_job(npz_path)
        if info:
            jobs.append(info)
            print(
                f"  {info['slug']:28s} @{info['resolution']:g}Å  "
                f"best RMSD={info['best_rmsd']:.3f} Å",
                flush=True,
            )

    # group catalog
    by_slug: dict[str, dict] = {}
    for j in jobs:
        L = by_slug.setdefault(j["slug"], {"slug": j["slug"], "label": j["label"], "resolutions": []})
        L["resolutions"].append({
            "resolution": j["resolution"],
            "tag": j["tag"],
            "path": j["path"],
            "best_rmsd": j["best_rmsd"],
            "best_E": j["best_E"],
            "n_models": j["n_models"],
        })

    catalog = {
        "default_slug": "prednisone" if "prednisone" in by_slug else (next(iter(by_slug), None)),
        "default_resolution": 2.0,
        "ligands": list(by_slug.values()),
        "n_exported": len(jobs),
    }
    (DATA / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"wrote {DATA / 'catalog.json'}  ({len(jobs)} jobs)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligands", type=str, default=None)
    args = ap.parse_args()
    slug_list = (
        [s.strip() for s in args.ligands.split(",") if s.strip()]
        if args.ligands else None
    )
    main(slugs=slug_list)
