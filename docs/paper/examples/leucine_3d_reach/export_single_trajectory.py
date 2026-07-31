#!/usr/bin/env python3
"""Run one seed and export a free→named→cleanup trajectory for the viewer.

  PYTHONPATH=../../../src python export_single_trajectory.py \\
      --sequence AFSSFN --resolution 2.75 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_viewer_assets import main as export_assets
from make_ot_name_refine_ensemble import build_scene, run_one

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DATA = ROOT / "viewer" / "data"


def main(sequence: str = "AFSSFN", resolution: float = 2.75, seed: int = 0):
    raw = sequence.replace("-", ",").upper()
    parts = tuple(s.strip() for s in raw.split(",") if s.strip())
    if len(parts) == 1 and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in parts[0]):
        seq = tuple(parts[0])
    else:
        seq = parts
    print(f"building scene {''.join(seq)} @ {resolution:g} Å …", flush=True)
    scene = build_scene(resolution, sequence=seq)
    print(
        f"  {scene['label']}  N={scene['n_atoms']}  running seed {seed} …",
        flush=True,
    )
    r = run_one(scene, seed, save_trajectory=True)
    traj = r["trajectory"]
    coords = traj["coords"]
    stages = traj["stages"]
    metrics = traj["metrics"]

    tag = f"{float(resolution):g}".replace(".", "p")
    stem = f"path_{''.join(seq)}_{tag}A_seed{seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    npz_path = OUT / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        coords=coords,
        stages=np.array(stages),
        metrics=metrics,
        seed=np.array(seed),
        resolution=np.array(resolution),
        n_match=np.array(r["n_match"]),
        free_nn=np.array(r["free_nn"]),
        named_rmsd=np.array(r["named_rmsd"]),
        cleanup_rmsd=np.array(r["admm_rmsd"]),
        polish_rmsd=np.array(r.get("polish_rmsd", r["admm_rmsd"])),
        free_stop=np.array(r.get("free_stop", "")),
        cleanup_stop=np.array(r.get("cleanup_stop", "")),
        polish_stop=np.array(r.get("polish_stop", "")),
    )
    print(
        f"  free NN {r['free_nn']:.3f} ({r.get('free_stop')}) · "
        f"named {r['named_rmsd']:.3f} ({r['n_match']}/{scene['n_atoms']}) · "
        f"ADMM {r['admm_rmsd']:.3f} · "
        f"polish {r.get('polish_rmsd', r['admm_rmsd']):.3f} "
        f"(wxc_scale={r.get('wxc_scale')}, geom_ok={r.get('geom_ok')}, "
        f"{r.get('polish_stop')})",
        flush=True,
    )
    print(f"wrote {npz_path}  ({len(stages)} frames)", flush=True)

    # Refresh density / structure / ensemble, then attach path trajectory.
    export_assets(resolution=resolution, sequence="".join(seq))
    structure = json.loads((DATA / "structure.json").read_text())
    structure["path_coords"] = coords.tolist()
    structure["path_stages"] = list(stages)
    structure["path_metrics"] = [
        None if not np.isfinite(m) else float(m) for m in metrics
    ]
    structure["path_meta"] = {
        "seed": int(seed),
        "n_frames": int(len(stages)),
        "n_free": int(sum(1 for s in stages if s in ("random", "free"))),
        "n_named": int(sum(1 for s in stages if s == "named")),
        "n_cleanup": int(sum(1 for s in stages if s == "cleanup")),
        "n_polish": int(sum(1 for s in stages if s == "polish")),
        "free_nn": float(r["free_nn"]),
        "named_rmsd": float(r["named_rmsd"]),
        "cleanup_rmsd": float(r["admm_rmsd"]),
        "polish_rmsd": float(r.get("polish_rmsd", r["admm_rmsd"])),
        "wxc_scale": r.get("wxc_scale"),
        "geom_ok": r.get("geom_ok"),
        "geom_final_max": r.get("geom_final_max"),
        "n_match": int(r["n_match"]),
        "free_stop": r.get("free_stop"),
        "cleanup_stop": r.get("cleanup_stop"),
        "polish_stop": r.get("polish_stop"),
    }
    (DATA / "structure.json").write_text(json.dumps(structure) + "\n")

    meta = json.loads((DATA / "meta.json").read_text())
    meta["n_path_frames"] = int(len(stages))
    meta["path_seed"] = int(seed)
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"viewer path: {structure['path_meta']['n_free']} free + "
        f"{structure['path_meta']['n_named']} named + "
        f"{structure['path_meta']['n_cleanup']} cleanup + "
        f"{structure['path_meta']['n_polish']} polish frames",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=str, default="AFSSFN")
    ap.add_argument("--resolution", type=float, default=2.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(sequence=args.sequence, resolution=args.resolution, seed=args.seed)
