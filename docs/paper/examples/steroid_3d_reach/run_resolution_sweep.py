#!/usr/bin/env python3
"""Run free-OT → Namer → ADMM cleanup for each steroid at several resolutions.

Per ligand writes under ``ligands/<slug>/out/``::

  trajectory_<slug>_<res>A_n<seeds>.npz
  <slug>_ot_name_refine_<res>A_n<seeds>.png / .pdf

Usage
-----
  uv run python run_resolution_sweep.py
  uv run python run_resolution_sweep.py --ligands DEX,PDN --resolutions 2,3 --n-seeds 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
LEUCINE = ROOT.parent / "leucine_3d_reach"
sys.path.insert(0, str(LEUCINE))
sys.path.insert(0, str(ROOT))

from ligand_refs import list_ligands, load_ligand  # noqa: E402
from make_ot_name_refine_ensemble import (  # noqa: E402
    L1Diff3D,
    LinearMap3D,
    SPACING,
    draw_overlay,
    molecular_radius,
    render_ortho,
    run_one,
)
from slicedot import (  # noqa: E402
    Geometry,
    Namer,
    SlicedOT,
    SlicedOTConfig,
    restraint_set_from_geometry,
    sigma_from_resolution,
)

torch.set_default_dtype(torch.float64)

DEFAULT_RESOLUTIONS = (1.5, 2.0, 2.5, 3.0, 3.5)


def build_scene(topo: dict, resolution: float):
    X_true = topo["X_ref"].copy()
    w = topo["W"].copy()
    sig = float(sigma_from_resolution(resolution))
    R = molecular_radius(X_true)
    half = R + 5.0 * sig + 4.0
    n = int(np.ceil(2.0 * half / SPACING))
    if n % 2 == 0:
        n += 1
    n = int(min(n, 81))
    NG = (n, n, n)
    T, org, sp = render_ortho(X_true, SPACING, NG, sig, w)
    ot = SlicedOT(
        torch.tensor(T),
        org,
        torch.tensor(sp),
        sig,
        SlicedOTConfig(
            n_dirs=32, dt=0.3, window=float(3.0 * half),
            map_cutoff=1e-7, backend="direct",
        ),
    )
    l1 = L1Diff3D(T, org, sp, sig)
    linmap = LinearMap3D(T, org, sp)
    geom = Geometry(
        topo["X_ref"],
        topo["bonds"],
        rotatable_bonds=topo["rotatable_bonds"],
        chiral_centres=topo["chiral_centres"],
        planar_groups=topo["planar_groups"],
        antibump=True,
    )
    rs = topo.get("restraint_set")
    if rs is None:
        rs = restraint_set_from_geometry(
            topo["X_ref"],
            topo["elements"],
            topo["bonds"],
            rotatable_bonds=topo["rotatable_bonds"],
            planar_groups=topo["planar_groups"],
            atom_ids=topo.get("names"),
            comp_id=str(topo.get("slug", "LIG")).upper()[:8],
            torsion14="planar",
        )
    namer = Namer(
        topo["X_ref"],
        restraint_set=rs,
        rotatable_bonds=topo["rotatable_bonds"],
        chiral_centres=topo["chiral_centres"],
        planar_groups=topo["planar_groups"],
        automorphisms=topo["automorphism_generators"],
    )
    return {
        "topo": topo,
        "X_true": X_true,
        "w": w,
        "sigma": sig,
        "half": half,
        "T": T,
        "origin": org,
        "spacing": sp,
        "ot": ot,
        "l1": l1,
        "linmap": linmap,
        "geom": geom,
        "namer": namer,
        "bonds": topo["bonds"],
        "resolution": float(resolution),
        "label": topo.get("label", topo.get("slug", "ligand")),
        "sequence": None,
        "n_atoms": int(X_true.shape[0]),
    }


def run_ligand_resolution(
    slug: str,
    resolution: float,
    *,
    n_seeds: int = 1,
    seed0: int = 0,
    atom_factor: float = 1.0,
    device_note: str = "",
) -> Path:
    topo = load_ligand(slug)
    out_dir = ROOT / "ligands" / slug / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{float(resolution):g}".replace(".", "p")
    stem = f"{slug}_ot_name_refine_{tag}A_n{n_seeds}"
    if abs(atom_factor - 1.0) > 1e-12:
        stem = f"{stem}_x{atom_factor:g}"

    print(
        f"\n{'=' * 72}\n"
        f"{topo['label']} ({slug}) @ {resolution:g} Å{device_note}\n"
        f"{'=' * 72}",
        flush=True,
    )
    scene = build_scene(topo, resolution)
    print(
        f"  N={scene['n_atoms']}  σ={scene['sigma']:.3f}  "
        f"half=±{scene['half']:.1f} Å  grid spacing={SPACING}",
        flush=True,
    )

    results = []
    t0 = time.perf_counter()
    for k in range(int(n_seeds)):
        seed = int(seed0) + k
        print(f"\n--- seed {seed} ({k + 1}/{n_seeds}) ---", flush=True)
        r = run_one(scene, seed, atom_factor=atom_factor)
        results.append(r)
        print(
            f"  free NN {r['free_nn']:.3f} Å · named {r['named_rmsd']:.3f} Å "
            f"({r['n_match']}/{scene['n_atoms']}) · "
            f"ADMM {r['admm_rmsd']:.3f} Å · polish {r['polish_rmsd']:.3f} Å",
            flush=True,
        )

    npz_path = out_dir / f"trajectory_{stem}.npz"
    np.savez_compressed(
        npz_path,
        seeds=np.array([r["seed"] for r in results]),
        free_nn=np.array([r["free_nn"] for r in results]),
        named_rmsd=np.array([r["named_rmsd"] for r in results]),
        n_match=np.array([r["n_match"] for r in results]),
        cleanup_rmsd=np.array([r["admm_rmsd"] for r in results]),
        polish_rmsd=np.array([r["polish_rmsd"] for r in results]),
        named_poses=np.stack([r["named"] for r in results], axis=0),
        admm_poses=np.stack([r["admm"] for r in results], axis=0),
        cleaned_poses=np.stack([r["cleaned"] for r in results], axis=0),
        resolution=np.array(resolution),
        atom_factor=np.array(atom_factor),
        slug=np.array(slug),
        n_ghosts=np.array([r["n_ghosts"] for r in results]),
    )
    # draw_overlay writes into leucine OUT_DIR; redirect via monkeypatch.
    import make_ot_name_refine_ensemble as ens

    old_out = ens.OUT_DIR
    ens.OUT_DIR = out_dir
    try:
        draw_overlay(scene, results, out_stem=stem)
    finally:
        ens.OUT_DIR = old_out

    summary = {
        "slug": slug,
        "label": topo["label"],
        "resolution": resolution,
        "n_seeds": n_seeds,
        "n_atoms": scene["n_atoms"],
        "free_nn_mean": float(np.mean([r["free_nn"] for r in results])),
        "named_rmsd_mean": float(np.mean([r["named_rmsd"] for r in results])),
        "cleanup_rmsd_mean": float(np.mean([r["admm_rmsd"] for r in results])),
        "n_match_mean": float(np.mean([r["n_match"] for r in results])),
        "elapsed_s": float(time.perf_counter() - t0),
        "npz": str(npz_path.relative_to(ROOT)),
    }
    af_tag = "" if abs(atom_factor - 1.0) <= 1e-12 else f"_x{atom_factor:g}"
    summary["atom_factor"] = float(atom_factor)
    (out_dir / f"summary_{tag}A{af_tag}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"  summary cleanup RMSD mean={summary['cleanup_rmsd_mean']:.3f} Å  "
        f"({summary['elapsed_s']:.1f}s) → {npz_path.name}",
        flush=True,
    )
    return npz_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ligands", type=str, default=None,
        help="Comma-separated slugs (default: all in ligands.json).",
    )
    ap.add_argument(
        "--resolutions", type=str, default=None,
        help="Comma-separated Å values (default: 1.5,2,2.5,3,3.5).",
    )
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--atom-factor", type=float, default=1.0)
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="Skip ligand×resolution if trajectory npz already exists.",
    )
    args = ap.parse_args()

    entries = list_ligands()
    if args.ligands:
        want = {s.strip() for s in args.ligands.split(",") if s.strip()}
        entries = [e for e in entries if e["slug"] in want]
        missing = want - {e["slug"] for e in entries}
        if missing:
            raise SystemExit(f"unknown ligand slug(s): {sorted(missing)}")

    if args.resolutions:
        resolutions = tuple(
            float(x) for x in args.resolutions.split(",") if x.strip()
        )
    else:
        resolutions = DEFAULT_RESOLUTIONS

    af = float(args.atom_factor)
    af_tag = "" if abs(af - 1.0) <= 1e-12 else f"_x{af:g}"
    print(
        f"steroid sweep: {len(entries)} ligands × {len(resolutions)} resolutions "
        f"× {args.n_seeds} seed(s)  atom_factor={af:g}",
        flush=True,
    )
    jobs = []
    for e in entries:
        for res in resolutions:
            jobs.append((e["slug"], res))

    t_all = time.perf_counter()
    done = 0
    for slug, res in jobs:
        tag = f"{float(res):g}".replace(".", "p")
        stem = f"{slug}_ot_name_refine_{tag}A_n{args.n_seeds}{af_tag}"
        npz = ROOT / "ligands" / slug / "out" / f"trajectory_{stem}.npz"
        if args.skip_existing and npz.is_file():
            print(f"skip existing {npz.relative_to(ROOT)}", flush=True)
            done += 1
            continue
        run_ligand_resolution(
            slug, res,
            n_seeds=args.n_seeds,
            seed0=args.seed0,
            atom_factor=af,
        )
        done += 1

    # Aggregate summary table (this atom_factor only).
    rows = []
    for slug, res in jobs:
        tag = f"{float(res):g}".replace(".", "p")
        p = ROOT / "ligands" / slug / "out" / f"summary_{tag}A{af_tag}.json"
        # Legacy ×1 summaries omit the suffix.
        if not p.is_file() and not af_tag:
            p = ROOT / "ligands" / slug / "out" / f"summary_{tag}A.json"
        if p.is_file():
            rows.append(json.loads(p.read_text()))
    agg_name = f"sweep_summary{af_tag}.json"
    agg = ROOT / "out" / agg_name
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"\n{'=' * 72}", flush=True)
    print(
        f"finished {done}/{len(jobs)} jobs in {time.perf_counter() - t_all:.1f}s",
        flush=True,
    )
    print(f"{'slug':28s} {'res':>5s} {'free':>7s} {'named':>7s} {'clean':>7s} {'match':>7s}", flush=True)
    for r in rows:
        print(
            f"{r['slug']:28s} {r['resolution']:5.1f} "
            f"{r['free_nn_mean']:7.3f} {r['named_rmsd_mean']:7.3f} "
            f"{r['cleanup_rmsd_mean']:7.3f} "
            f"{r['n_match_mean']:5.1f}/{r['n_atoms']}",
            flush=True,
        )
    print(f"wrote {agg}", flush=True)


if __name__ == "__main__":
    main()
