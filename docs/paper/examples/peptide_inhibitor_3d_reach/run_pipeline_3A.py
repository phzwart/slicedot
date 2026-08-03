#!/usr/bin/env python3
"""Peptide-inhibitor 3 Å pipeline: free-OT×2 → name/prune/ADMM, then torsion screen.

Stage A (all ligands)
  free OT with ``--atom-factor 2`` → ghost prune → Namer → standard ADMM
  cleanup → L1+geom polish  (same ``run_one`` as steroids / leucine).

Stage B (all ligands)
  torsion-randomized conformers → dedupe → PCA+OT → L1 top-K → same
  ``run_cleanup`` (free-atom schedule) + ``run_l1_geom_polish``.

Usage
-----
  uv run python run_pipeline_3A.py
  uv run python run_pipeline_3A.py --ligands ritonavir,darunavir
  uv run python run_pipeline_3A.py --skip-free   # torsion only
  uv run python run_pipeline_3A.py --skip-torsion
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    print(f"    log → {log.relative_to(ROOT)}", flush=True)
    with log.open("w") as fh:
        fh.write(f"# {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
    print(f"<<< exit {rc}", flush=True)
    return int(rc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligands", type=str, default=None)
    ap.add_argument("--resolution", type=float, default=3.0)
    ap.add_argument("--atom-factor", type=float, default=2.0)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n-conf", type=int, default=1000)
    ap.add_argument("--top-pca", type=int, default=50)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--skip-free", action="store_true")
    ap.add_argument("--skip-torsion", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--regen-confs", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    lig_args: list[str] = []
    if args.ligands:
        lig_args = ["--ligands", args.ligands]

    t0 = time.perf_counter()
    if not args.skip_free:
        cmd = [
            py, "run_resolution_sweep.py",
            *lig_args,
            "--resolutions", f"{args.resolution:g}",
            "--atom-factor", f"{args.atom_factor:g}",
            "--n-seeds", str(args.n_seeds),
            "--seed0", str(args.seed0),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        rc = _run(cmd, ROOT / "out" / "free_ot_x2_3A.log")
        if rc != 0:
            raise SystemExit(f"free-OT stage failed (rc={rc}); see out/free_ot_x2_3A.log")

    if not args.skip_torsion:
        cmd = [
            py, "screen_rdkit_conformers.py",
            *lig_args,
            "--resolutions", f"{args.resolution:g}",
            "--n-conf", str(args.n_conf),
            "--top-pca", str(args.top_pca),
            "--top", str(args.top),
            "--seed", str(args.seed0),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.regen_confs:
            cmd.append("--regen-confs")
        rc = _run(cmd, ROOT / "out" / "rdkit_screen.log")
        if rc != 0:
            raise SystemExit(f"torsion screen failed (rc={rc}); see out/rdkit_screen.log")

    print(
        f"\npipeline done in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
