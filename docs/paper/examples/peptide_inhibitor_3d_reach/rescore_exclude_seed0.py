#!/usr/bin/env python3
"""Re-rank / refine torsion screen excluding the ideal seed (conf index 0).

Reuses ``place_*`` PCA placements from a prior ``screen_rdkit_conformers`` run,
drops original conf 0, re-selects PCA→L1 top-K, then runs the same ADMM +
L1+geom polish. Cached refine poses from the prior screen are reused when the
uniq index was already refined.
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
    CLEANUP_SLACK0,
    CLEANUP_SLACK1,
    run_cleanup,
    run_l1_geom_polish,
    vg_ot,
)
from run_resolution_sweep import build_scene  # noqa: E402
from screen_rdkit_conformers import rmsd, score_l1  # noqa: E402
from slicedot import Geometry  # noqa: E402

torch.set_default_dtype(torch.float64)

TAG = "exseed0"


def _place_path(out_dir: Path, res: float, n_conf: int, top_pca: int, top: int) -> Path:
    return out_dir / (
        f"place_{res:g}A_n{n_conf}_pca{top_pca}_l1top{top}.npz".replace(".", "p")
        if False
        else f"place_{res:g}A_n{n_conf}_pca{top_pca}_l1top{top}.npz"
    )


def _find_place(out_dir: Path, res: float) -> Path:
    hits = sorted(out_dir.glob(f"place_{res:g}A_*.npz"))
    if not hits:
        # also try 3A vs 3.0A
        hits = sorted(out_dir.glob("place_*A_*.npz"))
        hits = [h for h in hits if f"{res:g}A" in h.name or f"{res}A" in h.name]
    if not hits:
        raise FileNotFoundError(f"no place_*.npz in {out_dir}")
    return hits[0]


def _find_screen(out_dir: Path, res: float) -> Path | None:
    hits = sorted(out_dir.glob(f"screen_{res:g}A_*.npz"))
    return hits[0] if hits else None


def _cache_from_screen(screen_npz: Path | None) -> dict[int, dict]:
    """Map original uniq conf index → prior refine/polish arrays."""
    if screen_npz is None or not screen_npz.is_file():
        return {}
    s = np.load(screen_npz, allow_pickle=True)
    out: dict[int, dict] = {}
    for j, u in enumerate(s["start_uniq"]):
        out[int(u)] = {
            "start_pose": np.asarray(s["start_poses"][j], dtype=np.float64),
            "start_rmsd": float(s["start_rmsd"][j]),
            "start_pca_E": float(s["start_pca_E"][j]),
            "start_l1_E": float(s["start_l1_E"][j]),
            "refine_pose": np.asarray(s["refine_poses"][j], dtype=np.float64),
            "refine_E": float(s["refine_E"][j]),
            "refine_L1": float(s["refine_L1"][j]),
            "refine_rmsd": float(s["refine_rmsd"][j]),
            "refine_steps": int(s["refine_steps"][j]),
            "refine_stop": str(s["refine_stop"][j]),
            "refine_time_s": float(s["refine_time_s"][j]),
            "polish_pose": np.asarray(s["polish_poses"][j], dtype=np.float64),
            "polish_L1": float(s["polish_L1"][j]),
            "polish_ot": float(s["polish_ot"][j]),
            "polish_rmsd": float(s["polish_rmsd"][j]),
            "polish_stop": str(s["polish_stop"][j]),
            "polish_time_s": float(s["polish_time_s"][j]),
        }
    return out


def rescore_one(
    slug: str,
    *,
    res: float,
    top_pca: int,
    top: int,
    project_slack: float,
    seed: int,
    exclude_orig: int = 0,
) -> dict:
    topo = load_ligand(slug)
    w = topo["W"]
    out_dir = ROOT / "ligands" / slug / "out" / "rdkit_screen"
    place_npz = _find_place(out_dir, res)
    p = np.load(place_npz)
    uniq = np.asarray(p["uniq_idx"], dtype=np.int64)
    pca_E = np.asarray(p["pca_E"], dtype=np.float64)
    pca_rmsd = np.asarray(p["pca_rmsd"], dtype=np.float64)
    placed = np.asarray(p["placed"], dtype=np.float64)
    n_conf_raw = int(p["confs_unique"].shape[0]) if "confs_unique" in p.files else len(uniq)
    # confs_unique length == len(uniq)
    n_conf_unique = int(len(uniq))

    keep = uniq != int(exclude_orig)
    pool = np.where(keep)[0]
    if pool.size == 0:
        raise RuntimeError(f"{slug}: nothing left after excluding conf {exclude_orig}")

    order_pca = pool[np.argsort(pca_E[pool])]
    n_pca = min(int(top_pca), len(order_pca))
    pca_top_idx = order_pca[:n_pca]

    scene = build_scene(topo, float(res))
    if not scene["geom"].chiral_centres and topo["chiral_centres"]:
        scene["geom"] = Geometry(
            topo["X_ref"],
            topo["bonds"],
            rotatable_bonds=topo["rotatable_bonds"],
            chiral_centres=topo["chiral_centres"],
            planar_groups=topo["planar_groups"],
            antibump=True,
        )
    l1 = scene["l1"]
    X_true = scene["X_true"]
    geom = scene["geom"]
    sig = scene["sigma"]

    t_l10 = time.perf_counter()
    l1_E = np.array(
        [score_l1(l1, placed[k], w) for k in pca_top_idx],
        dtype=np.float64,
    )
    order_l1 = np.argsort(l1_E)
    n_l1 = min(int(top), len(order_l1))
    l1_sel = order_l1[:n_l1]
    refine_src = pca_top_idx[l1_sel]
    t_l1 = time.perf_counter() - t_l10

    start_poses = np.full((n_l1, *placed.shape[1:]), np.nan)
    start_pca_E = np.full(n_l1, np.nan)
    start_l1_E = np.full(n_l1, np.nan)
    start_rmsd = np.full(n_l1, np.nan)
    start_uniq = np.zeros(n_l1, dtype=np.int64)

    cache = _cache_from_screen(_find_screen(out_dir, res))
    n_reuse = 0

    for j, k in enumerate(refine_src):
        u = int(uniq[k])
        start_uniq[j] = u
        start_pca_E[j] = float(pca_E[k])
        start_l1_E[j] = float(l1_E[l1_sel[j]])
        if u in cache:
            start_poses[j] = cache[u]["start_pose"]
            start_rmsd[j] = cache[u]["start_rmsd"]
            n_reuse += 1
        else:
            X0 = placed[k].copy()
            if project_slack is not None and float(project_slack) >= 0.0:
                Xp, _, _ = geom.project(
                    X0, tol=1e-3, max_iter=80, slack=float(project_slack),
                )
            else:
                Xp = X0
            start_poses[j] = Xp
            start_rmsd[j] = rmsd(Xp, X_true)

    tag = f"{res:g}A".replace(".", "p") if False else f"{res:g}A"
    # match existing naming: place_3A_n1000_pca50_l1top10.npz
    stem_bits = place_npz.stem  # place_3A_n1000_pca50_l1top10
    out_npz = out_dir / f"screen_{stem_bits[len('place_'):]}_{TAG}.npz"
    # cleaner names:
    out_npz = out_dir / f"{place_npz.stem.replace('place_', 'screen_')}_{TAG}.npz"
    out_json = out_npz.with_suffix(".json")
    place_out = out_dir / f"{place_npz.stem}_{TAG}.npz"
    place_json = place_out.with_suffix(".json")

    place_summary = {
        "slug": slug,
        "label": topo.get("label", slug),
        "resolution": float(res),
        "exclude_orig_conf": int(exclude_orig),
        "n_conf_unique_pool": int(pool.size),
        "n_conf_unique": int(n_conf_unique),
        "top_pca": int(n_pca),
        "top_l1": int(n_l1),
        "t_l1_s": float(t_l1),
        "pca_E_best": float(pca_E[pca_top_idx[0]]),
        "l1_E_best_pre": float(l1_E[l1_sel[0]]),
        "start_rmsd_best": float(np.nanmin(start_rmsd)),
        "start_rmsd_mean": float(np.nanmean(start_rmsd)),
        "source_place": str(place_npz.relative_to(ROOT)),
        "npz": str(place_out.relative_to(ROOT)),
        "cleanup_slack": [float(CLEANUP_SLACK0), float(CLEANUP_SLACK1)],
        "cleanup_schedule": "free_atom_named",
    }
    np.savez_compressed(
        place_out,
        resolution=np.array(res),
        sigma=np.array(sig),
        uniq_idx=uniq,
        exclude_orig_conf=np.array(exclude_orig),
        pca_E=pca_E,
        pca_rmsd=pca_rmsd,
        placed=placed,
        pca_top_idx=pca_top_idx,
        l1_E_pca_top=l1_E,
        l1_sel=l1_sel,
        refine_src=refine_src,
        start_poses=start_poses,
        start_pca_E=start_pca_E,
        start_l1_E=start_l1_E,
        start_rmsd=start_rmsd,
        start_uniq=start_uniq,
        X_true=X_true,
        t_l1_s=np.array(t_l1),
    )
    place_json.write_text(json.dumps(place_summary, indent=2) + "\n")
    print(
        f"  place (ex seed) start best={place_summary['start_rmsd_best']:.3f} Å  "
        f"mean={place_summary['start_rmsd_mean']:.3f} Å  "
        f"reuse_starts={n_reuse}/{n_l1}",
        flush=True,
    )

    refine_E = np.full(n_l1, np.nan)
    refine_L1 = np.full(n_l1, np.nan)
    refine_rmsd = np.full(n_l1, np.nan)
    refine_steps = np.zeros(n_l1, dtype=np.int32)
    refine_stop = np.array([""] * n_l1, dtype=object)
    refine_time = np.full(n_l1, np.nan)
    refine_poses = np.full_like(start_poses, np.nan)

    polish_L1 = np.full(n_l1, np.nan)
    polish_rmsd = np.full(n_l1, np.nan)
    polish_stop = np.array([""] * n_l1, dtype=object)
    polish_time = np.full(n_l1, np.nan)
    polish_poses = np.full_like(start_poses, np.nan)
    polish_ot = np.full(n_l1, np.nan)

    t0 = time.perf_counter()
    n_ran = 0
    for j, k in enumerate(refine_src):
        u = int(uniq[k])
        Xp = start_poses[j].copy()
        if u in cache:
            c = cache[u]
            refine_poses[j] = c["refine_pose"]
            refine_E[j] = c["refine_E"]
            refine_L1[j] = c["refine_L1"]
            refine_rmsd[j] = c["refine_rmsd"]
            refine_steps[j] = c["refine_steps"]
            refine_stop[j] = c["refine_stop"]
            refine_time[j] = c["refine_time_s"]
            polish_poses[j] = c["polish_pose"]
            polish_L1[j] = c["polish_L1"]
            polish_ot[j] = c["polish_ot"]
            polish_rmsd[j] = c["polish_rmsd"]
            polish_stop[j] = c["polish_stop"]
            polish_time[j] = c["polish_time_s"]
            print(
                f"  refine {j+1}/{n_l1}  uniq#{u}  REUSE  "
                f"start={start_rmsd[j]:.3f}  OT→{refine_rmsd[j]:.3f}  "
                f"L1→{polish_rmsd[j]:.3f}",
                flush=True,
            )
            continue

        seed_j = int(seed) + u
        t1 = time.perf_counter()
        cleanup = run_cleanup(
            scene, Xp, seed=seed_j, log_every=10**9, named_atoms=False,
        )
        Xf = cleanup["poses"][-1]
        refine_poses[j] = Xf
        refine_E[j] = float(cleanup["energies"][-1])
        refine_L1[j] = float(cleanup["l1_energies"][-1])
        refine_rmsd[j] = rmsd(Xf, X_true)
        refine_steps[j] = int(cleanup["n_steps"])
        refine_stop[j] = str(cleanup["stop_reason"])
        refine_time[j] = time.perf_counter() - t1

        t2 = time.perf_counter()
        polish = run_l1_geom_polish(scene, Xp, seed=seed_j)
        Xp_l1 = np.asarray(polish["poses"][-1], dtype=np.float64)
        polish_poses[j] = Xp_l1
        polish_L1[j] = float(scene["l1"].value_grad(Xp_l1, w)[0])
        polish_ot[j] = float(vg_ot(scene["ot"], Xp_l1, w, sig)[0])
        polish_rmsd[j] = rmsd(Xp_l1, X_true)
        polish_stop[j] = str(polish.get("stop_reason", "l1_geom"))
        polish_time[j] = time.perf_counter() - t2
        n_ran += 1
        print(
            f"  refine {j+1}/{n_l1}  uniq#{u}  NEW  "
            f"start={start_rmsd[j]:.3f} Å  "
            f"OT+ADMM→{refine_rmsd[j]:.3f} Å ({refine_time[j]:.1f}s)  "
            f"L1+geom→{polish_rmsd[j]:.3f} Å ({polish_time[j]:.1f}s)",
            flush=True,
        )

    j_best = int(np.nanargmin(refine_E))
    j_best_l1 = int(np.nanargmin(polish_L1))
    summary = {
        "slug": slug,
        "label": topo.get("label", slug),
        "resolution": float(res),
        "exclude_orig_conf": int(exclude_orig),
        "n_conf_unique_pool": int(pool.size),
        "n_conf_unique": int(n_conf_unique),
        "top_pca": int(n_pca),
        "top_l1": int(n_l1),
        "n_refine_reused": int(n_reuse),
        "n_refine_ran": int(n_ran),
        "t_l1_s": float(t_l1),
        "t_refine_total_s": float(np.nansum(refine_time)),
        "t_polish_total_s": float(np.nansum(polish_time)),
        "t_total_s": float(time.perf_counter() - t0),
        "pca_E_best": float(pca_E[pca_top_idx[0]]),
        "l1_E_best_pre": float(l1_E[l1_sel[0]]),
        "refine_E_best": float(refine_E[j_best]),
        "refine_L1_best": float(refine_L1[j_best]),
        "refine_rmsd_best": float(refine_rmsd[j_best]),
        "refine_rmsd_mean_top": float(np.nanmean(refine_rmsd)),
        "polish_L1_best": float(polish_L1[j_best_l1]),
        "polish_rmsd_best": float(polish_rmsd[j_best_l1]),
        "polish_rmsd_mean_top": float(np.nanmean(polish_rmsd)),
        "polish_ot_at_best_l1": float(polish_ot[j_best_l1]),
        "start_rmsd_best": float(np.nanmin(start_rmsd)),
        "best_slot_ot": int(j_best),
        "best_slot_l1polish": int(j_best_l1),
        "best_uniq_index": int(start_uniq[j_best]),
        "place_npz": str(place_out.relative_to(ROOT)),
        "npz": str(out_npz.relative_to(ROOT)),
        "source_place": str(place_npz.relative_to(ROOT)),
        "cleanup_slack": [float(CLEANUP_SLACK0), float(CLEANUP_SLACK1)],
        "cleanup_schedule": "free_atom_named",
        "tag": TAG,
    }
    np.savez_compressed(
        out_npz,
        resolution=np.array(res),
        sigma=np.array(sig),
        uniq_idx=uniq,
        exclude_orig_conf=np.array(exclude_orig),
        pca_E=pca_E,
        pca_rmsd=pca_rmsd,
        placed=placed,
        pca_top_idx=pca_top_idx,
        l1_E_pca_top=l1_E,
        l1_sel=l1_sel,
        refine_src=refine_src,
        start_poses=start_poses,
        start_pca_E=start_pca_E,
        start_l1_E=start_l1_E,
        start_rmsd=start_rmsd,
        start_uniq=start_uniq,
        refine_poses=refine_poses,
        refine_E=refine_E,
        refine_L1=refine_L1,
        refine_rmsd=refine_rmsd,
        refine_steps=refine_steps,
        refine_stop=np.asarray(refine_stop, dtype=object),
        refine_time_s=refine_time,
        polish_poses=polish_poses,
        polish_L1=polish_L1,
        polish_ot=polish_ot,
        polish_rmsd=polish_rmsd,
        polish_stop=np.asarray(polish_stop, dtype=object),
        polish_time_s=polish_time,
        X_true=X_true,
        t_l1_s=np.array(t_l1),
        t_total_s=np.array(summary["t_total_s"]),
    )
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"  done {slug}  R0={summary['start_rmsd_best']:.3f}  "
        f"OT={summary['refine_rmsd_best']:.3f}  "
        f"L1={summary['polish_rmsd_best']:.3f}  "
        f"best_uniq=#{summary['best_uniq_index']}",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligands", type=str, default=None)
    ap.add_argument("--resolution", type=float, default=3.0)
    ap.add_argument("--top-pca", type=int, default=50)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--project-slack", type=float, default=-1.0,
                    help="Geom project slack before refine; <0 skips (match screen default).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-orig", type=int, default=0)
    args = ap.parse_args()

    entries = list_ligands()
    if args.ligands:
        want = {s.strip() for s in args.ligands.split(",") if s.strip()}
        entries = [e for e in entries if e["slug"] in want]

    summaries = []
    t0 = time.perf_counter()
    for e in entries:
        slug = e["slug"]
        print(f"\n=== {slug} @ {args.resolution:g} Å  exclude conf {args.exclude_orig} ===", flush=True)
        summaries.append(
            rescore_one(
                slug,
                res=args.resolution,
                top_pca=args.top_pca,
                top=args.top,
                project_slack=args.project_slack,
                seed=args.seed,
                exclude_orig=args.exclude_orig,
            )
        )

    agg = ROOT / "out" / f"rdkit_screen_summary_{TAG}.json"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(summaries, indent=2) + "\n")

    print(f"\n{'=' * 72}", flush=True)
    print(
        f"finished {len(summaries)} ligands in {time.perf_counter()-t0:.1f}s  "
        f"(ideal seed excluded)",
        flush=True,
    )
    print(
        f"{'slug':28} {'R0':>7} {'R_OT':>7} {'R_L1':>7} {'uniq':>6}  reuse",
        flush=True,
    )
    for s in summaries:
        print(
            f"{s['slug']:28} {s['start_rmsd_best']:7.3f} "
            f"{s['refine_rmsd_best']:7.3f} {s['polish_rmsd_best']:7.3f} "
            f"{s['best_uniq_index']:6d}  "
            f"{s['n_refine_reused']}/{s['top_l1']}",
            flush=True,
        )
    print(f"wrote {agg}", flush=True)


if __name__ == "__main__":
    main()
