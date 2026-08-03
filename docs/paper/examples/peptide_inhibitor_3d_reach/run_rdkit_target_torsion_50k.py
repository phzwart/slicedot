#!/usr/bin/env python3
"""RDKit-target torsion screen: 50k → L1 top-1k retained → refine a small cut.

Reuses RDKit targets from ``--target-tag`` (default ``rdkit_tgt_s42``).
Writes under ``ligands/<slug>/out/<tag>/`` (default ``rdkit_tgt_s42_n50k``).

Protocol
--------
1. Generate ``--n-conf`` torsion-randomized conformers (default 50 000).
2. Drop near-duplicates of the target (``--target-rmsd-cut``).
3. Torsion-fingerprint dedupe.
4. COM+PCA place each survivor (best OT among axis flips).
5. Rank by **map density at atom centres** (Coot linearization
   ``E = −Σ wᵢ ρ(xᵢ)`` — not full-grid L1 render); **retain** ``--top``
   (default 1000) placements on disk.
6. Dual-refine only the best ``--refine-top`` of those (default **10**):
   - OT+ADMM → L1+geom polish
   - L1+geom alone

Full-grid L1 ranking of 50k is intentionally avoided (hours/ligand); centre
sampling is ~seconds. Full ADMM on all 1000 is also not the default.

Usage
-----
  uv run python run_rdkit_target_torsion_50k.py
  uv run python run_rdkit_target_torsion_50k.py --refine-top 25
  uv run python run_rdkit_target_torsion_50k.py --place-only
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
    run_cleanup,
    run_l1_geom_polish,
    vg_ot,
)
from run_rdkit_target_protocol import (  # noqa: E402
    kabsch_rmsd,
    scene_for_target,
)
from screen_rdkit_conformers import (  # noqa: E402
    dedupe_by_torsion,
    ensure_sdf,
    generate_conformers,
    map_pca,
    pca_placements_3d,
    rmsd,
    score_ot,
    torsion_quartets,
)

torch.set_default_dtype(torch.float64)


def _load_target(slug: str, target_tag: str) -> tuple[np.ndarray, dict]:
    d = ROOT / "ligands" / slug / "out" / target_tag
    npz = d / "target.npz"
    if not npz.is_file():
        raise FileNotFoundError(
            f"missing {npz}; run run_rdkit_target_protocol.py first "
            f"(or pass --target-tag)"
        )
    X = np.asarray(np.load(npz)["X"], dtype=np.float64)
    meta = {}
    j = d / "target.json"
    if j.is_file():
        meta = json.loads(j.read_text())
    return X, meta


def score_map_centers(linmap, X: np.ndarray, w: np.ndarray) -> float:
    """E = −Σ wᵢ ρ(xᵢ) via trilinear samples (value only; no grad)."""
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    T = linmap.T
    sp = linmap.spacing
    org = linmap.origin
    nx, ny, nz = linmap.shape
    t = (X - org) / sp
    t = np.clip(t, 0.0, np.array([nx - 1.001, ny - 1.001, nz - 1.001]))
    i0 = np.floor(t).astype(np.int64)
    i1 = np.minimum(i0 + 1, np.array(linmap.shape, dtype=np.int64) - 1)
    f = t - i0
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    c000 = T[i0[:, 0], i0[:, 1], i0[:, 2]]
    c001 = T[i0[:, 0], i0[:, 1], i1[:, 2]]
    c010 = T[i0[:, 0], i1[:, 1], i0[:, 2]]
    c011 = T[i0[:, 0], i1[:, 1], i1[:, 2]]
    c100 = T[i1[:, 0], i0[:, 1], i0[:, 2]]
    c101 = T[i1[:, 0], i0[:, 1], i1[:, 2]]
    c110 = T[i1[:, 0], i1[:, 1], i0[:, 2]]
    c111 = T[i1[:, 0], i1[:, 1], i1[:, 2]]
    rho = (
        c000 * (1 - fx) * (1 - fy) * (1 - fz)
        + c001 * (1 - fx) * (1 - fy) * fz
        + c010 * (1 - fx) * fy * (1 - fz)
        + c011 * (1 - fx) * fy * fz
        + c100 * fx * (1 - fy) * (1 - fz)
        + c101 * fx * (1 - fy) * fz
        + c110 * fx * fy * (1 - fz)
        + c111 * fx * fy * fz
    )
    return float(-(w * rho).sum())


def place_and_rank(
    scene: dict,
    topo: dict,
    confs_u: np.ndarray,
    uniq_idx: np.ndarray,
    *,
    top: int,
) -> dict:
    w = topo["W"]
    X_true = scene["X_true"]
    sig = scene["sigma"]
    ot = scene["ot"]
    linmap = scene["linmap"]
    _, map_axes = map_pca(scene["T"], scene["origin"], scene["spacing"])
    map_com = X_true.mean(0)

    n_u = len(confs_u)
    pca_E = np.empty(n_u, dtype=np.float64)
    pca_rmsd = np.empty(n_u, dtype=np.float64)
    placed = np.empty_like(confs_u)
    t0 = time.perf_counter()
    for k, Xc in enumerate(confs_u):
        best_E, best_X = np.inf, None
        for Xp in pca_placements_3d(Xc, w, map_com, map_axes):
            E = score_ot(ot, Xp, w, sig)
            if E < best_E:
                best_E, best_X = E, Xp
        placed[k] = best_X
        pca_E[k] = best_E
        pca_rmsd[k] = rmsd(best_X, X_true)
        if (k + 1) % 2000 == 0 or k + 1 == n_u:
            print(
                f"  [place] PCA+OT {k+1}/{n_u}  "
                f"best_OT={np.min(pca_E[:k+1]):.5g}  "
                f"({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )
    t_pca = time.perf_counter() - t0

    print(
        f"  [place] map@centres scoring {n_u} placed confs "
        f"(E=−Σ w ρ) …",
        flush=True,
    )
    t1 = time.perf_counter()
    map_E = np.array(
        [score_map_centers(linmap, placed[k], w) for k in range(n_u)],
        dtype=np.float64,
    )
    t_map = time.perf_counter() - t1
    # Lower (more negative) = denser fit at atom centres.
    order = np.argsort(map_E)
    n_top = min(int(top), len(order))
    top_idx = order[:n_top]

    start_poses = placed[top_idx].copy()
    start_rmsd = np.array([rmsd(start_poses[j], X_true) for j in range(n_top)])
    start_uniq = uniq_idx[top_idx].astype(np.int64)
    start_map = map_E[top_idx].copy()
    start_pca_E = pca_E[top_idx].copy()

    print(
        f"  [place] top-{n_top} by map@centres  "
        f"E∈[{start_map[0]:.5g}, {start_map[-1]:.5g}]  "
        f"start RMSD best={float(np.min(start_rmsd)):.3f} Å  "
        f"mean={float(np.mean(start_rmsd)):.3f} Å  "
        f"(pca {t_pca:.1f}s, map {t_map:.1f}s)",
        flush=True,
    )
    return {
        "pca_E": pca_E,
        "pca_rmsd": pca_rmsd,
        "placed": placed,
        "map_E": map_E,
        "l1_E": map_E,  # alias for downstream keys that still say start_l1
        "top_idx": top_idx,
        "start_poses": start_poses,
        "start_rmsd": start_rmsd,
        "start_uniq": start_uniq,
        "start_l1": start_map,
        "start_pca_E": start_pca_E,
        "t_pca_s": float(t_pca),
        "t_l1_s": float(t_map),
        "rank": "map_centers",
    }


def run_dual_refine(
    scene: dict,
    topo: dict,
    place: dict,
    *,
    seed: int,
    out_npz: Path,
    ckpt_every: int = 25,
) -> dict:
    """OT+ADMM→L1/geom and L1/geom-alone on every start in ``place``; resume-capable."""
    w = topo["W"]
    X_true = scene["X_true"]
    sig = scene["sigma"]
    start_poses = place["start_poses"]
    start_rmsd = place["start_rmsd"]
    start_uniq = place["start_uniq"]
    n = len(start_poses)

    def _empty():
        return {
            "refine_poses": np.full_like(start_poses, np.nan),
            "refine_E": np.full(n, np.nan),
            "refine_L1": np.full(n, np.nan),
            "refine_rmsd": np.full(n, np.nan),
            "refine_stop": np.array([""] * n, dtype=object),
            "refine_time_s": np.full(n, np.nan),
            "ot_then_l1_poses": np.full_like(start_poses, np.nan),
            "ot_then_l1_rmsd": np.full(n, np.nan),
            "ot_then_l1_stop": np.array([""] * n, dtype=object),
            "polish_poses": np.full_like(start_poses, np.nan),
            "polish_L1": np.full(n, np.nan),
            "polish_ot": np.full(n, np.nan),
            "polish_rmsd": np.full(n, np.nan),
            "polish_stop": np.array([""] * n, dtype=object),
            "polish_time_s": np.full(n, np.nan),
            "done_mask": np.zeros(n, dtype=bool),
        }

    state = _empty()
    if out_npz.is_file():
        prev = np.load(out_npz, allow_pickle=True)
        if (
            "done_mask" in prev.files
            and prev["start_uniq"].shape == start_uniq.shape
            and np.array_equal(prev["start_uniq"], start_uniq)
        ):
            for k in state:
                if k in prev.files:
                    state[k] = prev[k]
            print(
                f"  [refine] resume  {int(state['done_mask'].sum())}/{n} done",
                flush=True,
            )
        else:
            print("  [refine] existing npz mismatch — starting fresh", flush=True)

    def _save():
        np.savez_compressed(
            out_npz,
            resolution=np.array(scene["resolution"]),
            sigma=np.array(sig),
            start_poses=start_poses,
            start_rmsd=start_rmsd,
            start_uniq=start_uniq,
            start_l1=place["start_l1"],
            start_pca_E=place["start_pca_E"],
            X_true=X_true,
            **{k: state[k] for k in state},
        )

    t0 = time.perf_counter()
    for j in range(n):
        if state["done_mask"][j]:
            continue
        Xp = start_poses[j].copy()
        u = int(start_uniq[j])
        seed_j = int(seed) + u

        t1 = time.perf_counter()
        cleanup = run_cleanup(
            scene, Xp, seed=seed_j, log_every=10**9, named_atoms=False,
        )
        X_admm = cleanup["poses"][-1]
        state["refine_poses"][j] = X_admm
        state["refine_E"][j] = float(cleanup["energies"][-1])
        state["refine_L1"][j] = float(cleanup["l1_energies"][-1])
        state["refine_rmsd"][j] = rmsd(X_admm, X_true)
        state["refine_stop"][j] = str(cleanup["stop_reason"])
        state["refine_time_s"][j] = time.perf_counter() - t1

        pol_ot = run_l1_geom_polish(scene, X_admm, seed=seed_j)
        X_ot_l1 = np.asarray(pol_ot["poses"][-1], dtype=np.float64)
        state["ot_then_l1_poses"][j] = X_ot_l1
        state["ot_then_l1_rmsd"][j] = rmsd(X_ot_l1, X_true)
        state["ot_then_l1_stop"][j] = str(pol_ot.get("stop_reason", ""))

        t2 = time.perf_counter()
        polish = run_l1_geom_polish(scene, Xp, seed=seed_j + 17)
        Xp_l1 = np.asarray(polish["poses"][-1], dtype=np.float64)
        state["polish_poses"][j] = Xp_l1
        state["polish_L1"][j] = float(scene["l1"].value_grad(Xp_l1, w)[0])
        state["polish_ot"][j] = float(vg_ot(scene["ot"], Xp_l1, w, sig)[0])
        state["polish_rmsd"][j] = rmsd(Xp_l1, X_true)
        state["polish_stop"][j] = str(polish.get("stop_reason", "l1_geom"))
        state["polish_time_s"][j] = time.perf_counter() - t2
        state["done_mask"][j] = True

        if (j + 1) % 10 == 0 or j == 0 or j + 1 == n:
            nd = int(state["done_mask"].sum())
            print(
                f"  [refine] {nd}/{n}  uniq#{u}  "
                f"start={start_rmsd[j]:.3f}  "
                f"ADMM={state['refine_rmsd'][j]:.3f}  "
                f"OT→L1={state['ot_then_l1_rmsd'][j]:.3f}  "
                f"L1={state['polish_rmsd'][j]:.3f}  "
                f"({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
        if (j + 1) % int(ckpt_every) == 0 or j + 1 == n:
            _save()

    # energy-selected + oracle-min RMSD summaries
    done = state["done_mask"]
    assert done.all()
    j_ot = int(np.nanargmin(state["refine_E"]))
    j_l1 = int(np.nanargmin(state["polish_L1"]))
    summary = {
        "n_top": int(n),
        "start_rmsd_best": float(np.nanmin(start_rmsd)),
        "start_rmsd_mean": float(np.nanmean(start_rmsd)),
        "admm_rmsd_at_best_E": float(state["refine_rmsd"][j_ot]),
        "admm_E_best": float(state["refine_E"][j_ot]),
        "ot_then_l1_rmsd_at_best_E": float(state["ot_then_l1_rmsd"][j_ot]),
        "l1_alone_rmsd_at_best_L1": float(state["polish_rmsd"][j_l1]),
        "l1_alone_L1_best": float(state["polish_L1"][j_l1]),
        "min_admm_rmsd": float(np.nanmin(state["refine_rmsd"])),
        "min_ot_then_l1_rmsd": float(np.nanmin(state["ot_then_l1_rmsd"])),
        "min_l1_alone_rmsd": float(np.nanmin(state["polish_rmsd"])),
        "best_uniq_admm_E": int(start_uniq[j_ot]),
        "best_uniq_l1_E": int(start_uniq[j_l1]),
        "t_refine_total_s": float(np.nansum(state["refine_time_s"])),
        "t_polish_total_s": float(np.nansum(state["polish_time_s"])),
        "t_wall_s": float(time.perf_counter() - t0),
        "npz": str(out_npz.relative_to(ROOT)),
    }
    _save()
    out_npz.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"  [refine] done  R0={summary['start_rmsd_best']:.3f}  "
        f"min ADMM={summary['min_admm_rmsd']:.3f}  "
        f"min OT→L1={summary['min_ot_then_l1_rmsd']:.3f}  "
        f"min L1={summary['min_l1_alone_rmsd']:.3f}  "
        f"(E-pick OT→L1={summary['ot_then_l1_rmsd_at_best_E']:.3f}  "
        f"L1-pick={summary['l1_alone_rmsd_at_best_L1']:.3f})",
        flush=True,
    )
    return summary


def run_ligand(
    entry: dict,
    *,
    tag: str,
    target_tag: str,
    resolution: float,
    seed: int,
    n_conf: int,
    top: int,
    refine_top: int,
    torsion_bin: float,
    target_rmsd_cut: float,
    mmff: bool,
    place_only: bool,
    skip_existing: bool,
    ckpt_every: int,
) -> dict:
    slug = entry["slug"]
    topo = load_ligand(slug)
    out_dir = ROOT / "ligands" / slug / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    res = float(resolution)
    place_npz = out_dir / f"place_{res:g}A_n{n_conf}_l1top{top}.npz"
    screen_npz = out_dir / (
        f"screen_{res:g}A_n{n_conf}_l1top{top}_refine{refine_top}.npz"
    )
    summary_path = out_dir / "summary.json"

    if skip_existing and screen_npz.is_file() and screen_npz.with_suffix(".json").is_file():
        print(f"[skip] {slug} screen exists", flush=True)
        return json.loads(screen_npz.with_suffix(".json").read_text())

    print(
        f"\n{'=' * 72}\n"
        f"{topo.get('label', slug)} ({slug})  tag={tag}  "
        f"n_conf={n_conf}  L1-retain={top}  refine={refine_top}  @ {res:g} Å\n"
        f"{'=' * 72}",
        flush=True,
    )

    X_target, tgt_meta = _load_target(slug, target_tag)
    # copy target into this tag for self-contained folder
    tgt_out = out_dir / "target.npz"
    if not tgt_out.is_file():
        np.savez_compressed(
            tgt_out, X=X_target, names=np.array(topo["names"]), Z=topo["Z"],
        )
        (out_dir / "target.json").write_text(
            json.dumps({**tgt_meta, "source_tag": target_tag}, indent=2) + "\n"
        )
    scene = scene_for_target(topo, X_target, res)
    print(
        f"  target vs ideal "
        f"{tgt_meta.get('rmsd_vs_ideal_A', float('nan')):.3f} Å  "
        f"σ={scene['sigma']:.3f}",
        flush=True,
    )

    lib_seed = int(seed)
    conf_path = out_dir / f"conformers_n{n_conf}_seed{lib_seed}.npz"
    if conf_path.is_file():
        confs = np.asarray(np.load(conf_path)["confs"], dtype=np.float64)
        print(f"  [conf] reuse {conf_path.name} ({len(confs)})", flush=True)
        conf_meta = {"reused": True}
    else:
        print(
            f"  [conf] generate {n_conf}  seed={lib_seed}  mmff={mmff} …",
            flush=True,
        )
        t0 = time.perf_counter()
        sdf = ensure_sdf(entry, ROOT / "ligands" / slug)
        confs, conf_meta = generate_conformers(
            sdf, topo, n_conf=int(n_conf), seed=lib_seed, mmff=mmff,
        )
        np.savez_compressed(
            conf_path,
            confs=confs,
            names=np.array(topo["names"]),
            seed=np.array(lib_seed),
            bond_rms_vs_ref_A=np.array(conf_meta.get("bond_rms_vs_ref_A", np.nan)),
        )
        conf_path.with_suffix(".json").write_text(
            json.dumps(conf_meta, indent=2) + "\n"
        )
        print(
            f"  [conf] wrote {conf_path.name} in {time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    Xt0 = X_target - X_target.mean(0)
    keep = np.ones(len(confs), dtype=bool)
    n_near = 0
    for i, Xc in enumerate(confs):
        if kabsch_rmsd(Xc, Xt0) < float(target_rmsd_cut):
            keep[i] = False
            n_near += 1
    confs_f = confs[keep]
    orig_idx = np.where(keep)[0]
    print(
        f"  [conf] drop {n_near} within {target_rmsd_cut:g} Å of target  "
        f"→ {len(confs_f)}",
        flush=True,
    )

    quarts = torsion_quartets(topo["n"], topo["bonds"], topo["rotatable_bonds"])
    uniq_local = dedupe_by_torsion(confs_f, quarts, bin_deg=float(torsion_bin))
    confs_u = confs_f[uniq_local]
    uniq_idx = orig_idx[uniq_local]
    print(
        f"  [conf] unique={len(confs_u)} / {len(confs_f)} "
        f"(bin={torsion_bin:g}°)",
        flush=True,
    )

    if place_npz.is_file() and not place_only:
        # allow reuse of placement
        z = np.load(place_npz)
        place = {
            "start_poses": np.asarray(z["start_poses"], dtype=np.float64),
            "start_rmsd": np.asarray(z["start_rmsd"], dtype=np.float64),
            "start_uniq": np.asarray(z["start_uniq"], dtype=np.int64),
            "start_l1": np.asarray(z["start_l1"], dtype=np.float64),
            "start_pca_E": np.asarray(z["start_pca_E"], dtype=np.float64),
            "t_pca_s": float(z["t_pca_s"]) if "t_pca_s" in z.files else float("nan"),
            "t_l1_s": float(z["t_l1_s"]) if "t_l1_s" in z.files else float("nan"),
        }
        print(
            f"  [place] reuse {place_npz.name}  "
            f"top={len(place['start_poses'])}  "
            f"R0 best={float(np.min(place['start_rmsd'])):.3f}",
            flush=True,
        )
        place_summary = json.loads(place_npz.with_suffix(".json").read_text()) \
            if place_npz.with_suffix(".json").is_file() else {}
    else:
        raw = place_and_rank(
            scene, topo, confs_u, uniq_idx, top=top,
        )
        place = raw
        place_summary = {
            "slug": slug,
            "resolution": res,
            "n_conf_raw": int(len(confs)),
            "n_near_dropped": int(n_near),
            "n_conf_unique": int(len(confs_u)),
            "top_l1": int(len(raw["start_poses"])),
            "refine_top": int(refine_top),
            "t_pca_s": raw["t_pca_s"],
            "t_l1_s": raw["t_l1_s"],
            "start_rmsd_best": float(np.min(raw["start_rmsd"])),
            "start_rmsd_mean": float(np.mean(raw["start_rmsd"])),
            "l1_best": float(raw["start_l1"][0]),
            "l1_at_top_cut": float(raw["start_l1"][-1]),
            "npz": str(place_npz.relative_to(ROOT)),
            "seed": int(lib_seed),
            "mmff": bool(mmff),
            "target_rmsd_cut_A": float(target_rmsd_cut),
            "target_tag": target_tag,
        }
        # Retain top-K starts + full score vectors; skip 50k×N coordinate dumps.
        np.savez_compressed(
            place_npz,
            resolution=np.array(res),
            sigma=np.array(scene["sigma"]),
            uniq_idx=uniq_idx,
            pca_E=raw["pca_E"],
            pca_rmsd=raw["pca_rmsd"],
            l1_E=raw["l1_E"],
            top_idx=raw["top_idx"],
            start_poses=raw["start_poses"],
            start_rmsd=raw["start_rmsd"],
            start_uniq=raw["start_uniq"],
            start_l1=raw["start_l1"],
            start_pca_E=raw["start_pca_E"],
            X_true=scene["X_true"],
            t_pca_s=np.array(raw["t_pca_s"]),
            t_l1_s=np.array(raw["t_l1_s"]),
            n_near_dropped=np.array(n_near),
        )
        place_npz.with_suffix(".json").write_text(
            json.dumps(place_summary, indent=2) + "\n"
        )
        print(f"  [place] saved {place_npz.name}  (top-{len(raw['start_poses'])} retained)", flush=True)

    summary: dict = {
        "slug": slug,
        "tag": tag,
        "target_tag": target_tag,
        "resolution": res,
        "seed": int(lib_seed),
        "n_conf": int(n_conf),
        "top": int(top),
        "refine_top": int(refine_top),
        "target": tgt_meta,
        "place": place_summary,
    }

    if place_only:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    n_ref = min(int(refine_top), len(place["start_poses"]))
    place_ref = {
        "start_poses": place["start_poses"][:n_ref],
        "start_rmsd": place["start_rmsd"][:n_ref],
        "start_uniq": place["start_uniq"][:n_ref],
        "start_l1": place["start_l1"][:n_ref],
        "start_pca_E": place["start_pca_E"][:n_ref],
    }
    print(
        f"  [refine] dual-path on best {n_ref} / {len(place['start_poses'])} "
        f"L1-retained starts",
        flush=True,
    )
    ref = run_dual_refine(
        scene, topo, place_ref,
        seed=lib_seed,
        out_npz=screen_npz,
        ckpt_every=ckpt_every,
    )
    summary["torsion"] = ref
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligands", type=str, default=None)
    ap.add_argument("--resolution", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=10042,
                    help="Conformer library seed (default 10042 = prior lib seed).")
    ap.add_argument("--tag", type=str, default="rdkit_tgt_s42_n50k")
    ap.add_argument("--target-tag", type=str, default="rdkit_tgt_s42")
    ap.add_argument("--n-conf", type=int, default=50_000)
    ap.add_argument("--top", type=int, default=1000,
                    help="Retain after L1 ranking (default 1000).")
    ap.add_argument("--refine-top", type=int, default=10,
                    help="Dual-refine only this many best L1 starts (default 10).")
    ap.add_argument("--torsion-bin", type=float, default=15.0)
    ap.add_argument("--target-rmsd-cut", type=float, default=0.75)
    ap.add_argument("--mmff", action="store_true",
                    help="MMFF-relax confs (slow at 50k; off by default).")
    ap.add_argument("--place-only", action="store_true",
                    help="Stop after saving L1 top-K placements.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=25)
    args = ap.parse_args()

    entries = list_ligands()
    if args.ligands:
        want = {s.strip() for s in args.ligands.split(",") if s.strip()}
        entries = [e for e in entries if e["slug"] in want]
        missing = want - {e["slug"] for e in entries}
        if missing:
            raise SystemExit(f"unknown slug(s): {sorted(missing)}")

    print(
        f"50k torsion protocol: {len(entries)} ligands  tag={args.tag}  "
        f"n_conf={args.n_conf}  L1-retain={args.top}  "
        f"refine-top={args.refine_top}  mmff={args.mmff}  "
        f"place_only={args.place_only}",
        flush=True,
    )
    t0 = time.perf_counter()
    rows = []
    for e in entries:
        rows.append(
            run_ligand(
                e,
                tag=args.tag,
                target_tag=args.target_tag,
                resolution=args.resolution,
                seed=args.seed,
                n_conf=args.n_conf,
                top=args.top,
                refine_top=args.refine_top,
                torsion_bin=args.torsion_bin,
                target_rmsd_cut=args.target_rmsd_cut,
                mmff=args.mmff,
                place_only=args.place_only,
                skip_existing=args.skip_existing,
                ckpt_every=args.ckpt_every,
            )
        )

    agg = ROOT / "out" / f"{args.tag}_summary.json"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"\n{'=' * 72}", flush=True)
    print(
        f"finished {len(rows)} in {time.perf_counter()-t0:.1f}s  → {agg}",
        flush=True,
    )
    print(
        f"{'slug':14} {'R0':>7} {'minOT→L1':>9} {'minL1':>7} "
        f"{'E:OT→L1':>9} {'L1pick':>7}",
        flush=True,
    )
    for r in rows:
        t = r.get("torsion") or {}
        p = r.get("place") or {}
        print(
            f"{r['slug']:14} "
            f"{t.get('start_rmsd_best', p.get('start_rmsd_best', float('nan'))):7.3f} "
            f"{t.get('min_ot_then_l1_rmsd', float('nan')):9.3f} "
            f"{t.get('min_l1_alone_rmsd', float('nan')):7.3f} "
            f"{t.get('ot_then_l1_rmsd_at_best_E', float('nan')):9.3f} "
            f"{t.get('l1_alone_rmsd_at_best_L1', float('nan')):7.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
