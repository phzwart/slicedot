#!/usr/bin/env python3
"""New-target protocol: RDKit plausible conf as map/truth (does not overwrite prior runs).

For each ligand at 3 Å (default):

1. **Target** — RDKit torsion-randomize + MMFF from the CCD ideal SDF; pick a
   conf far from ideal. Saved under ``ligands/<slug>/out/<tag>/``.
2. **Free OT ×2** — free atoms → prune/name → from the *named* start run both:
   - OT+ADMM → L1+geom polish
   - L1+geom polish alone
3. **Torsion screen** — new ensemble (different seed), drop near-duplicates of
   the target, PCA+OT → L1 top-K → from each start run both refine paths.

Tag default ``rdkit_tgt_s<seed>`` keeps all products separate from the ideal-map
runs (``trajectory_*_x2.npz``, ``rdkit_screen/``, …).

Usage
-----
  uv run python run_rdkit_target_protocol.py
  uv run python run_rdkit_target_protocol.py --ligands amprenavir --seed 42
  uv run python run_rdkit_target_protocol.py --skip-free
  uv run python run_rdkit_target_protocol.py --skip-torsion
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
    aut_rmsd,
    run_cleanup,
    run_free_ot,
    run_l1_geom_polish,
    vg_ot,
)
from run_resolution_sweep import build_scene  # noqa: E402
from screen_rdkit_conformers import (  # noqa: E402
    dedupe_by_torsion,
    ensure_sdf,
    generate_conformers,
    pca_placements_3d,
    map_pca,
    rmsd,
    score_l1,
    score_ot,
    torsion_quartets,
)
from slicedot import Geometry, prune_ghosts  # noqa: E402

torch.set_default_dtype(torch.float64)


def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, float) - np.asarray(A, float).mean(0)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0)
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return float(np.sqrt(((A @ R - B) ** 2).sum(1).mean()))


def make_target_conf(
    entry: dict,
    topo: dict,
    *,
    seed: int,
    n_cand: int = 32,
    min_rmsd: float = 0.75,
    mmff: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return one RDKit plausible heavy-atom conf (COM-centred) + meta."""
    lig_dir = ROOT / "ligands" / entry["slug"]
    sdf = ensure_sdf(entry, lig_dir)
    # conf0 = ideal; rest = torsion-randomized
    confs, gen_meta = generate_conformers(
        sdf, topo, n_conf=int(n_cand), seed=int(seed), mmff=mmff,
    )
    X_ideal = confs[0]
    best_i, best_r = 1, -1.0
    chosen = None
    for i in range(1, len(confs)):
        r = kabsch_rmsd(confs[i], X_ideal)
        if r > best_r:
            best_i, best_r = i, r
        if r >= float(min_rmsd) and chosen is None:
            chosen = i
            break
    if chosen is None:
        chosen = best_i
    X = confs[chosen].copy()
    X -= X.mean(0)
    # also vs topology CCD coords
    r_topo = kabsch_rmsd(X, topo["X_ref"] - topo["X_ref"].mean(0))
    meta = {
        **{k: gen_meta[k] for k in (
            "sampler", "n_rotatable_sampled", "bond_rms_vs_ref_A",
            "mmff", "seed", "sdf",
        ) if k in gen_meta},
        "n_cand": int(n_cand),
        "chosen_index": int(chosen),
        "rmsd_vs_ideal_A": float(kabsch_rmsd(X, X_ideal - X_ideal.mean(0))),
        "rmsd_vs_topo_A": float(r_topo),
        "min_rmsd_req_A": float(min_rmsd),
    }
    return X, meta


def scene_for_target(topo: dict, X_target: np.ndarray, resolution: float) -> dict:
    """Build map/OT/L1/namer with RDKit target as X_true / X_ref."""
    topo_t = dict(topo)
    Xt = np.asarray(X_target, dtype=np.float64).copy()
    Xt = Xt - Xt.mean(0)
    topo_t["X_ref"] = Xt
    scene = build_scene(topo_t, float(resolution))
    if not scene["geom"].chiral_centres and topo["chiral_centres"]:
        scene["geom"] = Geometry(
            Xt,
            topo["bonds"],
            rotatable_bonds=topo["rotatable_bonds"],
            chiral_centres=topo["chiral_centres"],
            planar_groups=topo["planar_groups"],
            antibump=True,
        )
    scene["target_source"] = "rdkit_plausible"
    return scene


def run_free_branch(
    scene: dict,
    *,
    seed: int,
    atom_factor: float,
) -> dict:
    """Free OT → name/prune → OT+ADMM→L1/geom  and  L1/geom-alone from named."""
    rng = np.random.default_rng(int(seed))
    X_true = scene["X_true"]
    w_chem = np.asarray(scene["w"], dtype=np.float64)
    n = len(X_true)
    n_model = int(round(float(atom_factor) * n))
    half = scene["half"]
    X_start = X_true.mean(0) + rng.uniform(-half, half, size=(n_model, 3))
    w_free = (
        w_chem if n_model == n
        else np.full(n_model, 1.0 / n_model, dtype=np.float64)
    )

    print(
        f"  [free] OT  N_model={n_model} (true={n}, ×{atom_factor:g})",
        flush=True,
    )
    free = run_free_ot(scene, X_start, w=w_free)
    X_free = free["poses"][-1].copy()
    print(
        f"  [free] done  steps={free['n_steps']}  "
        f"NN={free['nn_rmsds'][-1]:.4f} Å  stop={free['stop_reason']}",
        flush=True,
    )
    scene["l1"].sigma = float(scene["sigma"])

    print("  [free] ghost prune / name …", flush=True)
    X_prior = (
        scene["topo"]["X_ref"]
        - scene["topo"]["X_ref"].mean(0)
        + X_free.mean(0)
    )
    prune = prune_ghosts(
        X_free,
        namer=scene["namer"],
        X_prior=X_prior,
        l1_oracle=scene["l1"],
        w_chem=w_chem,
        sigma=float(scene["sigma"]),
        verbose=True,
    )
    named = prune.Y_named.copy()
    n_match = 0
    for alpha in scene["namer"].automorphisms:
        ok = 0
        for i in range(n):
            j = int(np.argmin(np.linalg.norm(X_true - named[i], axis=1)))
            if j == int(alpha[i]):
                ok += 1
        n_match = max(n_match, ok)
    named_rmsd = float(aut_rmsd(named, X_true, scene["namer"]))
    print(
        f"  [free] named  RMSD={named_rmsd:.4f} Å  match={n_match}/{n}",
        flush=True,
    )

    # Path A: OT+ADMM → L1+geom
    print("  [free] OT+ADMM …", flush=True)
    cleanup = run_cleanup(scene, named, seed=seed, named_atoms=False)
    X_admm = cleanup["poses"][-1].copy()
    print("  [free] L1+geom after OT+ADMM …", flush=True)
    polish_after = run_l1_geom_polish(scene, X_admm, seed=seed)
    X_ot_l1 = polish_after["poses"][-1].copy()

    # Path B: L1+geom alone from named
    print("  [free] L1+geom alone from named …", flush=True)
    polish_alone = run_l1_geom_polish(scene, named, seed=seed + 17)
    X_l1 = polish_alone["poses"][-1].copy()

    out = {
        "seed": int(seed),
        "atom_factor": float(atom_factor),
        "free_nn": float(free["nn_rmsds"][-1]),
        "free_steps": int(free["n_steps"]),
        "free_stop": str(free["stop_reason"]),
        "named_rmsd": named_rmsd,
        "n_match": int(n_match),
        "n_ghosts": int(prune.ghost_idx.size),
        "admm_rmsd": float(cleanup["rmsds"][-1]),
        "admm_ot": float(cleanup["energies"][-1]),
        "admm_l1": float(cleanup["l1_energies"][-1]),
        "admm_stop": str(cleanup["stop_reason"]),
        "ot_then_l1_rmsd": float(polish_after["rmsds"][-1]),
        "ot_then_l1_stop": str(polish_after["stop_reason"]),
        "l1_alone_rmsd": float(polish_alone["rmsds"][-1]),
        "l1_alone_stop": str(polish_alone["stop_reason"]),
        "named": named,
        "admm": X_admm,
        "ot_then_l1": X_ot_l1,
        "l1_alone": X_l1,
    }
    print(
        f"  [free] summary  named={named_rmsd:.3f}  "
        f"ADMM={out['admm_rmsd']:.3f}  "
        f"OT→L1={out['ot_then_l1_rmsd']:.3f}  "
        f"L1-alone={out['l1_alone_rmsd']:.3f}",
        flush=True,
    )
    return out


def run_torsion_branch(
    scene: dict,
    topo: dict,
    entry: dict,
    *,
    seed: int,
    n_conf: int,
    top_pca: int,
    top: int,
    torsion_bin: float,
    target_rmsd_cut: float,
    mmff: bool,
    out_dir: Path,
) -> dict:
    """Torsion library (new seed) vs RDKit target; both refine paths."""
    slug = entry["slug"]
    w = topo["W"]
    X_true = scene["X_true"]
    sig = scene["sigma"]
    ot, l1 = scene["ot"], scene["l1"]
    geom = scene["geom"]
    res = scene["resolution"]

    conf_path = out_dir / f"conformers_n{n_conf}_seed{seed}.npz"
    if conf_path.is_file():
        confs = np.asarray(np.load(conf_path)["confs"], dtype=np.float64)
        print(f"  [torsion] reuse {conf_path.name} ({len(confs)})", flush=True)
        conf_meta = {"reused": True}
    else:
        print(f"  [torsion] generate {n_conf} confs seed={seed} …", flush=True)
        sdf = ensure_sdf(entry, ROOT / "ligands" / slug)
        # Use original topo (CCD) for atom-order check vs SDF
        confs, conf_meta = generate_conformers(
            sdf, topo, n_conf=int(n_conf), seed=int(seed), mmff=mmff,
        )
        np.savez_compressed(
            conf_path,
            confs=confs,
            names=np.array(topo["names"]),
            seed=np.array(seed),
            bond_rms_vs_ref_A=np.array(conf_meta.get("bond_rms_vs_ref_A", np.nan)),
        )
        conf_path.with_suffix(".json").write_text(
            json.dumps(conf_meta, indent=2) + "\n"
        )

    # Drop near-target and (usually) ideal-like conf 0 if too close to target
    Xt0 = X_true - X_true.mean(0)
    keep_mask = np.ones(len(confs), dtype=bool)
    near = []
    for i, Xc in enumerate(confs):
        r = kabsch_rmsd(Xc, Xt0)
        if r < float(target_rmsd_cut):
            keep_mask[i] = False
            near.append((i, r))
    confs_f = confs[keep_mask]
    orig_idx = np.where(keep_mask)[0]
    print(
        f"  [torsion] dropped {len(near)} confs within "
        f"{target_rmsd_cut:g} Å of target  remaining={len(confs_f)}",
        flush=True,
    )

    quarts = torsion_quartets(
        topo["n"], topo["bonds"], topo["rotatable_bonds"],
    )
    uniq_local = dedupe_by_torsion(confs_f, quarts, bin_deg=float(torsion_bin))
    confs_u = confs_f[uniq_local]
    uniq_idx = orig_idx[uniq_local]  # indices into raw generated confs
    print(
        f"  [torsion] unique={len(confs_u)} / {len(confs_f)} "
        f"(bin={torsion_bin:g}°)",
        flush=True,
    )

    _, map_axes = map_pca(scene["T"], scene["origin"], scene["spacing"])
    map_com = X_true.mean(0)

    t_pca0 = time.perf_counter()
    pca_E = np.empty(len(confs_u), dtype=np.float64)
    pca_rmsd = np.empty(len(confs_u), dtype=np.float64)
    placed = np.empty_like(confs_u)
    for k, Xc in enumerate(confs_u):
        best_E, best_X = np.inf, None
        for Xp in pca_placements_3d(Xc, w, map_com, map_axes):
            E = score_ot(ot, Xp, w, sig)
            if E < best_E:
                best_E, best_X = E, Xp
        placed[k] = best_X
        pca_E[k] = best_E
        pca_rmsd[k] = rmsd(best_X, X_true)
        if (k + 1) % 500 == 0 or k + 1 == len(confs_u):
            print(
                f"  [torsion] PCA+OT {k+1}/{len(confs_u)}  "
                f"best={np.min(pca_E[:k+1]):.5g}",
                flush=True,
            )
    t_pca = time.perf_counter() - t_pca0
    order_pca = np.argsort(pca_E)
    n_pca = min(int(top_pca), len(order_pca))
    pca_top_idx = order_pca[:n_pca]

    l1_E = np.array(
        [score_l1(l1, placed[k], w) for k in pca_top_idx], dtype=np.float64,
    )
    order_l1 = np.argsort(l1_E)
    n_l1 = min(int(top), len(order_l1))
    l1_sel = order_l1[:n_l1]
    refine_src = pca_top_idx[l1_sel]

    start_poses = np.array([placed[k].copy() for k in refine_src])
    start_rmsd = np.array([rmsd(start_poses[j], X_true) for j in range(n_l1)])
    start_uniq = np.array([int(uniq_idx[k]) for k in refine_src], dtype=np.int64)

    place_npz = out_dir / (
        f"place_{res:g}A_n{n_conf}_pca{top_pca}_l1top{top}.npz"
    )
    np.savez_compressed(
        place_npz,
        resolution=np.array(res),
        sigma=np.array(sig),
        uniq_idx=uniq_idx,
        confs_unique=confs_u,
        pca_E=pca_E,
        pca_rmsd=pca_rmsd,
        placed=placed,
        pca_top_idx=pca_top_idx,
        l1_E_pca_top=l1_E,
        refine_src=refine_src,
        start_poses=start_poses,
        start_rmsd=start_rmsd,
        start_uniq=start_uniq,
        X_true=X_true,
        dropped_near_target=np.array(len(near)),
    )
    print(
        f"  [torsion] place saved  start best={float(np.min(start_rmsd)):.3f} Å",
        flush=True,
    )

    refine_E = np.full(n_l1, np.nan)
    refine_rmsd = np.full(n_l1, np.nan)
    refine_L1 = np.full(n_l1, np.nan)
    refine_stop = np.array([""] * n_l1, dtype=object)
    refine_time = np.full(n_l1, np.nan)
    refine_poses = np.full_like(start_poses, np.nan)
    # OT+ADMM → L1 polish
    ot_then_l1_rmsd = np.full(n_l1, np.nan)
    ot_then_l1_poses = np.full_like(start_poses, np.nan)
    ot_then_l1_stop = np.array([""] * n_l1, dtype=object)
    # L1 alone
    polish_rmsd = np.full(n_l1, np.nan)
    polish_L1 = np.full(n_l1, np.nan)
    polish_ot = np.full(n_l1, np.nan)
    polish_stop = np.array([""] * n_l1, dtype=object)
    polish_time = np.full(n_l1, np.nan)
    polish_poses = np.full_like(start_poses, np.nan)

    for j in range(n_l1):
        Xp = start_poses[j].copy()
        u = int(start_uniq[j])
        seed_j = int(seed) + u

        t1 = time.perf_counter()
        cleanup = run_cleanup(
            scene, Xp, seed=seed_j, log_every=10**9, named_atoms=False,
        )
        X_admm = cleanup["poses"][-1]
        refine_poses[j] = X_admm
        refine_E[j] = float(cleanup["energies"][-1])
        refine_L1[j] = float(cleanup["l1_energies"][-1])
        refine_rmsd[j] = rmsd(X_admm, X_true)
        refine_stop[j] = str(cleanup["stop_reason"])
        refine_time[j] = time.perf_counter() - t1

        polish_ot_path = run_l1_geom_polish(scene, X_admm, seed=seed_j)
        X_ot_l1 = np.asarray(polish_ot_path["poses"][-1], dtype=np.float64)
        ot_then_l1_poses[j] = X_ot_l1
        ot_then_l1_rmsd[j] = rmsd(X_ot_l1, X_true)
        ot_then_l1_stop[j] = str(polish_ot_path.get("stop_reason", ""))

        t2 = time.perf_counter()
        polish = run_l1_geom_polish(scene, Xp, seed=seed_j + 17)
        Xp_l1 = np.asarray(polish["poses"][-1], dtype=np.float64)
        polish_poses[j] = Xp_l1
        polish_L1[j] = float(scene["l1"].value_grad(Xp_l1, w)[0])
        polish_ot[j] = float(vg_ot(scene["ot"], Xp_l1, w, sig)[0])
        polish_rmsd[j] = rmsd(Xp_l1, X_true)
        polish_stop[j] = str(polish.get("stop_reason", "l1_geom"))
        polish_time[j] = time.perf_counter() - t2

        print(
            f"  [torsion] refine {j+1}/{n_l1}  uniq#{u}  "
            f"start={start_rmsd[j]:.3f}  "
            f"ADMM={refine_rmsd[j]:.3f}  "
            f"OT→L1={ot_then_l1_rmsd[j]:.3f}  "
            f"L1-alone={polish_rmsd[j]:.3f}",
            flush=True,
        )

    j_ot = int(np.nanargmin(refine_E))
    j_l1 = int(np.nanargmin(polish_L1))
    summary = {
        "slug": slug,
        "resolution": float(res),
        "n_conf_raw": int(len(confs)),
        "n_conf_after_target_filter": int(len(confs_f)),
        "n_near_target_dropped": int(len(near)),
        "n_conf_unique": int(len(confs_u)),
        "top_pca": int(n_pca),
        "top_l1": int(n_l1),
        "t_pca_s": float(t_pca),
        "start_rmsd_best": float(np.nanmin(start_rmsd)),
        "admm_rmsd_best": float(refine_rmsd[j_ot]),
        "admm_E_best": float(refine_E[j_ot]),
        "ot_then_l1_rmsd_best": float(ot_then_l1_rmsd[j_ot]),
        "ot_then_l1_rmsd_at_best_admm_E": float(ot_then_l1_rmsd[j_ot]),
        "l1_alone_rmsd_best": float(polish_rmsd[j_l1]),
        "l1_alone_L1_best": float(polish_L1[j_l1]),
        "best_uniq_admm": int(start_uniq[j_ot]),
        "best_uniq_l1": int(start_uniq[j_l1]),
        "min_admm_rmsd": float(np.nanmin(refine_rmsd)),
        "min_ot_then_l1_rmsd": float(np.nanmin(ot_then_l1_rmsd)),
        "min_l1_alone_rmsd": float(np.nanmin(polish_rmsd)),
        "target_rmsd_cut_A": float(target_rmsd_cut),
        "seed": int(seed),
        "conformer_meta": conf_meta,
    }

    screen_npz = out_dir / (
        f"screen_{res:g}A_n{n_conf}_pca{top_pca}_l1top{top}.npz"
    )
    np.savez_compressed(
        screen_npz,
        resolution=np.array(res),
        sigma=np.array(sig),
        uniq_idx=uniq_idx,
        start_poses=start_poses,
        start_rmsd=start_rmsd,
        start_uniq=start_uniq,
        refine_poses=refine_poses,
        refine_E=refine_E,
        refine_L1=refine_L1,
        refine_rmsd=refine_rmsd,
        refine_stop=np.asarray(refine_stop, dtype=object),
        refine_time_s=refine_time,
        ot_then_l1_poses=ot_then_l1_poses,
        ot_then_l1_rmsd=ot_then_l1_rmsd,
        ot_then_l1_stop=np.asarray(ot_then_l1_stop, dtype=object),
        polish_poses=polish_poses,
        polish_L1=polish_L1,
        polish_ot=polish_ot,
        polish_rmsd=polish_rmsd,
        polish_stop=np.asarray(polish_stop, dtype=object),
        polish_time_s=polish_time,
        X_true=X_true,
        pca_E=pca_E,
        pca_rmsd=pca_rmsd,
    )
    screen_npz.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    summary["npz"] = str(screen_npz.relative_to(ROOT))
    summary["place_npz"] = str(place_npz.relative_to(ROOT))
    print(
        f"  [torsion] done  R0={summary['start_rmsd_best']:.3f}  "
        f"ADMM={summary['admm_rmsd_best']:.3f}  "
        f"OT→L1={summary['ot_then_l1_rmsd_best']:.3f}  "
        f"L1={summary['l1_alone_rmsd_best']:.3f}",
        flush=True,
    )
    return summary


def run_ligand(
    entry: dict,
    *,
    tag: str,
    resolution: float,
    seed: int,
    atom_factor: float,
    n_conf: int,
    top_pca: int,
    top: int,
    torsion_bin: float,
    target_rmsd_cut: float,
    n_cand: int,
    skip_free: bool,
    skip_torsion: bool,
    mmff: bool,
) -> dict:
    slug = entry["slug"]
    topo = load_ligand(slug)
    out_dir = ROOT / "ligands" / slug / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'=' * 72}\n"
        f"{topo.get('label', slug)} ({slug})  tag={tag}  "
        f"@ {resolution:g} Å  seed={seed}\n"
        f"{'=' * 72}",
        flush=True,
    )

    tgt_path = out_dir / "target.npz"
    tgt_json = out_dir / "target.json"
    if tgt_path.is_file():
        z = np.load(tgt_path)
        X_target = np.asarray(z["X"], dtype=np.float64)
        tgt_meta = json.loads(tgt_json.read_text()) if tgt_json.is_file() else {}
        print(
            f"  [target] reuse  rmsd_vs_ideal="
            f"{tgt_meta.get('rmsd_vs_ideal_A', float('nan')):.3f} Å",
            flush=True,
        )
    else:
        print(f"  [target] RDKit plausible conf (seed={seed}) …", flush=True)
        X_target, tgt_meta = make_target_conf(
            entry, topo, seed=seed, n_cand=n_cand, mmff=mmff,
        )
        np.savez_compressed(
            tgt_path,
            X=X_target,
            names=np.array(topo["names"]),
            Z=topo["Z"],
            seed=np.array(seed),
        )
        tgt_json.write_text(json.dumps(tgt_meta, indent=2) + "\n")
        print(
            f"  [target] wrote {tgt_path.relative_to(ROOT)}  "
            f"vs ideal {tgt_meta['rmsd_vs_ideal_A']:.3f} Å  "
            f"vs topo {tgt_meta['rmsd_vs_topo_A']:.3f} Å",
            flush=True,
        )

    scene = scene_for_target(topo, X_target, resolution)
    print(
        f"  scene N={scene['n_atoms']}  σ={scene['sigma']:.3f}  "
        f"half=±{scene['half']:.1f} Å",
        flush=True,
    )

    summary: dict = {
        "slug": slug,
        "label": topo.get("label", slug),
        "tag": tag,
        "resolution": float(resolution),
        "seed": int(seed),
        "target": tgt_meta,
        "target_npz": str(tgt_path.relative_to(ROOT)),
    }

    if not skip_free:
        free = run_free_branch(scene, seed=seed, atom_factor=atom_factor)
        free_npz = out_dir / (
            f"free_x{atom_factor:g}_{resolution:g}A_seed{seed}.npz"
        )
        np.savez_compressed(
            free_npz,
            seed=np.array(seed),
            atom_factor=np.array(atom_factor),
            resolution=np.array(resolution),
            X_true=scene["X_true"],
            free_nn=np.array(free["free_nn"]),
            named_rmsd=np.array(free["named_rmsd"]),
            n_match=np.array(free["n_match"]),
            admm_rmsd=np.array(free["admm_rmsd"]),
            ot_then_l1_rmsd=np.array(free["ot_then_l1_rmsd"]),
            l1_alone_rmsd=np.array(free["l1_alone_rmsd"]),
            named=free["named"],
            admm=free["admm"],
            ot_then_l1=free["ot_then_l1"],
            l1_alone=free["l1_alone"],
        )
        free_sum = {
            k: free[k] for k in free
            if k not in ("named", "admm", "ot_then_l1", "l1_alone")
        }
        free_sum["npz"] = str(free_npz.relative_to(ROOT))
        free_npz.with_suffix(".json").write_text(
            json.dumps(free_sum, indent=2) + "\n"
        )
        summary["free"] = free_sum

    if not skip_torsion:
        # Independent seed for the library so the target conf is not reused.
        lib_seed = int(seed) + 10_000
        tor = run_torsion_branch(
            scene, topo, entry,
            seed=lib_seed,
            n_conf=n_conf,
            top_pca=top_pca,
            top=top,
            torsion_bin=torsion_bin,
            target_rmsd_cut=target_rmsd_cut,
            mmff=mmff,
            out_dir=out_dir,
        )
        summary["torsion"] = tor

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligands", type=str, default=None)
    ap.add_argument("--resolution", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42,
                    help="Target + free-OT seed (library uses seed+10000).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Output folder tag (default rdkit_tgt_s<seed>).")
    ap.add_argument("--atom-factor", type=float, default=2.0)
    ap.add_argument("--n-conf", type=int, default=1000)
    ap.add_argument("--top-pca", type=int, default=50)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--torsion-bin", type=float, default=15.0)
    ap.add_argument("--target-rmsd-cut", type=float, default=0.75,
                    help="Drop library confs this close (Å) to the target.")
    ap.add_argument("--n-cand", type=int, default=32,
                    help="RDKit candidates to pick the target from.")
    ap.add_argument("--skip-free", action="store_true")
    ap.add_argument("--skip-torsion", action="store_true")
    ap.add_argument("--no-mmff", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"rdkit_tgt_s{args.seed}"
    entries = list_ligands()
    if args.ligands:
        want = {s.strip() for s in args.ligands.split(",") if s.strip()}
        entries = [e for e in entries if e["slug"] in want]
        missing = want - {e["slug"] for e in entries}
        if missing:
            raise SystemExit(f"unknown slug(s): {sorted(missing)}")

    print(
        f"RDKit-target protocol: {len(entries)} ligands  tag={tag}  "
        f"res={args.resolution:g}  seed={args.seed}  "
        f"atom_factor={args.atom_factor:g}  n_conf={args.n_conf}",
        flush=True,
    )
    t0 = time.perf_counter()
    rows = []
    for e in entries:
        rows.append(
            run_ligand(
                e,
                tag=tag,
                resolution=args.resolution,
                seed=args.seed,
                atom_factor=args.atom_factor,
                n_conf=args.n_conf,
                top_pca=args.top_pca,
                top=args.top,
                torsion_bin=args.torsion_bin,
                target_rmsd_cut=args.target_rmsd_cut,
                n_cand=args.n_cand,
                skip_free=args.skip_free,
                skip_torsion=args.skip_torsion,
                mmff=not args.no_mmff,
            )
        )

    agg = ROOT / "out" / f"{tag}_summary.json"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"\n{'=' * 72}", flush=True)
    print(
        f"finished {len(rows)} ligands in {time.perf_counter() - t0:.1f}s  "
        f"tag={tag}",
        flush=True,
    )
    print(
        f"{'slug':14} {'tgt':>5} | {'free named':>10} {'OT→L1':>7} {'L1':>7} | "
        f"{'tor R0':>7} {'ADMM':>7} {'OT→L1':>7} {'L1':>7}",
        flush=True,
    )
    for r in rows:
        tgt = r.get("target", {}).get("rmsd_vs_ideal_A", float("nan"))
        f = r.get("free", {})
        t = r.get("torsion", {})
        print(
            f"{r['slug']:14} {tgt:5.2f} | "
            f"{f.get('named_rmsd', float('nan')):10.3f} "
            f"{f.get('ot_then_l1_rmsd', float('nan')):7.3f} "
            f"{f.get('l1_alone_rmsd', float('nan')):7.3f} | "
            f"{t.get('start_rmsd_best', float('nan')):7.3f} "
            f"{t.get('admm_rmsd_best', float('nan')):7.3f} "
            f"{t.get('ot_then_l1_rmsd_best', float('nan')):7.3f} "
            f"{t.get('l1_alone_rmsd_best', float('nan')):7.3f}",
            flush=True,
        )
    print(f"wrote {agg}", flush=True)


if __name__ == "__main__":
    main()
