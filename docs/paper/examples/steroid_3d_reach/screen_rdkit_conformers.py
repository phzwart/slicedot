#!/usr/bin/env python3
"""RDKit conformer screen at fixed atom names (no remapping).

Protocol
--------
1. Generate ``--n-conf`` torsion-randomized conformers (default 1000).
2. Deduplicate by rotatable-bond torsion fingerprint.
3. COM+PCA place survivors; rank by sliced-OT; keep ``--top-pca`` (50).
4. Re-score those by L1 density fit; keep ``--top`` (10).
5. **Save placement** immediately (``place_*.npz``) — starts kept for reuse.
6. From each start, run both (same schedules as free-atom → name → cleanup):
   - ADMM **OT + L1 + geom** via ``run_cleanup`` (loose slack→anneal, not the
     separate ``named_atoms`` branch)
   - **L1 + geom** polish via ``run_l1_geom_polish`` (no OT)
7. Store starts + both refined finals.

Default: all ligands @ **3 Å** only.

Example
-------
  uv run python screen_rdkit_conformers.py
  uv run python screen_rdkit_conformers.py --ligands prednisone --n-conf 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
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
from slicedot import Geometry  # noqa: E402

torch.set_default_dtype(torch.float64)

DEFAULT_RESOLUTIONS = (3.0,)
RCSB_IDEAL_SDF = "https://files.rcsb.org/ligands/download/{code}_ideal.sdf"


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise SystemExit(
            "RDKit is required. Install with: uv sync --extra paper"
        ) from exc
    return Chem, AllChem


def ensure_sdf(entry: dict, lig_dir: Path) -> Path:
    local = lig_dir / "source.sdf"
    if local.is_file() and local.stat().st_size > 0:
        return local
    ccd = entry.get("ccd")
    if not ccd:
        raise FileNotFoundError(
            f"{entry['slug']}: no source.sdf and no CCD id to download"
        )
    url = RCSB_IDEAL_SDF.format(code=ccd)
    print(f"  download {url}", flush=True)
    dest = lig_dir / f"{ccd}_ideal.sdf"
    if not dest.is_file():
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def heavy_atom_indices(mol) -> np.ndarray:
    return np.asarray(
        [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1],
        dtype=np.int64,
    )


def conf_heavy_coords(mol, conf_id: int, heavy_idx: np.ndarray) -> np.ndarray:
    conf = mol.GetConformer(int(conf_id))
    return np.array(
        [list(conf.GetAtomPosition(int(i))) for i in heavy_idx],
        dtype=np.float64,
    )


def _bond_length_rms(X: np.ndarray, X_ref: np.ndarray, bonds) -> float:
    if not bonds:
        return 0.0
    err = [
        abs(float(np.linalg.norm(X[a] - X[b]) - np.linalg.norm(X_ref[a] - X_ref[b])))
        for a, b in bonds
    ]
    return float(np.sqrt(np.mean(np.square(err))))


def _rdkit_rotatable_quartets(mol) -> list[tuple[int, int, int, int]]:
    """Dihedral quartets on non-ring single bonds between heavy atoms."""
    Chem, _ = _require_rdkit()
    quarts = []
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        if a.GetAtomicNum() == 1 or c.GetAtomicNum() == 1:
            continue
        if a.GetDegree() < 2 or c.GetDegree() < 2:
            continue
        ni = [x.GetIdx() for x in a.GetNeighbors() if x.GetIdx() != c.GetIdx()]
        nl = [x.GetIdx() for x in c.GetNeighbors() if x.GetIdx() != a.GetIdx()]
        if not ni or not nl:
            continue
        quarts.append((ni[0], a.GetIdx(), c.GetIdx(), nl[0]))
    return quarts


def generate_conformers(
    sdf_path: Path,
    topo: dict,
    *,
    n_conf: int,
    seed: int,
    mmff: bool,
) -> tuple[np.ndarray, dict]:
    """Heavy-atom coords in fixed topology atom order (identity; no remap).

    CCD ideal SDFs often make ``EmbedMultipleConfs`` collapse to one pose.
    We keep the ideal 3-D seed and diversify by randomizing rotatable
    dihedrals (atom names/order unchanged).
    """
    Chem, AllChem = _require_rdkit()
    from rdkit.Chem import rdMolTransforms

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise RuntimeError(f"failed to read molecule from {sdf_path}")
    # Preserve ideal heavy-atom coords; place H geometrically.
    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"{sdf_path}: no 3-D coordinates after AddHs")
    heavy = heavy_atom_indices(mol)
    elements_probe = np.asarray(
        [mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in heavy],
        dtype=np.int64,
    )
    n = int(topo["n"])
    if len(heavy) != n:
        raise RuntimeError(f"{sdf_path}: heavy atoms {len(heavy)} != topology {n}")
    if not np.array_equal(elements_probe, np.asarray(topo["elements"], dtype=np.int64)):
        raise RuntimeError(
            f"{sdf_path}: heavy-atom element order ≠ topology — "
            "use the same CCD/PubChem SDF that built the ligand ref"
        )

    X0 = conf_heavy_coords(mol, mol.GetConformer().GetId(), heavy)
    bond_rms = _bond_length_rms(X0, topo["X_ref"], topo["bonds"])
    if bond_rms > 0.5:
        raise RuntimeError(
            f"{sdf_path}: identity atom-order bond RMS {bond_rms:.3f} Å vs ref"
        )

    quarts = _rdkit_rotatable_quartets(mol)
    rng = np.random.default_rng(int(seed))
    seed_cid = int(mol.GetConformer().GetId())

    t0 = time.perf_counter()
    # conf 0 = ideal; remaining = torsion-randomized copies
    cids = [seed_cid]
    for _k in range(int(n_conf) - 1):
        new_id = mol.AddConformer(Chem.Conformer(mol.GetConformer(seed_cid)), assignId=True)
        conf = mol.GetConformer(int(new_id))
        for q in quarts:
            rdMolTransforms.SetDihedralDeg(
                conf, int(q[0]), int(q[1]), int(q[2]), int(q[3]),
                float(rng.uniform(-180.0, 180.0)),
            )
        cids.append(int(new_id))
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    if mmff and quarts:
        try:
            # Short minimize per conf — keeps sidechain diversity.
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=50)
        except Exception as exc:  # noqa: BLE001
            print(f"  MMFF optimize skipped: {exc}", flush=True)
    t_mmff = time.perf_counter() - t1

    confs = np.empty((len(cids), n, 3), dtype=np.float64)
    for k, cid in enumerate(cids):
        X = conf_heavy_coords(mol, cid, heavy)
        confs[k] = X - X.mean(0)

    spread = 0.0
    if len(confs) > 1:
        idxs = np.linspace(0, len(confs) - 1, num=min(32, len(confs)), dtype=int)
        dsum, npr = 0.0, 0
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                dsum += float(
                    np.sqrt(((confs[idxs[i]] - confs[idxs[j]]) ** 2).sum(-1).mean())
                )
                npr += 1
        spread = dsum / max(npr, 1)

    print(
        f"  fixed atom order (no remap)  bond RMS vs ref = {bond_rms:.4f} Å  "
        f"n_chiral={len(topo['chiral_centres'])}  "
        f"torsion-sample {len(cids)} confs on {len(quarts)} rotbonds  "
        f"mean pairwise RMSD≈{spread:.3f} Å",
        flush=True,
    )
    if len(confs) > 1 and spread < 0.05:
        raise RuntimeError(
            f"{sdf_path}: conformer ensemble not diverse "
            f"(mean pairwise RMSD {spread:.4f} Å, n_rot={len(quarts)})"
        )

    meta = {
        "sdf": str(sdf_path),
        "n_embedded": int(len(cids)),
        "n_conf": int(confs.shape[0]),
        "n_atoms": int(confs.shape[1]),
        "atom_order": "identity",
        "sampler": "ideal_seed_torsion_random",
        "n_rotatable_sampled": int(len(quarts)),
        "bond_rms_vs_ref_A": bond_rms,
        "pairwise_rmsd_A": spread,
        "n_chiral": int(len(topo["chiral_centres"])),
        "t_embed_s": float(t_embed),
        "t_mmff_s": float(t_mmff),
        "mmff": bool(mmff),
        "seed": int(seed),
    }
    return confs, meta


# --- torsions -----------------------------------------------------------------

def _adjacency(n: int, bonds) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for a, b in bonds:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def torsion_quartets(n: int, bonds, rotatable_bonds) -> list[tuple[int, int, int, int]]:
    """One quartet (i-j-k-l) per rotatable bond j–k."""
    adj = _adjacency(n, bonds)
    quarts = []
    for j, k in rotatable_bonds:
        j, k = int(j), int(k)
        ni = [a for a in adj[j] if a != k]
        nl = [a for a in adj[k] if a != j]
        if not ni or not nl:
            continue
        quarts.append((min(ni), j, k, min(nl)))
    return quarts


def dihedral_deg(X: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    b1 = X[j] - X[i]
    b2 = X[k] - X[j]
    b3 = X[l] - X[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1n = np.linalg.norm(n1)
    n2n = np.linalg.norm(n2)
    b2n = np.linalg.norm(b2)
    if n1n < 1e-12 or n2n < 1e-12 or b2n < 1e-12:
        return 0.0
    n1 /= n1n
    n2 /= n2n
    m1 = np.cross(n1, b2 / b2n)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def torsion_fingerprint(X: np.ndarray, quarts) -> np.ndarray:
    if not quarts:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([dihedral_deg(X, *q) for q in quarts], dtype=np.float64)


def dedupe_by_torsion(
    confs: np.ndarray,
    quarts,
    *,
    bin_deg: float = 15.0,
) -> np.ndarray:
    """Greedy keep-first indices with distinct binned torsion fingerprints."""
    if not quarts:
        return np.arange(len(confs), dtype=np.int64)
    kept = []
    seen: set[tuple] = set()
    bins = float(bin_deg)
    for k, X in enumerate(confs):
        fp = torsion_fingerprint(X, quarts)
        key = tuple(int(np.floor((ang % 360.0) / bins)) for ang in fp)
        if key in seen:
            continue
        seen.add(key)
        kept.append(k)
    return np.asarray(kept, dtype=np.int64)


# --- PCA / scores -------------------------------------------------------------

def pca_frame(points: np.ndarray, weights: np.ndarray):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    P = np.asarray(points, dtype=np.float64)
    com = (w[:, None] * P).sum(axis=0)
    C = P - com
    cov = (w[:, None] * C).T @ C
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return com, axes


def map_pca(rho: np.ndarray, origin: np.ndarray, spacing):
    sp = np.atleast_1d(spacing).astype(np.float64) * np.ones(3)
    org = np.asarray(origin, dtype=np.float64).ravel()
    nx, ny, nz = rho.shape
    ax = [org[i] + np.arange(n) * sp[i] for i, n in enumerate((nx, ny, nz))]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    w = rho.ravel().astype(np.float64)
    w = w / w.sum()
    return pca_frame(G, w)


def pca_placements_3d(X, w, map_com, map_axes) -> list[np.ndarray]:
    com, axes = pca_frame(X, w)
    placements, seen = [], set()
    for s0 in (1.0, -1.0):
        for s1 in (1.0, -1.0):
            for s2 in (1.0, -1.0):
                A = axes.copy()
                A[:, 0] *= s0
                A[:, 1] *= s1
                A[:, 2] *= s2
                if np.linalg.det(A) < 0:
                    A[:, -1] *= -1
                key = tuple(np.round(A.ravel(), 8))
                if key in seen:
                    continue
                seen.add(key)
                placements.append((X - com) @ A @ map_axes.T + map_com)
    return placements


def score_ot(ot, X, w, sigma) -> float:
    return float(vg_ot(ot, X, w, sigma)[0])


def score_l1(l1, X, w) -> float:
    return float(l1.value_grad(X, w)[0])


def rmsd(a, b) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def screen_and_refine(
    slug: str,
    confs: np.ndarray,
    *,
    resolutions: tuple[float, ...],
    top_pca: int,
    top: int,
    torsion_bin: float,
    seed: int,
    project_slack: float,
    skip_existing: bool,
) -> list[dict]:
    topo = load_ligand(slug)
    out_dir = ROOT / "ligands" / slug / "out" / "rdkit_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    w = topo["W"]
    quarts = torsion_quartets(topo["n"], topo["bonds"], topo["rotatable_bonds"])
    summaries = []

    # --- torsion dedupe (shared across resolutions) ---
    t_dedup0 = time.perf_counter()
    uniq_idx = dedupe_by_torsion(confs, quarts, bin_deg=torsion_bin)
    confs_u = confs[uniq_idx]
    t_dedup = time.perf_counter() - t_dedup0
    print(
        f"  torsion dedupe: {len(confs)} → {len(confs_u)} unique  "
        f"(bin={torsion_bin:g}°, n_tors={len(quarts)})  {t_dedup:.2f}s",
        flush=True,
    )

    for res in resolutions:
        tag = f"{float(res):g}".replace(".", "p")
        place_stem = f"place_{tag}A_n{len(confs)}_pca{top_pca}_l1top{top}"
        stem = f"screen_{tag}A_n{len(confs)}_pca{top_pca}_l1top{top}"
        place_npz = out_dir / f"{place_stem}.npz"
        place_json = out_dir / f"{place_stem}_summary.json"
        out_npz = out_dir / f"{stem}.npz"
        out_json = out_dir / f"{stem}_summary.json"
        if skip_existing and out_npz.is_file() and out_json.is_file():
            print(f"[skip] {out_npz.relative_to(ROOT)}", flush=True)
            summaries.append(json.loads(out_json.read_text()))
            continue

        print(
            f"\n=== {slug} @ {res:g} Å  raw={len(confs)}  unique={len(confs_u)}  "
            f"pca_top={top_pca}  l1_top={top} ===",
            flush=True,
        )
        t_res0 = time.perf_counter()
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
        print(
            f"  geometry: {len(scene['geom'].chiral_centres)} chiral centres",
            flush=True,
        )
        ot, l1 = scene["ot"], scene["l1"]
        sig = scene["sigma"]
        X_true = scene["X_true"]
        _, map_axes = map_pca(scene["T"], scene["origin"], scene["spacing"])
        map_com = X_true.mean(0)

        # --- PCA place + OT rank → top_pca ---
        t_pca0 = time.perf_counter()
        pca_E = np.empty(len(confs_u), dtype=np.float64)
        pca_rmsd = np.empty(len(confs_u), dtype=np.float64)
        placed = np.empty_like(confs_u)
        place_flip = np.empty(len(confs_u), dtype=np.int32)
        for k, Xc in enumerate(confs_u):
            best_E, best_X, best_f = np.inf, None, -1
            for fi, Xp in enumerate(pca_placements_3d(Xc, w, map_com, map_axes)):
                E = score_ot(ot, Xp, w, sig)
                if E < best_E:
                    best_E, best_X, best_f = E, Xp, fi
            assert best_X is not None
            pca_E[k] = best_E
            placed[k] = best_X
            place_flip[k] = best_f
            pca_rmsd[k] = rmsd(best_X, X_true)
            if (k + 1) % 500 == 0 or k + 1 == len(confs_u):
                print(
                    f"  PCA+OT {k+1}/{len(confs_u)}  "
                    f"best={np.min(pca_E[:k+1]):.5g}",
                    flush=True,
                )
        t_pca = time.perf_counter() - t_pca0
        order_pca = np.argsort(pca_E)
        n_pca = min(int(top_pca), len(order_pca))
        pca_top_idx = order_pca[:n_pca]  # indices into confs_u / placed
        print(
            f"  PCA+OT done {t_pca:.1f}s  keep top-{n_pca}  "
            f"OT∈[{pca_E[pca_top_idx[0]]:.5g}, {pca_E[pca_top_idx[-1]]:.5g}]",
            flush=True,
        )

        # --- L1 score on PCA top → top ---
        t_l10 = time.perf_counter()
        l1_E = np.array(
            [score_l1(l1, placed[k], w) for k in pca_top_idx],
            dtype=np.float64,
        )
        order_l1 = np.argsort(l1_E)
        n_l1 = min(int(top), len(order_l1))
        l1_sel = order_l1[:n_l1]          # indices into pca_top_idx
        refine_src = pca_top_idx[l1_sel]  # indices into confs_u / placed
        t_l1 = time.perf_counter() - t_l10
        print(
            f"  L1 rank done {t_l1:.1f}s  keep top-{n_l1}  "
            f"L1∈[{l1_E[l1_sel[0]]:.5g}, {l1_E[l1_sel[-1]]:.5g}]",
            flush=True,
        )

        # --- Materialise top-L1 starts and save placement immediately ---
        geom = scene["geom"]
        start_poses = np.full((n_l1, *confs_u.shape[1:]), np.nan)
        start_pca_E = np.full(n_l1, np.nan)
        start_l1_E = np.full(n_l1, np.nan)
        start_rmsd = np.full(n_l1, np.nan)
        start_uniq = np.zeros(n_l1, dtype=np.int64)
        for j, k in enumerate(refine_src):
            X0 = placed[k].copy()
            if project_slack is not None and float(project_slack) >= 0.0:
                Xp, _, _ = geom.project(
                    X0, tol=1e-3, max_iter=80, slack=float(project_slack),
                )
            else:
                Xp = X0
            start_poses[j] = Xp
            start_pca_E[j] = float(pca_E[k])
            start_l1_E[j] = float(l1_E[l1_sel[j]])
            start_rmsd[j] = rmsd(Xp, X_true)
            start_uniq[j] = int(uniq_idx[k])

        place_summary = {
            "slug": slug,
            "label": topo.get("label", slug),
            "resolution": float(res),
            "n_conf_raw": int(len(confs)),
            "n_conf_unique": int(len(confs_u)),
            "n_torsion": int(len(quarts)),
            "torsion_bin_deg": float(torsion_bin),
            "top_pca": int(n_pca),
            "top_l1": int(n_l1),
            "t_dedup_s": float(t_dedup),
            "t_pca_s": float(t_pca),
            "t_l1_s": float(t_l1),
            "pca_E_best": float(pca_E[pca_top_idx[0]]),
            "l1_E_best_pre": float(l1_E[l1_sel[0]]),
            "start_rmsd_best": float(np.nanmin(start_rmsd)),
            "start_rmsd_mean": float(np.nanmean(start_rmsd)),
            "npz": str(place_npz.relative_to(ROOT)),
            "cleanup_slack": [float(CLEANUP_SLACK0), float(CLEANUP_SLACK1)],
            "cleanup_schedule": "free_atom_named",
        }
        np.savez_compressed(
            place_npz,
            resolution=np.array(res),
            sigma=np.array(sig),
            uniq_idx=uniq_idx,
            confs_unique=confs_u,
            torsion_quarts=np.asarray(quarts, dtype=np.int64).reshape(-1, 4),
            torsion_bin_deg=np.array(torsion_bin),
            pca_E=pca_E,
            pca_rmsd=pca_rmsd,
            placed=placed,
            place_flip=place_flip,
            pca_top_idx=pca_top_idx,
            pca_top_placed=placed[pca_top_idx],
            l1_E_pca_top=l1_E,
            l1_sel=l1_sel,
            refine_src=refine_src,
            start_poses=start_poses,
            start_pca_E=start_pca_E,
            start_l1_E=start_l1_E,
            start_rmsd=start_rmsd,
            start_uniq=start_uniq,
            X_true=X_true,
            t_dedup_s=np.array(t_dedup),
            t_pca_s=np.array(t_pca),
            t_l1_s=np.array(t_l1),
        )
        place_json.write_text(json.dumps(place_summary, indent=2) + "\n")
        print(
            f"  saved placement {place_npz.name}  "
            f"start RMSD best={place_summary['start_rmsd_best']:.3f} Å  "
            f"mean={place_summary['start_rmsd_mean']:.3f} Å",
            flush=True,
        )

        # --- From each start: OT+L1+geom ADMM  AND  L1+geom polish (no OT) ---
        refine_E = np.full(n_l1, np.nan)
        refine_L1 = np.full(n_l1, np.nan)
        refine_rmsd = np.full(n_l1, np.nan)
        refine_steps = np.zeros(n_l1, dtype=np.int32)
        refine_stop = np.array([""] * n_l1, dtype=object)
        refine_time = np.full(n_l1, np.nan)
        refine_poses = np.full((n_l1, *confs_u.shape[1:]), np.nan)

        polish_L1 = np.full(n_l1, np.nan)
        polish_rmsd = np.full(n_l1, np.nan)
        polish_stop = np.array([""] * n_l1, dtype=object)
        polish_time = np.full(n_l1, np.nan)
        polish_poses = np.full((n_l1, *confs_u.shape[1:]), np.nan)
        polish_ot = np.full(n_l1, np.nan)  # OT of L1-only result (diagnostic)

        for j, k in enumerate(refine_src):
            Xp = start_poses[j].copy()
            seed_j = int(seed) + int(uniq_idx[k])

            t1 = time.perf_counter()
            # Same ADMM schedule as free-atom → Namer → cleanup (named_atoms=False).
            cleanup = run_cleanup(
                scene, Xp,
                seed=seed_j,
                log_every=10**9,
                named_atoms=False,
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
            # Same L1+geom polish as the free-atom pipeline terminal stage.
            polish = run_l1_geom_polish(scene, Xp, seed=seed_j)
            Xp_l1 = np.asarray(polish["poses"][-1], dtype=np.float64)
            polish_poses[j] = Xp_l1
            polish_L1[j] = float(scene["l1"].value_grad(Xp_l1, w)[0])
            polish_ot[j] = float(vg_ot(scene["ot"], Xp_l1, w, sig)[0])
            polish_rmsd[j] = rmsd(Xp_l1, X_true)
            polish_stop[j] = str(polish.get("stop_reason", "l1_geom"))
            polish_time[j] = time.perf_counter() - t2

            print(
                f"  refine {j+1}/{n_l1}  uniq#{int(uniq_idx[k])}  "
                f"start={start_rmsd[j]:.3f} Å  "
                f"OT+ADMM→{refine_rmsd[j]:.3f} Å (OT={refine_E[j]:.5g}, "
                f"{refine_time[j]:.1f}s, {refine_stop[j]})  "
                f"L1+geom→{polish_rmsd[j]:.3f} Å (OT={polish_ot[j]:.5g}, "
                f"{polish_time[j]:.1f}s)",
                flush=True,
            )

        t_res = time.perf_counter() - t_res0
        j_best = int(np.nanargmin(refine_E))
        j_best_l1 = int(np.nanargmin(polish_L1))
        summary = {
            "slug": slug,
            "label": topo.get("label", slug),
            "resolution": float(res),
            "n_conf_raw": int(len(confs)),
            "n_conf_unique": int(len(confs_u)),
            "n_torsion": int(len(quarts)),
            "torsion_bin_deg": float(torsion_bin),
            "top_pca": int(n_pca),
            "top_l1": int(n_l1),
            "t_dedup_s": float(t_dedup),
            "t_pca_s": float(t_pca),
            "t_l1_s": float(t_l1),
            "t_refine_total_s": float(np.nansum(refine_time)),
            "t_refine_mean_s": float(np.nanmean(refine_time)),
            "t_polish_total_s": float(np.nansum(polish_time)),
            "t_polish_mean_s": float(np.nanmean(polish_time)),
            "t_total_s": float(t_res),
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
            "best_uniq_index": int(uniq_idx[refine_src[j_best]]),
            "place_npz": str(place_npz.relative_to(ROOT)),
            "npz": str(out_npz.relative_to(ROOT)),
            "cleanup_slack": [float(CLEANUP_SLACK0), float(CLEANUP_SLACK1)],
            "cleanup_schedule": "free_atom_named",
        }
        np.savez_compressed(
            out_npz,
            resolution=np.array(res),
            sigma=np.array(sig),
            uniq_idx=uniq_idx,
            confs_unique=confs_u,
            torsion_quarts=np.asarray(quarts, dtype=np.int64).reshape(-1, 4),
            torsion_bin_deg=np.array(torsion_bin),
            pca_E=pca_E,
            pca_rmsd=pca_rmsd,
            placed=placed,
            place_flip=place_flip,
            pca_top_idx=pca_top_idx,
            pca_top_placed=placed[pca_top_idx],
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
            t_dedup_s=np.array(t_dedup),
            t_pca_s=np.array(t_pca),
            t_l1_s=np.array(t_l1),
            t_total_s=np.array(t_res),
        )
        out_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"  done {slug}@{res:g}Å in {t_res:.1f}s  "
            f"start={summary['start_rmsd_best']:.3f} Å  "
            f"OT+ADMM={summary['refine_rmsd_best']:.3f} Å  "
            f"L1+geom={summary['polish_rmsd_best']:.3f} Å  "
            f"→ {out_npz.name}",
            flush=True,
        )
        summaries.append(summary)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligands", type=str, default=None)
    ap.add_argument("--resolutions", type=str, default="3.0",
                    help="Comma-separated Å (default: 3.0 only).")
    ap.add_argument("--n-conf", type=int, default=1000,
                    help="Conformers to generate (default 1000).")
    ap.add_argument("--top-pca", type=int, default=50,
                    help="Keep after PCA+OT ranking (default 50).")
    ap.add_argument("--top", type=int, default=10,
                    help="Keep after L1 ranking / refine (default 10).")
    ap.add_argument("--torsion-bin", type=float, default=15.0,
                    help="Torsion fingerprint bin width in degrees.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--project-slack", type=float, default=-1.0,
        help="Optional pre-ADMM geom clean slack; <0 skips (default).",
    )
    ap.add_argument("--no-mmff", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--regen-confs", action="store_true")
    args = ap.parse_args()

    _require_rdkit()
    entries = list_ligands()
    if args.ligands:
        want = {s.strip() for s in args.ligands.split(",") if s.strip()}
        entries = [e for e in entries if e["slug"] in want]
        missing = want - {e["slug"] for e in entries}
        if missing:
            raise SystemExit(f"unknown slug(s): {sorted(missing)}")

    resolutions = tuple(
        float(x) for x in args.resolutions.split(",") if x.strip()
    ) or DEFAULT_RESOLUTIONS

    print(
        f"RDKit screen: {len(entries)} ligands × {len(resolutions)} res  "
        f"n_conf={args.n_conf}  torsion_bin={args.torsion_bin:g}°  "
        f"pca_top={args.top_pca}  l1_top={args.top}",
        flush=True,
    )
    all_summaries = []
    t_all = time.perf_counter()

    for entry in entries:
        slug = entry["slug"]
        lig_dir = ROOT / "ligands" / slug
        topo = load_ligand(slug)
        conf_path = (
            lig_dir / "out" / "rdkit_screen"
            / f"conformers_n{args.n_conf}_seed{args.seed}.npz"
        )
        conf_path.parent.mkdir(parents=True, exist_ok=True)

        if conf_path.is_file() and not args.regen_confs:
            z = np.load(conf_path, allow_pickle=False)
            confs = np.asarray(z["confs"], dtype=np.float64)
            print(f"\n{slug}: reusing {conf_path.name}  ({len(confs)} confs)", flush=True)
            conf_meta = {
                "t_embed_s": float(z["t_embed_s"]) if "t_embed_s" in z.files else None,
                "t_mmff_s": float(z["t_mmff_s"]) if "t_mmff_s" in z.files else None,
                "bond_rms_vs_ref_A": (
                    float(z["bond_rms_vs_ref_A"]) if "bond_rms_vs_ref_A" in z.files else None
                ),
                "reused": True,
            }
        else:
            print(f"\n{slug}: generating {args.n_conf} ETKDG conformers …", flush=True)
            sdf = ensure_sdf(entry, lig_dir)
            confs, conf_meta = generate_conformers(
                sdf, topo,
                n_conf=args.n_conf,
                seed=args.seed,
                mmff=not args.no_mmff,
            )
            np.savez_compressed(
                conf_path,
                confs=confs,
                names=np.array(topo["names"]),
                t_embed_s=np.array(conf_meta["t_embed_s"]),
                t_mmff_s=np.array(conf_meta["t_mmff_s"]),
                bond_rms_vs_ref_A=np.array(conf_meta["bond_rms_vs_ref_A"]),
                seed=np.array(args.seed),
            )
            conf_path.with_suffix(".json").write_text(
                json.dumps(conf_meta, indent=2) + "\n"
            )
            print(
                f"  wrote {conf_path.relative_to(ROOT)}  "
                f"embed {conf_meta['t_embed_s']:.1f}s  "
                f"mmff {conf_meta['t_mmff_s']:.1f}s",
                flush=True,
            )

        summaries = screen_and_refine(
            slug, confs,
            resolutions=resolutions,
            top_pca=args.top_pca,
            top=args.top,
            torsion_bin=args.torsion_bin,
            seed=args.seed,
            project_slack=args.project_slack,
            skip_existing=args.skip_existing,
        )
        for s in summaries:
            s["conformer_meta"] = {
                k: conf_meta.get(k) for k in (
                    "t_embed_s", "t_mmff_s", "bond_rms_vs_ref_A",
                    "n_embedded", "n_chiral", "atom_order", "reused",
                )
            }
        all_summaries.extend(summaries)

    agg = ROOT / "out" / "rdkit_screen_summary.json"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(all_summaries, indent=2) + "\n")

    print(f"\n{'=' * 72}", flush=True)
    print(
        f"finished {len(all_summaries)} ligand×resolution jobs "
        f"in {time.perf_counter() - t_all:.1f}s",
        flush=True,
    )
    print(
        f"{'slug':28s} {'res':>5s} {'uniq':>6s} {'scr_s':>7s} {'ot_s':>7s} "
        f"{'l1_s':>7s} {'R0':>7s} {'R_OT':>7s} {'R_L1':>7s}",
        flush=True,
    )
    for s in all_summaries:
        print(
            f"{s['slug']:28s} {s['resolution']:5.1f} "
            f"{s['n_conf_unique']:6d} "
            f"{s['t_pca_s'] + s['t_l1_s']:7.1f} "
            f"{s['t_refine_total_s']:7.1f} "
            f"{s.get('t_polish_total_s', 0.0):7.1f} "
            f"{s['start_rmsd_best']:7.3f} "
            f"{s['refine_rmsd_best']:7.3f} "
            f"{s.get('polish_rmsd_best', float('nan')):7.3f}",
            flush=True,
        )
    print(f"wrote {agg}", flush=True)


if __name__ == "__main__":
    main()
