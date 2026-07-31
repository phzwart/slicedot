#!/usr/bin/env python3
"""Conformer screen: diverse sample → dedupe → PCA-align → score → refine top-K.

Pipeline
--------
  1. Build a density scene (true pose = target fold).
  2. Oversample in-plane chain folds from mixed bases / angle scales;
     lightly project; deduplicate by ring-aligned RMSD.
  3. PCA-align each unique model to the map (COM + principal axes; 4 flips).
  4. Score by sliced-W₁ (OT); greedily pick a diverse top-``--top`` by OT.
  5. ADMM-refine those starts; report best post-refine OT energy.

Example
-------
  uv run python screen_conformers.py --resolution 1.75 --chain zigzag
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from phenol import (
    BONDS,
    N_RING,
    ROTATABLE_BONDS,
    build_phenol,
    phenol_geometry,
    project_2d,
)
from targets2d import ConsistentSlicedW1, L1Diff, directions_2d, render
import make_figure as mf
from make_figure import (
    ADMM_OT_LR0,
    ADMM_OT_LR1,
    ADMM_RHO,
    GEOM_TOL,
    L1_LR,
    N_DIRS,
    OUT_DIR,
    build_scene,
    rmsd,
    run_admm,
    value_grad_fn,
)


def _downstream(n: int, a: int, b: int) -> list[int]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in BONDS:
        adj[i].append(j)
        adj[j].append(i)
    seen = {b}
    stack = [b]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v == a or v in seen:
                continue
            seen.add(v)
            stack.append(v)
    return sorted(seen)


def kabsch_rmsd(A: np.ndarray, B: np.ndarray, idxs: np.ndarray | None = None) -> float:
    """RMSD after Kabsch aligning ``A`` onto ``B`` (optionally on ``idxs`` only)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if idxs is None:
        idxs = np.arange(len(A))
    Aa = A[idxs] - A[idxs].mean(0)
    Bb = B[idxs] - B[idxs].mean(0)
    H = Aa.T @ Bb
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    A_al = (A - A[idxs].mean(0)) @ R + B[idxs].mean(0)
    return float(np.sqrt(((A_al - B) ** 2).sum(-1).mean()))


def ring_aligned_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """Shape RMSD after aligning on the rigid ring+OH (atoms 0..N_RING-1)."""
    return kabsch_rmsd(A, B, idxs=np.arange(N_RING))


def dedupe_greedy(
    confs: np.ndarray,
    thresh: float,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """Keep indices whose ring-aligned RMSD to all kept is ≥ ``thresh``."""
    n = len(confs)
    if order is None:
        order = np.arange(n)
    kept: list[int] = []
    for i in order:
        Xi = confs[i]
        if all(ring_aligned_rmsd(Xi, confs[j]) >= thresh for j in kept):
            kept.append(int(i))
    return np.asarray(kept, dtype=int)


def select_diverse_topk(
    scores: np.ndarray,
    placed: np.ndarray,
    top: int,
    thresh: float,
) -> np.ndarray:
    """Greedy: walk OT-sorted list, keep poses ≥ ``thresh`` Å apart (raw RMSD)."""
    order = np.argsort(scores)
    kept: list[int] = []
    for i in order:
        Xi = placed[i]
        if all(rmsd(Xi, placed[j]) >= thresh for j in kept):
            kept.append(int(i))
        if len(kept) >= top:
            break
    return np.asarray(kept, dtype=int)


def sample_conformers(
    bases: list[np.ndarray],
    n_conf: int,
    rng: np.random.Generator,
    geom=None,
    project_slack: float = 0.5,
    oversample: float = 3.0,
) -> np.ndarray:
    """Diverse planar chain folds from mixed bases and angle scales.

    Oversamples by ``oversample``, lightly projects (flat-bottom slack), then
    the caller should deduplicate down to ``n_conf``.
    """
    n_gen = max(int(np.ceil(oversample * n_conf)), n_conf)
    n = len(bases[0])
    distal = {
        (a, b): np.asarray(_downstream(n, a, b), dtype=int)
        for a, b in ROTATABLE_BONDS
    }
    # Mixture of kick sizes: local / medium / aggressive
    scales = np.array([0.35 * np.pi, 0.7 * np.pi, np.pi])
    out = np.empty((n_gen, n, 2), dtype=np.float64)
    for k in range(n_gen):
        X = np.asarray(bases[int(rng.integers(0, len(bases)))], dtype=np.float64).copy()
        scale = float(rng.choice(scales))
        # Perturb a random non-empty subset of rotatable bonds
        bonds = list(distal.items())
        rng.shuffle(bonds)
        n_kick = int(rng.integers(1, len(bonds) + 1))
        for (a, b), idxs in bonds[:n_kick]:
            if idxs.size == 0:
                continue
            ang = float(rng.uniform(-scale, scale))
            c = X[a]
            ca, sa = np.cos(ang), np.sin(ang)
            R = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
            X[idxs] = (X[idxs] - c) @ R.T + c
        if geom is not None:
            X, _, _ = project_2d(
                geom, X, tol=GEOM_TOL, max_iter=60, slack=float(project_slack),
            )
        out[k] = X - X.mean(axis=0)
    return out


def pca_frame(points: np.ndarray, weights: np.ndarray):
    """Return (COM, axes) with axes columns = principal directions (largest first)."""
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


def map_pca(rho: np.ndarray, V: np.ndarray):
    """PCA of the density as a weighted point cloud on voxel centres."""
    w = rho.ravel().astype(np.float64)
    w = w / w.sum()
    return pca_frame(V, w)


def pca_placements(
    X: np.ndarray, w: np.ndarray, map_com: np.ndarray, map_axes: np.ndarray,
) -> list[np.ndarray]:
    """Align model PCA frame to map PCA frame; return 4 sign-flip placements."""
    com, axes = pca_frame(X, w)
    placements = []
    for s0 in (1.0, -1.0):
        for s1 in (1.0, -1.0):
            A = axes.copy()
            A[:, 0] *= s0
            A[:, 1] *= s1
            if np.linalg.det(A) < 0:
                A[:, 1] *= -1
            Xp = (X - com) @ A @ map_axes.T + map_com
            placements.append(Xp)
    return placements


def score_ot(ot, X: np.ndarray, w: np.ndarray, sigma: float) -> float:
    E, _ = ot.value_grad(X, w, sigma)
    return float(E)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=float, default=1.75)
    ap.add_argument("--chain", choices=("extended", "zigzag"), default="zigzag")
    ap.add_argument("--n-conf", type=int, default=1000)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dedup", type=float, default=0.5,
        help="Min ring-aligned RMSD (Å) between kept samples / top-K starts.",
    )
    ap.add_argument(
        "--project-slack", type=float, default=0.5,
        help="ReLU slack (Å) when cleaning sampled folds (larger → more diversity).",
    )
    ap.add_argument(
        "--oversample", type=float, default=4.0,
        help="Generate this × n_conf raw samples before dedupe.",
    )
    ap.add_argument(
        "--objective", choices=("ot", "ot+l1"), default="ot+l1",
        help="ADMM blocks for the refine stage (screen always uses OT).",
    )
    args = ap.parse_args()

    mf.RESOLUTION = float(args.resolution)
    rng = np.random.default_rng(int(args.seed))
    t0 = time.perf_counter()

    scene = build_scene(
        misalign_deg=0.0, shift_radii=mf.SHIFT_RADII, chain_style=args.chain,
    )
    X0 = scene["X0"]
    w = scene["w"]
    true_com = scene["true_com"]
    X_true = X0 + true_com
    rhoT = render(X_true, w, scene["sigma"], scene["V"], scene["shape"])
    scene = {**scene, "X_true": X_true, "rhoT": rhoT}

    sig = scene["sigma"]
    V = scene["V"]
    print(
        f"scene @ {args.resolution:g} Å  chain={args.chain}  "
        f"σ={sig:.3f} Å  n_conf={args.n_conf}  top={args.top}  "
        f"dedup≥{args.dedup:g} Å",
        flush=True,
    )

    ot = ConsistentSlicedW1(rhoT, V, directions_2d(N_DIRS), nbins=320, pad=12.0)
    l1 = L1Diff(rhoT, V, sig)
    vg_ot = value_grad_fn("ot", ot, sig)
    vg_l1 = value_grad_fn("l1", l1, sig)

    geom = phenol_geometry(X0)
    X_ext, _ = build_phenol(chain_style="extended")
    X_zig, _ = build_phenol(chain_style="zigzag")
    bases = [X_ext, X_zig]

    print(
        f"sampling ~{int(args.oversample * args.n_conf)} raw conformers "
        f"(bases=extended+zigzag, slack={args.project_slack:g} Å) ...",
        flush=True,
    )
    raw = sample_conformers(
        bases, args.n_conf, rng, geom=geom,
        project_slack=args.project_slack, oversample=args.oversample,
    )
    # Prefer more-perturbed shapes first when deduping: sort by distance to both bases
    diversity_key = np.array([
        min(ring_aligned_rmsd(X, X_ext), ring_aligned_rmsd(X, X_zig))
        for X in raw
    ])
    order_div = np.argsort(-diversity_key)  # most different from bases first
    kept = dedupe_greedy(raw, thresh=args.dedup, order=order_div)
    if len(kept) > args.n_conf:
        kept = kept[: args.n_conf]
    confs = raw[kept]
    print(
        f"  kept {len(confs)}/{len(raw)} unique "
        f"(ring-aligned RMSD ≥ {args.dedup:g} Å)",
        flush=True,
    )
    if len(confs) < args.top:
        raise SystemExit(
            f"only {len(confs)} unique conformers after dedupe; "
            f"lower --dedup or raise --oversample"
        )

    # Pairwise diversity diagnostic on a subsample
    m = min(len(confs), 200)
    pair_rms = []
    for i in range(m):
        for j in range(i + 1, m):
            pair_rms.append(ring_aligned_rmsd(confs[i], confs[j]))
    pair_rms = np.asarray(pair_rms)
    print(
        f"  pairwise ring-RMSD (n={m}): "
        f"min={pair_rms.min():.3f}  median={np.median(pair_rms):.3f}  "
        f"mean={pair_rms.mean():.3f} Å",
        flush=True,
    )

    map_com, map_axes = map_pca(rhoT, V)
    print(
        f"map PCA COM=({map_com[0]:.2f}, {map_com[1]:.2f})  "
        f"true COM=({true_com[0]:.2f}, {true_com[1]:.2f})",
        flush=True,
    )

    print("PCA-align + OT score ...", flush=True)
    n_conf = len(confs)
    scores = np.empty(n_conf, dtype=np.float64)
    placed = np.empty_like(confs)
    for k in range(n_conf):
        best_E = np.inf
        best_X = None
        for Xp in pca_placements(confs[k], w, map_com, map_axes):
            E = score_ot(ot, Xp, w, sig)
            if E < best_E:
                best_E = E
                best_X = Xp
        scores[k] = best_E
        placed[k] = best_X
        if (k + 1) % 200 == 0 or k + 1 == n_conf:
            print(
                f"  scored {k+1}/{n_conf}  "
                f"best-so-far OT={scores[:k+1].min():.5g}",
                flush=True,
            )

    top_idx = select_diverse_topk(
        scores, placed, top=args.top, thresh=args.dedup,
    )
    print(
        f"\nscreen: OT min={scores.min():.6g}  "
        f"median={np.median(scores):.6g}  "
        f"p10={np.quantile(scores, 0.1):.6g}",
        flush=True,
    )
    print(
        f"diverse top-{len(top_idx)} (placed RMSD ≥ {args.dedup:g} Å) "
        f"for ADMM ({args.objective}) ...",
        flush=True,
    )
    if len(top_idx) < args.top:
        print(
            f"  warning: only {len(top_idx)} diverse hits "
            f"(wanted {args.top}); consider lowering --dedup",
            flush=True,
        )

    rows = []
    best_E = np.inf
    best_rec = None
    for rank, k in enumerate(top_idx, start=1):
        X_start = placed[k]
        X_start, _, _ = project_2d(geom, X_start, tol=GEOM_TOL, slack=0.0)
        kwargs = dict(vg_ot=vg_ot)
        if args.objective == "ot+l1":
            kwargs["vg_l1"] = vg_l1
        cache = run_admm(
            f"conf{k}", X_start, w, X_true, geom=geom,
            lr_ot0=ADMM_OT_LR0, lr_ot1=ADMM_OT_LR1, lr_l1=L1_LR,
            rho=ADMM_RHO, **kwargs,
        )
        poses = cache["poses"]
        E_ot = np.asarray(cache["energies"], dtype=np.float64)
        i_E = int(np.argmin(E_ot))
        i_R = int(cache["best_step"])
        rec = {
            "rank": rank,
            "conf_id": int(k),
            "screen_E": float(scores[k]),
            "screen_rmsd": rmsd(placed[k], X_true),
            "shape_rmsd": ring_aligned_rmsd(confs[k], X0),
            "E_best": float(E_ot[i_E]),
            "rmsd_at_E": rmsd(poses[i_E], X_true),
            "rmsd_best": float(cache["rmsds"].min()),
            "E_at_rmsd": float(E_ot[i_R]),
            "n_steps": int(cache["n_steps"]),
            "stop": str(cache["stop_reason"]),
        }
        rows.append(rec)
        print(
            f"  #{rank:2d} conf={k:4d}  screen_E={rec['screen_E']:.5g}  "
            f"init_RMSD={rec['screen_rmsd']:.3f} Å  "
            f"shape_RMSD={rec['shape_rmsd']:.3f} Å  "
            f"→ E*={rec['E_best']:.5g}  RMSD*={rec['rmsd_best']:.3f} Å  "
            f"({rec['n_steps']} steps, {rec['stop']})",
            flush=True,
        )
        if rec["E_best"] < best_E:
            best_E = rec["E_best"]
            best_rec = rec

    elapsed = time.perf_counter() - t0
    E_true, _ = ot.value_grad(X_true, w, sig)
    print("\n========== RESULT ==========")
    print(f"true-pose OT (oracle)     = {E_true:.6g}")
    print(f"best screen OT            = {float(scores.min()):.6g}")
    print(f"best refined OT (diverse top-{len(top_idx)}) = {best_E:.6g}")
    if best_rec is not None:
        print(
            f"  from conf #{best_rec['conf_id']}  "
            f"init_RMSD={best_rec['screen_rmsd']:.3f} Å  "
            f"RMSD@E*={best_rec['rmsd_at_E']:.3f} Å  "
            f"best RMSD={best_rec['rmsd_best']:.3f} Å"
        )
    print(f"elapsed {elapsed:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.chain}_{str(args.resolution).replace('.', 'p')}A"
    out = OUT_DIR / f"screen_conformers_{tag}.npz"
    np.savez_compressed(
        out,
        scores=scores,
        top_idx=top_idx,
        placed=placed[top_idx],
        confs=confs[top_idx],
        resolution=args.resolution,
        chain=args.chain,
        dedup=args.dedup,
        E_true=E_true,
        best_E=best_E,
        rows=np.array(rows, dtype=object),
    )
    print(f"wrote {out}")

    summary = OUT_DIR / f"screen_conformers_{tag}.txt"
    with summary.open("w") as f:
        f.write(f"resolution={args.resolution} chain={args.chain}\n")
        f.write(
            f"n_conf={n_conf} top={args.top} seed={args.seed} "
            f"dedup={args.dedup}\n"
        )
        f.write(f"E_true={E_true:.8g}\n")
        f.write(f"best_screen_E={float(scores.min()):.8g}\n")
        f.write(f"best_refined_E={best_E:.8g}\n")
        for rec in rows:
            f.write(
                f"rank={rec['rank']} conf={rec['conf_id']} "
                f"screen_E={rec['screen_E']:.8g} "
                f"init_rmsd={rec['screen_rmsd']:.6g} "
                f"shape_rmsd={rec['shape_rmsd']:.6g} "
                f"E_best={rec['E_best']:.8g} "
                f"rmsd_best={rec['rmsd_best']:.6g}\n"
            )
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
