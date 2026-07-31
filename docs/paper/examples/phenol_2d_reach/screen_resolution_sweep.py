#!/usr/bin/env python3
"""Resolution sweep: shared 1k conformers → PCA → OT screen → ADMM refine all.

For each resolution in ``--res-min`` … ``--res-max`` (step ``--res-step``):
  * rebuild the density map at that σ
  * PCA-align every shared conformer (4 axis flips; keep best OT)
  * ADMM-refine every conformer
  * save screen metrics + full refine traces (nan-padded)

The conformer library is generated once and reused across resolutions.

Example
-------
  uv run python screen_resolution_sweep.py --chain zigzag --n-conf 1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import make_figure as mf
from make_figure import (
    ADMM_OT_LR0,
    ADMM_OT_LR1,
    ADMM_RHO,
    GEOM_TOL,
    L1_LR,
    MAX_STEPS,
    N_DIRS,
    OUT_DIR,
    build_scene,
    rmsd,
    run_admm,
    value_grad_fn,
)
from phenol import build_phenol, phenol_geometry, project_2d
from targets2d import ConsistentSlicedW1, L1Diff, directions_2d, render

from screen_conformers import (
    dedupe_greedy,
    map_pca,
    pca_placements,
    ring_aligned_rmsd,
    sample_conformers,
    score_ot,
)


def _tag_res(resolution: float) -> str:
    return f"{resolution:g}".replace(".", "p") + "A"


def build_shared_conformers(
    n_conf: int,
    chain: str,
    seed: int,
    dedup: float,
    project_slack: float,
    oversample: float,
    out_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (confs COM-centred, X0 target fold, w)."""
    if out_path.is_file():
        z = np.load(out_path)
        print(f"reusing shared conformers {out_path}  (n={len(z['confs'])})", flush=True)
        return z["confs"], z["X0"], z["w"]

    rng = np.random.default_rng(seed)
    # Geometry / target fold at a nominal resolution (shape only).
    mf.RESOLUTION = 1.5
    scene = build_scene(
        misalign_deg=0.0, shift_radii=mf.SHIFT_RADII, chain_style=chain,
    )
    X0 = scene["X0"]
    w = scene["w"]
    geom = phenol_geometry(X0)
    X_ext, _ = build_phenol(chain_style="extended")
    X_zig, _ = build_phenol(chain_style="zigzag")

    print(
        f"sampling ~{int(oversample * n_conf)} raw conformers "
        f"(dedup≥{dedup:g} Å) ...",
        flush=True,
    )
    raw = sample_conformers(
        [X_ext, X_zig], n_conf, rng, geom=geom,
        project_slack=project_slack, oversample=oversample,
    )
    diversity_key = np.array([
        min(ring_aligned_rmsd(X, X_ext), ring_aligned_rmsd(X, X_zig))
        for X in raw
    ])
    kept = dedupe_greedy(raw, thresh=dedup, order=np.argsort(-diversity_key))
    if len(kept) < n_conf:
        print(
            f"  warning: only {len(kept)} unique; using all "
            f"(wanted {n_conf})",
            flush=True,
        )
    else:
        kept = kept[:n_conf]
    confs = raw[kept]
    print(f"  kept {len(confs)} unique conformers", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        confs=confs,
        X0=X0,
        w=w,
        chain=chain,
        seed=seed,
        dedup=dedup,
    )
    print(f"wrote {out_path}", flush=True)
    return confs, X0, w


def grad_screen_stats(ot, X, w, sigma, X_true):
    E, G = ot.value_grad(X, w, sigma)
    gn = np.linalg.norm(G, axis=1)
    g_mean = (w[:, None] * G).sum(0)
    g_norm = float(np.linalg.norm(g_mean))
    d_w = (w[:, None] * (X_true - X)).sum(0)
    d_norm = float(np.linalg.norm(d_w))
    cos_descent = float(np.dot(-g_mean, d_w) / (g_norm * d_norm + 1e-30))
    d_atom = X_true - X
    dn = np.linalg.norm(d_atom, axis=1) + 1e-30
    cos_atom = ((-G) * d_atom).sum(1) / (gn + 1e-30) / dn
    return {
        "screen_E": float(E),
        "g_mean_norm": g_norm,
        "g_atom_mean": float(gn.mean()),
        "g_atom_max": float(gn.max()),
        "g_atom_std": float(gn.std()),
        "cos_descent_com": cos_descent,
        "frac_atoms_acute": float((cos_atom > 0).mean()),
        "cos_atom_mean": float(cos_atom.mean()),
    }


def _pad_trace(arr: np.ndarray, T: int, fill=np.nan) -> np.ndarray:
    out = np.full(T, fill, dtype=np.float64)
    n = min(len(arr), T)
    out[:n] = np.asarray(arr[:n], dtype=np.float64)
    return out


def run_resolution(
    resolution: float,
    confs: np.ndarray,
    X0: np.ndarray,
    w: np.ndarray,
    chain: str,
    objective: str,
    out_path: Path,
    resume: bool,
) -> Path:
    if resume and out_path.is_file():
        z = np.load(out_path)
        done = int(np.sum(z["finished"]))
        if done >= len(confs):
            print(f"[skip] {out_path.name} already complete ({done}/{len(confs)})", flush=True)
            return out_path
        print(f"[resume] {out_path.name}  {done}/{len(confs)} done", flush=True)
        start_k = done
        # load partial arrays
        data = {k: z[k] for k in z.files}
    else:
        start_k = 0
        data = None

    mf.RESOLUTION = float(resolution)
    scene = build_scene(
        misalign_deg=0.0, shift_radii=mf.SHIFT_RADII, chain_style=chain,
    )
    # Prefer shared X0 fold; place at this scene's true_com / grid.
    true_com = scene["true_com"]
    X_true = X0 + true_com
    rhoT = render(X_true, w, scene["sigma"], scene["V"], scene["shape"])
    sig = scene["sigma"]
    V = scene["V"]
    geom = phenol_geometry(X0)

    ot = ConsistentSlicedW1(rhoT, V, directions_2d(N_DIRS), nbins=320, pad=12.0)
    l1 = L1Diff(rhoT, V, sig)
    vg_ot = value_grad_fn("ot", ot, sig)
    vg_l1 = value_grad_fn("l1", l1, sig)

    n = len(confs)
    T = MAX_STEPS + 1  # include step-0
    if data is None:
        data = {
            "resolution": np.array(resolution),
            "sigma": np.array(sig),
            "chain": np.array(chain),
            "X_true": X_true,
            "finished": np.zeros(n, dtype=np.bool_),
            "screen_E": np.full(n, np.nan),
            "screen_rmsd": np.full(n, np.nan),
            "shape_rmsd": np.full(n, np.nan),
            "g_mean_norm": np.full(n, np.nan),
            "g_atom_mean": np.full(n, np.nan),
            "g_atom_max": np.full(n, np.nan),
            "g_atom_std": np.full(n, np.nan),
            "cos_descent_com": np.full(n, np.nan),
            "frac_atoms_acute": np.full(n, np.nan),
            "cos_atom_mean": np.full(n, np.nan),
            "placed": np.full((n, *confs.shape[1:]), np.nan),
            "n_steps": np.zeros(n, dtype=np.int32),
            "stop_reason": np.array([""] * n, dtype=object),
            "best_step": np.full(n, -1, dtype=np.int32),
            "E_best": np.full(n, np.nan),
            "rmsd_best": np.full(n, np.nan),
            "rmsd_at_E": np.full(n, np.nan),
            "E_at_rmsd": np.full(n, np.nan),
            "E_final": np.full(n, np.nan),
            "rmsd_final": np.full(n, np.nan),
            "trace_E": np.full((n, T), np.nan),
            "trace_rmsd": np.full((n, T), np.nan),
            "trace_grad": np.full((n, T), np.nan),
            "trace_step": np.full((n, T), np.nan),
            "E_true": np.array(score_ot(ot, X_true, w, sig)),
        }

    map_com, map_axes = map_pca(rhoT, V)
    print(
        f"\n=== resolution {resolution:g} Å  σ={sig:.3f} Å  "
        f"n={n}  objective={objective} ===",
        flush=True,
    )
    t_res = time.perf_counter()

    for k in range(start_k, n):
        t0 = time.perf_counter()
        # PCA place
        best_E, best_X = np.inf, None
        for Xp in pca_placements(confs[k], w, map_com, map_axes):
            E = score_ot(ot, Xp, w, sig)
            if E < best_E:
                best_E, best_X = E, Xp
        assert best_X is not None
        st = grad_screen_stats(ot, best_X, w, sig, X_true)
        data["screen_E"][k] = st["screen_E"]
        data["screen_rmsd"][k] = rmsd(best_X, X_true)
        data["shape_rmsd"][k] = ring_aligned_rmsd(confs[k], X0)
        for key in (
            "g_mean_norm", "g_atom_mean", "g_atom_max", "g_atom_std",
            "cos_descent_com", "frac_atoms_acute", "cos_atom_mean",
        ):
            data[key][k] = st[key]
        data["placed"][k] = best_X

        X_start, _, _ = project_2d(geom, best_X, tol=GEOM_TOL, slack=0.0)
        kwargs = dict(vg_ot=vg_ot)
        if objective == "ot+l1":
            kwargs["vg_l1"] = vg_l1
        cache = run_admm(
            f"r{resolution:g}_c{k}", X_start, w, X_true, geom=geom,
            lr_ot0=ADMM_OT_LR0, lr_ot1=ADMM_OT_LR1, lr_l1=L1_LR,
            rho=ADMM_RHO, **kwargs,
        )
        E_tr = np.asarray(cache["energies"], dtype=np.float64)
        r_tr = np.asarray(cache["rmsds"], dtype=np.float64)
        g_tr = np.asarray(cache["grad_norms"], dtype=np.float64)
        s_tr = np.asarray(cache["step_sizes"], dtype=np.float64)
        i_E = int(np.nanargmin(E_tr))
        i_R = int(cache["best_step"])

        data["n_steps"][k] = int(cache["n_steps"])
        data["stop_reason"][k] = str(cache["stop_reason"])
        data["best_step"][k] = i_R
        data["E_best"][k] = float(E_tr[i_E])
        data["rmsd_best"][k] = float(r_tr.min())
        data["rmsd_at_E"][k] = float(r_tr[i_E])
        data["E_at_rmsd"][k] = float(E_tr[i_R])
        data["E_final"][k] = float(E_tr[-1])
        data["rmsd_final"][k] = float(r_tr[-1])
        data["trace_E"][k] = _pad_trace(E_tr, T)
        data["trace_rmsd"][k] = _pad_trace(r_tr, T)
        data["trace_grad"][k] = _pad_trace(g_tr, T)
        data["trace_step"][k] = _pad_trace(s_tr, T)
        data["finished"][k] = True

        # checkpoint every 25
        if (k + 1) % 25 == 0 or k + 1 == n:
            _save(out_path, data)
            dt = time.perf_counter() - t0
            print(
                f"  [{resolution:g}Å] {k+1}/{n}  "
                f"screen_E={data['screen_E'][k]:.4g}  "
                f"E*={data['E_best'][k]:.4g}  "
                f"R*={data['rmsd_best'][k]:.3f} Å  "
                f"({data['n_steps'][k]} steps, {dt:.1f}s)",
                flush=True,
            )

    _save(out_path, data)
    print(
        f"  done {resolution:g} Å in {time.perf_counter() - t_res:.1f}s  "
        f"best E*={np.nanmin(data['E_best']):.6g}  "
        f"best R*={np.nanmin(data['rmsd_best']):.3f} Å → {out_path}",
        flush=True,
    )
    return out_path


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # stop_reason as unicode
    payload = dict(data)
    payload["stop_reason"] = np.asarray(payload["stop_reason"], dtype=object)
    np.savez_compressed(path, **payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", choices=("extended", "zigzag"), default="zigzag")
    ap.add_argument("--n-conf", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup", type=float, default=0.5)
    ap.add_argument("--project-slack", type=float, default=0.5)
    ap.add_argument("--oversample", type=float, default=4.0)
    ap.add_argument("--res-min", type=float, default=1.0)
    ap.add_argument("--res-max", type=float, default=3.0)
    ap.add_argument("--res-step", type=float, default=0.5)
    ap.add_argument("--objective", choices=("ot", "ot+l1"), default="ot+l1")
    ap.add_argument(
        "--resume", action="store_true",
        help="Skip completed resolutions / continue partial npz files.",
    )
    args = ap.parse_args()

    resolutions = np.round(
        np.arange(args.res_min, args.res_max + 0.5 * args.res_step, args.res_step),
        6,
    )
    resolutions = [float(r) for r in resolutions if r <= args.res_max + 1e-9]

    sweep_dir = OUT_DIR / f"sweep_{args.chain}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    conf_path = sweep_dir / f"shared_conformers_n{args.n_conf}_seed{args.seed}.npz"

    meta = {
        "chain": args.chain,
        "n_conf": args.n_conf,
        "seed": args.seed,
        "dedup": args.dedup,
        "resolutions": resolutions,
        "objective": args.objective,
    }
    (sweep_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f"resolution sweep {resolutions}  chain={args.chain}  "
        f"n_conf={args.n_conf}  objective={args.objective}",
        flush=True,
    )
    confs, X0, w = build_shared_conformers(
        args.n_conf, args.chain, args.seed, args.dedup,
        args.project_slack, args.oversample, conf_path,
    )

    t0 = time.perf_counter()
    paths = []
    for res in resolutions:
        out = sweep_dir / f"refine_{_tag_res(res)}.npz"
        paths.append(
            run_resolution(
                res, confs, X0, w, args.chain, args.objective, out,
                resume=args.resume,
            )
        )
    print(
        f"\n========== SWEEP DONE ==========\n"
        f"elapsed {time.perf_counter() - t0:.1f}s\n"
        f"outputs in {sweep_dir}",
        flush=True,
    )
    for p in paths:
        print(f"  {p}", flush=True)


if __name__ == "__main__":
    main()
