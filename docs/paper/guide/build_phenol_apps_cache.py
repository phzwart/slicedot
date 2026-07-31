#!/usr/bin/env python3
"""Build committed pose caches for guide phenol application figures.

Scenarios
---------
* extended chain @ 1.5 Å → ``fig/cache/phenol_apps.npz`` (figs 24–27)
* zigzag chain @ 3 Å     → ``fig/cache/phenol_apps_zigzag_3A.npz`` (figs 28–31)

Each cache runs free-atom OT from a COM-matched 180° start and from a random
scatter, then Namer + ADMM cleanup on the random landing.

Usage
-----
  uv sync --extra paper
  uv run python docs/paper/guide/build_phenol_apps_cache.py
  uv run python docs/paper/guide/build_phenol_apps_cache.py --only zigzag_3A
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

GUIDE_DIR = Path(__file__).resolve().parent
CACHE_DIR = GUIDE_DIR / "fig" / "cache"
PHENOL_DIR = GUIDE_DIR.parent / "examples" / "phenol_2d_reach"

SCENARIOS = {
    "extended_1p5A": {
        "path": CACHE_DIR / "phenol_apps.npz",
        "resolution": 1.5,
        "chain": "extended",
        "seed": 0,
    },
    "zigzag_3A": {
        "path": CACHE_DIR / "phenol_apps_zigzag_3A.npz",
        "resolution": 3.0,
        "chain": "zigzag",
        "seed": 0,
    },
}

sys.path.insert(0, str(PHENOL_DIR))

import make_figure as mf  # noqa: E402
from make_figure import (  # noqa: E402
    N_DIRS,
    OT_LR,
    build_scene,
    rmsd,
    value_grad_fn,
)
from make_ot_name_refine import (  # noqa: E402
    _aligned_ideal_prior,
    _random_scatter,
    _run_admm_cleanup,
    _throw_away_names,
)
from make_ot_unrestrained import nn_rmsd, run_ot_unrestrained  # noqa: E402
from phenol import (  # noqa: E402
    BONDS,
    NAMES,
    build_phenol,
    embed3,
    phenol_geometry,
    phenol_namer,
)
from targets2d import ConsistentSlicedW1, L1Diff, directions_2d  # noqa: E402


def _free_ot(X_start, w, X_true, vg_ot, *, label: str):
    print(f"  free OT [{label}] ...", flush=True)
    cache = run_ot_unrestrained(
        X_start, w, X_true, vg_ot, lr=OT_LR,
        max_steps=5000, patience=80,
    )
    ib = int(cache["best_step"])
    X_free = cache["poses"][ib].copy()
    print(
        f"    {cache['n_steps']} steps ({cache['stop_reason']})  "
        f"NN {cache['nn_rmsds'][0]:.3f} → {cache['nn_rmsds'][ib]:.3f} Å  "
        f"label {cache['label_rmsds'][ib]:.3f} Å",
        flush=True,
    )
    return {
        "X_free": X_free,
        "nn0": float(cache["nn_rmsds"][0]),
        "nn": float(cache["nn_rmsds"][ib]),
        "label0": float(cache["label_rmsds"][0]),
        "label": float(cache["label_rmsds"][ib]),
        "n_steps": int(cache["n_steps"]),
    }


def build_one(*, resolution: float, chain: str, seed: int, out_path: Path) -> None:
    mf.RESOLUTION = float(resolution)
    rng = np.random.default_rng(int(seed))
    chain = str(chain).lower()

    print(
        f"Building phenol apps cache @ {resolution:g} Å  "
        f"chain={chain}  seed={seed}",
        flush=True,
    )

    scene = build_scene(
        misalign_deg=180.0, shift_radii=0.0, chain_style=chain,
    )
    X_true = scene["X_true"]
    w = scene["w"]
    ot = ConsistentSlicedW1(
        scene["rhoT"], scene["V"], directions_2d(N_DIRS), nbins=320, pad=12.0,
    )
    l1 = L1Diff(scene["rhoT"], scene["V"], scene["sigma"])
    vg_ot = value_grad_fn("ot", ot, scene["sigma"])
    vg_l1 = value_grad_fn("l1", l1, scene["sigma"])

    X_start_180 = scene["X_start"].copy()
    free180 = _free_ot(X_start_180, w, X_true, vg_ot, label="180°")

    half = float(scene["R_full"] + 4.0 * scene["sigma"] + scene["R"])
    X_start_rand = _random_scatter(
        len(X_true), rng, center=X_true.mean(0), half_width=half,
    )
    freerand = _free_ot(X_start_rand, w, X_true, vg_ot, label="random")

    print("  Namer + ADMM cleanup on random free landing ...", flush=True)
    Y, wY, order = _throw_away_names(freerand["X_free"], w, rng)
    X_ideal, _ = build_phenol(chain_style=chain)
    namer = phenol_namer(X_ideal)
    X_prior = _aligned_ideal_prior(X_ideal, Y)
    asn = namer.assign(embed3(Y), embed3(X_prior), weights=None)
    X_named = asn.Y_named[:, :2].copy()
    print(
        f"    naming unary={asn.unary_rms:.3f} restr={asn.restraint_rms:.3f}  "
        f"label-RMSD {rmsd(X_named, X_true):.3f} Å",
        flush=True,
    )

    geom = phenol_geometry(X_ideal)
    admm = _run_admm_cleanup(
        "phenol_apps", X_named, w, X_true, geom,
        vg_ot=vg_ot, vg_l1=vg_l1,
    )
    jb = int(admm["best_step"])
    X_admm = admm["poses"][jb].copy()
    print(
        f"    ADMM best RMSD {admm['rmsds'][jb]:.3f} Å "
        f"({admm['n_steps']} steps)",
        flush=True,
    )

    Ny, Nx = scene["shape"]
    origin = np.asarray(scene["origin"], dtype=np.float64)
    dx = float(scene["dx"])
    extent = np.array([
        origin[0] - 0.5 * dx,
        origin[0] + (Nx - 0.5) * dx,
        origin[1] - 0.5 * dx,
        origin[1] + (Ny - 0.5) * dx,
    ], dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rhoT=np.asarray(scene["rhoT"], dtype=np.float32),
        extent=extent,
        X_true=X_true.astype(np.float64),
        w=w.astype(np.float64),
        bonds=np.asarray(BONDS, dtype=np.int32),
        names=np.asarray(NAMES),
        X_start_180=X_start_180.astype(np.float64),
        X_free_180=free180["X_free"].astype(np.float64),
        nn_180=np.array(free180["nn"]),
        label_180=np.array(free180["label"]),
        nn0_180=np.array(free180["nn0"]),
        label0_180=np.array(free180["label0"]),
        X_start_rand=X_start_rand.astype(np.float64),
        X_free_rand=freerand["X_free"].astype(np.float64),
        nn_rand=np.array(freerand["nn"]),
        label_rand=np.array(freerand["label"]),
        nn0_rand=np.array(freerand["nn0"]),
        label0_rand=np.array(freerand["label0"]),
        X_shuffled=Y.astype(np.float64),
        shuffle_order=np.asarray(order, dtype=np.int32),
        X_named=X_named.astype(np.float64),
        perm_namer=np.asarray(asn.perm, dtype=np.int32),
        named_label_rmsd=np.array(rmsd(X_named, X_true)),
        named_nn_rmsd=np.array(nn_rmsd(X_named, X_true)),
        named_unary_rms=np.array(asn.unary_rms),
        named_restraint_rms=np.array(asn.restraint_rms),
        X_admm=X_admm.astype(np.float64),
        admm_rmsd=np.array(float(admm["rmsds"][jb])),
        resolution=np.array(float(resolution)),
        seed=np.array(int(seed)),
        sigma=np.array(float(scene["sigma"])),
        chain=np.array(chain),
    )
    print(f"wrote {out_path}", flush=True)


def main(only: str | None = None) -> None:
    keys = [only] if only else list(SCENARIOS)
    for key in keys:
        if key not in SCENARIOS:
            raise SystemExit(
                f"unknown scenario {key!r}; choose from {list(SCENARIOS)}"
            )
        cfg = SCENARIOS[key]
        build_one(
            resolution=cfg["resolution"],
            chain=cfg["chain"],
            seed=cfg["seed"],
            out_path=cfg["path"],
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        choices=sorted(SCENARIOS),
        default=None,
        help="Build a single scenario (default: all).",
    )
    args = ap.parse_args()
    main(only=args.only)
