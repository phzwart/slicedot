#!/usr/bin/env python3
"""Sweep OT→L1 weight-ramp schedules on the phenol 2-D reach scene.

Proposal
--------
Each cycle proposes Adam steps δ_OT, δ_L1 from the same pose, then
    x ← x + w_OT(t) δ_OT + w_L1(t) δ_L1
    x ← x + β (P_restr(x) - x)
with w_L1(t) = s(t), w_OT(t) = 1 - s(t), s increasing from 0 → 1
(OT does the reach; L1 takes over for polish).

Schedules for s(t):
  linear(T)          s = clip(t/T, 0, 1)
  cosine(T)          smooth half-cosine over T
  hold_linear(H,T)   s=0 for H steps, then linear over T
  progress_E         s from OT-energy drop (adaptive)
  fixed(w_L1)        constant mix (baselines)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phenol import build_phenol, phenol_geometry, project_2d  # noqa: E402
from targets2d import ConsistentSlicedW1, L1Diff, directions_2d  # noqa: E402
from make_figure import (  # noqa: E402
    Adam,
    GEOM_BETA,
    GEOM_TOL,
    L1_LR,
    N_DIRS,
    OT_LR,
    PATIENCE,
    RMSD_ATOL,
    STEP_ATOL,
    build_scene,
    rmsd,
    value_grad_fn,
)


def schedule_linear(t: int, T: int) -> float:
    return float(np.clip(t / max(T, 1), 0.0, 1.0))


def schedule_cosine(t: int, T: int) -> float:
    u = float(np.clip(t / max(T, 1), 0.0, 1.0))
    return 0.5 * (1.0 - math.cos(math.pi * u))


def schedule_hold_linear(t: int, hold: int, T: int) -> float:
    if t < hold:
        return 0.0
    return float(np.clip((t - hold) / max(T, 1), 0.0, 1.0))


def run_ramp(vg_ot, vg_l1, X0, w, X_true, geom, sched_fn, label,
             max_steps=400, patience=PATIENCE):
    X = X0.copy()
    opt_ot = Adam(X.shape, OT_LR)
    opt_l1 = Adam(X.shape, L1_LR)
    rmsds = [rmsd(X, X_true)]
    w_l1_hist = [0.0]
    best = rmsds[0]
    stag = 0
    reason = "max_steps"
    E0, _ = vg_ot(X, w)
    adaptive = label.startswith("progress")

    for t in range(max_steps):
        E_ot, G_ot = vg_ot(X, w)
        _, G_l1 = vg_l1(X, w)
        d_ot = opt_ot.step(X, G_ot) - X
        d_l1 = opt_l1.step(X, G_l1) - X

        if adaptive:
            # s rises as sliced-W1 energy falls from start toward ~5% of E0.
            E_floor = 0.05 * E0
            s = float(np.clip((E0 - E_ot) / max(E0 - E_floor, 1e-12), 0.0, 1.0))
        else:
            s = float(sched_fn(t))

        w_l1 = s
        w_ot = 1.0 - s
        X_new = X + w_ot * d_ot + w_l1 * d_l1
        Xp, _, _ = project_2d(geom, X_new, tol=GEOM_TOL, max_iter=80)
        X_new = X_new + GEOM_BETA * (Xp - X_new)

        for opt in (opt_ot, opt_l1):
            opt.m[:] = 0.0
            opt.v[:] = 0.0
            opt.t = 0

        ds = float(np.linalg.norm(X_new - X, axis=1).mean())
        X = X_new
        r = rmsd(X, X_true)
        rmsds.append(r)
        w_l1_hist.append(w_l1)

        if r < best - RMSD_ATOL:
            best = r
            stag = 0
        else:
            stag += 1
        if ds < STEP_ATOL:
            # count toward step patience via same stag? keep separate
            pass
        if stag >= patience:
            reason = "rmsd_plateau"
            break

    rmsds = np.asarray(rmsds)
    best_k = int(np.argmin(rmsds))
    return {
        "label": label,
        "n_steps": len(rmsds) - 1,
        "best_rmsd": float(rmsds.min()),
        "best_step": best_k,
        "final_rmsd": float(rmsds[-1]),
        "stop_reason": reason,
        "rmsds": rmsds,
        "w_l1": np.asarray(w_l1_hist),
        "w_l1_at_best": float(w_l1_hist[best_k]),
    }


def main():
    scene = build_scene()
    geom = phenol_geometry(build_phenol()[0])
    X0, _, _ = project_2d(geom, scene["X_start"], tol=GEOM_TOL)
    w, X_true = scene["w"], scene["X_true"]
    ot = ConsistentSlicedW1(
        scene["rhoT"], scene["V"], directions_2d(N_DIRS), nbins=320, pad=12.0,
    )
    l1 = L1Diff(scene["rhoT"], scene["V"], scene["sigma"])
    vg_ot = value_grad_fn("ot", ot, scene["sigma"])
    vg_l1 = value_grad_fn("l1", l1, scene["sigma"])

    configs = []
    # Baselines
    configs.append(("fixed_OT", lambda t: 0.0))
    configs.append(("fixed_L1", lambda t: 1.0))
    configs.append(("fixed_50_50", lambda t: 0.5))
    configs.append(("fixed_80_20", lambda t: 0.2))  # mostly OT
    configs.append(("fixed_20_80", lambda t: 0.8))  # mostly L1
    # Linear ramps
    for T in (10, 20, 40, 60, 80):
        configs.append((f"linear_{T}", lambda t, T=T: schedule_linear(t, T)))
    # Cosine ramps
    for T in (20, 40, 60):
        configs.append((f"cosine_{T}", lambda t, T=T: schedule_cosine(t, T)))
    # Hold OT then ramp
    for H, T in ((10, 20), (15, 30), (20, 40)):
        configs.append(
            (f"hold{H}_lin{T}", lambda t, H=H, T=T: schedule_hold_linear(t, H, T))
        )
    # Adaptive energy progress
    configs.append(("progress_E", lambda t: 0.0))

    print(
        f"{'schedule':<16} {'best':>8} {'@step':>6} {'final':>8} "
        f"{'n':>5} {'wL1@best':>8}  reason"
    )
    print("-" * 72)
    results = []
    for label, fn in configs:
        r = run_ramp(vg_ot, vg_l1, X0, w, X_true, geom, fn, label)
        results.append(r)
        print(
            f"{label:<16} {r['best_rmsd']:8.3f} {r['best_step']:6d} "
            f"{r['final_rmsd']:8.3f} {r['n_steps']:5d} "
            f"{r['w_l1_at_best']:8.3f}  {r['stop_reason']}"
        )

    # Rank by best RMSD, then by steps to best
    ranked = sorted(results, key=lambda r: (r["best_rmsd"], r["best_step"]))
    print("\nTop 5 by best RMSD:")
    for r in ranked[:5]:
        print(
            f"  {r['label']:<16}  best={r['best_rmsd']:.3f} Å "
            f"@ {r['best_step']}  (w_L1={r['w_l1_at_best']:.2f})"
        )

    out = Path(__file__).resolve().parent / "out" / "weight_ramp_sweep.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        labels=np.array([r["label"] for r in results]),
        best_rmsd=np.array([r["best_rmsd"] for r in results]),
        best_step=np.array([r["best_step"] for r in results]),
        final_rmsd=np.array([r["final_rmsd"] for r in results]),
    )
    print(f"\nwrote {out}")
    return ranked[0]


if __name__ == "__main__":
    main()
