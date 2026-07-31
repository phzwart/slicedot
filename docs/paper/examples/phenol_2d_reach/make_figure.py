#!/usr/bin/env python3
"""Phenol 2-D reach figure: ADMM OT vs L1 vs OT+L1 (+ P_restr).

Scene
-----
  * Planar ortho-pentyl phenol (ring + OH + 5 floppy chain carbons) at 1.5 Å.
  * Start pose: 90° rotation about the COM, then translated 3 molecular radii
    away from the true pose along +x.
  * Headline: consensus ADMM with OT + L1 + annealed P_restr.
    Contrast panels: ADMM with OT-only or L1-only (+ P_restr).
    Display frames are equal-RMSD spaced from start → best on the OT+L1 run.

Output: out/phenol_2d_reach.pdf/.png and out/trajectory_*.npz
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from phenol import (
    N_RING,
    build_phenol,
    geom_rms_2d,
    molecular_radius,
    phenol_geometry,
    project_2d,
    rotate,
)
from targets2d import (
    ConsistentSlicedW1,
    L1Diff,
    directions_2d,
    make_grid,
    render,
    sigma_from_resolution,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
RESOLUTION = 1.5
DX = 0.25
N_DIRS = 64
SHIFT_RADII = 3.0
MISALIGN_DEG = 90.0
OT_LR = 0.5
L1_LR = 0.1
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
MAX_STEPS = 5000
# Convergence: mean per-atom displacement below atol, or no RMSD improvement,
# for `patience` consecutive steps.
STEP_ATOL = 1e-3  # Å
RMSD_ATOL = 5e-3  # Å — treat as plateau if best-RMSD has not improved
PATIENCE = 40
N_SHOW = 5  # OT panels: start + intermediates + best/final
ROI_PAD = 1.8
USE_GEOMETRY = True  # interleave P_restr after each Adam step
GEOM_TOL = 1e-3
GEOM_BETA = 1.49  # Adam-path over-relax: x <- x + β (P_restr(x) - x)
# ADMM geom prox over-relax: x_G ← y_G + β (P_restr(y_G) - y_G), with
# β drawn in a window that cosine-anneals [β_low, β_high] → {β_low}.
GEOM_BETA_LOW = 1.0
GEOM_BETA_HIGH = 1.49
GEOM_BETA_SEED = 0
# OT→L1 weight ramp:
#   x ← x + (1-s) δ_OT + s δ_L1 , then P_restr^β
# s rises 0→1 over RAMP_STEPS, then restarts at s=0 (high OT) each round.
USE_WEIGHT_RAMP = True
RAMP_KIND = "linear"   # "linear" | "cosine"
RAMP_STEPS = 40
RAMP_CYCLIC = True      # after one round, restart with high OT
RAMP_MIN_ROUNDS = 2    # do not plateau-stop before this many full ramps
USE_GEOM_TRUST = False
GEOM_TRUST_EPS = 1e-6
# ReLU flat-bottom slack (Å) for P_restr: loose early, sharpened later.
GEOM_SLACK0 = 1.5
GEOM_SLACK1 = 0.30
GEOM_SLACK_ANNEAL = RAMP_STEPS * RAMP_MIN_ROUNDS  # steps to reach GEOM_SLACK1
# Consensus ADMM (ρ=1 in scaled duals: prox centre is z − u, u ← u + x − z).
ADMM_RHO = 1.0
# OT prox step size: large early for reach, annealed down so L1+geom settle.
ADMM_OT_LR0 = OT_LR
ADMM_OT_LR1 = 0.02
ADMM_OT_LR_ANNEAL = GEOM_SLACK_ANNEAL
GEOM_BETA_ANNEAL = GEOM_SLACK_ANNEAL


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


class Adam:
    def __init__(self, shape, lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                 eps=ADAM_EPS):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(self, X: np.ndarray, G: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * G
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (G * G)
        mhat = self.m / (1.0 - self.beta1 ** self.t)
        vhat = self.v / (1.0 - self.beta2 ** self.t)
        return X - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def start_pose(
    X0: np.ndarray,
    true_com: np.ndarray,
    R: float,
    misalign_deg: float,
    shift_radii: float,
) -> np.ndarray:
    """Rotate about COM by ``misalign_deg``, then translate by ``shift_radii * R``."""
    return (
        rotate(X0, np.deg2rad(misalign_deg))
        + np.asarray(true_com, dtype=np.float64)
        + np.array([float(shift_radii) * float(R), 0.0])
    )


def build_scene(
    misalign_deg: float | None = None,
    shift_radii: float | None = None,
    chain_style: str = "extended",
):
    """Build density scene and a start pose.

    Grid / true-pose placement always uses ``SHIFT_RADII`` so rotate-only
    starts (``shift_radii=0``) share the same map as the reach problem.
    ``chain_style`` selects the target (and start) chain fold
    (``"extended"`` or ``"zigzag"``).
    """
    misalign_deg = MISALIGN_DEG if misalign_deg is None else float(misalign_deg)
    shift_radii = SHIFT_RADII if shift_radii is None else float(shift_radii)
    chain_style = str(chain_style)

    X0, w = build_phenol(chain_style=chain_style)
    # Full extent for the grid; shift uses ring+OH core radius so the start
    # pose stays within reach of the 3σ-truncated density (the chain inflates R).
    R_full = molecular_radius(X0)
    R = molecular_radius(X0[:N_RING])
    sig = sigma_from_resolution(RESOLUTION)

    half = 3.0 * R + 8.0 * sig + R_full
    n = int(math.ceil(2.0 * half / DX))
    if n % 2 == 0:
        n += 1
    shape = (n, n)
    V, origin = make_grid(shape, DX)

    true_com = np.array([-SHIFT_RADII * R / 2.0, 0.0])
    X_true = X0 + true_com
    X_start = start_pose(X0, true_com, R, misalign_deg, shift_radii)
    rhoT = render(X_true, w, sig, V, shape)
    return {
        "X0": X0,
        "X_true": X_true,
        "X_start": X_start,
        "true_com": true_com,
        "w": w,
        "R": R,
        "R_full": R_full,
        "sigma": sig,
        "chain_style": chain_style,
        "rhoT": rhoT,
        "V": V,
        "origin": origin,
        "dx": DX,
        "shape": shape,
        "misalign_deg": misalign_deg,
        "shift_radii": shift_radii,
    }


def value_grad_fn(name, obj, sigma):
    if name == "ot":
        return lambda X, w: obj.value_grad(X, w, sigma)
    return lambda X, w: obj.value_grad(X, w)


def geom_slack(t: int, T: int = GEOM_SLACK_ANNEAL,
               s0: float = GEOM_SLACK0, s1: float = GEOM_SLACK1) -> float:
    """Linear anneal of ReLU flat-bottom half-width (Å)."""
    u = float(np.clip(t / max(int(T), 1), 0.0, 1.0))
    return float(s0 + (s1 - s0) * u)


def admm_ot_lr(t: int, T: int = ADMM_OT_LR_ANNEAL,
               lr0: float = ADMM_OT_LR0, lr1: float = ADMM_OT_LR1) -> float:
    """Cosine anneal of OT Adam lr: high early, near lr1 for late settling."""
    u = float(np.clip(t / max(int(T), 1), 0.0, 1.0))
    # half-cosine: slow drop at start, faster into the settle regime
    w = 0.5 * (1.0 - math.cos(math.pi * u))
    return float(lr0 + (lr1 - lr0) * w)


def admm_geom_beta(
    t: int,
    T: int = GEOM_BETA_ANNEAL,
    lo: float = GEOM_BETA_LOW,
    hi: float = GEOM_BETA_HIGH,
    rng: np.random.Generator | None = None,
) -> float:
    """Effective ADMM geom over-relax β ∈ [lo, hi_t].

    The upper edge cosine-anneals ``hi → lo`` so early steps can over-relax
    (and, if ``lo < hi``, randomly sample inside the window to break cycles);
    late steps collapse to ``lo``.  If ``lo == hi``, returns that constant.
    """
    lo_f, hi_f = float(lo), float(hi)
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    if hi_f <= lo_f + 1e-15:
        return lo_f
    u = float(np.clip(t / max(int(T), 1), 0.0, 1.0))
    w = 0.5 * (1.0 - math.cos(math.pi * u))
    hi_t = hi_f + (lo_f - hi_f) * w
    if rng is None:
        return float(0.5 * (lo_f + hi_t))
    return float(rng.uniform(lo_f, hi_t))


def ramp_s(t: int, kind: str = RAMP_KIND, T: int = RAMP_STEPS,
           cyclic: bool = RAMP_CYCLIC) -> float:
    """L1 weight s(t) ∈ [0, 1]; OT weight is 1 − s.

    If ``cyclic``, the phase is ``t % T`` so each round restarts at s=0 (pure OT).
    """
    T = max(int(T), 1)
    phase = (int(t) % T) if cyclic else int(t)
    u = float(np.clip(phase / T, 0.0, 1.0))
    if kind == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * u))
    if kind == "linear":
        return u
    raise ValueError(f"unknown ramp kind {kind!r}")


def run_admm(name, X0, w, X_true, geom, *,
             vg_ot=None, vg_l1=None,
             lr_ot0=ADMM_OT_LR0, lr_ot1=ADMM_OT_LR1, lr_l1=L1_LR,
             rho=ADMM_RHO,
             beta_low=GEOM_BETA_LOW, beta_high=GEOM_BETA_HIGH,
             beta_seed=GEOM_BETA_SEED,
             max_steps=MAX_STEPS, atol=STEP_ATOL,
             rmsd_atol=RMSD_ATOL, patience=PATIENCE):
    """Consensus ADMM: optional OT / L1 fidelities + annealed P_restr.

    Pass ``vg_ot`` and/or ``vg_l1``.  Scaled duals (ρ absorbed): prox centres
    are ``z - u_i``; after the average, ``u_i ← u_i + x_i - z``.  Data prox
    is one Adam step; geometry prox is over-relaxed ``P_restr`` at the current
    slack: ``x_G = y_G + β (P_restr(y_G) - y_G)`` with β annealed/randomized
    in ``[beta_low, beta_high]``.  When OT is active its Adam lr cosine-
    anneals ``lr_ot0`` → ``lr_ot1``.
    """
    if geom is None:
        raise ValueError("run_admm requires a Geometry operator")
    use_ot = vg_ot is not None
    use_l1 = vg_l1 is not None
    if not use_ot and not use_l1:
        raise ValueError("run_admm needs vg_ot and/or vg_l1")
    rho = float(rho)
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    beta_low = float(beta_low)
    beta_high = float(beta_high)
    rng = np.random.default_rng(int(beta_seed))

    vg_E = vg_ot if use_ot else vg_l1
    z = np.asarray(X0, dtype=np.float64).copy()
    u_ot = np.zeros_like(z) if use_ot else None
    u_l1 = np.zeros_like(z) if use_l1 else None
    u_g = np.zeros_like(z)
    opt_ot = Adam(z.shape, lr=float(lr_ot0)) if use_ot else None
    opt_l1 = Adam(z.shape, lr=float(lr_l1)) if use_l1 else None

    E0, G0 = vg_E(z, w)
    energies = [float(E0)]
    grad_norms = [float(np.linalg.norm(G0, axis=1).mean())]
    rmsds = [rmsd(z, X_true)]
    poses = [z.copy()]
    step_sizes = [0.0]
    geom_rms = [0.0]
    slack_hist = []
    u_norm_hist = []
    lr_ot_hist = []
    beta_hist = []

    best_rmsd = rmsds[0]
    stagnant_step = 0
    stagnant_rmsd = 0
    reason = "max_steps"
    n_blocks = 1 + int(use_ot) + int(use_l1)

    for t in range(max_steps):
        z_prev = z
        slack_t = geom_slack(t)
        lr_ot_t = admm_ot_lr(t, lr0=lr_ot0, lr1=lr_ot1) if use_ot else None
        if use_ot:
            opt_ot.lr = lr_ot_t

        xs = []
        # Inexact prox of active fidelities at (z − u)
        if use_ot:
            y_ot = z - u_ot
            _, G_ot = vg_ot(y_ot, w)
            x_ot = opt_ot.step(y_ot, G_ot)
            xs.append(x_ot)
        if use_l1:
            y_l1 = z - u_l1
            _, G_l1 = vg_l1(y_l1, w)
            x_l1 = opt_l1.step(y_l1, G_l1)
            xs.append(x_l1)
        y_g = z - u_g
        x_hat, g_rms, _ = project_2d(
            geom, y_g, tol=GEOM_TOL, max_iter=120, slack=slack_t,
        )
        beta_t = admm_geom_beta(t, lo=beta_low, hi=beta_high, rng=rng)
        x_g = y_g + beta_t * (x_hat - y_g)
        xs.append(x_g)

        z = sum(xs) / float(n_blocks)
        # Scaled duals (u ≡ λ/ρ): u ← u + x − z
        u_norms = []
        if use_ot:
            u_ot = u_ot + (x_ot - z)
            u_norms.append(np.linalg.norm(u_ot))
            opt_ot.m[:] = 0.0
            opt_ot.v[:] = 0.0
            opt_ot.t = 0
        if use_l1:
            u_l1 = u_l1 + (x_l1 - z)
            u_norms.append(np.linalg.norm(u_l1))
            opt_l1.m[:] = 0.0
            opt_l1.v[:] = 0.0
            opt_l1.t = 0
        u_g = u_g + (x_g - z)
        u_norms.append(np.linalg.norm(u_g))

        E_z, G_z = vg_E(z, w)
        ds = float(np.linalg.norm(z - z_prev, axis=1).mean())
        r = rmsd(z, X_true)
        u_norm = float(sum(u_norms) / len(u_norms))

        poses.append(z.copy())
        energies.append(float(E_z))
        grad_norms.append(float(np.linalg.norm(G_z, axis=1).mean()))
        rmsds.append(r)
        step_sizes.append(ds)
        geom_rms.append(g_rms)
        slack_hist.append(slack_t)
        u_norm_hist.append(u_norm)
        beta_hist.append(beta_t)
        if lr_ot_t is not None:
            lr_ot_hist.append(lr_ot_t)

        if r < best_rmsd - rmsd_atol:
            best_rmsd = r
            stagnant_rmsd = 0
        else:
            stagnant_rmsd += 1

        if ds < atol:
            stagnant_step += 1
        else:
            stagnant_step = 0

        if stagnant_step >= patience:
            reason = "step_atol"
            break
        if stagnant_rmsd >= patience:
            reason = "rmsd_plateau"
            break

    blocks = []
    if use_ot:
        blocks.append("ot")
    if use_l1:
        blocks.append("l1")
    blocks.append("geom")
    lrs = []
    if use_ot:
        lrs.append(float(lr_ot0))
    if use_l1:
        lrs.append(float(lr_l1))

    return {
        "name": name,
        "poses": np.stack(poses, axis=0),
        "energies": np.asarray(energies),
        "grad_norms": np.asarray(grad_norms),
        "rmsds": np.asarray(rmsds),
        "step_sizes": np.asarray(step_sizes),
        "geom_rms": np.asarray(geom_rms),
        "n_steps": len(poses) - 1,
        "converged": reason != "max_steps",
        "stop_reason": reason,
        "best_step": int(np.argmin(rmsds)),
        "used_geometry": True,
        "n_data_proj": int(use_ot) + int(use_l1),
        "lrs": lrs,
        "lr_ot_end": float(lr_ot1) if use_ot else None,
        "geom_beta": float(np.mean(beta_hist)) if beta_hist else None,
        "geom_beta_low": beta_low,
        "geom_beta_high": beta_high,
        "beta_hist": np.asarray(beta_hist),
        "mean_trust": [1.0] * len(lrs),
        "mean_damage": [0.0] * len(lrs),
        "trust_hist": [[] for _ in lrs],
        "mix_hist": np.asarray([]),
        "slack_hist": np.asarray(slack_hist),
        "u_norm_hist": np.asarray(u_norm_hist),
        "lr_ot_hist": np.asarray(lr_ot_hist),
        "used_weight_ramp": False,
        "ramp_kind": None,
        "ramp_steps": None,
        "ramp_cyclic": False,
        "n_ramp_rounds": 0,
        "method": "admm",
        "admm_rho": rho,
        "admm_blocks": blocks,
    }


def run_adam(name, vg, X0, w, X_true, lr, max_steps=MAX_STEPS,
             atol=STEP_ATOL, rmsd_atol=RMSD_ATOL, patience=PATIENCE,
             geom=None, geom_beta=GEOM_BETA, use_geom_trust=False,
             use_weight_ramp=False):
    """Run Adam (+ optional over-relaxed P_restr) to convergence."""
    return run_alt_proj(
        name, [vg], X0, w, X_true, lrs=[lr], max_steps=max_steps,
        atol=atol, rmsd_atol=rmsd_atol, patience=patience, geom=geom,
        geom_beta=geom_beta, use_geom_trust=use_geom_trust,
        use_weight_ramp=False,
    )


def run_alt_proj(name, vgs, X0, w, X_true, lrs, max_steps=MAX_STEPS,
                 atol=STEP_ATOL, rmsd_atol=RMSD_ATOL, patience=PATIENCE,
                 geom=None, geom_beta=GEOM_BETA, use_geom_trust=USE_GEOM_TRUST,
                 use_weight_ramp=USE_WEIGHT_RAMP,
                 ramp_kind=RAMP_KIND, ramp_steps=RAMP_STEPS,
                 ramp_cyclic=RAMP_CYCLIC, ramp_min_rounds=RAMP_MIN_ROUNDS):
    """Data projectors then optional over-relaxed P_restr.

    With two projectors (OT, L1), the default mix is a weight ramp:
        s = ramp(t);  x ← x + (1-s) δ_OT + s δ_L1
    With ``ramp_cyclic``, s restarts at 0 after every ``ramp_steps`` (high OT
    again). Plateau stopping is disabled until ``ramp_min_rounds`` finish.
    Then if ``geom`` is set: x ← x + β (P_restr(x) - x).
    """
    if len(lrs) != len(vgs):
        raise ValueError(f"lrs length {len(lrs)} != n projectors {len(vgs)}")
    X = X0.copy()
    opts = [Adam(X.shape, lr=float(lr)) for lr in lrs]
    E0, G0 = vgs[0](X, w)
    energies = [float(E0)]
    grad_norms = [float(np.linalg.norm(G0, axis=1).mean())]
    rmsds = [rmsd(X, X_true)]
    poses = [X.copy()]
    step_sizes = [0.0]
    geom_rms = [0.0]
    mix_hist = []  # L1 weight s(t) when ramping
    slack_hist = []
    trust_hist = [[] for _ in vgs]
    damage_hist = [[] for _ in vgs]

    best_rmsd = rmsds[0]
    stagnant_step = 0
    stagnant_rmsd = 0
    reason = "max_steps"
    beta = float(geom_beta)
    do_ramp = bool(use_weight_ramp) and len(vgs) == 2
    min_steps_before_plateau = (
        int(ramp_min_rounds) * max(int(ramp_steps), 1) if do_ramp and ramp_cyclic else 0
    )
    for t in range(max_steps):
        # New ramp round: give OT another chance; clear RMSD stagnation.
        if do_ramp and ramp_cyclic and t > 0 and (t % max(int(ramp_steps), 1) == 0):
            stagnant_rmsd = 0
            stagnant_step = 0

        X_prev = X
        E_rep, G_rep = None, None
        slack_t = geom_slack(t) if geom is not None else 0.0

        deltas, damages = [], []
        for opt, vg in zip(opts, vgs):
            E, G = vg(X, w)
            if E_rep is None:
                E_rep, G_rep = E, G
            X_prop = opt.step(X, G)
            delta = X_prop - X
            if geom is not None and use_geom_trust:
                rms0 = geom_rms_2d(geom, X)
                rms1 = geom_rms_2d(geom, X_prop)
                dmg = max(0.0, rms1 - rms0)
            else:
                dmg = 0.0
            deltas.append(delta)
            damages.append(dmg)

        # Optional relative geom-trust on the proposed deltas.
        if geom is not None and use_geom_trust and len(vgs) > 1:
            positive = [d for d in damages if d > GEOM_TRUST_EPS]
            d_ref = min(positive) if positive else GEOM_TRUST_EPS
            alphas = [
                1.0 if d <= GEOM_TRUST_EPS else float(d_ref / d)
                for d in damages
            ]
            deltas = [a * dlt for a, dlt in zip(alphas, deltas)]
        else:
            alphas = [1.0] * len(vgs)

        if do_ramp:
            s = ramp_s(t, kind=ramp_kind, T=ramp_steps, cyclic=ramp_cyclic)
            weights = [1.0 - s, s]
            X_new = X + sum(wi * dlt for wi, dlt in zip(weights, deltas))
            mix_hist.append(s)
        elif len(vgs) > 1:
            X_new = X + sum(deltas)
            mix_hist.append(0.5)
        else:
            X_new = X + deltas[0]
            mix_hist.append(0.0)

        for i, (a, dmg) in enumerate(zip(alphas, damages)):
            trust_hist[i].append(a)
            damage_hist[i].append(dmg)

        g_rms = 0.0
        if geom is not None:
            Xp, g_rms, _ = project_2d(
                geom, X_new, tol=GEOM_TOL, max_iter=120, slack=slack_t,
            )
            X_new = X_new + beta * (Xp - X_new)
            for opt in opts:
                opt.m[:] = 0.0
                opt.v[:] = 0.0
                opt.t = 0
        slack_hist.append(slack_t)
        ds = float(np.linalg.norm(X_new - X_prev, axis=1).mean())
        X = X_new
        r = rmsd(X, X_true)
        poses.append(X.copy())
        energies.append(float(E_rep))
        grad_norms.append(float(np.linalg.norm(G_rep, axis=1).mean()))
        rmsds.append(r)
        step_sizes.append(ds)
        geom_rms.append(g_rms)

        if r < best_rmsd - rmsd_atol:
            best_rmsd = r
            stagnant_rmsd = 0
        else:
            stagnant_rmsd += 1

        if ds < atol:
            stagnant_step += 1
        else:
            stagnant_step = 0

        if stagnant_step >= patience and t + 1 >= min_steps_before_plateau:
            reason = "step_atol"
            break
        if stagnant_rmsd >= patience and t + 1 >= min_steps_before_plateau:
            reason = "rmsd_plateau"
            break

    mean_trust = [float(np.mean(h)) if h else 1.0 for h in trust_hist]
    mean_damage = [float(np.mean(h)) if h else 0.0 for h in damage_hist]
    n_rounds = (
        int(np.ceil(len(mix_hist) / max(int(ramp_steps), 1))) if do_ramp else 0
    )
    return {
        "name": name,
        "poses": np.stack(poses, axis=0),          # (T+1, N, 2)
        "energies": np.asarray(energies),
        "grad_norms": np.asarray(grad_norms),
        "rmsds": np.asarray(rmsds),
        "step_sizes": np.asarray(step_sizes),
        "geom_rms": np.asarray(geom_rms),
        "n_steps": len(poses) - 1,
        "converged": reason != "max_steps",
        "stop_reason": reason,
        "best_step": int(np.argmin(rmsds)),
        "used_geometry": geom is not None,
        "n_data_proj": len(vgs),
        "lrs": list(lrs),
        "geom_beta": beta if geom is not None else None,
        "mean_trust": mean_trust,
        "mean_damage": mean_damage,
        "trust_hist": trust_hist,
        "mix_hist": np.asarray(mix_hist),
        "slack_hist": np.asarray(slack_hist),
        "used_weight_ramp": do_ramp,
        "ramp_kind": ramp_kind if do_ramp else None,
        "ramp_steps": ramp_steps if do_ramp else None,
        "ramp_cyclic": bool(ramp_cyclic) if do_ramp else False,
        "n_ramp_rounds": n_rounds,
    }


def save_cache(cache: dict, path: Path):
    np.savez_compressed(
        path,
        poses=cache["poses"],
        energies=cache["energies"],
        grad_norms=cache["grad_norms"],
        rmsds=cache["rmsds"],
        step_sizes=cache["step_sizes"],
        geom_rms=cache["geom_rms"],
        n_steps=np.array(cache["n_steps"]),
        converged=np.array(cache["converged"]),
        stop_reason=np.array(cache["stop_reason"]),
        best_step=np.array(cache["best_step"]),
        used_geometry=np.array(cache["used_geometry"]),
    )


def select_frames(cache: dict, n_show: int = N_SHOW) -> list[int]:
    """Pick display indices along the descent to the best-RMSD iterate.

    Frames are spaced evenly in RMSD from start → best (the part of the run
    that tells the reach story). The plateau after the best step is dropped.
    Always includes start (0) and the best-RMSD step.
    """
    rmsds = cache["rmsds"]
    best = int(cache["best_step"])
    T = best + 1  # only analyse [0, best]
    if T == 1:
        return [0]
    n_show = max(2, min(n_show, T))

    r0, r1 = float(rmsds[0]), float(rmsds[best])
    if abs(r0 - r1) < 1e-9:
        return [0, best]

    # Equal RMSD targets from start down to best.
    targets = np.linspace(r0, r1, n_show)
    idxs = []
    for t in targets:
        # among steps ≤ best, closest RMSD to target
        i = int(np.argmin(np.abs(rmsds[:T] - t)))
        idxs.append(i)

    idxs = sorted(set(idxs))
    if 0 not in idxs:
        idxs.insert(0, 0)
    if best not in idxs:
        idxs.append(best)
    idxs = sorted(set(idxs))

    # Thin if needed, keeping start and best.
    while len(idxs) > n_show:
        mid = [i for i in idxs if i not in (0, best)]
        if not mid:
            break
        # drop the frame with the smallest RMSD gap to its neighbours
        gaps = []
        for i in mid:
            pos = idxs.index(i)
            gaps.append((abs(rmsds[idxs[pos + 1]] - rmsds[idxs[pos - 1]]), i))
        idxs.remove(min(gaps)[1])
    return idxs


def _admm_tag(cache: dict) -> str:
    blocks = cache.get("admm_blocks")
    if blocks:
        body = "+".join(
            "P_restr" if b == "geom" else b.upper() for b in blocks
        )
    else:
        body = "OT+L1+P_restr"
    return f"ADMM {body} (ρ={cache.get('admm_rho', ADMM_RHO):g})"


def analyze_selection(cache: dict, idxs: list[int]) -> None:
    if cache.get("method") == "admm":
        tag = _admm_tag(cache)
    else:
        n_data = int(cache.get("n_data_proj", 1))
        if n_data > 1:
            if cache.get("used_weight_ramp"):
                cyc = " cyclic" if cache.get("ramp_cyclic") else ""
                tag = (
                    f"OT→L1 ramp({cache.get('ramp_kind')},{cache.get('ramp_steps')}"
                    f"{cyc})"
                )
            else:
                tag = "OT+L1"
        else:
            tag = "Adam"
        if cache.get("used_geometry"):
            tag += "→P_restr"
    print(f"\n[{cache['name']}] {tag} steps={cache['n_steps']}  "
          f"converged={cache['converged']} ({cache['stop_reason']})")
    print(f"  RMSD: {cache['rmsds'][0]:.3f} → min {cache['rmsds'].min():.3f} "
          f"(step {cache['best_step']}) → final {cache['rmsds'][-1]:.3f} Å")
    print(f"  E:    {cache['energies'][0]:.6g} → final {cache['energies'][-1]:.6g}")
    if cache.get("slack_hist") is not None and len(cache["slack_hist"]):
        sh = np.asarray(cache["slack_hist"])
        print(f"  geom slack: {sh[0]:.3f} → {sh[-1]:.3f} Å (ReLU flat-bottom)")
    if cache.get("method") == "admm" and cache.get("u_norm_hist") is not None:
        uh = np.asarray(cache["u_norm_hist"])
        if len(uh):
            print(f"  mean ‖u‖: start={uh[0]:.3f} end={uh[-1]:.3f}")
    if cache.get("method") == "admm" and cache.get("lr_ot_hist") is not None:
        lh = np.asarray(cache["lr_ot_hist"])
        if len(lh):
            print(f"  OT lr: {lh[0]:.3g} → {lh[-1]:.3g} (cosine anneal)")
    if cache.get("method") == "admm" and cache.get("beta_hist") is not None:
        bh = np.asarray(cache["beta_hist"])
        if len(bh):
            print(
                f"  geom β: [{cache.get('geom_beta_low', GEOM_BETA_LOW):g},"
                f"{cache.get('geom_beta_high', GEOM_BETA_HIGH):g}]  "
                f"used {bh[0]:.3f} → {bh[-1]:.3f}  "
                f"(mean {bh.mean():.3f})"
            )
    if cache.get("used_weight_ramp") and cache.get("mix_hist") is not None:
        mh = np.asarray(cache["mix_hist"])
        if len(mh):
            b = int(cache["best_step"])
            s_best = float(mh[b - 1]) if b > 0 and b - 1 < len(mh) else float(mh[-1])
            print(f"  ramp w_L1: start={mh[0]:.2f} @best={s_best:.2f} end={mh[-1]:.2f}"
                  f"  rounds≈{cache.get('n_ramp_rounds', '?')}")
    print(f"  show frames (step index): {idxs}")
    for i in idxs:
        if i == 0:
            tag = "start"
        elif i == cache["best_step"]:
            tag = "best"
        else:
            tag = f"step {i}"
        gnote = ""
        if cache.get("used_geometry") and i > 0:
            gnote = f"  geom_rms={cache['geom_rms'][i]:.2e}"
        print(f"    [{tag:8s}]  RMSD={cache['rmsds'][i]:.3f} Å  "
              f"⟨‖g‖⟩={cache['grad_norms'][i]:.3e}  E={cache['energies'][i]:.6g}{gnote}")


def roi_limits(poses: list[np.ndarray], pad: float):
    pts = np.concatenate(poses, axis=0)
    xmin, ymin = pts.min(axis=0) - pad
    xmax, ymax = pts.max(axis=0) + pad
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    return (cx - half, cx + half, cy - half, cy + half)


def _panel(ax, rhoT, full_extent, vmax, xlim, ylim, X_true, X_cur,
           G=None, quiver_scale=None, cur_color="#0b5fff"):
    ax.imshow(
        rhoT, origin="lower", extent=full_extent, cmap="YlOrBr",
        vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal",
    )
    ax.plot(
        X_true[:, 0], X_true[:, 1],
        "o", ms=4.5, mfc="none", mec="0.4", mew=0.9, zorder=3,
    )
    ax.plot(
        X_cur[:, 0], X_cur[:, 1],
        "o", ms=5.0, mfc=cur_color, mec=cur_color, zorder=4,
    )
    if G is not None:
        ax.quiver(
            X_cur[:, 0], X_cur[:, 1], -G[:, 0], -G[:, 1],
            angles="xy", scale_units="xy", scale=quiver_scale,
            width=0.012, color="#b33a3a", zorder=5,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def draw_figure(scene, caches, show_idxs, g0):
    X_true = scene["X_true"]
    rhoT, origin, dx, shape = (
        scene["rhoT"], scene["origin"], scene["dx"], scene["shape"],
    )
    Ny, Nx = shape
    full_extent = [
        origin[0] - 0.5 * dx,
        origin[0] + (Nx - 0.5) * dx,
        origin[1] - 0.5 * dx,
        origin[1] + (Ny - 0.5) * dx,
    ]

    ot, ot_only, l1 = caches["ot"], caches["ot_only"], caches["l1"]
    g_ot0, g_ot_only0, g_l10 = g0["ot"], g0["ot_only"], g0["l1"]

    mean_g = float(np.linalg.norm(g_ot0, axis=1).mean())
    quiver_scale = mean_g / (0.45 * scene["R"]) if mean_g > 0 else 1.0
    vmax = float(rhoT.max())

    show_poses = [ot["poses"][i] for i in show_idxs]
    xmin, xmax, ymin, ymax = roi_limits(
        [X_true, *show_poses, ot_only["poses"][-1], l1["poses"][-1]], ROI_PAD,
    )
    xlim, ylim = (xmin, xmax), (ymin, ymax)

    n_ot = len(show_idxs)
    fig = plt.figure(figsize=(4.6, 2.0 * n_ot + 2.2))
    gs = fig.add_gridspec(
        n_ot + 1, 2, height_ratios=[1] * n_ot + [1.05],
        hspace=0.28, wspace=0.18,
    )

    for row, idx in enumerate(show_idxs):
        ax = fig.add_subplot(gs[row, :])
        X_cur = ot["poses"][idx]
        is_start = idx == 0
        _panel(
            ax, rhoT, full_extent, vmax, xlim, ylim, X_true, X_cur,
            G=g_ot0 if is_start else None,
            quiver_scale=quiver_scale,
            cur_color="#1b1b1b" if is_start else "#0b5fff",
        )
        if is_start:
            label = "start"
        elif idx == ot["best_step"]:
            label = f"best\n({idx})"
        else:
            label = f"step {idx}"
        gnote = ""
        if is_start:
            gnote = f"   $\\langle\\|g\\|\\rangle$={ot['grad_norms'][0]:.2e}"
        if ot.get("method") == "admm":
            row_name = "OT+L1"
        elif ot.get("n_data_proj", 1) > 1:
            row_name = "OT+L1"
        else:
            row_name = "OT"
        ax.set_ylabel(f"{row_name}\n{label}", fontsize=8, rotation=0, ha="right",
                      va="center", labelpad=32)
        ax.set_title(
            f"RMSD {ot['rmsds'][idx]:.2f} Å{gnote}",
            fontsize=8, loc="left", pad=2,
        )
        if row < n_ot - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$x$ (Å)", fontsize=8)

    # OT-only / L1-only ADMM: start (quiver) + final
    for col, (cache, G0, name) in enumerate((
        (ot_only, g_ot_only0, "OT"),
        (l1, g_l10, "L1"),
    )):
        ax = fig.add_subplot(gs[-1, col])
        _panel(
            ax, rhoT, full_extent, vmax, xlim, ylim,
            X_true, cache["poses"][0], G=G0, quiver_scale=quiver_scale,
            cur_color="#1b1b1b",
        )
        ax.plot(
            cache["poses"][-1, :, 0], cache["poses"][-1, :, 1],
            "o", ms=5.0, mfc="none", mec="#0b5fff", mew=1.1, zorder=6,
        )
        ax.set_title(
            f"ADMM {name}  $\\langle\\|g\\|\\rangle_0$="
            f"{cache['grad_norms'][0]:.2e}\n"
            f"RMSD {cache['rmsds'][0]:.2f}$\\rightarrow$"
            f"{cache['rmsds'][-1]:.2f} Å  ({cache['n_steps']} steps)",
            fontsize=8, loc="left", pad=2,
        )
        ax.set_xlabel(r"$x$ (Å)", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{name}\nfinal", fontsize=8, rotation=0,
                          ha="right", va="center", labelpad=32)
        else:
            ax.set_yticklabels([])

    handles = [
        Line2D([0], [0], marker="o", color="0.4", mfc="none", ms=5, lw=0,
               label="true"),
        Line2D([0], [0], marker="o", color="#1b1b1b", ms=5, lw=0,
               label="current / start"),
        Line2D([0], [0], color="#b33a3a", lw=2,
               label=r"$-\nabla E$ at start (shared scale)"),
        Line2D([0], [0], marker="o", color="#0b5fff", mfc="none", ms=5, lw=1,
               label="selected / final"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        fontsize=8, bbox_to_anchor=(0.55, -0.01),
    )
    if ot.get("method") == "admm":
        cycle = (
            rf"ADMM OT$+$L1$+P_{{\mathrm{{restr}}}}$ "
            rf"(OT lr {ADMM_OT_LR0}$\to${ADMM_OT_LR1}/{L1_LR}, "
            rf"$\rho$={ADMM_RHO:g}, "
            rf"$\beta$ {GEOM_BETA_LOW}$\to${GEOM_BETA_HIGH}, "
            rf"slack {GEOM_SLACK0}$\to${GEOM_SLACK1})"
        )
    elif ot.get("n_data_proj", 1) > 1:
        if ot.get("used_weight_ramp"):
            cyc = r" cyclic" if ot.get("ramp_cyclic") else ""
            cycle = (
                rf"OT$\to$L1 {ot.get('ramp_kind')} ramp/{ot.get('ramp_steps')}"
                rf"{cyc} $\to P_{{\mathrm{{restr}}}}^\beta$ "
                rf"(lr={OT_LR}/{L1_LR}, $\beta$={GEOM_BETA})"
            )
        else:
            cycle = (
                rf"OT$+$L1$\to P_{{\mathrm{{restr}}}}^\beta$ "
                rf"(lr={OT_LR}/{L1_LR}, $\beta$={GEOM_BETA})"
            )
    elif ot.get("used_geometry"):
        cycle = rf"Adam$+P_{{\mathrm{{restr}}}}^\beta$ ($\beta$={GEOM_BETA})"
    else:
        cycle = "Adam"
    fig.suptitle(
        f"ortho-pentyl phenol @ {RESOLUTION} Å · {MISALIGN_DEG:.0f}° · {SHIFT_RADII}$R$ "
        f"($R$={scene['R']:.2f} Å) · {cycle} · "
        f"{ot['n_steps']} steps",
        fontsize=10, y=0.995,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "phenol_2d_reach.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "phenol_2d_reach.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    scene = build_scene()
    rhoT, V, origin, dx, sig = (
        scene["rhoT"], scene["V"], scene["origin"], scene["dx"], scene["sigma"],
    )
    ot = ConsistentSlicedW1(rhoT, V, directions_2d(N_DIRS), nbins=320, pad=12.0)
    l1 = L1Diff(rhoT, V, sig)

    # Ideal geometry from the centred phenol; P_restr is pose-invariant.
    X_ideal, _ = build_phenol()
    geom = phenol_geometry(X_ideal) if USE_GEOMETRY else None
    if geom is not None:
        # Stage A: start already on the manifold (exact projection, β=1)
        X0, _, _ = project_2d(geom, scene["X_start"], tol=GEOM_TOL, slack=0.0)
        scene = {**scene, "X_start": X0}
        print(
            f"P_restr enabled (tol={GEOM_TOL}, "
            f"ADMM β∈[{GEOM_BETA_LOW:g},{GEOM_BETA_HIGH:g}], "
            f"slack {GEOM_SLACK0}→{GEOM_SLACK1} Å); start idealised",
            flush=True,
        )

    vg_ot = value_grad_fn("ot", ot, sig)
    vg_l1 = value_grad_fn("l1", l1, sig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    caches = {}
    g0 = {}
    X0, w, X_true = scene["X_start"], scene["w"], scene["X_true"]

    jobs = [
        ("ot", "OT+L1", dict(vg_ot=vg_ot, vg_l1=vg_l1)),
        ("ot_only", "OT", dict(vg_ot=vg_ot)),
        ("l1", "L1", dict(vg_l1=vg_l1)),
    ]
    for key, label, kwargs in jobs:
        lr_note = (
            f"OT lr {ADMM_OT_LR0}→{ADMM_OT_LR1}"
            if "vg_ot" in kwargs else f"L1 lr {L1_LR}"
        )
        if "vg_ot" in kwargs and "vg_l1" in kwargs:
            lr_note = f"OT lr {ADMM_OT_LR0}→{ADMM_OT_LR1}/{L1_LR}"
        print(
            f"running ADMM {label}+P_restr  "
            f"({lr_note}, ρ={ADMM_RHO:g}, "
            f"slack {GEOM_SLACK0}→{GEOM_SLACK1} Å) ...",
            flush=True,
        )
        caches[key] = run_admm(
            key, X0, w, X_true, geom=geom,
            lr_ot0=ADMM_OT_LR0, lr_ot1=ADMM_OT_LR1, lr_l1=L1_LR,
            rho=ADMM_RHO, **kwargs,
        )
        save_cache(caches[key], OUT_DIR / f"trajectory_{key}.npz")
        print(
            f"  cached {OUT_DIR / f'trajectory_{key}.npz'}  "
            f"({caches[key]['n_steps']} steps, {caches[key]['stop_reason']})",
            flush=True,
        )

    _, g0["ot"] = vg_ot(X0, w)
    _, g0["ot_only"] = vg_ot(X0, w)
    _, g0["l1"] = vg_l1(X0, w)

    show_idxs = select_frames(caches["ot"], N_SHOW)
    for name in ("ot", "ot_only", "l1"):
        idxs = show_idxs if name == "ot" else [0, caches[name]["n_steps"]]
        analyze_selection(caches[name], idxs)

    draw_figure(scene, caches, show_idxs, g0)
    print(f"\nwrote {OUT_DIR / 'phenol_2d_reach.pdf'}")
    print(f"wrote {OUT_DIR / 'phenol_2d_reach.png'}")


if __name__ == "__main__":
    main()
