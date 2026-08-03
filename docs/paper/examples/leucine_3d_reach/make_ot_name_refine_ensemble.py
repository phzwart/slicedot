#!/usr/bin/env python3
"""Leucine 3-D: random scatter → free-atom OT → naming → ADMM OT+L1+P_restr.

Ensemble of ``n_seeds`` runs against a rendered ortho density; wire-frame
overlay of named vs cleaned poses (no true structure drawn).

Usage
-----
  PYTHONPATH=../../../src python make_ot_name_refine_ensemble.py \\
      --resolution 3 --n-seeds 10
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.optimize import linear_sum_assignment

from slicedot import (
    Geometry,
    Namer,
    SlicedOT,
    SlicedOTConfig,
    prune_ghosts,
    restraint_set_from_geometry,
    sigma_from_resolution,
)
from slicedot.fixtures import W as W_NP, leucine_topology
from slicedot.fixtures_peptide import lrp_pdb_topology, oligopeptide_topology

try:
    from peptide_refs import load_peptide_ref, list_peptide_refs
except ImportError:  # running from another cwd
    list_peptide_refs = None
    load_peptide_ref = None

torch.set_default_dtype(torch.float64)

OUT_DIR = Path(__file__).resolve().parent / "out"

OT_LR = 0.4
MAX_FREE_STEPS = 1500
FREE_PATIENCE = 60
# ADMM cleanup (mirrors phenol 2-D OT+L1+P_restr finish).
# Equal OT/L1/geom consensus; density ceilings reject walk-off; stop when
# geometry is satisfied at the annealed slack (not at best-L1 alone).
CLEANUP_SLACK0 = 1.0
CLEANUP_SLACK1 = 0.20
CLEANUP_ANNEAL = 40
CLEANUP_MIN_STEPS = CLEANUP_ANNEAL + 50
ADMM_RHO = 1.0
ADMM_OT_LR0 = 0.20
ADMM_OT_LR1 = 0.015
L1_LR = 0.12
GEOM_BETA_LOW = 1.0
GEOM_BETA_HIGH = 1.0
GEOM_BETA_FINAL = 1.0
GEOM_CONSENSUS_WEIGHT = 1.0
# Reject updates that tip OT above this × OT(at ADMM start).
OT_CEIL_FACTOR = 5.0
# Reject updates that tip L1 above this × L1(at ADMM start).
L1_CEIL_FACTOR = 1.35
GEOM_TOL = 1e-3
# Geometry OK when distance/planar residuals sit inside slack + tol.
GEOM_OK_PATIENCE = 8
STEP_ATOL = 3e-3  # Å mean per-atom step for ADMM stop
ADMM_PATIENCE = 25
GAUSS_TRUNC = 3.0
SPACING = 0.5  # Å
# Final polish: Coot-style linearized map + geometry (slack=0), Phenix weight.
# Phenix/CNS (Adams et al. 1997): wxc ≈ ||∇E_geom|| / ||∇E_data||, then
#   E = wxc_scale * wxc * E_data + E_geom
# phenix.real_space_refine auto-picks weight to hit target bond/angle RMSD
# (defaults 0.01 Å / 1°).  Coot: E_map = −Σ Z_i ρ(x_i), map weight scales ∇ρ.
POLISH_STEPS = 80
POLISH_LR = 0.08
POLISH_PATIENCE = 15
# Phenix-like bond target (Å); angle target folded into distance 1–3 restraints.
TARGET_BONDS_RMSD = 0.02
TARGET_BONDS_MAX = 0.05
TARGET_CHIRAL_MAX = 0.05  # Å³-scale residual (Phenix-like chirality tolerance)
WXC_SCALE_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
L1_POLISH_CEIL = 1.35  # × L1 at polish start


class Adam:
    def __init__(self, shape, lr, beta1=0.9, beta2=0.999, eps=1e-8):
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


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def nn_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    ri, cj = linear_sum_assignment(d2)
    return float(np.sqrt(d2[ri, cj].mean()))


def aut_rmsd(X: np.ndarray, X_true: np.ndarray, namer: Namer) -> float:
    """Label RMSD minimised over graph automorphisms (CD1↔CD2)."""
    best = np.inf
    for alpha in namer.automorphisms:
        best = min(best, rmsd(X, X_true[alpha]))
    return float(best)


def render_ortho(X, sp, NG, sigma, weights):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.asarray(NG, dtype=np.float64) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG, dtype=np.float64)
    for p, w in zip(X, weights):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2.0 * sigma * sigma))
    return T / T.sum(), org, sp


def molecular_radius(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    return float(np.linalg.norm(X - X.mean(0), axis=1).max())


def build_scene(resolution: float = 3.0, sequence: tuple[str, ...] | None = None):
    """``sequence=None`` → capped leucine fixture; else peptide from ``data/peptides/``."""
    if sequence is None:
        topo = leucine_topology()
        w = W_NP.copy()
        label = "ACE–Leu–NME"
    else:
        seq_str = "".join(sequence)
        topo = None
        if load_peptide_ref is not None:
            try:
                topo = load_peptide_ref(sequence=seq_str)
            except KeyError:
                topo = None
        if topo is None and tuple(sequence) == ("L", "R", "P"):
            topo = lrp_pdb_topology()
            src = topo.get("source", {}).get("pdb_id", "PDB")
            topo["label"] = f"L–R–P ({src})"
        if topo is None:
            topo = oligopeptide_topology(sequence)
            topo["label"] = "–".join(sequence)
        w = topo["W"].copy()
        label = topo.get("label", "–".join(sequence))
    X_true = topo["X_ref"].copy()
    sig = float(sigma_from_resolution(resolution))
    R = molecular_radius(X_true)
    # Scatter / map box: molecule + a few σ of padding for free-atom OT.
    half = R + 5.0 * sig + 4.0
    n = int(np.ceil(2.0 * half / SPACING))
    if n % 2 == 0:
        n += 1
    # Cap grid so Arg/Pro peptides stay tractable (~80³).
    n = int(min(n, 81))
    NG = (n, n, n)
    T, org, sp = render_ortho(X_true, SPACING, NG, sig, w)
    ot = SlicedOT(
        torch.tensor(T),
        org,
        torch.tensor(sp),
        sig,
        SlicedOTConfig(
            n_dirs=32, dt=0.3, window=float(3.0 * half),
            map_cutoff=1e-7, backend="direct",
        ),
    )
    l1 = L1Diff3D(T, org, sp, sig)
    linmap = LinearMap3D(T, org, sp)
    geom = Geometry(
        topo["X_ref"],
        topo["bonds"],
        rotatable_bonds=topo["rotatable_bonds"],
        chiral_centres=topo["chiral_centres"],
        planar_groups=topo["planar_groups"],
        antibump=True,
    )
    # Naming prior: CIF 1–2 / 1–3 + planes/rings + plane-scoped 1–4 only.
    rs = topo.get("restraint_set")
    if rs is None:
        rs = restraint_set_from_geometry(
            topo["X_ref"],
            topo["elements"],
            topo["bonds"],
            rotatable_bonds=topo["rotatable_bonds"],
            planar_groups=topo["planar_groups"],
            atom_ids=topo.get("names"),
            comp_id="".join(sequence) if sequence else "LEU",
            torsion14="planar",
        )
    namer = Namer(
        topo["X_ref"],
        restraint_set=rs,
        rotatable_bonds=topo["rotatable_bonds"],
        chiral_centres=topo["chiral_centres"],
        planar_groups=topo["planar_groups"],
        automorphisms=topo["automorphism_generators"],
    )
    return {
        "topo": topo,
        "X_true": X_true,
        "w": w,
        "sigma": sig,
        "half": half,
        "T": T,
        "origin": org,
        "spacing": sp,
        "ot": ot,
        "l1": l1,
        "linmap": linmap,
        "geom": geom,
        "namer": namer,
        "bonds": topo["bonds"],
        "resolution": float(resolution),
        "label": label,
        "sequence": sequence,
        "n_atoms": int(X_true.shape[0]),
    }


def vg_ot(ot: SlicedOT, X: np.ndarray, w: np.ndarray, sigma: float):
    x = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w, dtype=torch.float64)
    loss = ot(x, wt, float(sigma))
    loss.backward()
    return float(loss.detach()), x.grad.detach().numpy()


class L1Diff3D:
    """E = Σ_v |ρ_T - ρ_M| on an ortho grid (unit-mass densities, truncated Gaussians)."""

    def __init__(self, rhoT: np.ndarray, origin, spacing, sigma: float,
                 trunc: float = GAUSS_TRUNC):
        T = np.asarray(rhoT, dtype=np.float64)
        self.shape = T.shape
        self.T = (T / T.sum()).ravel()
        self.origin = np.asarray(origin, dtype=np.float64).ravel()
        self.spacing = np.atleast_1d(spacing).astype(np.float64) * np.ones(3)
        self.sigma = float(sigma)
        self.trunc = float(trunc)
        ax = [
            self.origin[i] + np.arange(n) * self.spacing[i]
            for i, n in enumerate(self.shape)
        ]
        G = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
        self.V = G.reshape(-1, 3)

    def value_grad(self, x: np.ndarray, w: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        s2 = self.sigma * self.sigma
        r2max = (self.trunc * self.sigma) ** 2
        n = x.shape[0]
        nv = self.V.shape[0]
        # Sparse render: accumulate only voxels near each atom.
        raw = np.zeros(nv, dtype=np.float64)
        # Keep per-atom (voxel_idx, g) lists for the backward.
        hits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            d = self.V - x[i]
            d2 = (d * d).sum(-1)
            mask = d2 <= r2max
            if not np.any(mask):
                hits.append((np.empty(0, dtype=np.int64),
                             np.empty(0), np.empty((0, 3))))
                continue
            idx = np.nonzero(mask)[0]
            g = np.exp(-d2[idx] / (2.0 * s2))
            raw[idx] += w[i] * g
            hits.append((idx, g, d[idx]))
        Z = float(raw.sum())
        if Z <= 0:
            return 0.0, np.zeros_like(x)
        M = raw / Z
        resid = M - self.T
        val = float(np.abs(resid).sum())
        s = np.sign(resid)
        s_c = s - float((s * M).sum())
        G = np.zeros_like(x)
        for i, (idx, g, d) in enumerate(hits):
            if idx.size == 0:
                continue
            coef = (w[i] * g) * s_c[idx] / Z
            G[i] = (coef[:, None] * d).sum(0) / s2
        return val, G


def _admm_slack(t: int, T: int = CLEANUP_ANNEAL,
                s0: float = CLEANUP_SLACK0, s1: float = CLEANUP_SLACK1) -> float:
    u = float(np.clip(t / max(int(T), 1), 0.0, 1.0))
    return float(s0 + (s1 - s0) * u)


def _admm_ot_lr(t: int, T: int = CLEANUP_ANNEAL,
                lr0: float = ADMM_OT_LR0, lr1: float = ADMM_OT_LR1) -> float:
    u = float(np.clip(t / max(int(T), 1), 0.0, 1.0))
    ww = 0.5 * (1.0 - math.cos(math.pi * u))
    return float(lr0 + (lr1 - lr0) * ww)


def _admm_geom_beta(t: int, T: int = CLEANUP_ANNEAL,
                    lo: float = GEOM_BETA_LOW, hi: float = GEOM_BETA_HIGH,
                    rng: np.random.Generator | None = None) -> float:
    lo_f, hi_f = float(lo), float(hi)
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    if hi_f <= lo_f + 1e-15:
        return lo_f
    u = float(np.clip(t / max(int(T), 1), 0.0, 1.0))
    ww = 0.5 * (1.0 - math.cos(math.pi * u))
    hi_t = hi_f + (lo_f - hi_f) * ww
    if rng is None:
        return float(0.5 * (lo_f + hi_t))
    return float(rng.uniform(lo_f, hi_t))


def run_free_ot(scene, X0, *, w=None, lr=OT_LR, max_steps=MAX_FREE_STEPS,
                patience=FREE_PATIENCE, log_every: int = 25):
    """Free-atom OT until the OT loss plateaus (no use of the true pose).

    ``w`` defaults to ``scene["w"]`` (chemical).  Overcomplete free OT should
    pass a local uniform ``w_free`` and leave ``scene["w"]`` untouched.
    """
    import time as _time
    X_true = scene["X_true"]
    w = scene["w"] if w is None else np.asarray(w, dtype=np.float64)
    ot = scene["ot"]
    sig = scene["sigma"]
    X = np.asarray(X0, dtype=np.float64).copy()
    if X.shape[0] != w.shape[0]:
        raise ValueError(
            f"free OT weight length {w.shape[0]} != atom count {X.shape[0]}"
        )
    opt = Adam(X.shape, lr=lr)
    E0, _ = vg_ot(ot, X, w, sig)
    energies = [float(E0)]
    nn_rmsds = [nn_rmsd(X, X_true)]  # diagnostic only
    poses = [X.copy()]
    best_E = energies[0]
    stagnant = 0
    reason = "max_steps"
    t0 = _time.perf_counter()
    print(
        f"  [free OT] N={X.shape[0]}  step 0  OT={energies[0]:.5g}  "
        f"NN={nn_rmsds[0]:.4f} Å",
        flush=True,
    )
    for k in range(1, max_steps + 1):
        E, G = vg_ot(ot, X, w, sig)
        X = opt.step(X, G)
        poses.append(X.copy())
        energies.append(float(E))
        nn_rmsds.append(nn_rmsd(X, X_true))
        # Relative improvement in OT loss (blind).
        if E < best_E - max(1e-4, 1e-3 * abs(best_E)):
            best_E = float(E)
            stagnant = 0
        else:
            stagnant += 1
        if k % log_every == 0 or stagnant >= patience:
            print(
                f"  [free OT] step {k}  OT={E:.5g}  NN={nn_rmsds[-1]:.4f} Å  "
                f"({_time.perf_counter() - t0:.1f}s, stagnant={stagnant})",
                flush=True,
            )
        if stagnant >= patience:
            reason = "ot_loss_plateau"
            break
    return {
        "poses": np.asarray(poses),
        "energies": np.asarray(energies, dtype=np.float64),
        "nn_rmsds": np.asarray(nn_rmsds, dtype=np.float64),
        "n_steps": len(poses) - 1,
        "stop_reason": reason,
    }


class LinearMap3D:
    """Coot / Diamond linearized real-space map term: E = −Σ w_i ρ(x_i).

    Density and ∇ρ are trilinear samples of the target map at atom centres
    (the classic interactive RSR linearization used in Coot).
    """

    def __init__(self, rhoT: np.ndarray, origin, spacing):
        self.T = np.asarray(rhoT, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64).ravel()
        self.spacing = np.atleast_1d(spacing).astype(np.float64) * np.ones(3)
        self.shape = self.T.shape

    def value_grad(self, X: np.ndarray, w: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        sp = self.spacing
        org = self.origin
        nx, ny, nz = self.shape
        E = 0.0
        G = np.zeros_like(X)
        for i, (p, wi) in enumerate(zip(X, w)):
            t = (p - org) / sp
            # Clamp to grid interior for stable gradients.
            t = np.clip(t, 0.0, np.array([nx - 1.001, ny - 1.001, nz - 1.001]))
            i0 = np.floor(t).astype(int)
            i1 = np.minimum(i0 + 1, np.array(self.shape) - 1)
            f = t - i0
            # 8 corners
            c000 = self.T[i0[0], i0[1], i0[2]]
            c001 = self.T[i0[0], i0[1], i1[2]]
            c010 = self.T[i0[0], i1[1], i0[2]]
            c011 = self.T[i0[0], i1[1], i1[2]]
            c100 = self.T[i1[0], i0[1], i0[2]]
            c101 = self.T[i1[0], i0[1], i1[2]]
            c110 = self.T[i1[0], i1[1], i0[2]]
            c111 = self.T[i1[0], i1[1], i1[2]]
            fx, fy, fz = f
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
            # ∂ρ/∂t
            dtx = (
                -c000 * (1 - fy) * (1 - fz) - c001 * (1 - fy) * fz
                - c010 * fy * (1 - fz) - c011 * fy * fz
                + c100 * (1 - fy) * (1 - fz) + c101 * (1 - fy) * fz
                + c110 * fy * (1 - fz) + c111 * fy * fz
            )
            dty = (
                -c000 * (1 - fx) * (1 - fz) - c001 * (1 - fx) * fz
                + c010 * (1 - fx) * (1 - fz) + c011 * (1 - fx) * fz
                - c100 * fx * (1 - fz) - c101 * fx * fz
                + c110 * fx * (1 - fz) + c111 * fx * fz
            )
            dtz = (
                -c000 * (1 - fx) * (1 - fy) + c001 * (1 - fx) * (1 - fy)
                - c010 * (1 - fx) * fy + c011 * (1 - fx) * fy
                - c100 * fx * (1 - fy) + c101 * fx * (1 - fy)
                - c110 * fx * fy + c111 * fx * fy
            )
            E -= float(wi) * float(rho)
            # ∂ρ/∂x = (∂ρ/∂t) / spacing; E = -Σ w ρ → ∇E = -w ∇ρ
            G[i] = -float(wi) * np.array([dtx, dty, dtz]) / sp
        return float(E), G


def vg_geom_ls(geom: Geometry, X: np.ndarray, *, slack: float = 0.0):
    """E = ½‖r‖², G = Jᵀ r at the given ReLU slack (0 ⇒ no flat-bottom)."""
    prev = geom.slack
    geom.slack = float(slack)
    try:
        r = np.asarray(geom._pack(X), dtype=np.float64)
        if r.size == 0:
            return 0.0, np.zeros_like(X)
        r = np.clip(np.nan_to_num(r, nan=0.0, posinf=1e2, neginf=-1e2), -1e2, 1e2)
        J = np.asarray(geom._jac(X), dtype=np.float64)
        J = np.clip(np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0), -1e2, 1e2)
        E = 0.5 * float(np.dot(r, r))
        G = np.dot(J.T, r).reshape(X.shape)
        G = np.clip(np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0), -1e3, 1e3)
        return E, G
    finally:
        geom.slack = prev


def phenix_data_weight(G_data: np.ndarray, G_geom: np.ndarray,
                       wxc_scale: float = 1.0) -> float:
    """Adams et al. 1997 / Phenix ``wxc``: ‖∇E_geom‖ / ‖∇E_data‖ × scale."""
    nd = float(np.linalg.norm(G_data))
    ng = float(np.linalg.norm(G_geom))
    if nd < 1e-16:
        return 0.0
    return float(wxc_scale) * ng / nd


def _geom_satisfied(
    geom,
    X: np.ndarray,
    slack: float,
    tol: float = GEOM_TOL,
    slack_kinds: dict | None = None,
) -> bool:
    """True when distance / planar / chiral residuals sit inside the slack well."""
    info = geom.residual(X)
    t = float(tol)
    if slack_kinds is not None:
        sk = {k: float(v) for k, v in slack_kinds.items()}
        return (
            float(info["bond"]["max"]) <= sk.get("bond", 0.0) + t
            and float(info["angle"]["max"]) <= sk.get("angle", 0.0) + t
            and float(info["torsion14"]["max"]) <= sk.get("torsion14", 0.0) + t
            and float(info["planar"]["max"]) <= 5.0 * t + sk.get("planar", 0.0)
            and float(info["chiral"]["max"]) <= 5.0 * t + sk.get("chiral", 0.0)
            and float(info["bump"]["max"]) <= sk.get("bump", 0.0) + t
        )
    s = float(slack)
    return (
        float(info["distance_max_A"]) <= s + t
        and float(info["planar"]["max"]) <= 5.0 * t + s
        and float(info["chiral"]["max"]) <= 5.0 * t
        and float(info["bump"]["max"]) <= s + t
    )


def _fmt_geom_line(
    info: dict,
    *,
    slack: float | None = None,
    slack_kinds: dict | None = None,
) -> str:
    """Compact geometry residual string for ADMM logs."""
    bond = info["bond"]
    ang = info["angle"]
    t14 = info["torsion14"]
    pl = info["planar"]
    ch = info["chiral"]
    bump = info["bump"]
    parts = [
        f"dmax={info['distance_max_A']:.3f}",
        f"bond={bond['rms']:.3f}/{bond['max']:.3f}",
        f"ang={ang['rms']:.3f}/{ang['max']:.3f}",
        f"t14={t14['rms']:.3f}/{t14['max']:.3f}",
        f"plan={pl['rms']:.3f}/{pl['max']:.3f}",
        f"chir={ch['rms']:.3f}/{ch['max']:.3f}",
        f"bump={bump['max']:.3f}(n={bump.get('active', 0)})",
        f"wrms={info['weighted_rms']:.3g}",
    ]
    if slack_kinds is not None:
        parts.insert(
            0,
            "slack("
            f"b={float(slack_kinds.get('bond', 0)):.2f},"
            f"a={float(slack_kinds.get('angle', 0)):.2f},"
            f"p={float(slack_kinds.get('planar', 0)):.2f},"
            f"c={float(slack_kinds.get('chiral', 0)):.2f})",
        )
    elif slack is not None:
        parts.insert(0, f"slack={float(slack):.3f}")
    return "  ".join(parts)


# Named-atom terminal flat-bottoms: 0.05 Å bonds, ~10° angles (≈0.15 Å as 1–3),
# matching planar / torsion / bump, and a modest chiral-volume dead zone.
NAMED_ANGLE_SLACK_10DEG_A = 0.15
NAMED_SLACK_KINDS0 = {
    "bond": 0.35,
    "angle": 0.50,
    "torsion14": 0.50,
    "planar": 0.35,
    "bump": 0.35,
    "chiral": 0.80,
}
NAMED_SLACK_KINDS1 = {
    "bond": 0.05,
    "angle": NAMED_ANGLE_SLACK_10DEG_A,
    "torsion14": NAMED_ANGLE_SLACK_10DEG_A,
    "planar": 0.05,
    "bump": 0.05,
    "chiral": 0.25,
}


def _anneal_slack_kinds(t: int, T: int, kinds0: dict, kinds1: dict) -> dict:
    return {
        k: _admm_slack(t, T=T, s0=float(kinds0[k]), s1=float(kinds1[k]))
        for k in kinds1
    }


def run_cleanup(
    scene,
    X0,
    *,
    seed: int = 0,
    log_every: int = 10,
    named_atoms: bool = False,
):
    """Consensus ADMM: OT + L1 + annealed P_restr (phenol-style finish).

    Density ceilings: reject updates with OT > ``OT_CEIL_FACTOR × OT_start`` or
    L1 > ``L1_CEIL_FACTOR × L1_start``.  After slack anneal, stop only once
    geometry is satisfied at the current slack (blind; not argmin vs truth).

    ``named_atoms=True`` uses a gentler schedule for already-labelled chemical
    models (RDKit conformers, etc.): smaller OT/L1 steps, stronger geometry
    consensus, and a terminal P_restr at bond≤0.05 Å / angle≈10° (not forced
    slack=0, which was destroying density-feasible placements).
    """
    import time as _time

    if named_atoms:
        slack0, slack1 = 0.35, 0.05
        slack_kinds0 = dict(NAMED_SLACK_KINDS0)
        slack_kinds1 = dict(NAMED_SLACK_KINDS1)
        anneal_T = 100
        max_steps = anneal_T + 80
        ot_lr0, ot_lr1 = 0.04, 0.004
        l1_lr = 0.03
        w_g = 2.5
        ot_ceil_factor = 2.0
        l1_ceil_factor = 1.5
        force_terminal_slack0 = False
        geom_ok_patience = 12
    else:
        slack0, slack1 = float(CLEANUP_SLACK0), float(CLEANUP_SLACK1)
        slack_kinds0 = slack_kinds1 = None
        anneal_T = int(CLEANUP_ANNEAL)
        max_steps = max(int(CLEANUP_MIN_STEPS), anneal_T + 50)
        ot_lr0, ot_lr1 = float(ADMM_OT_LR0), float(ADMM_OT_LR1)
        l1_lr = float(L1_LR)
        w_g = float(GEOM_CONSENSUS_WEIGHT)
        ot_ceil_factor = float(OT_CEIL_FACTOR)
        l1_ceil_factor = float(L1_CEIL_FACTOR)
        force_terminal_slack0 = False
        geom_ok_patience = int(GEOM_OK_PATIENCE)

    X_true = scene["X_true"]
    w = scene["w"]
    ot = scene["ot"]
    l1 = scene["l1"]
    geom = scene["geom"]
    namer = scene["namer"]
    sig = scene["sigma"]
    prev_chiral_w = float(geom.w.get("chiral", 0.1))
    if named_atoms:
        geom.w["chiral"] = min(prev_chiral_w, 0.05)
    z = np.asarray(X0, dtype=np.float64).copy()
    u_ot = np.zeros_like(z)
    u_l1 = np.zeros_like(z)
    u_g = np.zeros_like(z)
    opt_ot = Adam(z.shape, lr=ot_lr0)
    opt_l1 = Adam(z.shape, lr=l1_lr)
    rng = np.random.default_rng(int(seed) + 17)
    w_sum = 2.0 + w_g
    log_every = max(int(log_every), 1)

    def _vg_ot(X, wt):
        return vg_ot(ot, X, wt, sig)

    def _vg_l1(X, wt):
        return l1.value_grad(X, wt)

    E0, _ = _vg_ot(z, w)
    E0 = float(E0)
    L10 = float(_vg_l1(z, w)[0])
    ot_ceil = float(ot_ceil_factor) * max(E0, 1e-12)
    l1_ceil = float(l1_ceil_factor) * max(L10, 1e-12)
    best_ot = E0
    best_l1 = L10
    best_rmsd = aut_rmsd(z, X_true, namer)
    energies = [E0]
    l1_energies = [L10]
    steps = [0.0]
    true_rmsds = [best_rmsd]
    poses = [z.copy()]
    slack_hist: list[float] = []
    beta_hist: list[float] = []
    geom_dist_max: list[float] = []
    n_ot_rejects = 0
    n_l1_rejects = 0
    n_soft_accepts = 0
    n_rollbacks = 0

    info0 = geom.residual(z)
    geom_dist_max.append(float(info0["distance_max_A"]))

    T = int(anneal_T)
    stagnant_step = 0
    geom_ok_streak = 0
    reason = "max_steps"
    t0 = _time.perf_counter()

    # Placeholders so the nested logger can close over current step state.
    E_z, E_l1, ds = E0, L10, 0.0
    slack_t, beta_t, lr_ot_t = float(slack0), float(GEOM_BETA_LOW), float(ot_lr0)
    kinds_t = dict(slack_kinds0) if slack_kinds0 is not None else None
    info = info0

    print(
        f"  [ADMM] start  N={z.shape[0]}  OT₀={E0:.5g}  L1₀={L10:.5g}  "
        f"ceil OT×{ot_ceil_factor:g}={ot_ceil:.5g}  L1×{l1_ceil_factor:g}={l1_ceil:.5g}"
        + ("  named_atoms" if named_atoms else ""),
        flush=True,
    )
    if slack_kinds1 is not None:
        print(
            f"  [ADMM] schedule  anneal={T}  max_steps={max_steps}  "
            f"slack_kinds bond {slack_kinds0['bond']:g}→{slack_kinds1['bond']:g} Å  "
            f"angle {slack_kinds0['angle']:g}→{slack_kinds1['angle']:g} Å(~10°)  "
            f"planar {slack_kinds0['planar']:g}→{slack_kinds1['planar']:g}  "
            f"chiral {slack_kinds0['chiral']:g}→{slack_kinds1['chiral']:g}  "
            f"lr_OT {ot_lr0:g}→{ot_lr1:g}  w_geom={w_g:g}",
            flush=True,
        )
    else:
        print(
            f"  [ADMM] schedule  anneal={T}  max_steps={max_steps}  "
            f"slack {slack0:g}→{slack1:g}  "
            f"lr_OT {ot_lr0:g}→{ot_lr1:g}  "
            f"β_geom={GEOM_BETA_LOW:g}…{GEOM_BETA_HIGH:g}→{GEOM_BETA_FINAL:g}  "
            f"w_geom={w_g:g}",
            flush=True,
        )
    print(
        f"  [ADMM] step 0  OT={E0:.5g}  L1={L10:.5g}  "
        f"RMSD={best_rmsd:.4f} Å  "
        f"{_fmt_geom_line(info0, slack=slack0, slack_kinds=kinds_t)}",
        flush=True,
    )

    def _log_step(t_idx: int, *, force: bool = False, tag: str = ""):
        if not force and (t_idx % log_every != 0):
            return
        phase = "anneal" if t_idx < T else "settle"
        ok = (
            _geom_satisfied(geom, z, slack_t, slack_kinds=kinds_t)
            if t_idx > 0 else False
        )
        prefix = f"  [ADMM] step {t_idx}"
        if tag:
            prefix += f" {tag}"
        print(
            f"{prefix}  phase={phase}  "
            f"OT={E_z:.5g} ({100.0 * E_z / ot_ceil:.0f}%ceil)  "
            f"L1={E_l1:.5g} ({100.0 * E_l1 / l1_ceil:.0f}%ceil)  "
            f"RMSD={true_rmsds[-1]:.4f} Å  Δx̄={ds:.4f} Å  "
            f"β={beta_t:.2f}  lr={lr_ot_t:.3g}  "
            f"rejects OT/L1/soft/roll={n_ot_rejects}/{n_l1_rejects}/"
            f"{n_soft_accepts}/{n_rollbacks}  "
            f"geom_ok={int(ok)}×{geom_ok_streak}  "
            f"{_fmt_geom_line(info, slack=slack_t, slack_kinds=kinds_t)}",
            flush=True,
        )

    for t in range(max_steps):
        z_prev = z.copy()
        u_ot_prev, u_l1_prev, u_g_prev = u_ot.copy(), u_l1.copy(), u_g.copy()
        slack_t = _admm_slack(t, T=T, s0=slack0, s1=slack1)
        if slack_kinds0 is not None and slack_kinds1 is not None:
            kinds_t = _anneal_slack_kinds(t, T, slack_kinds0, slack_kinds1)
        else:
            kinds_t = None
        lr_ot_t = _admm_ot_lr(t, T=T, lr0=ot_lr0, lr1=ot_lr1)
        opt_ot.lr = lr_ot_t
        if t >= T:
            beta_t = float(GEOM_BETA_FINAL)
        else:
            beta_t = _admm_geom_beta(t, T=T, rng=rng)

        y_ot = z - u_ot
        _, G_ot = _vg_ot(y_ot, w)
        x_ot = opt_ot.step(y_ot, G_ot)

        y_l1 = z - u_l1
        _, G_l1 = _vg_l1(y_l1, w)
        x_l1 = opt_l1.step(y_l1, G_l1)

        y_g = z - u_g
        if kinds_t is not None:
            x_hat, _, _ = geom.project(
                y_g, tol=GEOM_TOL, max_iter=80, slack_kinds=kinds_t,
            )
        else:
            x_hat, _, _ = geom.project(
                y_g, tol=GEOM_TOL, max_iter=80, slack=slack_t,
            )
        x_g = y_g + beta_t * (x_hat - y_g)

        z = (x_ot + x_l1 + w_g * x_g) / w_sum
        u_ot = u_ot + (x_ot - z)
        u_l1 = u_l1 + (x_l1 - z)
        u_g = u_g + (x_g - z)
        opt_ot.m[:] = 0.0
        opt_ot.v[:] = 0.0
        opt_ot.t = 0
        opt_l1.m[:] = 0.0
        opt_l1.v[:] = 0.0
        opt_l1.t = 0

        E_z, _ = _vg_ot(z, w)
        E_z = float(E_z)
        E_l1, _ = _vg_l1(z, w)
        E_l1 = float(E_l1)
        guard_tag = ""

        def _rollback():
            nonlocal z, u_ot, u_l1, u_g, E_z, E_l1
            z = z_prev
            u_ot, u_l1, u_g = u_ot_prev, u_l1_prev, u_g_prev
            E_z = float(_vg_ot(z, w)[0])
            E_l1 = float(_vg_l1(z, w)[0])

        # Density guard: never accept a step that leaves the OT/L1 envelope.
        # Prefer a lighter geom blend over dropping geometry entirely.
        if E_z > ot_ceil or E_l1 > l1_ceil:
            which = "OT" if E_z > ot_ceil else "L1"
            if E_z > ot_ceil:
                n_ot_rejects += 1
            else:
                n_l1_rejects += 1
            z_soft = (x_ot + x_l1 + 0.25 * w_g * x_g) / (2.0 + 0.25 * w_g)
            E_s, _ = _vg_ot(z_soft, w)
            L_s, _ = _vg_l1(z_soft, w)
            if float(E_s) <= ot_ceil and float(L_s) <= l1_ceil:
                z = z_soft
                u_ot = u_ot_prev + (x_ot - z)
                u_l1 = u_l1_prev + (x_l1 - z)
                u_g = u_g_prev + (x_g - z)
                E_z = float(E_s)
                E_l1 = float(L_s)
                n_soft_accepts += 1
                guard_tag = f"soft←{which}"
            else:
                _rollback()
                n_rollbacks += 1
                guard_tag = f"rollback←{which}"

        ds = float(np.linalg.norm(z - z_prev, axis=1).mean())
        info = geom.residual(z)
        dmax = float(info["distance_max_A"])
        rms_t = aut_rmsd(z, X_true, namer)
        best_ot = min(best_ot, E_z)
        best_l1 = min(best_l1, E_l1)
        best_rmsd = min(best_rmsd, rms_t)
        poses.append(z.copy())
        energies.append(E_z)
        l1_energies.append(E_l1)
        steps.append(ds)
        true_rmsds.append(rms_t)
        slack_hist.append(slack_t)
        beta_hist.append(beta_t)
        geom_dist_max.append(dmax)

        if t + 1 >= T:
            if _geom_satisfied(geom, z, slack_t, slack_kinds=kinds_t):
                geom_ok_streak += 1
            else:
                geom_ok_streak = 0
            if ds < STEP_ATOL:
                stagnant_step += 1
            else:
                stagnant_step = 0

        # Log regularly; also force at anneal→settle and on the first
        # density-guard event after a quiet stretch (avoid rollback spam).
        force_log = (t + 1 == T) or (t == 0)
        if guard_tag and (t == 0 or steps[-2] > 1e-12 or (t + 1) % log_every == 0):
            force_log = True
        _log_step(t + 1, force=force_log, tag=guard_tag)

        if t + 1 < T:
            continue

        if geom_ok_streak >= geom_ok_patience:
            reason = "geom_ok"
            break
        if geom_ok_streak >= 3 and stagnant_step >= ADMM_PATIENCE:
            reason = "geom_ok_step"
            break

    # Final density-feasible geometry polish.
    polished = False
    polish_slack = None
    polish_kinds = None
    if slack_kinds1 is not None:
        # Named atoms: terminal bond≤0.05 Å / angle≈10°, never force slack=0.
        kinds_trials = (
            dict(slack_kinds1),
            {**slack_kinds1, "bond": 0.08, "angle": 0.20, "torsion14": 0.20,
             "planar": 0.08, "bump": 0.08, "chiral": 0.35},
        )
        print(
            "  [ADMM] terminal P_restr  trying named slack_kinds "
            f"(bond→{slack_kinds1['bond']:g} Å, angle→{slack_kinds1['angle']:g} Å≈10°) "
            "under OT/L1 ceilings …",
            flush=True,
        )
        for kinds_p in kinds_trials:
            z_prev = z.copy()
            Xp, wrms_p, nfev_p = geom.project(
                z, tol=GEOM_TOL, max_iter=200, slack_kinds=kinds_p,
            )
            E_p, _ = _vg_ot(Xp, w)
            L_p, _ = _vg_l1(Xp, w)
            info_p = geom.residual(Xp)
            ok_density = float(E_p) <= ot_ceil and float(L_p) <= l1_ceil
            print(
                f"  [ADMM]   kinds bond={kinds_p['bond']:g} angle={kinds_p['angle']:g}  "
                f"OT={float(E_p):.5g} ({'ok' if float(E_p) <= ot_ceil else 'FAIL'})  "
                f"L1={float(L_p):.5g} ({'ok' if float(L_p) <= l1_ceil else 'FAIL'})  "
                f"RMSD={aut_rmsd(Xp, X_true, namer):.4f} Å  "
                f"proj_wrms={float(wrms_p):.3g}  nfev={int(nfev_p)}  "
                f"{_fmt_geom_line(info_p, slack_kinds=kinds_p)}",
                flush=True,
            )
            if not ok_density:
                continue
            z = Xp
            info = geom.residual(z)
            poses.append(z.copy())
            energies.append(float(E_p))
            l1_energies.append(float(L_p))
            steps.append(float(np.linalg.norm(z - z_prev, axis=1).mean()))
            true_rmsds.append(aut_rmsd(z, X_true, namer))
            slack_hist.append(float(kinds_p["bond"]))
            beta_hist.append(1.0)
            geom_dist_max.append(float(info["distance_max_A"]))
            polished = True
            polish_slack = float(kinds_p["bond"])
            polish_kinds = dict(kinds_p)
            reason = "geom_polish"
            break
    else:
        slack_trials = (0.0, 0.05, 0.10, float(slack1)) if slack1 > 0 else (0.0, 0.05, 0.10)
        slack_trials = tuple(sorted({float(s) for s in slack_trials}))
        print(
            f"  [ADMM] terminal P_restr  trying slack∈{{{', '.join(f'{s:g}' for s in slack_trials)}}} "
            f"under OT/L1 ceilings …",
            flush=True,
        )
        for slack_p in slack_trials:
            z_prev = z.copy()
            Xp, wrms_p, nfev_p = geom.project(
                z, tol=GEOM_TOL, max_iter=200, slack=slack_p,
            )
            E_p, _ = _vg_ot(Xp, w)
            L_p, _ = _vg_l1(Xp, w)
            info_p = geom.residual(Xp)
            ok_density = float(E_p) <= ot_ceil and float(L_p) <= l1_ceil
            print(
                f"  [ADMM]   slack={slack_p:g}  "
                f"OT={float(E_p):.5g} ({'ok' if float(E_p) <= ot_ceil else 'FAIL'})  "
                f"L1={float(L_p):.5g} ({'ok' if float(L_p) <= l1_ceil else 'FAIL'})  "
                f"RMSD={aut_rmsd(Xp, X_true, namer):.4f} Å  "
                f"proj_wrms={float(wrms_p):.3g}  nfev={int(nfev_p)}  "
                f"{_fmt_geom_line(info_p, slack=slack_p)}",
                flush=True,
            )
            if not ok_density:
                continue
            z = Xp
            info = geom.residual(z)
            poses.append(z.copy())
            energies.append(float(E_p))
            l1_energies.append(float(L_p))
            steps.append(float(np.linalg.norm(z - z_prev, axis=1).mean()))
            true_rmsds.append(aut_rmsd(z, X_true, namer))
            slack_hist.append(float(slack_p))
            beta_hist.append(1.0)
            geom_dist_max.append(float(info["distance_max_A"]))
            polished = True
            polish_slack = float(slack_p)
            reason = "geom_polish"
            break

    # Legacy named-atom force slack=0 (disabled for the new schedule).
    if force_terminal_slack0 and (not polished or float(polish_slack or 1.0) > 0.0):
        z_prev = z.copy()
        Xp, wrms_p, nfev_p = geom.project(
            z, tol=GEOM_TOL, max_iter=400, slack=0.0,
        )
        E_p, _ = _vg_ot(Xp, w)
        L_p, _ = _vg_l1(Xp, w)
        info_p = geom.residual(Xp)
        print(
            f"  [ADMM]   force slack=0  "
            f"OT={float(E_p):.5g}  L1={float(L_p):.5g}  "
            f"RMSD={aut_rmsd(Xp, X_true, namer):.4f} Å  "
            f"proj_wrms={float(wrms_p):.3g}  nfev={int(nfev_p)}  "
            f"{_fmt_geom_line(info_p, slack=0.0)}",
            flush=True,
        )
        z = Xp
        poses.append(z.copy())
        energies.append(float(E_p))
        l1_energies.append(float(L_p))
        steps.append(float(np.linalg.norm(z - z_prev, axis=1).mean()))
        true_rmsds.append(aut_rmsd(z, X_true, namer))
        slack_hist.append(0.0)
        beta_hist.append(1.0)
        geom_dist_max.append(float(info_p["distance_max_A"]))
        polished = True
        polish_slack = 0.0
        reason = "geom_polish_forced"

    if prev_chiral_w is not None:
        geom.w["chiral"] = prev_chiral_w

    elapsed = _time.perf_counter() - t0
    info_f = geom.residual(z)
    print(
        f"  [ADMM] done  steps={len(poses) - 1}  stop={reason}  "
        f"{elapsed:.1f}s  OT {E0:.5g}→{energies[-1]:.5g} (best {best_ot:.5g})  "
        f"L1 {L10:.5g}→{l1_energies[-1]:.5g} (best {best_l1:.5g})  "
        f"RMSD {true_rmsds[0]:.4f}→{true_rmsds[-1]:.4f} "
        f"(best {best_rmsd:.4f}) Å",
        flush=True,
    )
    print(
        f"  [ADMM]       rejects OT={n_ot_rejects}  L1={n_l1_rejects}  "
        f"soft={n_soft_accepts}  rollback={n_rollbacks}  "
        f"geom_polish={'yes' if polished else 'no'}"
        + (f"@{polish_slack:g}" if polished else "")
        + f"  {_fmt_geom_line(info_f)}",
        flush=True,
    )

    return {
        "poses": np.asarray(poses),
        "energies": np.asarray(energies, dtype=np.float64),
        "l1_energies": np.asarray(l1_energies, dtype=np.float64),
        "steps": np.asarray(steps, dtype=np.float64),
        "rmsds": np.asarray(true_rmsds, dtype=np.float64),
        "slack_hist": np.asarray(slack_hist, dtype=np.float64),
        "beta_hist": np.asarray(beta_hist, dtype=np.float64),
        "geom_dist_max": np.asarray(geom_dist_max, dtype=np.float64),
        "ot_start": E0,
        "l1_start": L10,
        "ot_ceil": ot_ceil,
        "l1_ceil": l1_ceil,
        "ot_best": float(best_ot),
        "l1_best": float(best_l1),
        "rmsd_best": float(best_rmsd),
        "geom_final_max": float(geom_dist_max[-1]),
        "geom_polished": bool(polished),
        "polish_slack": polish_slack,
        "n_ot_rejects": int(n_ot_rejects),
        "n_l1_rejects": int(n_l1_rejects),
        "n_soft_accepts": int(n_soft_accepts),
        "n_rollbacks": int(n_rollbacks),
        "n_steps": len(poses) - 1,
        "stop_reason": reason,
        "elapsed_s": float(elapsed),
        "method": "admm_ot_l1_geom_named" if named_atoms else "admm_ot_l1_geom",
        "named_atoms": bool(named_atoms),
    }



def _minimize_map_geom(scene, X0, *, wxc_scale: float, max_steps: int = POLISH_STEPS):
    """Adam on Phenix-weighted linearized map + slack=0 geometry."""
    w = scene["w"]
    lin = scene["linmap"]
    l1 = scene["l1"]
    geom = scene["geom"]
    X = np.asarray(X0, dtype=np.float64).copy()
    opt = Adam(X.shape, lr=POLISH_LR)
    L10 = float(l1.value_grad(X, w)[0])
    l1_ceil = float(L1_POLISH_CEIL) * max(L10, 1e-12)
    poses = [X.copy()]
    l1_hist = [L10]
    map_hist = [float(lin.value_grad(X, w)[0])]
    geom_hist = [float(vg_geom_ls(geom, X, slack=0.0)[0])]
    dist_max = [float(geom.residual(X)["distance_max_A"])]
    best_X = X.copy()
    best_score = None
    stagnant = 0
    reason = "max_steps"

    for _ in range(int(max_steps)):
        E_map, G_map = lin.value_grad(X, w)
        E_l1, G_l1 = l1.value_grad(X, w)
        E_g, G_g = vg_geom_ls(geom, X, slack=0.0)
        # Data gradient: L1 (global density) + linearized atom-centred map (Coot).
        G_data = G_l1 + G_map
        wxc = phenix_data_weight(G_data, G_g, wxc_scale=wxc_scale)
        G = wxc * G_data + G_g
        X_prev = X
        X = opt.step(X, G)
        E_l1_n = float(l1.value_grad(X, w)[0])
        if E_l1_n > l1_ceil:
            X = X_prev
            stagnant += 1
            if stagnant >= POLISH_PATIENCE:
                reason = "l1_ceil"
                break
            continue
        info = geom.residual(X)
        dmax = float(info["distance_max_A"])
        bond_rms = float(info["bond"]["rms"])
        # Prefer geom-ok + better (more negative) linearized map.
        E_map_n = float(lin.value_grad(X, w)[0])
        geom_ok = (
            bond_rms <= TARGET_BONDS_RMSD
            and dmax <= TARGET_BONDS_MAX
            and float(info["planar"]["max"]) <= TARGET_BONDS_MAX
            and float(info["chiral"]["max"]) <= TARGET_CHIRAL_MAX
        )
        score = (0 if geom_ok else 1, dmax, E_map_n)
        if best_score is None or score < best_score:
            best_score = score
            best_X = X.copy()
            stagnant = 0
        else:
            stagnant += 1
        poses.append(X.copy())
        l1_hist.append(E_l1_n)
        map_hist.append(E_map_n)
        geom_hist.append(float(vg_geom_ls(geom, X, slack=0.0)[0]))
        dist_max.append(dmax)
        ds = float(np.linalg.norm(X - X_prev, axis=1).mean())
        if geom_ok and ds < STEP_ATOL:
            reason = "geom_ok"
            break
        if stagnant >= POLISH_PATIENCE:
            reason = "plateau"
            break

    return {
        "X": best_X,
        "poses": np.asarray(poses),
        "l1": np.asarray(l1_hist, dtype=np.float64),
        "map": np.asarray(map_hist, dtype=np.float64),
        "geom": np.asarray(geom_hist, dtype=np.float64),
        "dist_max": np.asarray(dist_max, dtype=np.float64),
        "wxc_scale": float(wxc_scale),
        "stop_reason": reason,
    }


def run_l1_geom_polish(scene, X0, *, seed: int = 0):
    """Final RSR polish: L1 + Coot linearized map + geometry (slack=0).

    Weighting follows Phenix/CNS (Adams et al. 1997):
        wxc = ‖∇E_geom‖ / ‖∇E_data‖,
        E ∼ wxc_scale · wxc · E_data + E_geom.
    ``wxc_scale`` is chosen on a short grid (as in phenix.real_space_refine
    auto weight search) to meet ``TARGET_BONDS_RMSD`` / ``TARGET_BONDS_MAX``
    while keeping L1 under ``L1_POLISH_CEIL × L1_start`` and maximizing map fit.
    """
    X_true = scene["X_true"]
    namer = scene["namer"]
    geom = scene["geom"]
    l1 = scene["l1"]
    w = scene["w"]
    X0 = np.asarray(X0, dtype=np.float64)
    L10 = float(l1.value_grad(X0, w)[0])

    trials = []
    for scale in WXC_SCALE_GRID:
        tr = _minimize_map_geom(scene, X0, wxc_scale=float(scale))
        info = geom.residual(tr["X"])
        dmax = float(info["distance_max_A"])
        bond_rms = float(info["bond"]["rms"])
        E_map = float(scene["linmap"].value_grad(tr["X"], w)[0])
        E_l1 = float(l1.value_grad(tr["X"], w)[0])
        geom_ok = (
            bond_rms <= TARGET_BONDS_RMSD
            and dmax <= TARGET_BONDS_MAX
            and float(info["planar"]["max"]) <= TARGET_BONDS_MAX
            and float(info["chiral"]["max"]) <= TARGET_CHIRAL_MAX
        )
        trials.append({
            "scale": float(scale),
            "tr": tr,
            "dmax": dmax,
            "bond_rms": bond_rms,
            "E_map": E_map,
            "E_l1": E_l1,
            "geom_ok": geom_ok,
            "rmsd": aut_rmsd(tr["X"], X_true, namer),
        })

    # Prefer geometry-ok trials with best (lowest) linearized map energy;
    # else closest to the bond target.
    ok = [t for t in trials if t["geom_ok"] and t["E_l1"] <= L1_POLISH_CEIL * L10]
    if ok:
        best = min(ok, key=lambda t: (t["E_map"], t["dmax"]))
    else:
        best = min(trials, key=lambda t: (t["dmax"], t["E_map"]))

    tr = best["tr"]
    poses = np.asarray(tr["poses"], dtype=np.float64)
    l1_e = np.asarray(tr["l1"], dtype=np.float64)
    map_e = np.asarray(tr["map"], dtype=np.float64)
    geom_e = np.asarray(tr["geom"], dtype=np.float64)
    dmax_e = np.asarray(tr["dist_max"], dtype=np.float64)
    X_best = np.asarray(best["tr"]["X"], dtype=np.float64)
    if not np.allclose(poses[-1], X_best):
        poses = np.concatenate([poses, X_best[None]], axis=0)
        l1_e = np.concatenate([l1_e, [best["E_l1"]]])
        map_e = np.concatenate([map_e, [best["E_map"]]])
        geom_e = np.concatenate([geom_e, [float(vg_geom_ls(geom, X_best, slack=0.0)[0])]])
        dmax_e = np.concatenate([dmax_e, [best["dmax"]]])
    return {
        "poses": poses,
        "l1_energies": l1_e,
        "map_energies": map_e,
        "geom_energies": geom_e,
        "dist_max": dmax_e,
        "rmsds": np.array(
            [aut_rmsd(P, X_true, namer) for P in poses], dtype=np.float64
        ),
        "wxc_scale": float(best["scale"]),
        "geom_ok": bool(best["geom_ok"]),
        "geom_final_max": float(best["dmax"]),
        "l1_final": float(best["E_l1"]),
        "map_final": float(best["E_map"]),
        "n_steps": len(poses) - 1,
        "stop_reason": tr["stop_reason"] if best["geom_ok"] else "best_weight",
        "method": "l1_linmap_geom_phenix_wxc",
        "trials": [(t["scale"], t["dmax"], t["E_map"], t["geom_ok"]) for t in trials],
    }


def _subsample_indices(n: int, max_frames: int) -> np.ndarray:
    """Evenly spaced indices including first and last."""
    n = int(n)
    if n <= max_frames:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, num=max_frames, dtype=np.int64))


def build_label_trajectory(
    free_poses: np.ndarray,
    free_energies: np.ndarray,
    order: np.ndarray,
    perm: np.ndarray,
    named: np.ndarray,
    cleanup_poses: np.ndarray,
    cleanup_energies: np.ndarray,
    polish_poses: np.ndarray | None = None,
    polish_metrics: np.ndarray | None = None,
    *,
    max_free_frames: int = 80,
    max_cleanup_frames: int = 40,
    max_polish_frames: int = 30,
) -> dict:
    """Reorder free-atom poses into final label slots so topology bonds apply.

    Stages: ``random`` → ``free`` → ``named`` → ``cleanup`` → ``polish``.
    Metrics are OT (free/cleanup) or L1 (polish); blind, not RMSD-to-truth.
    """
    # label i ← free cloud index order[perm[i]]
    cloud_of_label = np.asarray(order, dtype=np.int64)[
        np.asarray(perm, dtype=np.int64)
    ]
    free_idx = _subsample_indices(len(free_poses), max_free_frames)
    coords = []
    stages = []
    metrics = []
    for t in free_idx:
        P = np.asarray(free_poses[t], dtype=np.float64)[cloud_of_label]
        coords.append(P)
        stages.append("random" if t == 0 else "free")
        metrics.append(float(free_energies[t]))
    coords.append(np.asarray(named, dtype=np.float64).copy())
    stages.append("named")
    metrics.append(float("nan"))

    cleanup_poses = np.asarray(cleanup_poses, dtype=np.float64)
    cleanup_energies = np.asarray(cleanup_energies, dtype=np.float64)
    clean_idx = _subsample_indices(len(cleanup_poses), max_cleanup_frames)
    for t in clean_idx:
        coords.append(np.asarray(cleanup_poses[t], dtype=np.float64).copy())
        stages.append("cleanup")
        metrics.append(float(cleanup_energies[t]))

    if polish_poses is not None and len(polish_poses) > 0:
        polish_poses = np.asarray(polish_poses, dtype=np.float64)
        if polish_metrics is None:
            polish_metrics = np.full(len(polish_poses), np.nan)
        else:
            polish_metrics = np.asarray(polish_metrics, dtype=np.float64)
        p_idx = _subsample_indices(len(polish_poses), max_polish_frames)
        for t in p_idx:
            coords.append(np.asarray(polish_poses[t], dtype=np.float64).copy())
            stages.append("polish")
            metrics.append(float(polish_metrics[t]))

    return {
        "coords": np.stack(coords, axis=0),
        "stages": stages,
        "metrics": np.asarray(metrics, dtype=np.float64),
        "cloud_of_label": cloud_of_label,
    }


def run_one(
    scene,
    seed: int,
    *,
    atom_factor: float = 1.0,
    save_trajectory: bool = False,
) -> dict:
    rng = np.random.default_rng(int(seed))
    X_true = scene["X_true"]
    w_chem = np.asarray(scene["w"], dtype=np.float64)
    n = len(X_true)
    atom_factor = float(atom_factor)
    n_model = int(round(atom_factor * n))
    if n_model < n:
        raise ValueError("atom_factor < 1 is not supported (undercomplete)")
    half = scene["half"]
    X_start = X_true.mean(0) + rng.uniform(-half, half, size=(n_model, 3))
    w_free = (
        w_chem if n_model == n
        else np.full(n_model, 1.0 / n_model, dtype=np.float64)
    )

    print(
        f"  [stage] free OT  N_model={n_model} (true={n}, ×{atom_factor:g})",
        flush=True,
    )
    free = run_free_ot(scene, X_start, w=w_free)
    X_free = free["poses"][-1].copy()  # last iterate after OT-loss plateau
    print(
        f"  [stage] free OT done  steps={free['n_steps']}  "
        f"NN={free['nn_rmsds'][-1]:.4f} Å  stop={free['stop_reason']}",
        flush=True,
    )

    # L1 / refine always use chemical weights + map-resolution σ.
    scene["l1"].sigma = float(scene["sigma"])

    prune = None
    if n_model != n:
        print("  [stage] ghost prune …", flush=True)
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
        asn = prune.assignment
        named = prune.Y_named.copy()
        # label i ← free index kept_idx[i]  ⇒  order=kept_idx, perm=id
        order = np.asarray(prune.kept_idx, dtype=np.int64)
        perm_for_traj = np.arange(n, dtype=np.int64)
        print(
            f"  [stage] prune done  ghosts={prune.ghost_idx.size}  "
            f"L1={prune.l1:.5g}  restr_rms={prune.restraint_rms:.4f} Å  "
            f"models={prune.n_models}",
            flush=True,
        )
    else:
        print("  [stage] naming …", flush=True)
        order = rng.permutation(n)
        Y = X_free[order]
        X_prior = (
            scene["topo"]["X_ref"]
            - scene["topo"]["X_ref"].mean(0)
            + Y.mean(0)
        )
        asn = scene["namer"].assign(Y, X_prior, weights=None)
        named = asn.Y_named.copy()
        perm_for_traj = np.asarray(asn.perm, dtype=np.int64)

    # Chemical recovery up to Aut: nearest true atom to named[i] should be alpha[i].
    n_match = 0
    for alpha in scene["namer"].automorphisms:
        ok = 0
        for i in range(n):
            j = int(np.argmin(np.linalg.norm(X_true - named[i], axis=1)))
            if j == int(alpha[i]):
                ok += 1
        n_match = max(n_match, ok)
    print(
        f"  [stage] named  RMSD={aut_rmsd(named, X_true, scene['namer']):.4f} Å  "
        f"match={n_match}/{n}  restr_rms={asn.restraint_rms:.4f} Å",
        flush=True,
    )

    print("  [stage] ADMM cleanup …", flush=True)
    cleanup = run_cleanup(scene, named, seed=seed)
    X_admm = cleanup["poses"][-1].copy()
    print(
        f"  [stage] ADMM summary  steps={cleanup['n_steps']}  "
        f"stop={cleanup['stop_reason']}  "
        f"RMSD={cleanup['rmsds'][-1]:.4f} Å  "
        f"OT={cleanup['energies'][-1]:.5g}  L1={cleanup['l1_energies'][-1]:.5g}  "
        f"rejects OT/L1/soft/roll="
        f"{cleanup['n_ot_rejects']}/{cleanup['n_l1_rejects']}/"
        f"{cleanup.get('n_soft_accepts', 0)}/{cleanup.get('n_rollbacks', 0)}  "
        f"dmax={cleanup['geom_final_max']:.3f} Å",
        flush=True,
    )
    print("  [stage] L1+geom polish …", flush=True)
    polish = run_l1_geom_polish(scene, X_admm, seed=seed)
    X_clean = polish["poses"][-1].copy()
    print(
        f"  [stage] polish done  steps={polish['n_steps']}  "
        f"RMSD={polish['rmsds'][-1]:.4f} Å  stop={polish['stop_reason']}",
        flush=True,
    )
    out = {
        "seed": int(seed),
        "free_nn": float(free["nn_rmsds"][-1]),
        "named_rmsd": float(aut_rmsd(named, X_true, scene["namer"])),
        "n_match": int(n_match),
        "restr_rms": float(asn.restraint_rms),
        "admm_rmsd": float(cleanup["rmsds"][-1]),
        "admm_final": float(cleanup["rmsds"][-1]),
        "polish_rmsd": float(polish["rmsds"][-1]),
        "named": named,
        "admm": X_admm,
        "cleaned": X_clean,
        "free_steps": int(free["n_steps"]),
        "cleanup_steps": int(cleanup["n_steps"]),
        "polish_steps": int(polish["n_steps"]),
        "free_stop": free["stop_reason"],
        "cleanup_stop": cleanup["stop_reason"],
        "polish_stop": polish["stop_reason"],
        "cleanup_method": cleanup.get("method", "admm_ot_l1_geom"),
        "polish_method": polish.get("method"),
        "wxc_scale": float(polish["wxc_scale"]),
        "geom_ok": bool(polish["geom_ok"]),
        "geom_final_max": float(polish["geom_final_max"]),
        "l1_polish": float(polish["l1_final"]),
        "ot_start": float(cleanup["ot_start"]),
        "ot_ceil": float(cleanup["ot_ceil"]),
        "ot_best": float(cleanup["ot_best"]),
        "ot_final": float(cleanup["energies"][-1]),
        "l1_start": float(cleanup["l1_start"]),
        "l1_best": float(cleanup["l1_best"]),
        "n_ot_rejects": int(cleanup["n_ot_rejects"]),
        "n_l1_rejects": int(cleanup["n_l1_rejects"]),
        "atom_factor": float(atom_factor),
        "n_model": int(n_model),
        "n_ghosts": int(0 if prune is None else prune.ghost_idx.size),
        "prune_l1": float("nan" if prune is None else prune.l1),
        "prune_restr_rms": float(
            "nan" if prune is None else prune.restraint_rms
        ),
        "prune_score": float("nan" if prune is None else prune.score),
        "n_prune_models": int(0 if prune is None else prune.n_models),
        "ghost_mask": (
            None if prune is None
            else np.asarray(prune.ghost_mask, dtype=bool)
        ),
        "kept_idx": (
            None if prune is None
            else np.asarray(prune.kept_idx, dtype=np.int64)
        ),
    }
    if save_trajectory:
        traj = build_label_trajectory(
            free["poses"], free["energies"], order, perm_for_traj, named,
            cleanup["poses"], cleanup["energies"],
            polish["poses"], polish["l1_energies"],
        )
        out["trajectory"] = traj
        out["perm"] = np.asarray(asn.perm, dtype=np.int64)
        out["order"] = np.asarray(order, dtype=np.int64)
    return out


def _wire_segments(X, bonds):
    segs = []
    for i, j in bonds:
        segs.append([X[i], X[j]])
    return np.asarray(segs, dtype=np.float64)


def _draw_wire_2d(ax, X, bonds, *, ia, ib, color, lw=1.0, alpha=0.55, ms=2.2):
    """Project bonds onto axis pair (ia, ib)."""
    X = np.asarray(X, dtype=np.float64)
    for i, j in bonds:
        ax.plot(
            [X[i, ia], X[j, ia]], [X[i, ib], X[j, ib]],
            "-", color=color, lw=lw, alpha=alpha, zorder=3,
            solid_capstyle="round",
        )
    ax.plot(
        X[:, ia], X[:, ib],
        "o", ms=ms, mfc=color, mec=color, alpha=alpha, zorder=4,
    )


def _map_extent_2d(origin, spacing, shape, ia, ib):
    """imshow extent for a 2-D slice/projection on axes (ia, ib)."""
    org = np.asarray(origin, dtype=np.float64)
    sp = np.atleast_1d(spacing) * np.ones(3)
    na, nb = int(shape[ia]), int(shape[ib])
    return [
        org[ia] - 0.5 * sp[ia],
        org[ia] + (na - 0.5) * sp[ia],
        org[ib] - 0.5 * sp[ib],
        org[ib] + (nb - 0.5) * sp[ib],
    ]


def draw_overlay(scene, results, *, out_stem: str) -> None:
    """Density max-projections + 3-D wireframes (no true structure)."""
    X_true = scene["X_true"]
    bonds = scene["bonds"]
    T = np.asarray(scene["T"], dtype=np.float64)
    org = scene["origin"]
    sp = scene["spacing"]
    rms = np.array([r["admm_rmsd"] for r in results], dtype=np.float64)
    r_lo, r_hi = float(rms.min()), float(max(rms.max(), rms.min() + 1e-6))
    cmap = plt.get_cmap("viridis")

    com = X_true.mean(0)
    span = molecular_radius(X_true) + 2.5
    # ROI from all poses + true COM box
    pts = np.vstack([X_true, *[r["named"] for r in results],
                     *[r["cleaned"] for r in results]])
    pad = 1.5
    lo = pts.min(0) - pad
    hi = pts.max(0) + pad

    # Max projections along each lab axis (density summed through the view).
    # T has indexing (x, y, z) from meshgrid indexing="ij".
    proj = {
        "xy": (T.max(axis=2).T, 0, 1),   # imshow row=y, col=x → transpose
        "xz": (T.max(axis=1).T, 0, 2),
        "yz": (T.max(axis=0).T, 1, 2),
    }
    vmax = float(max(p[0].max() for p in proj.values()))

    fig = plt.figure(figsize=(10.2, 9.0), constrained_layout=True)
    # Row 0–2: density max-proj (named | cleaned); row 3: 3-D wireframes
    axes_2d = []
    for row, (tag, (img, ia, ib)) in enumerate(proj.items()):
        for col, (key, stage) in enumerate(
            (("named", "after naming"), ("cleaned", "cleaned (final)"))
        ):
            ax = fig.add_subplot(4, 2, row * 2 + col + 1)
            extent = _map_extent_2d(org, sp, T.shape, ia, ib)
            ax.imshow(
                img, origin="lower", extent=extent, cmap="YlOrBr",
                vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal",
                zorder=0,
            )
            for r in results:
                t = (r["admm_rmsd"] - r_lo) / (r_hi - r_lo)
                colc = cmap(0.15 + 0.75 * t)
                _draw_wire_2d(
                    ax, r[key], bonds, ia=ia, ib=ib, color=colc,
                )
            ax.set_xlim(lo[ia], hi[ia])
            ax.set_ylim(lo[ib], hi[ib])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            labs = "xyz"
            ax.set_xlabel(f"{labs[ia]} (Å)", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{labs[ib]} (Å)", fontsize=7)
            ax.set_title(
                f"{stage} · max-{labs[3 - ia - ib]} proj",
                fontsize=8, loc="left", pad=2,
            )
            axes_2d.append(ax)

    axes_3d = []
    for col, (key, stage) in enumerate(
        (("named", "after naming"), ("cleaned", "cleaned (final)"))
    ):
        ax = fig.add_subplot(4, 2, 6 + col + 1, projection="3d")
        for r in results:
            t = (r["admm_rmsd"] - r_lo) / (r_hi - r_lo)
            colc = cmap(0.15 + 0.75 * t)
            segs = _wire_segments(r[key], bonds)
            ax.add_collection3d(
                Line3DCollection(
                    segs, colors=[colc], linewidths=1.0, alpha=0.55,
                )
            )
            P = r[key]
            ax.scatter(
                P[:, 0], P[:, 1], P[:, 2],
                c=[colc], s=7, alpha=0.55, depthshade=False,
            )
        ax.set_xlim(com[0] - span, com[0] + span)
        ax.set_ylim(com[1] - span, com[1] + span)
        ax.set_zlim(com[2] - span, com[2] + span)
        ax.set_xlabel("x", fontsize=6)
        ax.set_ylabel("y", fontsize=6)
        ax.set_zlabel("z", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_title(f"{stage} · 3-D", fontsize=8, loc="left", pad=2)
        ax.view_init(elev=18, azim=-60)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        axes_3d.append(ax)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=r_lo, vmax=r_hi),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_2d + axes_3d, fraction=0.02, pad=0.02)
    cbar.set_label("cleanup final RMSD (Å)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{scene['label']} @ {scene['resolution']:g} Å (3-D density) · "
        f"{len(results)} seeds · free OT → Namer → ADMM OT+L1+P_restr "
        f"(slack {CLEANUP_SLACK0:g}→{CLEANUP_SLACK1:g} Å)",
        fontsize=10,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=cmap(0.5), lw=1.2, alpha=0.7,
                   label=f"seed runs (n={len(results)})"),
        ],
        loc="lower center", ncol=1, frameon=False, fontsize=8,
        bbox_to_anchor=(0.45, -0.02),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(
    resolution: float = 3.0,
    n_seeds: int = 10,
    seed0: int = 0,
    sequence: str | None = None,
    atom_factor: float = 1.0,
):
    resolution = float(resolution)
    atom_factor = float(atom_factor)
    tag = f"{resolution:g}".replace(".", "p")
    seq_tuple = None
    if sequence:
        seq_tuple = tuple(s.strip().upper() for s in sequence.replace("-", ",").split(",") if s.strip())
        if len(seq_tuple) == 1 and len(seq_tuple[0]) > 1 and all(
            c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_tuple[0]
        ):
            # Allow compact "AFSSFN" / "LRP" form.
            seq_tuple = tuple(seq_tuple[0])
        stem_seq = "".join(seq_tuple)
        out_stem = f"peptide_{stem_seq}_ot_name_refine_{tag}A_n{n_seeds}"
    else:
        out_stem = f"leucine_ot_name_refine_{tag}A_n{n_seeds}"
    if abs(atom_factor - 1.0) > 1e-12:
        out_stem = f"{out_stem}_x{atom_factor:g}"

    print(
        f"building scene @ {resolution:g} Å  "
        f"sequence={seq_tuple or 'leucine fixture'} ...",
        flush=True,
    )
    scene = build_scene(resolution, sequence=seq_tuple)
    n_model = int(round(atom_factor * scene["n_atoms"]))
    print(
        f"  {scene['label']}  grid half-width ±{scene['half']:.1f} Å  "
        f"σ={scene['sigma']:.3f} Å  N={scene['n_atoms']}  "
        f"free_atoms={n_model} (×{atom_factor:g})",
        flush=True,
    )

    results = []
    for k in range(int(n_seeds)):
        seed = int(seed0) + k
        print(f"\n=== seed {seed} ({k + 1}/{n_seeds}) ===", flush=True)
        r = run_one(scene, seed, atom_factor=atom_factor)
        results.append(r)
        n_at = scene["n_atoms"]
        ghost_txt = (
            "" if r["n_ghosts"] == 0
            else f" · ghosts {r['n_ghosts']} (L1={r['prune_l1']:.4g})"
        )
        print(
            f"  free NN {r['free_nn']:.3f} Å ({r['free_steps']} steps) · "
            f"named {r['named_rmsd']:.3f} Å ({r['n_match']}/{n_at}) · "
            f"cleanup final {r['admm_rmsd']:.3f} Å "
            f"({r['cleanup_steps']} steps, {r.get('cleanup_stop', '?')})"
            f"{ghost_txt}",
            flush=True,
        )

    n_at = scene["n_atoms"]
    print("\nsummary:", flush=True)
    print(
        f"  naming match: "
        f"{np.mean([r['n_match'] for r in results]):.1f}/{n_at}  "
        f"(min {min(r['n_match'] for r in results)}, "
        f"max {max(r['n_match'] for r in results)})",
        flush=True,
    )
    print(
        f"  cleanup final RMSD: "
        f"mean {np.mean([r['admm_rmsd'] for r in results]):.3f} Å  "
        f"median {np.median([r['admm_rmsd'] for r in results]):.3f} Å  "
        f"min {min(r['admm_rmsd'] for r in results):.3f}  "
        f"max {max(r['admm_rmsd'] for r in results):.3f}",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"trajectory_{out_stem}.npz",
        seeds=np.array([r["seed"] for r in results]),
        free_nn=np.array([r["free_nn"] for r in results]),
        named_rmsd=np.array([r["named_rmsd"] for r in results]),
        n_match=np.array([r["n_match"] for r in results]),
        cleanup_rmsd=np.array([r["admm_rmsd"] for r in results]),
        named_poses=np.stack([r["named"] for r in results], axis=0),
        cleaned_poses=np.stack([r["cleaned"] for r in results], axis=0),
        resolution=np.array(resolution),
        atom_factor=np.array(atom_factor),
        n_ghosts=np.array([r["n_ghosts"] for r in results]),
        prune_l1=np.array([r["prune_l1"] for r in results]),
        prune_restr_rms=np.array([r["prune_restr_rms"] for r in results]),
        n_prune_models=np.array([r["n_prune_models"] for r in results]),
    )
    draw_overlay(scene, results, out_stem=out_stem)
    print(f"\nwrote {OUT_DIR / f'{out_stem}.pdf'}")
    print(f"wrote {OUT_DIR / f'{out_stem}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=float, default=3.0)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument(
        "--sequence", type=str, default=None,
        help='Residue sequence, e.g. "LRP" or "L,R,P". Uses data/peptides/ '
             'when available. Default: leucine fixture.',
    )
    ap.add_argument(
        "--list-refs", action="store_true",
        help="List crystallographic peptide references and exit.",
    )
    ap.add_argument(
        "--atom-factor", type=float, default=1.0,
        help="Free-atom count multiplier vs chemistry (ghosts pruned before naming).",
    )
    args = ap.parse_args()
    if args.list_refs:
        if list_peptide_refs is None:
            raise SystemExit("peptide_refs unavailable")
        for e in list_peptide_refs():
            print(
                f"{e['id']:16s}  {e['sequence']:8s}  "
                f"N={e['n_atoms']:2d}  {e['pdb_id']} chain {e['chain']}"
            )
        raise SystemExit(0)
    main(
        resolution=args.resolution,
        n_seeds=args.n_seeds,
        seed0=args.seed0,
        sequence=args.sequence,
        atom_factor=args.atom_factor,
    )
