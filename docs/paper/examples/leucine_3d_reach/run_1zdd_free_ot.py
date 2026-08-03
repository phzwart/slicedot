#!/usr/bin/env python3
"""1ZDD (Z34C): global free-atom OT, then overlapping-sphere local OT.

Phase 1 — global sliced W₁ from a uniform box start until OT-loss plateau.
Phase 2 — cover the cloud with overlapping spheres of radius R; for each
sphere compute unbalanced local OT (atoms + voxels inside the ball),
accumulate gradients, then one Adam update after all spheres.  Shrink
R ← R / √3 and repeat until R_min.

Usage
-----
  uv run python docs/paper/examples/leucine_3d_reach/run_1zdd_free_ot.py
  uv run python docs/paper/examples/leucine_3d_reach/run_1zdd_free_ot.py \\
      --device cuda --sphere-r0 20 --sphere-r-min 2
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from slicedot import Namer, SlicedOT, SlicedOTConfig, prune_ghosts, sigma_from_resolution

from build_peptide_refs import TARGETS, try_extract
from make_ot_name_refine_ensemble import L1Diff3D

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

# Special-case HLH; oligopeptides come from TARGETS in build_peptide_refs.
HLH_TARGETS = {
    "1ZDD": ("1ZDD", "A", 6, 39, "FNMQCQRRFYEALHDPNLNEEQRNAKIKSIRDDC"),
}
PEPTIDE_BY_SEQ = {seq: (pdb, ch, s0, s1, seq) for pdb, ch, s0, s1, seq in TARGETS}

OT_LR = 0.4
MAX_STEPS = 1500
PATIENCE = 60
SPACING = 0.5  # Å
N_DIRS = 32
SHRINK = 1.0 / math.sqrt(3.0)


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


def nn_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    ri, cj = linear_sum_assignment(d2)
    return float(np.sqrt(d2[ri, cj].mean()))


def molecular_radius(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    return float(np.linalg.norm(X - X.mean(0), axis=1).max())


def render_ortho(X, sp, NG, sigma, weights):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.asarray(NG, dtype=np.float64) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG, dtype=np.float64)
    for p, w in zip(X, weights):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2.0 * sigma * sigma))
    return T / T.sum(), org, sp


def vg_ot(
    ot: SlicedOT,
    X: np.ndarray,
    w: np.ndarray,
    sigma: float,
    device: torch.device | str | None = None,
    Tq: torch.Tensor | None = None,
):
    if device is None:
        device = next(ot.buffers()).device
    x = torch.tensor(X, dtype=torch.float64, device=device, requires_grad=True)
    wt = torch.tensor(w, dtype=torch.float64, device=device)
    loss = ot(x, wt, float(sigma), Tq=Tq, local_radius=None)
    loss.backward()
    return float(loss.detach().cpu()), x.grad.detach().cpu().numpy()


def resolve_target(sequence: str) -> tuple[str, str, int, int, str]:
    """Return (pdb_id, chain, start, end, sequence)."""
    key = sequence.strip().upper().replace("-", "").replace(",", "")
    if key in HLH_TARGETS:
        return HLH_TARGETS[key]
    if key in PEPTIDE_BY_SEQ:
        return PEPTIDE_BY_SEQ[key]
    # Allow pdb_id lookup for HLH / peptides.
    for pdb, ch, s0, s1, seq in list(HLH_TARGETS.values()) + list(TARGETS):
        if key == pdb.upper() or key == f"{pdb}_{seq}":
            return pdb, ch, s0, s1, seq
    known = sorted(set(list(HLH_TARGETS) + list(PEPTIDE_BY_SEQ)))
    raise SystemExit(f"unknown target {sequence!r}; try one of {known}")


def load_target(sequence: str) -> dict:
    pdb_id, chains, start, end, seq = resolve_target(sequence)
    got, chain = try_extract(pdb_id, chains, start, end, seq)
    if got is None:
        raise RuntimeError(f"failed to extract {pdb_id} {seq}")
    X = np.asarray(got["X"], dtype=np.float64)
    Z = np.asarray(got["Z"], dtype=np.float64)
    planar_raw = got.get("planar_groups")
    if planar_raw is None:
        planar: list = []
    else:
        planar = [list(map(int, g)) for g in planar_raw]
    swaps = np.asarray(got.get("swaps", np.zeros((0, 2))), dtype=np.int64).reshape(-1, 2)
    n = X.shape[0]
    identity = np.arange(n, dtype=np.int64)
    aut = [identity.copy()]
    for a, b in swaps:
        p = identity.copy()
        p[int(a)], p[int(b)] = int(b), int(a)
        aut.append(p)
    return {
        "X": X,
        "W": Z / Z.sum(),
        "Z": Z,
        "elements": Z.astype(np.int64),
        "names": [str(n) for n in got["names"]],
        "bonds": np.asarray(got["bonds"], dtype=np.int64),
        "rotatable_bonds": [
            (int(a), int(b))
            for a, b in np.asarray(got.get("rotatable_bonds", np.zeros((0, 2))))
        ],
        "chiral_centres": [
            tuple(int(x) for x in row)
            for row in np.asarray(got.get("chiral_centres", np.zeros((0, 4))))
        ],
        "planar_groups": planar,
        "automorphism_generators": aut,
        "chain": chain,
        "n_residues": len(seq),
        "sequence": seq,
        "pdb_id": pdb_id.upper(),
        "ref_id": f"{pdb_id.upper()}_{seq}",
    }


def build_scene(
    topo: dict,
    resolution: float = 2.0,
    *,
    device: torch.device | str = "cpu",
    n_dirs: int = N_DIRS,
):
    X_true = topo["X"].copy()
    w = topo["W"].copy()
    sig = float(sigma_from_resolution(resolution))
    R = molecular_radius(X_true)
    half = R + 5.0 * sig + 4.0
    n = int(np.ceil(2.0 * half / SPACING))
    if n % 2 == 0:
        n += 1
    n = int(min(n, 81))
    half = 0.5 * (n - 1) * SPACING
    NG = (n, n, n)
    T, org, sp = render_ortho(X_true, SPACING, NG, sig, w)
    dev = torch.device(device)
    n_dirs = int(n_dirs)
    ot = SlicedOT(
        torch.tensor(T, device=dev),
        org,
        torch.tensor(sp, device=dev),
        sig,
        SlicedOTConfig(
            n_dirs=n_dirs,
            dt=0.3,
            window=float(3.0 * half),
            map_cutoff=1e-7,
            backend="direct",
        ),
        device=dev,
    )
    return {
        "X_true": X_true,
        "w": w,
        "sigma": sig,
        "half": half,
        "T": T,
        "origin": org,
        "spacing": sp,
        "ot": ot,
        "NG": NG,
        "R": R,
        "resolution": float(resolution),
        "n_atoms": int(X_true.shape[0]),
        "device": str(dev),
        "n_dirs": n_dirs,
    }


def run_global_ot(
    scene,
    X0,
    *,
    w=None,
    lr=OT_LR,
    max_steps=MAX_STEPS,
    patience=PATIENCE,
    log_every: int = 25,
    label: str = "global",
):
    """Full-domain free-atom OT with numpy Adam (no localization).

    Pass ``w`` for overcomplete free OT; leave ``scene["w"]`` as chemical weights.
    """
    X_true = scene["X_true"]
    w = scene["w"] if w is None else np.asarray(w, dtype=np.float64)
    ot = scene["ot"]
    sig = scene["sigma"]
    device = scene.get("device")
    X = np.asarray(X0, dtype=np.float64).copy()
    if X.shape[0] != w.shape[0]:
        raise ValueError(
            f"weight length {w.shape[0]} != atom count {X.shape[0]}"
        )
    opt = Adam(X.shape, lr=lr)
    E0, _ = vg_ot(ot, X, w, sig, device=device)
    energies = [float(E0)]
    nn_rmsds = [nn_rmsd(X, X_true)]
    poses = [X.copy()]
    best_E = energies[0]
    stagnant = 0
    reason = "max_steps"
    t0 = time.perf_counter()
    print(
        f"  [{label}] step 0  OT={energies[0]:.6g}  NN-RMSD={nn_rmsds[0]:.4f} Å",
        flush=True,
    )
    for k in range(1, max_steps + 1):
        E, G = vg_ot(ot, X, w, sig, device=device)
        X = opt.step(X, G)
        poses.append(X.copy())
        energies.append(float(E))
        nn_rmsds.append(nn_rmsd(X, X_true))
        if E < best_E - max(1e-4, 1e-3 * abs(best_E)):
            best_E = float(E)
            stagnant = 0
        else:
            stagnant += 1
        if k % log_every == 0 or stagnant >= patience:
            dt = time.perf_counter() - t0
            print(
                f"  [{label}] step {k}  OT={energies[-1]:.6g}  "
                f"NN-RMSD={nn_rmsds[-1]:.4f} Å  "
                f"({dt:.1f}s, stagnant={stagnant})",
                flush=True,
            )
        if stagnant >= patience:
            reason = "ot_loss_plateau"
            break
    return {
        "X_final": X,
        "poses": np.asarray(poses, dtype=np.float64),
        "energies": np.asarray(energies, dtype=np.float64),
        "nn_rmsds": np.asarray(nn_rmsds, dtype=np.float64),
        "local_radii": np.full(len(energies), np.nan),
        "n_steps": len(energies) - 1,
        "stop_reason": reason,
        "elapsed_s": time.perf_counter() - t0,
        "best_E": float(best_E),
    }


def run_torch_sgd_ot(
    scene,
    X0,
    *,
    w=None,
    lr: float = 0.1,
    max_steps=MAX_STEPS,
    patience=PATIENCE,
    log_every: int = 25,
    label: str = "sgd",
):
    """Post-landing refine with ``torch.optim.SGD`` on atom coordinates."""
    X_true = scene["X_true"]
    w = scene["w"] if w is None else np.asarray(w, dtype=np.float64)
    ot = scene["ot"]
    sig = float(scene["sigma"])
    device = scene.get("device")
    X0 = np.asarray(X0, dtype=np.float64)
    if X0.shape[0] != w.shape[0]:
        raise ValueError(
            f"weight length {w.shape[0]} != atom count {X0.shape[0]}"
        )
    x = torch.nn.Parameter(
        torch.tensor(X0, dtype=torch.float64, device=device)
    )
    wt = torch.tensor(w, dtype=torch.float64, device=device)
    opt = torch.optim.SGD([x], lr=float(lr))

    with torch.no_grad():
        E0 = float(ot(x, wt, sig).detach().cpu())
    X = x.detach().cpu().numpy()
    energies = [E0]
    nn_rmsds = [nn_rmsd(X, X_true)]
    poses = [X.copy()]
    best_E = energies[0]
    stagnant = 0
    reason = "max_steps"
    t0 = time.perf_counter()
    print(
        f"  [{label}] step 0  OT={energies[0]:.6g}  NN-RMSD={nn_rmsds[0]:.4f} Å  "
        f"(torch.optim.SGD lr={lr:g})",
        flush=True,
    )
    for k in range(1, max_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = ot(x, wt, sig)
        loss.backward()
        opt.step()
        E = float(loss.detach().cpu())
        X = x.detach().cpu().numpy()
        poses.append(X.copy())
        energies.append(E)
        nn_rmsds.append(nn_rmsd(X, X_true))
        if E < best_E - max(1e-4, 1e-3 * abs(best_E)):
            best_E = E
            stagnant = 0
        else:
            stagnant += 1
        if k % log_every == 0 or stagnant >= patience:
            dt = time.perf_counter() - t0
            print(
                f"  [{label}] step {k}  OT={energies[-1]:.6g}  "
                f"NN-RMSD={nn_rmsds[-1]:.4f} Å  "
                f"({dt:.1f}s, stagnant={stagnant})",
                flush=True,
            )
        if stagnant >= patience:
            reason = "ot_loss_plateau"
            break
    return {
        "X_final": X,
        "poses": np.asarray(poses, dtype=np.float64),
        "energies": np.asarray(energies, dtype=np.float64),
        "nn_rmsds": np.asarray(nn_rmsds, dtype=np.float64),
        "local_radii": np.full(len(energies), np.nan),
        "n_steps": len(energies) - 1,
        "stop_reason": reason,
        "elapsed_s": time.perf_counter() - t0,
        "best_E": float(best_E),
    }


def sphere_centers(X: np.ndarray, R: float, *, pad: float | None = None) -> np.ndarray:
    """Grid of sphere centres covering the atom cloud AABB (+ pad).

    Centre spacing = R so neighbouring balls of radius R overlap.
    Centres with no atom inside their ball are dropped.
    """
    X = np.asarray(X, dtype=np.float64)
    pad = float(0.5 * R if pad is None else pad)
    lo = X.min(axis=0) - pad
    hi = X.max(axis=0) + pad
    step = float(R)
    axes = []
    for a in range(3):
        if hi[a] <= lo[a]:
            axes.append(np.array([0.5 * (lo[a] + hi[a])]))
            continue
        n = max(1, int(np.ceil((hi[a] - lo[a]) / step)) + 1)
        axes.append(np.linspace(lo[a], hi[a], n))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    # Keep only centres that cover at least one atom.
    d2 = ((grid[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    keep = (d2 <= (R * R)).any(axis=1)
    kept = grid[keep]
    return kept if len(kept) else grid[:1]


def _box_target_spectrum(ot: SlicedOT, center: torch.Tensor, R: float, sigma: float):
    """Unbalanced Tq from voxels inside the ball (centre, R)."""
    V_abs = ot.V_vox + ot.centre
    d2 = ((V_abs - center) ** 2).sum(-1)
    mv = ot.m_vox * (d2 <= (R * R)).to(ot.m_vox.dtype)
    if float(mv.sum()) <= 0.0:
        return None
    Tq = ot._structure_factor_from_proj(ot.vox_proj, mv, ot.qk, ot.cdtype)
    extra2 = max(float(sigma) ** 2 - ot.sigma_data ** 2, 0.0)
    if extra2 > 0.0:
        blur = torch.exp(-2 * math.pi ** 2 * extra2 * ot.qk ** 2).to(ot.cdtype)
        Tq = Tq * blur
    return Tq


def gather_sphere_gradients(
    ot: SlicedOT,
    X: np.ndarray,
    w: np.ndarray,
    sigma: float,
    centers: np.ndarray,
    R: float,
    device: torch.device | str,
):
    """Local OT on each overlapping sphere; accumulate then average grads."""
    dev = torch.device(device)
    x = torch.tensor(X, dtype=torch.float64, device=dev, requires_grad=True)
    wt = torch.tensor(w, dtype=torch.float64, device=dev)
    G_acc = torch.zeros_like(x)
    n_acc = torch.zeros(x.shape[0], dtype=torch.float64, device=dev)
    E_sum = 0.0
    n_active = 0
    R = float(R)

    for c in centers:
        c_t = torch.as_tensor(c, dtype=torch.float64, device=dev)
        d2_a = ((x.detach() - c_t) ** 2).sum(-1)
        atom_m = d2_a <= (R * R)
        if int(atom_m.sum().item()) == 0:
            continue
        Tq = _box_target_spectrum(ot, c_t, R, sigma)
        if Tq is None:
            continue
        w_box = wt * atom_m.to(wt.dtype)
        if float(w_box.sum().item()) <= 0.0:
            continue
        loss = ot(x, w_box, float(sigma), Tq=Tq, local_radius=None)
        (g,) = torch.autograd.grad(loss, x, retain_graph=False)
        G_acc = G_acc + g
        n_acc = n_acc + atom_m.to(n_acc.dtype)
        E_sum += float(loss.detach().cpu())
        n_active += 1

    if n_active == 0:
        return float("nan"), np.zeros_like(X), 0
    G = (G_acc / n_acc.clamp_min(1.0).unsqueeze(-1)).detach().cpu().numpy()
    return E_sum / n_active, G, n_active


def run_overlapping_spheres(
    scene,
    X0,
    *,
    w=None,
    r0: float,
    r_min: float,
    lr=OT_LR,
    max_steps_per_scale: int = 80,
    patience: int = 25,
    log_every: int = 5,
    label: str = "spheres",
):
    """Multiscale overlapping-sphere OT after a global settle."""
    X_true = scene["X_true"]
    w = scene["w"] if w is None else np.asarray(w, dtype=np.float64)
    ot = scene["ot"]
    sig = scene["sigma"]
    device = scene.get("device")
    X = np.asarray(X0, dtype=np.float64).copy()
    if X.shape[0] != w.shape[0]:
        raise ValueError(
            f"weight length {w.shape[0]} != atom count {X.shape[0]}"
        )
    opt = Adam(X.shape, lr=lr)

    poses = [X.copy()]
    energies = []
    nn_rmsds = [nn_rmsd(X, X_true)]
    radii = []
    t0 = time.perf_counter()
    R = float(r0)
    scale = 0
    total_steps = 0
    best_E_global = float("inf")
    best_nn = float(nn_rmsds[0])
    X_best_nn = X.copy()

    while R >= float(r_min) - 1e-12:
        centers = sphere_centers(X, R)
        print(
            f"  [{label}] scale {scale}  R={R:.3f} Å  "
            f"n_spheres={len(centers)}  "
            f"cover=[{X.min(0)} … {X.max(0)}]",
            flush=True,
        )
        best_E = float("inf")
        stagnant = 0
        scale_best_nn = float(nn_rmsd(X, X_true))
        X_scale_best = X.copy()
        for k in range(1, max_steps_per_scale + 1):
            centers = sphere_centers(X, R)
            E, G, n_active = gather_sphere_gradients(
                ot, X, w, sig, centers, R, device,
            )
            if n_active == 0:
                print(f"  [{label}] no active spheres at R={R:.3f}; stop scale",
                      flush=True)
                break
            X = opt.step(X, G)
            total_steps += 1
            nn = nn_rmsd(X, X_true)
            poses.append(X.copy())
            energies.append(float(E))
            nn_rmsds.append(nn)
            radii.append(float(R))
            if nn < scale_best_nn:
                scale_best_nn = nn
                X_scale_best = X.copy()
            if nn < best_nn:
                best_nn = nn
                X_best_nn = X.copy()
            if E < best_E - max(1e-4, 1e-3 * abs(best_E if np.isfinite(best_E) else 1.0)):
                best_E = float(E)
                stagnant = 0
            else:
                stagnant += 1
            if k % log_every == 0 or stagnant >= patience:
                dt = time.perf_counter() - t0
                print(
                    f"  [{label}] R={R:.3f} step {k}  "
                    f"OT_local={E:.6g}  NN-RMSD={nn:.4f} Å  "
                    f"spheres={n_active}/{len(centers)}  "
                    f"({dt:.1f}s, stagnant={stagnant})",
                    flush=True,
                )
            if stagnant >= patience:
                break
        # Carry the best-NN pose of this scale into the next (finer) radius.
        if scale_best_nn < nn_rmsd(X, X_true) - 1e-12:
            print(
                f"  [{label}] restore scale-best NN={scale_best_nn:.4f} Å "
                f"(was {nn_rmsd(X, X_true):.4f})",
                flush=True,
            )
            X = X_scale_best.copy()
            poses.append(X.copy())
            energies.append(float("nan"))
            nn_rmsds.append(scale_best_nn)
            radii.append(float(R))
            opt = Adam(X.shape, lr=lr)  # fresh momentum after restore
        best_E_global = min(best_E_global, best_E)
        R *= SHRINK
        scale += 1

    if not energies:
        # No local steps; still record diagnostics.
        E_g, _ = vg_ot(ot, X, w, sig, device=device)
        energies = [E_g]
        radii = [float(r0)]

    return {
        "X_final": X,
        "X_best_nn": X_best_nn,
        "poses": np.asarray(poses, dtype=np.float64),
        "energies": np.asarray(energies, dtype=np.float64),
        "nn_rmsds": np.asarray(nn_rmsds, dtype=np.float64),
        "local_radii": np.asarray(radii, dtype=np.float64),
        "n_steps": int(total_steps),
        "stop_reason": "r_min",
        "elapsed_s": time.perf_counter() - t0,
        "best_E": float(best_E_global) if np.isfinite(best_E_global) else float("nan"),
        "best_nn": float(best_nn),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sequence", type=str, default="1ZDD",
        help="Peptide sequence (e.g. AFSSFN) or target id (1ZDD).",
    )
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--lr", type=float, default=OT_LR)
    ap.add_argument(
        "--device", type=str, default="auto",
        help="'auto' (CUDA if available), 'cpu', or a torch device string.",
    )
    ap.add_argument(
        "--sphere-r0", type=float, default=None,
        help="Initial sphere radius (Å). Default: 1.5 × molecular radius.",
    )
    ap.add_argument(
        "--sphere-r-min", type=float, default=2.0,
        help="Stop shrinking spheres below this radius (Å).",
    )
    ap.add_argument(
        "--sphere-steps", type=int, default=40,
        help="Max gather→update cycles per radius scale.",
    )
    ap.add_argument(
        "--sphere-patience", type=int, default=15,
        help="Plateau patience within a radius scale.",
    )
    ap.add_argument(
        "--skip-local", action="store_true",
        help="Run global OT only.",
    )
    ap.add_argument(
        "--n-dirs", type=int, default=N_DIRS,
        help="Number of slice directions L.",
    )
    ap.add_argument(
        "--atom-factor", type=float, default=1.0,
        help="Multiply free-atom count vs true structure (map unchanged). "
             "Model weights are uniform 1/N.",
    )
    ap.add_argument(
        "--sgd-lr", type=float, default=None,
        help="After the global Adam land, continue with torch.optim.SGD at this "
             "lr (e.g. 0.1). Default: off.",
    )
    ap.add_argument(
        "--sgd-steps", type=int, default=MAX_STEPS,
        help="Max SGD steps after landing (default: same as --max-steps).",
    )
    ap.add_argument(
        "--sgd-patience", type=int, default=PATIENCE,
        help="OT-loss plateau patience for the post-landing SGD phase.",
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"extracting {args.sequence} …", flush=True)
    topo = load_target(args.sequence)
    sequence = topo["sequence"]
    print(
        f"  {topo['ref_id']}  chain={topo['chain']}  "
        f"residues={topo['n_residues']}  atoms={len(topo['X'])}",
        flush=True,
    )

    scene = build_scene(
        topo, resolution=args.resolution, device=device, n_dirs=args.n_dirs,
    )
    r0 = float(
        1.5 * scene["R"] if args.sphere_r0 is None else args.sphere_r0
    )
    atom_factor = float(args.atom_factor)
    if atom_factor <= 0.0:
        raise SystemExit("--atom-factor must be > 0")
    n_true = scene["n_atoms"]
    n_model = int(round(atom_factor * n_true))
    if n_model < 1:
        raise SystemExit("atom-factor yields zero free atoms")
    # Keep scene["w"] / sigma as chemical + resolution; free OT uses w_free.
    w_chem = np.asarray(scene["w"], dtype=np.float64).copy()
    w_free = (
        w_chem if n_model == n_true
        else np.full(n_model, 1.0 / n_model, dtype=np.float64)
    )

    print(
        f"device={scene['device']}  "
        f"map {args.resolution:.2f} Å  σ={scene['sigma']:.3f}  "
        f"grid={scene['NG'][0]}³  spacing={SPACING}  "
        f"R_mol={scene['R']:.2f} Å  half={scene['half']:.2f} Å  "
        f"n_dirs={scene['n_dirs']}  "
        f"atoms={n_model} (true={n_true}, ×{atom_factor:g})",
        flush=True,
    )

    rng = np.random.default_rng(int(args.seed))
    half = scene["half"]
    X0 = scene["X_true"].mean(0) + rng.uniform(-half, half, size=(n_model, 3))
    nn0 = nn_rmsd(X0, scene["X_true"])
    print(
        f"uniform start  NN-RMSD={nn0:.4f} Å  seed={args.seed}  "
        f"(Hungarian true←model)",
        flush=True,
    )

    print("\n=== Phase 1: global OT ===", flush=True)
    global_res = run_global_ot(
        scene, X0, w=w_free, lr=args.lr, max_steps=args.max_steps,
        patience=PATIENCE, label="global",
    )
    stages = [{"name": "global", "X_start": X0, "result": global_res}]
    X_cur = global_res["X_final"].copy()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.resolution:g}".replace(".", "p")
    stem = topo["ref_id"].lower()
    af_tag = f"_x{atom_factor:g}" if abs(atom_factor - 1.0) > 1e-12 else ""
    sgd_tag = f"_sgd{args.sgd_lr:g}" if args.sgd_lr is not None else ""
    out_path = OUT / (
        f"{stem}_free_ot_{tag}A_L{scene['n_dirs']}{af_tag}{sgd_tag}_seed{args.seed}.npz"
    )
    # Checkpoint after global so a killed local phase still leaves a usable pose.
    np.savez_compressed(
        out_path,
        X_true=scene["X_true"],
        X0=X0,
        X_after_global=global_res["X_final"],
        X_final=global_res["X_final"],
        poses=global_res["poses"],
        nn_rmsds=global_res["nn_rmsds"],
        energies=global_res["energies"],
        stage_names=np.array(["global"]),
        W=w_chem,
        W_free=w_free,
        origin=scene["origin"],
        spacing=scene["spacing"],
        NG=np.asarray(scene["NG"]),
        resolution=np.array(args.resolution),
        sigma=np.array(scene["sigma"]),
        seed=np.array(args.seed),
        device=np.array(scene["device"]),
        sequence=np.array(sequence),
        names=np.array(topo["names"]),
        Z=topo["Z"],
        bonds=topo["bonds"],
        global_nn_final=np.array(global_res["nn_rmsds"][-1]),
        checkpoint=np.array("after_global"),
        ref_id=np.array(topo["ref_id"]),
        n_dirs=np.array(scene["n_dirs"]),
        atom_factor=np.array(atom_factor),
        n_model=np.array(n_model),
        n_true=np.array(n_true),
    )
    print(f"  checkpoint → {out_path}", flush=True)

    if not args.skip_local:
        print(
            f"\n=== Phase 2: overlapping spheres  "
            f"R0={r0:.3f} → Rmin={args.sphere_r_min:g}  "
            f"shrink=1/√3 ===",
            flush=True,
        )
        local_res = run_overlapping_spheres(
            scene,
            X_cur,
            w=w_free,
            r0=r0,
            r_min=args.sphere_r_min,
            lr=args.lr,
            max_steps_per_scale=args.sphere_steps,
            patience=args.sphere_patience,
            label="spheres",
        )
        stages.append({"name": "spheres", "X_start": X_cur, "result": local_res})
        X_cur = local_res["X_final"].copy()

    if args.sgd_lr is not None:
        print(
            f"\n=== Post-landing torch.optim.SGD  lr={args.sgd_lr:g}  "
            f"max_steps={args.sgd_steps} ===",
            flush=True,
        )
        sgd_res = run_torch_sgd_ot(
            scene, X_cur, w=w_free,
            lr=float(args.sgd_lr),
            max_steps=int(args.sgd_steps),
            patience=int(args.sgd_patience),
            label="sgd",
        )
        stages.append({"name": "sgd", "X_start": X_cur, "result": sgd_res})
        X_cur = sgd_res["X_final"].copy()

    prune = None
    if n_model != n_true:
        print("\n=== Ghost prune (geometry + L1) ===", flush=True)
        namer = Namer(
            scene["X_true"],
            topo["elements"],
            topo["bonds"],
            rotatable_bonds=topo["rotatable_bonds"],
            chiral_centres=topo["chiral_centres"],
            planar_groups=topo["planar_groups"],
            automorphisms=topo["automorphism_generators"],
        )
        l1 = L1Diff3D(
            scene["T"], scene["origin"], scene["spacing"], scene["sigma"],
        )
        X_prior = scene["X_true"] - scene["X_true"].mean(0) + X_cur.mean(0)
        prune = prune_ghosts(
            X_cur,
            namer=namer,
            X_prior=X_prior,
            l1_oracle=l1,
            w_chem=w_chem,
            sigma=float(scene["sigma"]),
        )
        print(
            f"  kept {n_true}/{n_model}  ghosts={prune.ghost_idx.size}  "
            f"L1={prune.l1:.6g}  restr_rms={prune.restraint_rms:.4f} Å  "
            f"score={prune.score:.6g}  models={prune.n_models}  "
            f"σ_reset={prune.sigma:.4f}",
            flush=True,
        )

    last = stages[-1]["result"]
    nn_last = last["nn_rmsds"]
    best_k = int(np.argmin(nn_last))
    X_best_nn = last.get("X_best_nn", last["X_final"])
    if "X_best_nn" not in last:
        # Global-only: take min-NN pose along the trajectory.
        X_best_nn = last["poses"][int(np.argmin(last["nn_rmsds"]))]

    all_poses, all_nn, all_E, all_R, stage_breaks = [], [], [], [], []
    for st in stages:
        r = st["result"]
        stage_breaks.append(sum(len(p) for p in all_poses))
        all_poses.append(r["poses"])
        all_nn.append(r["nn_rmsds"])
        rad = r.get("local_radii", np.full(len(r["nn_rmsds"]), np.nan))
        if len(rad) < len(r["nn_rmsds"]):
            rad = np.concatenate(
                [np.full(len(r["nn_rmsds"]) - len(rad), np.nan), rad]
            )
        all_R.append(rad)
    poses_cat = np.concatenate(all_poses, axis=0)
    nn_cat = np.concatenate(all_nn, axis=0)
    E_parts = []
    for st in stages:
        r = st["result"]
        e = r["energies"]
        if len(e) == len(r["nn_rmsds"]):
            E_parts.append(e)
        elif len(e) + 1 == len(r["nn_rmsds"]):
            E_parts.append(np.concatenate([[e[0] if len(e) else np.nan], e]))
        else:
            pad = np.full(len(r["nn_rmsds"]) - len(e), np.nan)
            E_parts.append(np.concatenate([pad, e]))
    E_cat = np.concatenate(E_parts, axis=0)
    R_cat = np.concatenate(all_R, axis=0)

    save_kw = dict(
        X_true=scene["X_true"],
        X0=X0,
        X_after_global=global_res["X_final"],
        X_final=last["X_final"],
        X_best_nn=np.asarray(X_best_nn, dtype=np.float64),
        poses=poses_cat,
        nn_rmsds=nn_cat,
        energies=E_cat,
        local_radius_schedule=R_cat,
        stage_breaks=np.asarray(stage_breaks, dtype=np.int64),
        stage_names=np.array([st["name"] for st in stages]),
        W=w_chem,
        W_free=w_free,
        origin=scene["origin"],
        spacing=scene["spacing"],
        NG=np.asarray(scene["NG"]),
        resolution=np.array(args.resolution),
        sigma=np.array(scene["sigma"]),
        seed=np.array(args.seed),
        device=np.array(scene["device"]),
        sequence=np.array(sequence),
        stop_reason=np.array(last["stop_reason"]),
        sphere_r0=np.array(r0),
        sphere_r_min=np.array(args.sphere_r_min),
        sphere_shrink=np.array(SHRINK),
        names=np.array(topo["names"]),
        Z=topo["Z"],
        bonds=topo["bonds"],
        global_nn_final=np.array(global_res["nn_rmsds"][-1]),
        global_best_E=np.array(global_res["best_E"]),
        final_best_E=np.array(last["best_E"]),
        final_best_nn=np.array(nn_cat.min()),
        checkpoint=np.array("final"),
        ref_id=np.array(topo["ref_id"]),
        n_dirs=np.array(scene["n_dirs"]),
        atom_factor=np.array(atom_factor),
        n_model=np.array(n_model),
        n_true=np.array(n_true),
    )
    if prune is not None:
        save_kw.update(
            X_named=prune.Y_named,
            ghost_mask=prune.ghost_mask,
            kept_idx=prune.kept_idx,
            ghost_idx=prune.ghost_idx,
            prune_l1=np.array(prune.l1),
            prune_restr_rms=np.array(prune.restraint_rms),
            prune_score=np.array(prune.score),
            n_prune_models=np.array(prune.n_models),
            sigma_reset=np.array(prune.sigma),
        )
    np.savez_compressed(out_path, **save_kw)

    print()
    print("--- summary ---")
    for st in stages:
        r = st["result"]
        print(
            f"  {st['name']}: stop={r['stop_reason']} steps={r['n_steps']}  "
            f"OT_best={r['best_E']:.6g}  "
            f"NN_final={r['nn_rmsds'][-1]:.4f} Å  "
            f"NN_best={r['nn_rmsds'].min():.4f} Å",
            flush=True,
        )
    print(f"NN-RMSD start  : {nn_cat[0]:.4f} Å")
    print(f"NN-RMSD global : {global_res['nn_rmsds'][-1]:.4f} Å")
    print(f"NN-RMSD final  : {nn_last[-1]:.4f} Å")
    print(f"NN-RMSD best   : {nn_cat.min():.4f} Å  "
          f"(last-stage best idx {best_k})")
    if prune is not None:
        print(
            f"prune: ghosts={prune.ghost_idx.size}  L1={prune.l1:.6g}  "
            f"restr_rms={prune.restraint_rms:.4f} Å  σ={prune.sigma:.4f}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
