#!/usr/bin/env python3
"""Omit a 3-residue stretch from 1ZDD and rebuild it into a difference map.

Pipeline
--------
1. Render ρ_obs from the full true model.
2. Delete residues ``[r0, r0+3)``; render ρ_calc from the fixed remainder.
3. Difference target  Δρ = max(ρ_obs − ρ_calc, 0), renormalised.
4. Free-atom OT of the omitted atoms into Δρ (Adam, then optional torch SGD).
5. Name against the omitted fragment topology (tiny Hungarian).
6. Geometry idealisation of the fragment with **fixed flanking anchors**
   (C of the preceding residue, N of the following) so peptide links close.

Usage
-----
  uv run python run_1zdd_omit_rebuild.py --r0 14 --device cuda
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares, linear_sum_assignment

from slicedot import Geometry, Namer, SlicedOT, SlicedOTConfig, sigma_from_resolution

from run_1zdd_free_ot import (
    OUT,
    SPACING,
    Adam,
    load_target,
    molecular_radius,
    nn_rmsd,
    render_ortho,
    vg_ot,
)

torch.set_default_dtype(torch.float64)

BB = ("N", "CA", "C", "O")


def _res_groups(names: list[str]) -> dict[str, dict[str, int]]:
    g: dict[str, dict[str, int]] = defaultdict(dict)
    for i, n in enumerate(names):
        r, a = str(n).split("_", 1)
        g[r][a] = i
    return g


def _res_order(groups: dict[str, dict[str, int]]) -> list[str]:
    return sorted(groups, key=lambda r: int(r[1:]))


def render_abs(X, sp, NG, sigma, weights):
    """Unnormalised ortho Gaussian map (same kernel as ``render_ortho``)."""
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.asarray(NG, dtype=np.float64) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG, dtype=np.float64)
    for p, w in zip(X, weights):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2.0 * sigma * sigma))
    return T, org, sp


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t with P @ R.T + t ≈ Q (row vectors)."""
    Pc = P.mean(0)
    Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = Qc - Pc @ R.T
    return R, t


def link_idealise(
    X_omit: np.ndarray,
    *,
    X_ref_omit: np.ndarray,
    bonds_omit: np.ndarray,
    anchor_C: np.ndarray,
    anchor_N: np.ndarray,
    idx_N_first: int,
    idx_C_last: int,
    d_CN: float = 1.33,
    w_link: float = 20.0,
    w_bond: float = 5.0,
    w_anchor_pos: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """Idealise omit fragment with fixed peptide-link anchors.

    Anchors stay fixed in space.  Moving atoms feel:
      * internal 1–2 bonds (to X_ref lengths)
      * link springs  |C_prev − N_first| and |C_last − N_next| → d_CN
      * soft unary pull toward a Kabsch-aligned reference (keeps register)
    """
    X0 = np.asarray(X_omit, dtype=np.float64).copy()
    X_ref = np.asarray(X_ref_omit, dtype=np.float64)
    R, t = kabsch(X_ref, X0)
    X_ref_al = X_ref @ R.T + t
    bond_d = {
        (int(a), int(b)) if a < b else (int(b), int(a)): float(
            np.linalg.norm(X_ref[a] - X_ref[b])
        )
        for a, b in bonds_omit
    }
    n = X0.shape[0]
    iN, iC = int(idx_N_first), int(idx_C_last)
    aC = np.asarray(anchor_C, dtype=np.float64)
    aN = np.asarray(anchor_N, dtype=np.float64)

    def pack(v):
        X = v.reshape(n, 3)
        res = []
        for (a, b), d0 in bond_d.items():
            res.append(w_bond * (np.linalg.norm(X[a] - X[b]) - d0))
        res.append(w_link * (np.linalg.norm(X[iN] - aC) - d_CN))
        res.append(w_link * (np.linalg.norm(X[iC] - aN) - d_CN))
        res.extend((w_anchor_pos * (X - X_ref_al).ravel()).tolist())
        return np.asarray(res, dtype=np.float64)

    sol = least_squares(pack, X0.ravel(), method="trf", ftol=1e-10, xtol=1e-10, gtol=1e-10)
    X = sol.x.reshape(n, 3)
    info = {
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "link_CN_N": float(np.linalg.norm(X[iN] - aC)),
        "link_C_Nnext": float(np.linalg.norm(X[iC] - aN)),
        "rmsd_to_true": float(nn_rmsd(X, X_ref)),  # same N, labelled
    }
    # labelled RMSD (not Hungarian)
    info["label_rmsd"] = float(np.sqrt(((X - X_ref) ** 2).sum(-1).mean()))
    return X, info


def run_ot(ot, X0, w, sigma, device, *, lr, max_steps, patience, label, optimizer):
    """Adam (numpy) or torch.optim.SGD loop against ``ot``."""
    X_true_frag = None  # filled by caller for logging only via nn vs start
    del X_true_frag
    X = np.asarray(X0, dtype=np.float64).copy()
    energies = []
    poses = [X.copy()]
    t0 = time.perf_counter()
    if optimizer == "adam":
        opt = Adam(X.shape, lr=lr)
        E, _ = vg_ot(ot, X, w, sigma, device=device)
        energies.append(E)
        best_E = E
        stagnant = 0
        print(f"  [{label}] step 0  OT={E:.6g}", flush=True)
        reason = "max_steps"
        for k in range(1, max_steps + 1):
            E, G = vg_ot(ot, X, w, sigma, device=device)
            X = opt.step(X, G)
            poses.append(X.copy())
            energies.append(E)
            if E < best_E - max(1e-4, 1e-3 * abs(best_E)):
                best_E = E
                stagnant = 0
            else:
                stagnant += 1
            if k % 25 == 0 or stagnant >= patience:
                print(
                    f"  [{label}] step {k}  OT={E:.6g}  "
                    f"({time.perf_counter() - t0:.1f}s, stagnant={stagnant})",
                    flush=True,
                )
            if stagnant >= patience:
                reason = "ot_loss_plateau"
                break
    elif optimizer == "sgd":
        x = torch.nn.Parameter(torch.tensor(X, dtype=torch.float64, device=device))
        wt = torch.tensor(w, dtype=torch.float64, device=device)
        opt = torch.optim.SGD([x], lr=float(lr))
        with torch.no_grad():
            E = float(ot(x, wt, float(sigma)).detach().cpu())
        energies.append(E)
        best_E = E
        stagnant = 0
        reason = "max_steps"
        print(
            f"  [{label}] step 0  OT={E:.6g}  (torch.optim.SGD lr={lr:g})",
            flush=True,
        )
        for k in range(1, max_steps + 1):
            opt.zero_grad(set_to_none=True)
            loss = ot(x, wt, float(sigma))
            loss.backward()
            opt.step()
            E = float(loss.detach().cpu())
            X = x.detach().cpu().numpy()
            poses.append(X.copy())
            energies.append(E)
            if E < best_E - max(1e-4, 1e-3 * abs(best_E)):
                best_E = E
                stagnant = 0
            else:
                stagnant += 1
            if k % 25 == 0 or stagnant >= patience:
                print(
                    f"  [{label}] step {k}  OT={E:.6g}  "
                    f"({time.perf_counter() - t0:.1f}s, stagnant={stagnant})",
                    flush=True,
                )
            if stagnant >= patience:
                reason = "ot_loss_plateau"
                break
    else:
        raise ValueError(optimizer)
    return {
        "X_final": X,
        "poses": np.asarray(poses),
        "energies": np.asarray(energies),
        "n_steps": len(energies) - 1,
        "stop_reason": reason,
        "best_E": float(best_E),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r0", type=int, default=14, help="First omitted residue (0-based).")
    ap.add_argument("--n-res", type=int, default=3, help="Length of omitted stretch.")
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--n-dirs", type=int, default=32)
    ap.add_argument("--atom-factor", type=float, default=1.0,
                    help="Overcomplete factor for omit free atoms only.")
    ap.add_argument("--lr", type=float, default=0.4, help="Adam lr for free OT.")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--sgd-lr", type=float, default=0.1,
                    help="Post-land torch SGD lr (None/negative to skip).")
    ap.add_argument("--sgd-steps", type=int, default=400)
    args = ap.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    print("loading 1ZDD …", flush=True)
    topo = load_target("1ZDD")
    X_true = topo["X"].copy()
    W = topo["W"].copy()
    names = [str(n) for n in topo["names"]]
    bonds = np.asarray(topo["bonds"], dtype=np.int64)
    groups = _res_groups(names)
    order = _res_order(groups)
    n_res = len(order)
    r0 = int(args.r0)
    n_omit_res = int(args.n_res)
    if r0 < 1 or r0 + n_omit_res >= n_res:
        raise SystemExit(
            f"--r0/--n-res must leave a flanking residue on each side "
            f"(r0≥1, r0+n_res≤{n_res - 1}); got r0={r0} n_res={n_omit_res}"
        )
    omit_res = order[r0: r0 + n_omit_res]
    prev_res = order[r0 - 1]
    next_res = order[r0 + n_omit_res]
    omit_idx = np.array(
        sorted(i for r in omit_res for i in groups[r].values()),
        dtype=np.int64,
    )
    fixed_mask = np.ones(X_true.shape[0], dtype=bool)
    fixed_mask[omit_idx] = False
    fixed_idx = np.flatnonzero(fixed_mask)

    seq_omit = topo["sequence"][r0: r0 + n_omit_res]
    print(
        f"omit residues {omit_res}  seq={seq_omit}  "
        f"n_atoms={omit_idx.size}  flanks {prev_res} … {next_res}  "
        f"device={device}",
        flush=True,
    )

    sig = float(sigma_from_resolution(args.resolution))
    R = molecular_radius(X_true)
    half = R + 5.0 * sig + 4.0
    n = int(np.ceil(2.0 * half / SPACING))
    if n % 2 == 0:
        n += 1
    n = int(min(n, 81))
    half = 0.5 * (n - 1) * SPACING
    NG = (n, n, n)

    # Absolute maps → positive difference density for the hole.
    T_full, org, sp = render_abs(X_true, SPACING, NG, sig, W)
    T_fix, _, _ = render_abs(X_true[fixed_idx], SPACING, NG, sig, W[fixed_idx])
    T_diff = np.maximum(T_full - T_fix, 0.0)
    mass = float(T_diff.sum())
    if mass <= 0:
        raise SystemExit("difference map has no positive mass")
    T_diff /= mass
    frac = mass / float(T_full.sum())
    print(
        f"map {args.resolution:g} Å  σ={sig:.3f}  grid={n}³  "
        f"Δρ mass fraction={frac:.3f} (expect ~{omit_idx.size / X_true.shape[0]:.3f})",
        flush=True,
    )

    X_omit_true = X_true[omit_idx].copy()
    W_omit = W[omit_idx].copy()
    W_omit /= W_omit.sum()
    names_omit = [names[i] for i in omit_idx]
    # Remap bonds into fragment-local indices.
    inv = {int(g): k for k, g in enumerate(omit_idx)}
    bonds_omit = np.array(
        [(inv[a], inv[b]) for a, b in bonds if a in inv and b in inv],
        dtype=np.int64,
    )
    # Local element / topology for Namer.
    Z_omit = topo["Z"][omit_idx].astype(np.int64)
    # Rotatable / chiral / planar restricted to omit atoms.
    def _remap_pairs(pairs):
        out = []
        for pair in pairs:
            if all(int(x) in inv for x in pair):
                out.append(tuple(inv[int(x)] for x in pair))
        return out

    rot_omit = _remap_pairs(topo["rotatable_bonds"])
    chi_omit = _remap_pairs(topo["chiral_centres"])
    planar_omit = []
    for g in topo["planar_groups"]:
        if all(int(x) in inv for x in g):
            planar_omit.append([inv[int(x)] for x in g])

    # Anchor atoms (fixed in lab frame).
    if "C" not in groups[prev_res] or "N" not in groups[next_res]:
        raise SystemExit("flanking residues missing C/N for peptide links")
    if "N" not in groups[omit_res[0]] or "C" not in groups[omit_res[-1]]:
        raise SystemExit("omit stretch missing terminal N/C")
    anchor_C = X_true[groups[prev_res]["C"]].copy()
    anchor_N = X_true[groups[next_res]["N"]].copy()
    idx_N_first = inv[groups[omit_res[0]]["N"]]
    idx_C_last = inv[groups[omit_res[-1]]["C"]]
    d_link0 = float(np.linalg.norm(
        X_true[groups[prev_res]["C"]] - X_true[groups[omit_res[0]]["N"]]
    ))
    d_link1 = float(np.linalg.norm(
        X_true[groups[omit_res[-1]]["C"]] - X_true[groups[next_res]["N"]]
    ))
    print(
        f"true link lengths: C({prev_res})–N={d_link0:.3f} Å  "
        f"C–N({next_res})={d_link1:.3f} Å",
        flush=True,
    )

    # Free OT into Δρ.
    n_true = X_omit_true.shape[0]
    n_model = int(round(float(args.atom_factor) * n_true))
    w_free = (
        W_omit if n_model == n_true
        else np.full(n_model, 1.0 / n_model, dtype=np.float64)
    )
    rng = np.random.default_rng(args.seed)
    com = X_omit_true.mean(0)
    span = 0.5 * (X_omit_true.max(0) - X_omit_true.min(0)) + 2.0 * sig
    X0 = com + rng.uniform(-span, span, size=(n_model, 3))

    ot = SlicedOT(
        torch.tensor(T_diff, device=device),
        org,
        torch.tensor(sp, device=device),
        sig,
        SlicedOTConfig(
            n_dirs=args.n_dirs, dt=0.3, window=float(3.0 * half),
            map_cutoff=1e-7, backend="direct",
        ),
        device=device,
    )
    print(
        f"\n=== Free OT into Δρ  N={n_model} (true omit={n_true}) ===",
        flush=True,
    )
    print(f"  start NN-RMSD={nn_rmsd(X0, X_omit_true):.4f} Å", flush=True)
    adam_res = run_ot(
        ot, X0, w_free, sig, device,
        lr=args.lr, max_steps=args.max_steps, patience=60,
        label="adam", optimizer="adam",
    )
    X_cur = adam_res["X_final"]
    print(f"  after Adam  NN-RMSD={nn_rmsd(X_cur, X_omit_true):.4f} Å", flush=True)

    sgd_res = None
    if args.sgd_lr is not None and args.sgd_lr > 0:
        print(
            f"\n=== Post-land torch.optim.SGD  lr={args.sgd_lr:g} ===",
            flush=True,
        )
        sgd_res = run_ot(
            ot, X_cur, w_free, sig, device,
            lr=args.sgd_lr, max_steps=args.sgd_steps, patience=80,
            label="sgd", optimizer="sgd",
        )
        X_cur = sgd_res["X_final"]
        print(f"  after SGD  NN-RMSD={nn_rmsd(X_cur, X_omit_true):.4f} Å", flush=True)

    # If overcomplete, keep Hungarian nearest to true omit (seed prune).
    if n_model != n_true:
        d2 = ((X_omit_true[:, None, :] - X_cur[None, :, :]) ** 2).sum(-1)
        _ri, cj = linear_sum_assignment(d2)
        X_cur = X_cur[cj].copy()
        print(
            f"  Hungarian keep {n_true}/{n_model}  "
            f"NN-RMSD={nn_rmsd(X_cur, X_omit_true):.4f} Å",
            flush=True,
        )

    # Name fragment (small).
    print("\n=== Name omitted fragment ===", flush=True)
    namer = Namer(
        X_omit_true,
        Z_omit,
        bonds_omit,
        rotatable_bonds=rot_omit,
        chiral_centres=chi_omit,
        planar_groups=planar_omit,
    )
    # Prior: COM-aligned true omit (no orientation cheat beyond translation).
    X_prior = X_omit_true - X_omit_true.mean(0) + X_cur.mean(0)
    asn = namer.assign(X_cur, X_prior)
    X_named = asn.Y_named.copy()
    print(
        f"  named  label-RMSD={float(np.sqrt(((X_named - X_omit_true)**2).sum(-1).mean())):.4f} Å  "
        f"restr={asn.restraint_rms:.4f}  unary={asn.unary_rms:.4f}  "
        f"flags={asn.flags}",
        flush=True,
    )

    # Internal geometry (fragment alone).
    geom = Geometry(
        X_omit_true,
        bonds_omit,
        rotatable_bonds=rot_omit,
        chiral_centres=chi_omit,
        planar_groups=planar_omit,
        antibump=True,
    )
    X_geom, g_rms, g_it = geom.project(X_named, slack=0.2)
    print(
        f"\n=== Geometry project (fragment)  rms={g_rms:.4g}  iters={g_it}  "
        f"label-RMSD={float(np.sqrt(((X_geom - X_omit_true)**2).sum(-1).mean())):.4f} Å ===",
        flush=True,
    )
    print(
        f"  links before idealise: "
        f"|C–N|={np.linalg.norm(X_geom[idx_N_first] - anchor_C):.3f}  "
        f"|C–N'|={np.linalg.norm(X_geom[idx_C_last] - anchor_N):.3f} Å",
        flush=True,
    )

    X_linked, link_info = link_idealise(
        X_geom,
        X_ref_omit=X_omit_true,
        bonds_omit=bonds_omit,
        anchor_C=anchor_C,
        anchor_N=anchor_N,
        idx_N_first=idx_N_first,
        idx_C_last=idx_C_last,
        d_CN=0.5 * (d_link0 + d_link1),
    )
    print(
        f"\n=== Link idealise (fixed flanks) ===\n"
        f"  |C({prev_res})–N|={link_info['link_CN_N']:.3f} Å  "
        f"(true {d_link0:.3f})\n"
        f"  |C–N({next_res})|={link_info['link_C_Nnext']:.3f} Å  "
        f"(true {d_link1:.3f})\n"
        f"  label-RMSD to omit truth={link_info['label_rmsd']:.4f} Å",
        flush=True,
    )

    # Reassemble full coordinates for export / viewer.
    X_full = X_true.copy()
    X_full[omit_idx] = X_linked
    label_rmsd_full = float(
        np.sqrt(((X_full[omit_idx] - X_omit_true) ** 2).sum(-1).mean())
    )

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.resolution:g}".replace(".", "p")
    out = OUT / (
        f"1zdd_omit_r{r0}n{n_omit_res}_{tag}A_L{args.n_dirs}"
        f"_x{args.atom_factor:g}_seed{args.seed}.npz"
    )
    np.savez_compressed(
        out,
        X_true=X_true,
        X_fixed=X_true[fixed_idx],
        X_omit_true=X_omit_true,
        X0=X0,
        X_after_adam=adam_res["X_final"],
        X_after_sgd=(
            np.zeros((0, 3), dtype=np.float64)
            if sgd_res is None
            else sgd_res["X_final"]
        ),
        X_named=X_named,
        X_geom=X_geom,
        X_linked=X_linked,
        X_full_rebuilt=X_full,
        omit_idx=omit_idx,
        fixed_idx=fixed_idx,
        names=np.array(names),
        names_omit=np.array(names_omit),
        bonds=bonds,
        bonds_omit=bonds_omit,
        T_diff=T_diff,
        origin=org,
        spacing=sp,
        NG=np.asarray(NG),
        resolution=np.array(args.resolution),
        sigma=np.array(sig),
        seed=np.array(args.seed),
        r0=np.array(r0),
        n_omit_res=np.array(n_omit_res),
        omit_sequence=np.array(seq_omit),
        anchor_C=anchor_C,
        anchor_N=anchor_N,
        idx_N_first=np.array(idx_N_first),
        idx_C_last=np.array(idx_C_last),
        link_CN_N=np.array(link_info["link_CN_N"]),
        link_C_Nnext=np.array(link_info["link_C_Nnext"]),
        label_rmsd_omit=np.array(link_info["label_rmsd"]),
        label_rmsd_full_omit=np.array(label_rmsd_full),
        diff_mass_fraction=np.array(frac),
        free_nn_final=np.array(nn_rmsd(X_cur, X_omit_true)),
        named_label_rmsd=np.array(
            float(np.sqrt(((X_named - X_omit_true) ** 2).sum(-1).mean()))
        ),
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
