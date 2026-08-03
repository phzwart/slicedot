#!/usr/bin/env python3
"""EM-style local remodel: full density, fixed atoms + free stretch.

Unlike the omit/Δρ demo, the target is the **full** map ρ_obs.  Fixed atoms
still enter the model (so free atoms are not sucked into already-explained
density) but only the free stretch is optimised.

Pipeline
--------
1. Build ρ_obs from the true full model (optional molecular crop).
2. Hold all atoms fixed except residues ``[r0, r0+n)``.
3. Jitter / displace the free stretch as the remodel starting guess.
4. Sliced OT (optional Gabor) on the composite model; gradients only on free.
5. Name free stretch → geometry → peptide-link idealise to fixed flanks.

Usage
-----
  uv run python run_1zdd_em_remodel.py --r0 8 --n-res 3 --device cuda
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from slicedot import Geometry, Namer, SlicedOT, SlicedOTConfig, WindowedSlicedOT

from run_1zdd_free_ot import (
    OUT,
    SPACING,
    Adam,
    load_target,
    molecular_radius,
    nn_rmsd,
    render_ortho,
)
from run_1zdd_omit_gabor_refine import crop_diff_map
from run_1zdd_omit_rebuild import (
    _res_groups,
    _res_order,
    kabsch,
    link_idealise,
)

torch.set_default_dtype(torch.float64)


def vg_composite(
    ot,
    X_fixed: np.ndarray,
    X_free: np.ndarray,
    w_fixed: np.ndarray,
    w_free: np.ndarray,
    sigma: float,
    device,
):
    """OT value + grad w.r.t. free atoms only; fixed coords are detached."""
    xf = torch.tensor(X_fixed, dtype=torch.float64, device=device)
    xr = torch.tensor(X_free, dtype=torch.float64, device=device, requires_grad=True)
    wf = torch.tensor(w_fixed, dtype=torch.float64, device=device)
    wr = torch.tensor(w_free, dtype=torch.float64, device=device)
    x = torch.cat([xf.detach(), xr], dim=0)
    w = torch.cat([wf, wr], dim=0)
    loss = ot(x, w, float(sigma))
    loss.backward()
    return float(loss.detach().cpu()), xr.grad.detach().cpu().numpy()


def vg_gabor_composite(
    win: WindowedSlicedOT,
    X_fixed: np.ndarray,
    X_free: np.ndarray,
    w_fixed: np.ndarray,
    w_free: np.ndarray,
    sigma: float,
    device,
):
    xf = torch.tensor(X_fixed, dtype=torch.float64, device=device)
    xr = torch.tensor(X_free, dtype=torch.float64, device=device, requires_grad=True)
    wf = torch.tensor(w_fixed, dtype=torch.float64, device=device)
    wr = torch.tensor(w_free, dtype=torch.float64, device=device)
    x = torch.cat([xf.detach(), xr], dim=0)
    w = torch.cat([wf, wr], dim=0)
    loss = win(x, w, float(sigma))
    loss.backward()
    return float(loss.detach().cpu()), xr.grad.detach().cpu().numpy()


def run_free_ot(
    score_fn,
    X_fixed,
    X_free0,
    w_fixed,
    w_free,
    sigma,
    device,
    *,
    X_free_true,
    lr,
    steps,
    patience,
    label,
    s_schedule=None,
):
    """Adam on free atoms only.  ``score_fn`` returns (E, G_free)."""
    X = np.asarray(X_free0, dtype=np.float64).copy()
    opt = Adam(X.shape, lr=lr)
    nn0 = nn_rmsd(X, X_free_true)
    nns = [nn0]
    Es = []
    best_nn = nn0
    best_X = X.copy()
    stagnant = 0
    t0 = time.perf_counter()
    print(f"\n=== {label}  lr={lr:g}  steps≤{steps} ===", flush=True)
    print(f"  start free NN={nn0:.4f} Å", flush=True)
    for k in range(steps):
        if s_schedule is not None:
            s_schedule(k, steps)
        E, G = score_fn(X_fixed, X, w_fixed, w_free, sigma, device)
        X = opt.step(X, G)
        nn = nn_rmsd(X, X_free_true)
        Es.append(E)
        nns.append(nn)
        if nn < best_nn - 1e-5:
            best_nn = nn
            best_X = X.copy()
            stagnant = 0
        else:
            stagnant += 1
        if k % 25 == 0 or k + 1 == steps or stagnant >= patience:
            extra = ""
            if s_schedule is not None and hasattr(s_schedule, "s_max"):
                extra = f"  s_max={s_schedule.s_max:.2f}"
            print(
                f"  [{label} {k:4d}] E={E:.6g}  NN={nn:.4f}  "
                f"best={best_nn:.4f}{extra}  "
                f"({time.perf_counter() - t0:.1f}s, stagnant={stagnant})",
                flush=True,
            )
        if stagnant >= patience:
            print(f"  stop: NN plateau ({patience})", flush=True)
            break
    return {
        "X_final": X,
        "X_best_nn": best_X,
        "nn_rmsds": np.asarray(nns),
        "energies": np.asarray(Es),
        "nn_best": float(best_nn),
        "nn_final": float(nns[-1]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r0", type=int, default=8, help="First free residue (0-based).")
    ap.add_argument("--n-res", type=int, default=3, help="Length of free stretch.")
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--n-dirs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument(
        "--init",
        choices=("jitter", "rigid", "random"),
        default="jitter",
        help="How to displace the free stretch from truth before remodel.",
    )
    ap.add_argument("--jitter", type=float, default=1.5, help="Å noise for --init jitter.")
    ap.add_argument(
        "--gabor",
        action="store_true",
        help="After global land, continue with lean Gabor on free stretch.",
    )
    ap.add_argument("--gabor-steps", type=int, default=120)
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--gabor-dirs", type=int, default=16)
    ap.add_argument("--crop-pad", type=float, default=5.0)
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args()

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
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
    r0, n_free_res = int(args.r0), int(args.n_res)
    if r0 < 1 or r0 + n_free_res >= n_res:
        raise SystemExit(
            f"need flanks on both sides (r0≥1, r0+n_res≤{n_res - 1}); "
            f"got r0={r0} n_res={n_free_res}"
        )
    free_res = order[r0 : r0 + n_free_res]
    prev_res, next_res = order[r0 - 1], order[r0 + n_free_res]
    free_idx = np.array(
        sorted(i for r in free_res for i in groups[r].values()), dtype=np.int64,
    )
    fixed_mask = np.ones(X_true.shape[0], dtype=bool)
    fixed_mask[free_idx] = False
    fixed_idx = np.flatnonzero(fixed_mask)

    seq_free = topo["sequence"][r0 : r0 + n_free_res]
    print(
        f"EM remodel  free={free_res} seq={seq_free}  "
        f"n_free={free_idx.size}  n_fixed={fixed_idx.size}  "
        f"flanks {prev_res}…{next_res}  device={device}",
        flush=True,
    )

    from slicedot import sigma_from_resolution

    sig = float(sigma_from_resolution(args.resolution))
    R = molecular_radius(X_true)
    half = R + 5.0 * sig + 4.0
    n = int(np.ceil(2.0 * half / SPACING))
    if n % 2 == 0:
        n += 1
    n = int(min(n, 81))
    half = 0.5 * (n - 1) * SPACING
    NG = (n, n, n)

    # Full density (no difference map).
    T_full, org_full, sp = render_ortho(X_true, SPACING, NG, sig, W)
    if args.no_crop:
        T, org = T_full, org_full
        NG_use = NG
        crop_note = f"full {n}³"
    else:
        T, org, sp, NG_use = crop_diff_map(
            T_full, org_full, sp, X_true, pad=float(args.crop_pad),
        )
        crop_note = f"mol crop → {NG_use[0]}×{NG_use[1]}×{NG_use[2]}"
    print(
        f"ρ_obs {args.resolution:g} Å  σ={sig:.3f}  {crop_note}  (full density)",
        flush=True,
    )

    X_fixed = X_true[fixed_idx].copy()
    W_fixed = W[fixed_idx].copy()
    X_free_true = X_true[free_idx].copy()
    W_free = W[free_idx].copy()
    # Joint model weights must sum to 1 (same convention as render_ortho target).
    w_all = np.concatenate([W_fixed, W_free])
    w_all = w_all / w_all.sum()
    n_f = X_fixed.shape[0]
    w_fixed = w_all[:n_f]
    w_free = w_all[n_f:]

    names_free = [names[i] for i in free_idx]
    inv = {int(g): k for k, g in enumerate(free_idx)}
    bonds_free = np.array(
        [(inv[a], inv[b]) for a, b in bonds if a in inv and b in inv],
        dtype=np.int64,
    )
    Z_free = topo["Z"][free_idx].astype(np.int64)

    def _remap_pairs(pairs):
        out = []
        for pair in pairs:
            if all(int(x) in inv for x in pair):
                out.append(tuple(inv[int(x)] for x in pair))
        return out

    rot_free = _remap_pairs(topo["rotatable_bonds"])
    chi_free = _remap_pairs(topo["chiral_centres"])
    planar_free = [
        [inv[int(x)] for x in g]
        for g in topo["planar_groups"]
        if all(int(x) in inv for x in g)
    ]

    anchor_C = X_true[groups[prev_res]["C"]].copy()
    anchor_N = X_true[groups[next_res]["N"]].copy()
    idx_N_first = inv[groups[free_res[0]]["N"]]
    idx_C_last = inv[groups[free_res[-1]]["C"]]
    d_link0 = float(np.linalg.norm(
        X_true[groups[prev_res]["C"]] - X_true[groups[free_res[0]]["N"]]
    ))
    d_link1 = float(np.linalg.norm(
        X_true[groups[free_res[-1]]["C"]] - X_true[groups[next_res]["N"]]
    ))

    # Displace free stretch away from truth.
    rng = np.random.default_rng(args.seed)
    if args.init == "jitter":
        X_free0 = X_free_true + rng.normal(0.0, args.jitter, size=X_free_true.shape)
    elif args.init == "rigid":
        # Random small rigid motion about COM.
        com = X_free_true.mean(0)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = np.deg2rad(25.0)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        Rmat = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        X_free0 = (X_free_true - com) @ Rmat.T + com + rng.normal(0, 0.5, 3)
    else:
        com = X_free_true.mean(0)
        span = 0.5 * (X_free_true.max(0) - X_free_true.min(0)) + 2.0 * sig
        X_free0 = com + rng.uniform(-span, span, size=X_free_true.shape)

    print(
        f"init={args.init}  free NN={nn_rmsd(X_free0, X_free_true):.4f} Å  "
        f"true links {d_link0:.3f} / {d_link1:.3f} Å",
        flush=True,
    )

    half_win = 0.5 * max(NG_use) * float(np.atleast_1d(sp).ravel()[0])
    cfg = SlicedOTConfig(
        n_dirs=args.n_dirs, dt=0.3, window=float(3.0 * half_win),
        map_cutoff=1e-7, backend="direct",
    )
    ot = SlicedOT(
        torch.tensor(T, device=device),
        org,
        torch.tensor(sp, device=device),
        sig,
        cfg,
        device=device,
    )

    def score_global(Xf, Xr, wf, wr, sigma, dev):
        return vg_composite(ot, Xf, Xr, wf, wr, sigma, dev)

    global_res = run_free_ot(
        score_global, X_fixed, X_free0, w_fixed, w_free, sig, device,
        X_free_true=X_free_true, lr=args.lr, steps=args.max_steps,
        patience=args.patience, label="global-composite",
    )
    X_cur = global_res["X_best_nn"].copy()

    gabor_res = None
    if args.gabor:
        s_floor = 3.0 * sig
        s_max0 = max(6.0, 5.0 * sig)

        class _Sched:
            def __init__(self, win, s0, s1):
                self.win = win
                self.s0 = s0
                self.s1 = s1
                self.s_max = s0

            def __call__(self, k, steps):
                frac = k / max(steps - 1, 1)
                self.s_max = self.s0 * (1.0 - 0.7 * frac) + self.s1 * (0.7 * frac)
                self.s_max = max(self.s_max, self.s1 + 1e-6)
                self.win.s_max = self.s_max

        win = WindowedSlicedOT(
            torch.tensor(T, device=device),
            org,
            torch.tensor(sp, device=device),
            sig,
            s_range=(s_floor, s_max0),
            n_windows=args.n_windows,
            n_directions=args.gabor_dirs,
            pi_a="model",
            backend="direct",
            seed=args.seed,
            config=cfg,
            device=device,
            memory_warn_gb=1.5,
        )
        sched = _Sched(win, s_max0, s_floor)

        def score_gabor(Xf, Xr, wf, wr, sigma, dev):
            return vg_gabor_composite(win, Xf, Xr, wf, wr, sigma, dev)

        gabor_res = run_free_ot(
            score_gabor, X_fixed, X_cur, w_fixed, w_free, sig, device,
            X_free_true=X_free_true, lr=args.lr, steps=args.gabor_steps,
            patience=args.patience, label="gabor-composite", s_schedule=sched,
        )
        X_cur = gabor_res["X_best_nn"].copy()

    # Name + geometry + links (free stretch only).
    print("\n=== Name free stretch ===", flush=True)
    namer = Namer(
        X_free_true, Z_free, bonds_free,
        rotatable_bonds=rot_free, chiral_centres=chi_free, planar_groups=planar_free,
    )
    X_prior = X_free_true - X_free_true.mean(0) + X_cur.mean(0)
    asn = namer.assign(X_cur, X_prior)
    X_named = asn.Y_named.copy()
    print(
        f"  named  label-RMSD="
        f"{float(np.sqrt(((X_named - X_free_true) ** 2).sum(-1).mean())):.4f} Å  "
        f"restr={asn.restraint_rms:.4f}  flags={asn.flags}",
        flush=True,
    )

    geom = Geometry(
        X_free_true, bonds_free,
        rotatable_bonds=rot_free, chiral_centres=chi_free,
        planar_groups=planar_free, antibump=True,
    )
    X_geom, g_rms, g_it = geom.project(X_named, slack=0.2)
    print(
        f"\n=== Geometry  rms={g_rms:.4g}  iters={g_it}  "
        f"label-RMSD={float(np.sqrt(((X_geom - X_free_true) ** 2).sum(-1).mean())):.4f} Å ===",
        flush=True,
    )

    X_linked, link_info = link_idealise(
        X_geom,
        X_ref_omit=X_free_true,
        bonds_omit=bonds_free,
        anchor_C=anchor_C,
        anchor_N=anchor_N,
        idx_N_first=idx_N_first,
        idx_C_last=idx_C_last,
        d_CN=0.5 * (d_link0 + d_link1),
    )
    print(
        f"\n=== Link idealise ===\n"
        f"  |C({prev_res})–N|={link_info['link_CN_N']:.3f} Å (true {d_link0:.3f})\n"
        f"  |C–N({next_res})|={link_info['link_C_Nnext']:.3f} Å (true {d_link1:.3f})\n"
        f"  label-RMSD={link_info['label_rmsd']:.4f} Å",
        flush=True,
    )

    X_full = X_true.copy()
    X_full[free_idx] = X_linked

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.resolution:g}".replace(".", "p")
    out = OUT / (
        f"1zdd_em_remodel_r{r0}n{n_free_res}_{tag}A"
        f"{'_gabor' if args.gabor else ''}_seed{args.seed}.npz"
    )
    payload = dict(
        X_true=X_true,
        X_fixed=X_fixed,
        X_free_true=X_free_true,
        X_free0=X_free0,
        X_after_global=global_res["X_best_nn"],
        X_named=X_named,
        X_geom=X_geom,
        X_linked=X_linked,
        X_full_rebuilt=X_full,
        free_idx=free_idx,
        fixed_idx=fixed_idx,
        names=np.array(names),
        names_free=np.array(names_free),
        bonds_free=bonds_free,
        origin=org,
        spacing=sp,
        NG=np.asarray(NG_use),
        resolution=np.array(args.resolution),
        sigma=np.array(sig),
        seed=np.array(args.seed),
        r0=np.array(r0),
        n_free_res=np.array(n_free_res),
        free_sequence=np.array(seq_free),
        global_nn_rmsds=global_res["nn_rmsds"],
        global_nn_best=np.array(global_res["nn_best"]),
        link_CN_N=np.array(link_info["link_CN_N"]),
        link_C_Nnext=np.array(link_info["link_C_Nnext"]),
        label_rmsd=np.array(link_info["label_rmsd"]),
        init_nn=np.array(nn_rmsd(X_free0, X_free_true)),
        mode=np.array("em_full_density_composite"),
    )
    if gabor_res is not None:
        payload.update(
            X_after_gabor=gabor_res["X_best_nn"],
            gabor_nn_rmsds=gabor_res["nn_rmsds"],
            gabor_nn_best=np.array(gabor_res["nn_best"]),
        )
    np.savez_compressed(out, **payload)

    print("\n========== summary ==========")
    print(f"init free NN     : {nn_rmsd(X_free0, X_free_true):.4f} Å")
    print(f"after global OT  : {global_res['nn_best']:.4f} Å")
    if gabor_res is not None:
        print(f"after Gabor      : {gabor_res['nn_best']:.4f} Å")
    print(f"after name+geom+link: {link_info['label_rmsd']:.4f} Å")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
