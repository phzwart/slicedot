#!/usr/bin/env python3
"""1ZDD (Z34C helix–loop–helix): 2 Å map → free-atom OT from uniform box.

Starts from N atoms placed uniformly in the map box (same count/weights as
the reference), runs Adam on sliced W₁, then optionally shakes the settled
cloud with a Gaussian of given RMSD and re-runs OT to escape local minima.
Reports nearest-neighbour (Hungarian) RMSD only.

Usage
-----
  uv run python docs/paper/examples/leucine_3d_reach/run_1zdd_free_ot.py
  uv run python docs/paper/examples/leucine_3d_reach/run_1zdd_free_ot.py \\
      --shake-rmsd 1 --n-shakes 1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from slicedot import SlicedOT, SlicedOTConfig, sigma_from_resolution

from build_peptide_refs import try_extract

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

PDB_ID = "1ZDD"
SEQUENCE = "FNMQCQRRFYEALHDPNLNEEQRNAKIKSIRDDC"
# Deposited numbering (auth); extractor falls back to sequence scan if needed.
START_SEQ, END_SEQ = 6, 39

OT_LR = 0.4
MAX_STEPS = 1500
PATIENCE = 60
SPACING = 0.5  # Å
N_DIRS = 32


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


def vg_ot(ot: SlicedOT, X: np.ndarray, w: np.ndarray, sigma: float):
    x = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w, dtype=torch.float64)
    loss = ot(x, wt, float(sigma))
    loss.backward()
    return float(loss.detach()), x.grad.detach().numpy()


def load_1zdd() -> dict:
    got, chain = try_extract(PDB_ID, "A", START_SEQ, END_SEQ, SEQUENCE)
    if got is None:
        raise RuntimeError(f"failed to extract {PDB_ID} {SEQUENCE}")
    X = np.asarray(got["X"], dtype=np.float64)
    Z = np.asarray(got["Z"], dtype=np.float64)
    return {
        "X": X,
        "W": Z / Z.sum(),
        "Z": Z,
        "names": [str(n) for n in got["names"]],
        "bonds": np.asarray(got["bonds"], dtype=np.int64),
        "chain": chain,
        "n_residues": len(SEQUENCE),
        "sequence": SEQUENCE,
    }


def build_scene(topo: dict, resolution: float = 2.0):
    X_true = topo["X"].copy()
    w = topo["W"].copy()
    sig = float(sigma_from_resolution(resolution))
    R = molecular_radius(X_true)
    half = R + 5.0 * sig + 4.0
    n = int(np.ceil(2.0 * half / SPACING))
    if n % 2 == 0:
        n += 1
    n = int(min(n, 81))
    # Keep scatter box consistent with the actual grid extent.
    half = 0.5 * (n - 1) * SPACING
    NG = (n, n, n)
    T, org, sp = render_ortho(X_true, SPACING, NG, sig, w)
    ot = SlicedOT(
        torch.tensor(T),
        org,
        torch.tensor(sp),
        sig,
        SlicedOTConfig(
            n_dirs=N_DIRS,
            dt=0.3,
            window=float(3.0 * half),
            map_cutoff=1e-7,
            backend="direct",
        ),
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
    }


def run_free_ot(scene, X0, *, lr=OT_LR, max_steps=MAX_STEPS, patience=PATIENCE,
                log_every: int = 25, label: str = ""):
    X_true = scene["X_true"]
    w = scene["w"]
    ot = scene["ot"]
    sig = scene["sigma"]
    X = np.asarray(X0, dtype=np.float64).copy()
    opt = Adam(X.shape, lr=lr)
    E0, _ = vg_ot(ot, X, w, sig)
    energies = [float(E0)]
    nn_rmsds = [nn_rmsd(X, X_true)]
    poses = [X.copy()]
    best_E = energies[0]
    stagnant = 0
    reason = "max_steps"
    t0 = time.perf_counter()
    prefix = f"[{label}] " if label else ""
    print(
        f"  {prefix}step 0  OT={energies[0]:.6g}  NN-RMSD={nn_rmsds[0]:.4f} Å",
        flush=True,
    )
    for k in range(1, max_steps + 1):
        E, G = vg_ot(ot, X, w, sig)
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
                f"  {prefix}step {k}  OT={energies[-1]:.6g}  "
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
        "n_steps": len(energies) - 1,
        "stop_reason": reason,
        "elapsed_s": time.perf_counter() - t0,
        "best_E": float(best_E),
    }


def gaussian_shake(X: np.ndarray, rmsd_target: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Isotropic Gaussian displacement, scaled to exact RMSD ``rmsd_target``."""
    X = np.asarray(X, dtype=np.float64)
    delta = rng.normal(size=X.shape)
    cur = float(np.sqrt((delta ** 2).sum(-1).mean()))
    if cur < 1e-15:
        return X.copy(), 0.0
    delta *= float(rmsd_target) / cur
    return X + delta, float(np.sqrt((delta ** 2).sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--lr", type=float, default=OT_LR)
    ap.add_argument(
        "--shake-rmsd", type=float, default=1.0,
        help="After first OT plateau, shake atoms to this Gaussian RMSD "
             "and re-run OT (0 disables).",
    )
    ap.add_argument(
        "--n-shakes", type=int, default=1,
        help="Number of settle → shake → OT cycles after the first settle.",
    )
    args = ap.parse_args()

    print(f"extracting {PDB_ID} {SEQUENCE} …", flush=True)
    topo = load_1zdd()
    print(
        f"  chain={topo['chain']}  residues={topo['n_residues']}  "
        f"atoms={len(topo['X'])}",
        flush=True,
    )

    scene = build_scene(topo, resolution=args.resolution)
    print(
        f"map {args.resolution:.2f} Å  σ={scene['sigma']:.3f}  "
        f"grid={scene['NG'][0]}³  spacing={SPACING}  "
        f"R={scene['R']:.2f} Å  half={scene['half']:.2f} Å  "
        f"n_dirs={N_DIRS}",
        flush=True,
    )

    rng = np.random.default_rng(int(args.seed))
    n = scene["n_atoms"]
    half = scene["half"]
    X0 = scene["X_true"].mean(0) + rng.uniform(-half, half, size=(n, 3))
    nn0 = nn_rmsd(X0, scene["X_true"])
    print(f"uniform start  NN-RMSD={nn0:.4f} Å  seed={args.seed}", flush=True)

    print("\n=== OT pass 0 (from uniform) ===", flush=True)
    free = run_free_ot(
        scene, X0, lr=args.lr, max_steps=args.max_steps, patience=PATIENCE,
        label="pass0",
    )
    stages = [{
        "name": "pass0",
        "X_start": X0,
        "result": free,
    }]

    X_cur = free["X_final"].copy()
    if args.shake_rmsd > 0 and args.n_shakes > 0:
        for s in range(int(args.n_shakes)):
            X_shake, got_rmsd = gaussian_shake(X_cur, args.shake_rmsd, rng)
            nn_shake = nn_rmsd(X_shake, scene["X_true"])
            E_shake, _ = vg_ot(scene["ot"], X_shake, scene["w"], scene["sigma"])
            print(
                f"\n=== shake {s + 1}  "
                f"ΔRMSD={got_rmsd:.4f} Å  "
                f"NN-RMSD={nn_shake:.4f} Å  OT={E_shake:.6g} ===",
                flush=True,
            )
            print(f"=== OT pass {s + 1} (from shake) ===", flush=True)
            nxt = run_free_ot(
                scene, X_shake, lr=args.lr, max_steps=args.max_steps,
                patience=PATIENCE, label=f"pass{s + 1}",
            )
            stages.append({
                "name": f"pass{s + 1}",
                "X_start": X_shake,
                "shake_rmsd": got_rmsd,
                "result": nxt,
            })
            X_cur = nxt["X_final"].copy()

    # Pick best stage by OT loss (blind); also report NN for diagnosis.
    best_stage = min(stages, key=lambda st: st["result"]["best_E"])
    last = stages[-1]["result"]
    nn_last = last["nn_rmsds"]
    best_k = int(np.argmin(nn_last))

    # Concatenate poses / metrics across stages for the viewer.
    all_poses = []
    all_nn = []
    all_E = []
    stage_breaks = []
    for st in stages:
        r = st["result"]
        stage_breaks.append(len(all_poses))
        all_poses.append(r["poses"])
        all_nn.append(r["nn_rmsds"])
        all_E.append(r["energies"])
    poses_cat = np.concatenate(all_poses, axis=0)
    nn_cat = np.concatenate(all_nn, axis=0)
    E_cat = np.concatenate(all_E, axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.resolution:g}".replace(".", "p")
    out_path = OUT / f"1zdd_free_ot_{tag}A_seed{args.seed}.npz"
    np.savez_compressed(
        out_path,
        X_true=scene["X_true"],
        X0=X0,
        X_after_pass0=stages[0]["result"]["X_final"],
        X_final=last["X_final"],
        X_best_ot=best_stage["result"]["X_final"],
        poses=poses_cat,
        nn_rmsds=nn_cat,
        energies=E_cat,
        stage_breaks=np.asarray(stage_breaks, dtype=np.int64),
        stage_names=np.array([st["name"] for st in stages]),
        W=scene["w"],
        origin=scene["origin"],
        spacing=scene["spacing"],
        NG=np.asarray(scene["NG"]),
        resolution=np.array(args.resolution),
        sigma=np.array(scene["sigma"]),
        seed=np.array(args.seed),
        sequence=np.array(SEQUENCE),
        stop_reason=np.array(last["stop_reason"]),
        shake_rmsd=np.array(args.shake_rmsd),
        n_shakes=np.array(args.n_shakes),
        names=np.array(topo["names"]),
        Z=topo["Z"],
        bonds=topo["bonds"],
        pass0_nn_final=np.array(stages[0]["result"]["nn_rmsds"][-1]),
        pass0_best_E=np.array(stages[0]["result"]["best_E"]),
        final_best_E=np.array(last["best_E"]),
        best_ot_stage=np.array(best_stage["name"]),
    )

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
    print(
        f"best OT stage: {best_stage['name']}  "
        f"(E={best_stage['result']['best_E']:.6g}, "
        f"NN={best_stage['result']['nn_rmsds'][-1]:.4f} Å)"
    )
    print(f"NN-RMSD start : {nn_cat[0]:.4f} Å")
    print(f"NN-RMSD after pass0 : {stages[0]['result']['nn_rmsds'][-1]:.4f} Å")
    print(f"NN-RMSD final : {nn_last[-1]:.4f} Å  "
          f"(best along last pass: {nn_last[best_k]:.4f} @ step {best_k})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
