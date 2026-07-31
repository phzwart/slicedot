"""Acceptance tests for Geometry / P_restr."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from slicedot import Geometry, SlicedOT, SlicedOTConfig
from slicedot.fixtures import leucine_topology, W as W_np
from slicedot.geometry import chiral_volume, build_distance_pairs
from slicedot.perturb import dihedral, rotation_matrix, set_torsion, backrub


TOL = 1e-4


@pytest.fixture(scope="module")
def leu():
    return leucine_topology()


@pytest.fixture(scope="module")
def geom(leu):
    return Geometry(
        leu["X_ref"],
        leu["bonds"],
        rotatable_bonds=leu["rotatable_bonds"],
        chiral_centres=leu["chiral_centres"],
        planar_groups=leu["planar_groups"],
        antibump=True,
    )


def _rms_perturb(X, rms, rng):
    noise = rng.standard_normal(X.shape)
    noise *= rms / (np.sqrt((noise ** 2).mean()) + 1e-30)
    return X + noise


def test_no_14_across_rotatable(leu):
    pairs = build_distance_pairs(
        leu["X_ref"], leu["bonds"], leu["rotatable_bonds"], maxsep=3,
    )
    kinds = {(i, j): k for i, j, _, k in pairs}
    idx = leu["idx"]
    # N–CG is 1-4 across χ1 (CA–CB); must be absent
    key = tuple(sorted((idx["N"], idx["CG"])))
    assert key not in kinds or kinds.get(key) != "torsion14"
    # CA–CD1 is 1-4 across χ2 (CB–CG); must be absent as torsion14
    key2 = tuple(sorted((idx["CA"], idx["CD1"])))
    assert key2 not in {tuple(sorted((i, j))) for i, j, _, k in pairs if k == "torsion14"}


def test_idempotence(geom, leu):
    rng = np.random.default_rng(0)
    X = _rms_perturb(leu["X_ref"], 0.3, rng)
    Xp, r1, _ = geom.project(X, tol=TOL, max_iter=400)
    Xp2, r2, _ = geom.project(Xp, tol=TOL, max_iter=400)
    assert geom.residual(Xp)["distance_max_A"] < 0.05
    assert r1 <= TOL * 5
    assert np.max(np.abs(Xp2 - Xp)) < TOL * 5
    assert r2 <= r1 + 1e-6


def test_se3_invariance(geom, leu):
    rng = np.random.default_rng(1)
    X = _rms_perturb(leu["X_ref"], 0.2, rng)
    axis = rng.standard_normal(3)
    R = rotation_matrix(axis, 0.7)
    t = rng.standard_normal(3) * 2.0
    Xp, _, _ = geom.project(X, tol=TOL, max_iter=300)
    Yp, _, _ = geom.project(X @ R.T + t, tol=TOL, max_iter=300)
    # Restrained distances must match (free torsions across rotatable bonds may differ).
    errs = []
    for i, j, d0, _k in geom.pairs:
        errs.append(abs(np.linalg.norm(Yp[j] - Yp[i]) - np.linalg.norm(Xp[j] - Xp[i])))
    assert max(errs) < 0.02
    assert geom.residual(Yp)["distance_max_A"] < 0.05
    # Rigid transform of the projected pose is also a near-minimizer.
    expect = Xp @ R.T + t
    assert geom.residual(expect)["distance_max_A"] < 0.05


def test_convergence_table(geom, leu, capsys):
    rng = np.random.default_rng(2)
    rows = []
    for rms in (0.05, 0.3, 1.0):
        max_err = []
        for trial in range(5):
            X = _rms_perturb(leu["X_ref"], rms, rng)
            Xp, wrms, nit = geom.project(X, tol=TOL, max_iter=600)
            info = geom.residual(Xp)
            max_err.append(info["distance_max_A"])
            # Beat GS stall (~0.2 Å at 1 Å pert); tight for small perts.
            lim = 0.02 if rms < 1.0 else 0.08
            assert info["distance_max_A"] < lim, (
                f"rms_pert={rms} trial={trial} max_dist={info['distance_max_A']:.3e} "
                f"wrms={wrms:.3e} nfev={nit}"
            )
        rows.append((rms, float(np.median(max_err)), float(np.max(max_err))))
    print("\nconvergence table (median / max distance error Å over 5 trials):")
    for rms, med, mx in rows:
        print(f"  pert {rms:4.2f} Å → med {med:.3e}  max {mx:.3e}")
    assert rows[-1][2] < 0.08


def test_chirality_preserved(geom, leu):
    rng = np.random.default_rng(3)
    c, a, b, d = leu["chiral_centres"][0]
    V0 = np.sign(chiral_volume(leu["X_ref"], c, a, b, d))
    assert V0 != 0
    flips = 0
    for _ in range(100):
        X = _rms_perturb(leu["X_ref"], 1.2, rng)
        Xp, _, _ = geom.project(X, tol=TOL, max_iter=400)
        if np.sign(chiral_volume(Xp, c, a, b, d)) != V0:
            flips += 1
    assert flips == 0, f"{flips}/100 chirality inversions"


def test_rotamer_reachable(geom, leu):
    idx = leu["idx"]
    X_ref = leu["X_ref"]
    chi0 = dihedral(X_ref, idx["N"], idx["CA"], idx["CB"], idx["CG"])
    X_rot = set_torsion(
        X_ref, idx["CA"], idx["CB"], None, np.deg2rad(120.0), bonds=leu["bonds"],
    )
    chi1 = dihedral(X_rot, idx["N"], idx["CA"], idx["CB"], idx["CG"])
    # unwrap difference
    dchi = (chi1 - chi0 + np.pi) % (2 * np.pi) - np.pi
    assert abs(abs(dchi) - np.deg2rad(120.0)) < np.deg2rad(5.0)

    Xp, _, _ = geom.project(X_rot, tol=TOL, max_iter=300)
    chip = dihedral(Xp, idx["N"], idx["CA"], idx["CB"], idx["CG"])
    d_to_rot = (chip - chi1 + np.pi) % (2 * np.pi) - np.pi
    d_to_ref = (chip - chi0 + np.pi) % (2 * np.pi) - np.pi
    # Must stay near the new rotamer, not be dragged back to reference χ1
    assert abs(d_to_rot) < np.deg2rad(25.0), f"Δχ to rotamer {np.rad2deg(d_to_rot):.1f}°"
    assert abs(d_to_ref) > np.deg2rad(60.0), f"dragged toward ref (Δ={np.rad2deg(d_to_ref):.1f}°)"


def test_antibump(geom, leu):
    idx = leu["idx"]
    X = leu["X_ref"].copy()
    # ACE_CH3 and CD2 are far in topology; force clash
    i, j = idx["ACE_CH3"], idx["CD2"]
    assert geom.D[i, j] > 3
    mid = 0.5 * (X[i] + X[j])
    X[i] = mid + np.array([0.75, 0.0, 0.0])
    X[j] = mid - np.array([0.75, 0.0, 0.0])
    assert np.linalg.norm(X[j] - X[i]) < 2.0
    info0 = geom.residual(X)
    assert info0["bump"]["active"] >= 1
    Xp, _, _ = geom.project(X, tol=TOL, max_iter=400)
    sep = float(np.linalg.norm(Xp[j] - Xp[i]))
    assert sep >= 2.5, f"separation after project {sep:.2f} Å"


def test_antibump_slack_anneals(geom, leu):
    """Loose slack should dead-zone a mild clash; tight slack should resolve it."""
    idx = leu["idx"]
    X = leu["X_ref"].copy()
    i, j = idx["ACE_CH3"], idx["CD2"]
    pair = (i, j) if i < j else (j, i)
    assert pair in geom.bump_pairs
    k = geom.bump_pairs.index(pair)
    mid = 0.5 * (X[i] + X[j])
    # Separation 2.2 Å → 0.6 Å penetration vs r0=2.8
    X[i] = mid + np.array([1.1, 0.0, 0.0])
    X[j] = mid - np.array([1.1, 0.0, 0.0])
    assert abs(np.linalg.norm(X[j] - X[i]) - 2.2) < 1e-9

    prev = geom.slack
    try:
        geom.slack = 0.8  # dead zone covers the 0.6 Å penetration
        bump_loose, _ = geom._bump_residuals(X)
        assert float(bump_loose[k]) == 0.0

        geom.slack = 0.0
        bump_tight, _ = geom._bump_residuals(X)
        assert float(bump_tight[k]) > 0.0
    finally:
        geom.slack = prev


def test_ot_roundtrip(geom, leu):
    """One over-relaxed deformation step then P_restr."""
    from slicedot.fixtures import sigma_of

    X_true = leu["X_ref"]
    w = torch.tensor(W_np)
    sig = sigma_of(2.5)
    # render target
    sp = 0.5
    NG = (40, 40, 40)
    org = -0.5 * (np.array(NG) - 1) * sp
    ax = [np.arange(n) * sp for n in NG]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG)
    for p, wt in zip(X_true, W_np):
        T += wt * np.exp(-((G - p) ** 2).sum(-1) / (2 * sig * sig))
    T /= T.sum()

    ot = SlicedOT(
        torch.tensor(T), org, sp, sig,
        SlicedOTConfig(n_dirs=24, dt=0.35, window=48.0, map_cutoff=1e-7),
    )
    # start displaced
    X = X_true + np.array([3.0, 0.5, -0.5])
    dv = ot.deformation(torch.tensor(X), w, sig).numpy()
    beta = 1.6
    X_data = X + beta * dv
    com_before = X.mean(0)
    Xp, wrms, _ = geom.project(X_data, tol=1e-3, max_iter=400)
    info = geom.residual(Xp)
    assert info["distance_max_A"] < 0.05
    # moved toward the density / true pose
    assert np.linalg.norm(Xp.mean(0) - X_true.mean(0)) < np.linalg.norm(
        com_before - X_true.mean(0)
    )


def test_backrub_moves_peptide(leu):
    idx = leu["idx"]
    # Toy backrub about N–NME_N axis moving CA (not chemically real; API smoke)
    X2 = backrub(
        leu["X_ref"], idx["N"], idx["NME_N"],
        [idx["CA"], idx["CB"], idx["C"]], 0.3,
    )
    assert not np.allclose(X2[idx["CA"]], leu["X_ref"][idx["CA"]])
