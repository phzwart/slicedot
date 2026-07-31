"""Acceptance tests for free-atom cloud naming (Namer)."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from slicedot import Geometry, Namer
from slicedot.fixtures import leucine_topology, Z as LEU_Z
from slicedot.fixtures_peptide import oligopeptide_topology
from slicedot.perturb import rotation_matrix

# Full operating-point grid is slow; CI uses fewer trials unless RUNSLOW=1.
N_TRIALS = 50 if os.environ.get("RUNSLOW") else 10
RMS_GRID = (0.2, 0.35, 0.5, 0.7, 1.0)


def _namer_from(topo, *, use_aut=True):
    aut = topo.get("automorphism_generators") if use_aut else [np.arange(len(topo["X_ref"]))]
    return Namer(
        topo["X_ref"],
        topo["elements"],
        topo["bonds"],
        rotatable_bonds=topo["rotatable_bonds"],
        chiral_centres=topo["chiral_centres"],
        planar_groups=topo["planar_groups"],
        automorphisms=aut,
    )


def _geom_from(topo) -> Geometry:
    return Geometry(
        topo["X_ref"],
        topo["bonds"],
        rotatable_bonds=topo["rotatable_bonds"],
        chiral_centres=topo["chiral_centres"],
        planar_groups=topo["planar_groups"],
        antibump=False,
    )


def _rms_perturb(X, rms, rng):
    noise = rng.standard_normal(X.shape)
    noise *= rms / (np.sqrt((noise ** 2).mean()) + 1e-30)
    return X + noise


def _landing_cloud(X, rms, rng, geom: Geometry):
    """Geometry-preserving free-atom landing at ~rms displacement.

    Independent Gaussian noise destroys bonds; a light P_restr pass restores
    local chemistry the way OT landings sit on atomic sites.
    """
    Ylab = _rms_perturb(X, rms, rng)
    slack = max(0.25, 0.6 * float(rms))
    Ylab, _, _ = geom.project(Ylab, tol=1e-3, max_iter=60, slack=slack)
    return Ylab


def _shuffle_cloud(Ylab, W, rng):
    order = rng.permutation(len(Ylab))
    return Ylab[order], W[order], np.argsort(order)


def _wrong_labels(namer: Namer, perm, truth) -> int:
    if namer.equivalent_to(perm, truth):
        return 0
    best = len(perm)
    for alpha in namer.automorphisms:
        best = min(best, int(np.sum(perm != truth[alpha])))
    return best


@pytest.fixture(scope="module")
def leu():
    return leucine_topology()


@pytest.fixture(scope="module")
def oligo():
    return oligopeptide_topology()


@pytest.fixture(scope="module")
def oligo_namer(oligo):
    return _namer_from(oligo)


@pytest.fixture(scope="module")
def oligo_geom(oligo):
    return _geom_from(oligo)


def test_weight_dz_groups():
    from slicedot.namer import MIN_WEIGHT_DZ, _weight_discriminable_groups

    # Organic C/N/O: one indiscriminable group (ΔZ ≤ 2)
    g = _weight_discriminable_groups([6, 7, 8], min_dz=MIN_WEIGHT_DZ)
    assert len(g) == 1 and set(g[0]) == {6, 7, 8}
    # S (16) is only 8 e⁻ from O — still not weight-separable from C/N/O
    g = _weight_discriminable_groups([6, 7, 8, 16], min_dz=MIN_WEIGHT_DZ)
    assert len(g) == 1
    # Fe (26) is ≥10 e⁻ from C/N/O → two groups
    g = _weight_discriminable_groups([6, 7, 8, 26], min_dz=MIN_WEIGHT_DZ)
    assert len(g) == 2
    flats = {frozenset(x) for x in g}
    assert frozenset({6, 7, 8}) in flats and frozenset({26}) in flats
    # C vs Fe: fully separable
    g = _weight_discriminable_groups([6, 26], min_dz=MIN_WEIGHT_DZ)
    assert len(g) == 2


def test_organic_weights_do_not_type_by_z(leu):
    """C/N/O ΔZ < 10: passing Z-weights must not change typing vs geometry."""
    namer = _namer_from(leu)
    X = leu["X_ref"]
    W = leu["elements"].astype(np.float64) / leu["elements"].sum()
    rng = np.random.default_rng(11)
    order = rng.permutation(len(X))
    Y, wY = X[order], W[order]
    a_w = namer.assign(Y, X, wY)
    a_g = namer.assign(Y, X, None)
    assert np.array_equal(a_w.perm, a_g.perm)


def test_noop_identity(leu):
    namer = _namer_from(leu)
    X = leu["X_ref"]
    a = namer.assign(X.copy(), X, leu["elements"].astype(np.float64) / leu["elements"].sum())
    assert namer.equivalent_to(a.perm, np.arange(len(X)))
    assert a.n_repaired == 0
    assert a.restraint_rms < 1e-6


def test_noop_oligo(oligo, oligo_namer):
    X = oligo["X_ref"]
    a = oligo_namer.assign(X.copy(), X, oligo["W"])
    assert oligo_namer.equivalent_to(a.perm, np.arange(oligo["n"]))
    assert a.n_repaired == 0


def test_leucine_shuffle_smoke(leu):
    namer = _namer_from(leu)
    X = leu["X_ref"]
    W = LEU_Z / LEU_Z.sum()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    Y, wY, truth = X[order], W[order], np.argsort(order)
    a = namer.assign(Y, X, wY)
    assert namer.equivalent_to(a.perm, truth)


def test_operating_point_swap_rates(oligo, oligo_namer, oligo_geom, capsys):
    """Distance-only fails at 0.5 Å; full pipeline stays near-perfect."""
    X = oligo["X_ref"]
    W = oligo["W"]
    namer = oligo_namer
    geom = oligo_geom
    rng = np.random.default_rng(2)

    rows = []
    for rms in RMS_GRID:
        dist = el = full = 0
        n_perfect = 0
        for _ in range(N_TRIALS):
            Ylab = _landing_cloud(X, rms, rng, geom)
            Y, wY, truth = _shuffle_cloud(Ylab, W, rng)
            ad = namer.assign(Y, X, distance_only=True)
            ae = namer.assign(Y, X, wY, element_blocked_only=True)
            af = namer.assign(Y, X, wY)
            dist += _wrong_labels(namer, ad.perm, truth)
            el += _wrong_labels(namer, ae.perm, truth)
            wf = _wrong_labels(namer, af.perm, truth)
            full += wf
            if wf == 0:
                n_perfect += 1
        rows.append((rms, dist / N_TRIALS, el / N_TRIALS, full / N_TRIALS, n_perfect))

    print("\noperating-point wrong-label counts (mean over trials):")
    for rms, d, e, f, npf in rows:
        print(
            f"  rms={rms:4.2f}  dist={d:6.2f}  elem={e:6.2f}  full={f:6.2f}  "
            f"perfect={npf}/{N_TRIALS}"
        )

    by_rms = {r[0]: r for r in rows}
    # Distance-only should show errors by 0.5 Å
    assert by_rms[0.5][1] > 0.5, "distance-only should fail at 0.5 Å"
    # Full pipeline: near-zero wrong labels at ≤0.5 Å
    assert by_rms[0.2][3] < 0.5
    assert by_rms[0.35][3] < 1.0
    assert by_rms[0.5][3] < 2.0
    # At higher noise, restraints should not be worse than unary-only.
    assert by_rms[0.5][3] <= by_rms[0.5][1] + 0.5
    assert by_rms[0.7][3] <= by_rms[0.7][1]


def test_crossing_regime(leu):
    """Rotated unary prior: fingerprint + restraints beat distance-only.

    Uses the smaller leucine fixture — an oligopeptide has too many near-tied
    bond fingerprints for a pure spatial init to disambiguate under a crossed
    prior.
    """
    namer = _namer_from(leu)
    geom = _geom_from(leu)
    X = leu["X_ref"]
    W = LEU_Z / LEU_Z.sum()
    rng = np.random.default_rng(3)
    better = 0
    n_tot = 0
    n_fp = 0
    for deg in (90.0, 150.0):
        R = rotation_matrix(rng.standard_normal(3), np.deg2rad(deg))
        X_prior = (X - X.mean(0)) @ R.T + X.mean(0)
        for _ in range(max(5, N_TRIALS // 2)):
            Ylab = _landing_cloud(X, 0.35, rng, geom)
            Y, wY, truth = _shuffle_cloud(Ylab, W, rng)
            ad = namer.assign(Y, X_prior, distance_only=True)
            af = namer.assign(Y, X_prior, wY)
            wd = _wrong_labels(namer, ad.perm, truth)
            wf = _wrong_labels(namer, af.perm, truth)
            n_tot += 1
            if "fingerprint_init" in af.flags:
                n_fp += 1
            if wf < wd or (wf == 0):
                better += 1
            assert np.isfinite(af.restraint_rms)
    assert n_fp >= 1, "crossed prior should trigger fingerprint init"
    assert better >= int(0.35 * n_tot)


def test_chirality_zero_uncorrected(oligo, oligo_namer, oligo_geom):
    X = oligo["X_ref"]
    W = oligo["W"]
    namer = oligo_namer
    geom = oligo_geom
    rng = np.random.default_rng(4)
    n_trials = 100 if os.environ.get("RUNSLOW") else 20
    uncorrected = 0
    for _ in range(n_trials):
        Ylab = _landing_cloud(X, 0.5, rng, geom)
        Y, wY, truth = _shuffle_cloud(Ylab, W, rng)
        a = namer.assign(Y, X, wY)
        # Count chiral centres still inverted after assign, ignoring Aut
        # that flip no stereo (CD methyl swaps don't flip Cα).
        Yn = a.Y_named
        for (c, a_i, b, d), V0 in zip(oligo["chiral_centres"], namer.V_ref):
            if abs(V0) < 1e-8:
                continue
            from slicedot.geometry import chiral_volume
            if np.sign(chiral_volume(Yn, c, a_i, b, d)) != np.sign(V0):
                # Only count if assignment is otherwise correct (Aut-eq)
                if namer.equivalent_to(a.perm, truth):
                    uncorrected += 1
                elif f"chiral_inversion:{c}" not in a.flags:
                    uncorrected += 1
    assert uncorrected == 0, f"{uncorrected} uncorrected inversions"


def test_automorphism_invariance(oligo, oligo_namer):
    X = oligo["X_ref"]
    W = oligo["W"]
    namer = oligo_namer
    rng = np.random.default_rng(5)
    # Apply random automorphism to ground-truth labelling of Y
    alpha = namer.automorphisms[int(rng.integers(0, len(namer.automorphisms)))]
    # Y positions = X, but we present them shuffled; truth is argsort
    order = rng.permutation(oligo["n"])
    Y = X[order]
    wY = W[order]
    truth = np.argsort(order)
    # Relabel truth by alpha: equivalent answer truth ∘ alpha
    a = namer.assign(Y, X, wY)
    assert namer.equivalent_to(a.perm, truth)
    # Ambiguous groups should mention CD swaps when present
    # (structure-level, independent of which Aut we applied to GT)
    a2 = namer.assign(Y, X, wY)
    g1 = sorted(tuple(sorted(g[0])) for g in a.ambiguous_groups)
    g2 = sorted(tuple(sorted(g[0])) for g in a2.ambiguous_groups)
    assert g1 == g2


def test_timing_under_ot_iteration(oligo, oligo_namer, oligo_geom):
    """Naming must stay cheaper than a ballpark OT iteration."""
    X = oligo["X_ref"]
    W = oligo["W"]
    namer = oligo_namer
    rng = np.random.default_rng(6)
    Ylab = _landing_cloud(X, 0.5, rng, oligo_geom)
    Y, wY, _ = _shuffle_cloud(Ylab, W, rng)

    # Warm up
    namer.assign(Y, X, wY)
    t0 = time.perf_counter()
    n_rep = 20
    for _ in range(n_rep):
        namer.assign(Y, X, wY)
    ms = (time.perf_counter() - t0) * 1e3 / n_rep
    print(f"\nnaming wall time: {ms:.2f} ms/call (N={oligo['n']})")
    # One OT iteration on a ~100-atom 2-D scene is typically tens of ms; keep a
    # generous CI bound so the 50+name+cleanup path stays the cheap one.
    assert ms < 200.0, f"naming too slow: {ms:.1f} ms"


def test_compose_with_geometry_project(leu):
    """Named cloud can be handed to P_restr cleanup."""
    namer = _namer_from(leu)
    geom = _geom_from(leu)
    X = leu["X_ref"]
    W = LEU_Z / LEU_Z.sum()
    rng = np.random.default_rng(7)
    Ylab = _landing_cloud(X, 0.4, rng, geom)
    order = rng.permutation(len(X))
    Y, wY, truth = Ylab[order], W[order], np.argsort(order)
    a = namer.assign(Y, X, wY)
    assert namer.equivalent_to(a.perm, truth)
    Xp, wrms, _ = geom.project(a.Y_named, tol=1e-3, max_iter=100)
    assert wrms < 1.0


def test_namer_uses_geometry_compat_restraint_set(leu):
    """Without a CIF, Namer builds a harmonic RestraintSet from X_ref."""
    namer = _namer_from(leu)
    assert namer.restraint_set is not None
    assert len(namer.restraints) == len(namer.restraint_set.to_naming_pairs())
    assert all(r.d_lo == r.d_hi for r in namer.restraints)
    perm = np.arange(len(leu["X_ref"]))
    assert namer.restraint_rms(leu["X_ref"], perm) < 1e-6


def test_namer_from_explicit_restraint_set(leu):
    from slicedot import restraint_set_from_geometry

    topo = leu
    rs = restraint_set_from_geometry(
        topo["X_ref"],
        topo["elements"],
        topo["bonds"],
        rotatable_bonds=topo["rotatable_bonds"],
        atom_ids=topo.get("names"),
    )
    namer = Namer(
        topo["X_ref"],
        restraint_set=rs,
        rotatable_bonds=topo["rotatable_bonds"],
        chiral_centres=topo["chiral_centres"],
        planar_groups=topo["planar_groups"],
    )
    perm = np.arange(len(leu["X_ref"]))
    assert namer.restraint_rms(leu["X_ref"], perm) < 1e-6
