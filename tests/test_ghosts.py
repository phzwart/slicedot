"""Ghost marking / prune-by-geometry+L1."""
from __future__ import annotations

import numpy as np
import pytest

from slicedot import Namer, prune_ghosts, seed_kept_hungarian, sigma_from_resolution
from slicedot.fixtures import W as W_NP, leucine_topology
from slicedot.ghosts import map_density_at


class _TinyL1:
    """Unit-mass ortho L1 oracle (same contract as ensemble L1Diff3D)."""

    def __init__(self, X_true, w, sigma, spacing=0.5, pad=6.0):
        self.sigma = float(sigma)
        half = float(np.linalg.norm(X_true - X_true.mean(0), axis=1).max()) + pad
        n = int(np.ceil(2.0 * half / spacing))
        if n % 2 == 0:
            n += 1
        self.shape = (n, n, n)
        self.origin = -0.5 * (n - 1) * spacing * np.ones(3)
        self.spacing = spacing * np.ones(3)
        ax = [self.origin[i] + np.arange(n) * spacing for i in range(3)]
        G = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
        self.V = G.reshape(-1, 3)
        raw = np.zeros(self.V.shape[0], dtype=np.float64)
        s2 = self.sigma * self.sigma
        for p, wi in zip(X_true, w):
            d2 = ((self.V - p) ** 2).sum(-1)
            raw += wi * np.exp(-d2 / (2.0 * s2))
        self.T = (raw / raw.sum()).reshape(self.shape)

    def value_grad(self, x, w):
        x = np.asarray(x, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        s2 = self.sigma * self.sigma
        raw = np.zeros(self.V.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            d2 = ((self.V - x[i]) ** 2).sum(-1)
            raw += w[i] * np.exp(-d2 / (2.0 * s2))
        Z = float(raw.sum())
        M = raw / Z
        T = self.T.ravel()
        val = float(np.abs(M - T).sum())
        return val, np.zeros_like(x)


def _namer(topo):
    return Namer(
        topo["X_ref"],
        topo["elements"],
        topo["bonds"],
        rotatable_bonds=topo["rotatable_bonds"],
        chiral_centres=topo["chiral_centres"],
        planar_groups=topo["planar_groups"],
        automorphisms=topo.get("automorphism_generators"),
    )


@pytest.fixture(scope="module")
def leu_scene():
    topo = leucine_topology()
    X = topo["X_ref"].copy()
    w = W_NP.copy()
    sig = float(sigma_from_resolution(2.0))
    return {
        "topo": topo,
        "X": X,
        "w": w,
        "sigma": sig,
        "namer": _namer(topo),
        "l1": _TinyL1(X, w, sig),
    }


def test_equal_n_no_ghosts(leu_scene):
    X = leu_scene["X"]
    n = len(X)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    Y = X[order]
    prior = X - X.mean(0) + Y.mean(0)
    res = prune_ghosts(
        Y,
        namer=leu_scene["namer"],
        X_prior=prior,
        l1_oracle=leu_scene["l1"],
        w_chem=leu_scene["w"],
        sigma=leu_scene["sigma"],
        max_swap_models=8,
    )
    assert res.ghost_idx.size == 0
    assert not res.ghost_mask.any()
    assert res.kept_idx.shape == (n,)
    assert set(res.kept_idx.tolist()) == set(range(n))
    np.testing.assert_allclose(res.w_chem.sum(), 1.0)
    np.testing.assert_allclose(res.w_chem, leu_scene["w"])
    assert res.sigma == pytest.approx(leu_scene["sigma"])
    assert res.Y_named.shape == (n, 3)


def test_surplus_outliers_marked_ghosts(leu_scene):
    X = leu_scene["X"]
    n = len(X)
    # Perfect copy of chemistry + far outliers.
    outliers = X.mean(0) + np.array(
        [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [-40.0, 0.0, 0.0]]
    )
    X_free = np.vstack([X, outliers])
    prior = X.copy()
    dens = map_density_at(leu_scene["l1"], X_free)
    assert dens[:n].min() > dens[n:].max()

    res = prune_ghosts(
        X_free,
        namer=leu_scene["namer"],
        X_prior=prior,
        l1_oracle=leu_scene["l1"],
        w_chem=leu_scene["w"],
        sigma=leu_scene["sigma"],
        max_swap_models=16,
    )
    assert res.ghost_idx.size == 3
    assert set(res.ghost_idx.tolist()) == {n, n + 1, n + 2}
    assert res.kept_idx.shape == (n,)
    assert set(res.kept_idx.tolist()) == set(range(n))
    # Near-perfect keep-set should beat a random L1 on the true pose floor.
    l1_true, _ = leu_scene["l1"].value_grad(X, leu_scene["w"])
    assert res.l1 <= l1_true + 0.05
    np.testing.assert_allclose(res.w_chem, leu_scene["w"])
    assert res.sigma == pytest.approx(leu_scene["sigma"])


def test_seed_hungarian_rectangular():
    prior = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    free = np.vstack([prior, [[10.0, 10.0, 10.0], [11.0, 0.0, 0.0]]])
    kept = seed_kept_hungarian(free, prior)
    assert kept.shape == (3,)
    assert set(kept.tolist()) == {0, 1, 2}


def test_undercomplete_raises(leu_scene):
    X = leu_scene["X"][:3]
    with pytest.raises(ValueError, match="at least"):
        prune_ghosts(
            X,
            namer=leu_scene["namer"],
            X_prior=leu_scene["X"],
            l1_oracle=leu_scene["l1"],
            w_chem=leu_scene["w"],
            sigma=leu_scene["sigma"],
        )
