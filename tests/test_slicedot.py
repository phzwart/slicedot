"""End-to-end validation of slicedot across grid geometries.

Stages (from the NumPy protocol / paper §Validation):
  1 floor vs clearance      map diagnostic; detects inadequate padding
  2 analytic anchor         rigid translation must give t * (1/L) sum |u.e|
  3 gradient vs FD          autograd on the log-cosh surrogate
  4 reach                   value and |grad| flat to large displacement
  5 deformation oracle      one step recovers a rigid translation
  6 backend agreement       direct / grid / grid_custom
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from slicedot import (
    CrystalSlicedOT,
    SlicedOT,
    SlicedOTConfig,
    fibonacci_directions,
    orthogonalization_matrix,
    sigma_from_resolution,
)
from slicedot.fixtures import X0, W as W_np, sigma_of

torch.set_default_dtype(torch.float64)

SIG = sigma_of(2.5)
W = torch.tensor(W_np)
E = np.array([1.0, 0.0, 0.0])
CFG = dict(n_dirs=32, dt=0.3, window=96.0, map_cutoff=1e-7)


def render_ortho(X, sp, NG, sigma):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.array(NG) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG)
    for p, w in zip(X, W_np):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2 * sigma * sigma))
    return T / T.sum(), org, sp


def render_cell(X, cell, NG, sigma):
    O = orthogonalization_matrix(*cell).numpy()
    Xc = X + (O @ np.array([0.5, 0.5, 0.5]))
    ii = np.stack(np.meshgrid(*[np.arange(n) for n in NG], indexing="ij"), -1).reshape(
        -1, 3
    )
    V = (ii / np.array(NG)) @ O.T
    T = np.zeros(len(V))
    for p, w in zip(Xc, W_np):
        T += w * np.exp(-((V - p) ** 2).sum(-1) / (2 * sigma * sigma))
    return (T / T.sum()).reshape(NG), Xc


def diagnostics(m, X):
    xt = torch.tensor(X)
    ex = np.abs(m.U.numpy() @ E).mean()
    floor = m(xt, W, SIG).item()
    anchors = [
        100
        * (m(torch.tensor(X + t * E), W, SIG).item() - t * ex)
        / (t * ex)
        for t in (1.0, 3.0, 6.0)
    ]
    rng = np.random.default_rng(3)
    Y = X + 0.3 * rng.standard_normal(X.shape)
    y = torch.tensor(Y, requires_grad=True)
    m(y, W, SIG, delta=1e-4).backward()
    ga = y.grad.numpy()
    h = 1e-6
    err = []
    for i in range(0, len(X), 4):
        for k in range(3):
            Yp = Y.copy()
            Yp[i, k] += h
            Ym = Y.copy()
            Ym[i, k] -= h
            gn = (
                m(torch.tensor(Yp), W, SIG, delta=1e-4).item()
                - m(torch.tensor(Ym), W, SIG, delta=1e-4).item()
            ) / (2 * h)
            err.append(abs(ga[i, k] - gn) / max(abs(gn), 1e-30))
    yy = torch.tensor(X + 12.0 * E, requires_grad=True)
    v = m(yy, W, SIG)
    v.backward()
    reach_pct = (v.item() / (12.0 * ex) - 1) * 100
    g_over_w = np.linalg.norm(yy.grad.numpy(), axis=1).mean() / W_np.mean()
    shift = np.array([5.0, 2.0, -3.0])
    dv = m.deformation(torch.tensor(X + shift), W, SIG).numpy()
    step_err = np.linalg.norm((X + shift) + dv - X, axis=1).max()
    return {
        "floor": floor,
        "anchors": anchors,
        "fd_med": float(np.median(err)),
        "fd_max": float(max(err)),
        "reach_pct": reach_pct,
        "g_over_w": g_over_w,
        "step_err": step_err,
    }


def assert_ok(d, *, floor_tol=2e-5, anchor_tol=0.5, fd_med_tol=1e-6, step_tol=0.05):
    assert d["floor"] < floor_tol, f"floor {d['floor']:.2e}"
    assert all(abs(a) < anchor_tol for a in d["anchors"]), d["anchors"]
    assert d["fd_med"] < fd_med_tol, d["fd_med"]
    assert abs(d["reach_pct"]) < 0.5, d["reach_pct"]
    assert d["g_over_w"] > 0.3, d["g_over_w"]
    assert d["step_err"] < step_tol, d["step_err"]


@pytest.fixture(scope="module")
def cubic_case():
    T, org, sp = render_ortho(X0, 0.45, (48, 48, 48), SIG)
    m = SlicedOT(torch.tensor(T), org, torch.tensor(sp), SIG, SlicedOTConfig(**CFG))
    return m, X0


@pytest.fixture(scope="module")
def aniso_case():
    T, org, sp = render_ortho(X0, [0.40, 0.55, 0.45], (52, 40, 46), SIG)
    m = SlicedOT(torch.tensor(T), org, torch.tensor(sp), SIG, SlicedOTConfig(**CFG))
    return m, X0


@pytest.fixture(scope="module")
def monoclinic_case():
    cell = (28.4, 31.7, 25.1, 90.0, 105.3, 90.0)
    T, Xc = render_cell(X0, cell, (60, 66, 54), SIG)
    m = CrystalSlicedOT(torch.tensor(T), cell, SIG, SlicedOTConfig(**CFG))
    return m, Xc


@pytest.fixture(scope="module")
def triclinic_case():
    cell = (40.0, 42.0, 44.0, 58.0, 63.0, 112.0)
    T, Xc = render_cell(X0, cell, (96, 100, 104), SIG)
    m = CrystalSlicedOT(torch.tensor(T), cell, SIG, SlicedOTConfig(**CFG))
    return m, Xc


@pytest.mark.parametrize(
    "case",
    ["cubic_case", "aniso_case", "monoclinic_case", "triclinic_case"],
)
def test_grid_geometry(case, request):
    m, X = request.getfixturevalue(case)
    assert_ok(diagnostics(m, X))


def test_backend_agreement(cubic_case):
    rng = np.random.default_rng(0)
    n = 200
    Xb = rng.normal(0, 4.0, (n, 3))
    Xb -= Xb.mean(0)
    Zb = rng.choice([6.0, 7.0, 8.0], n)
    wb = torch.tensor(Zb / Zb.sum())
    T, org, sp = render_ortho(X0, 0.45, (48, 48, 48), SIG)
    ref_v = ref_g = None
    for be in ("direct", "grid", "grid_custom"):
        c = dict(CFG)
        c.update(backend=be, sigma_grid=0.3)
        mm = SlicedOT(
            torch.tensor(T), org, torch.tensor(sp), SIG, SlicedOTConfig(**c)
        )
        y = torch.tensor(Xb, requires_grad=True)
        v = mm(y, wb, SIG)
        v.backward()
        if ref_v is None:
            ref_v, ref_g = v.item(), y.grad.detach().clone()
        else:
            assert abs(v.item() - ref_v) / abs(ref_v) < 5e-3
            assert (
                (y.grad - ref_g).abs().max().item() / ref_g.abs().max().item() < 5e-2
            )


def test_sigma_from_resolution_matches_fixture():
    assert math.isclose(sigma_from_resolution(2.5), SIG, rel_tol=0, abs_tol=1e-12)


def test_fibonacci_directions_unit():
    U = fibonacci_directions(48)
    assert U.shape == (48, 3)
    norms = U.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)


# ---------------------------------------------------------------- localization
def test_local_radius_none_matches_baseline(cubic_case):
    m, X = cubic_case
    xt = torch.tensor(X)
    v0 = m(xt, W, SIG).item()
    v1 = m(xt, W, SIG, local_radius=None).item()
    assert abs(v0 - v1) < 1e-14
    assert m.V_vox.ndim == 2 and m.m_vox.ndim == 1
    assert m.V_vox.shape[0] == m.m_vox.shape[0]


def test_large_local_radius_renorm_matches_global(cubic_case):
    """Soft weights ≈ 1 on support for huge R + renorm → global Tq."""
    m, X = cubic_case
    kind0, bal0 = m.cfg.local_kind, m.cfg.local_balance
    m.cfg.local_kind = "soft"
    m.cfg.local_balance = "renorm"
    try:
        xt = torch.tensor(X)
        T_global = m.stage(SIG)
        T_local = m._localized_target_spectrum(xt[None], 1e3, SIG)
        if T_local.ndim == 3:
            T_local = T_local[0]
        err = (T_local - T_global).abs().max().item() / T_global.abs().max().item()
        assert err < 1e-6, err
    finally:
        m.cfg.local_kind, m.cfg.local_balance = kind0, bal0


def test_localization_suppresses_distant_target_mass(cubic_case):
    """Small R zeros target mass when the cloud is far from the map."""
    m, X = cubic_case
    kind0, bal0 = m.cfg.local_kind, m.cfg.local_balance
    m.cfg.local_kind = "soft"
    m.cfg.local_balance = "unbalanced"
    try:
        xt = torch.tensor(X)
        w_near = m._voxel_weights(xt[None], 4.0)
        mass_near = float((m.m_vox * w_near[0]).sum().item())
        assert mass_near > 0.5  # most of the unit-mass map

        Y = X + np.array([40.0, 0.0, 0.0])
        w_far = m._voxel_weights(torch.tensor(Y)[None], 2.0)
        mass_far = float((m.m_vox * w_far[0]).sum().item())
        assert mass_far < 1e-8, mass_far

        T_far = m._localized_target_spectrum(torch.tensor(Y)[None], 2.0, SIG)
        if T_far.ndim == 3:
            T_far = T_far[0]
        assert float(T_far.abs().max().item()) < 1e-8
    finally:
        m.cfg.local_kind, m.cfg.local_balance = kind0, bal0


def test_local_soft_fd_gradient(cubic_case):
    """Autograd through soft voxel weights matches finite differences."""
    m, X = cubic_case
    kind0, bal0 = m.cfg.local_kind, m.cfg.local_balance
    m.cfg.local_kind = "soft"
    m.cfg.local_balance = "unbalanced"
    try:
        rng = np.random.default_rng(1)
        Y = X + 0.4 * rng.standard_normal(X.shape)
        y = torch.tensor(Y, requires_grad=True)
        m(y, W, SIG, delta=1e-4, local_radius=4.0).backward()
        ga = y.grad.numpy()
        h = 1e-6
        err = []
        for i in range(0, len(X), 3):
            for k in range(3):
                Yp = Y.copy()
                Yp[i, k] += h
                Ym = Y.copy()
                Ym[i, k] -= h
                gn = (
                    m(torch.tensor(Yp), W, SIG, delta=1e-4, local_radius=4.0).item()
                    - m(torch.tensor(Ym), W, SIG, delta=1e-4, local_radius=4.0).item()
                ) / (2 * h)
                err.append(abs(ga[i, k] - gn) / max(abs(gn), 1e-30))
        assert float(np.median(err)) < 1e-5, float(np.median(err))
    finally:
        m.cfg.local_kind, m.cfg.local_balance = kind0, bal0
