"""Tests for Gabor-windowed sliced W1 (WindowedSlicedOT)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from slicedot import (
    GaborTarget,
    SlicedOT,
    SlicedOTConfig,
    WindowedSlicedOT,
    suggest_L,
    window_atoms,
)
from slicedot.fixtures import X0, sigma_of
from slicedot.fixtures import W as W_np

torch.set_default_dtype(torch.float64)

SIG = sigma_of(2.5)
W = torch.tensor(W_np)
CFG = {"n_dirs": 32, "dt": 0.3, "window": 96.0, "map_cutoff": 1e-7}
S_FLOOR = 3.0 * SIG


def render_ortho(X, sp, NG, sigma):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.array(NG) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG)
    for p, w in zip(X, W_np):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2 * sigma * sigma))
    return T / T.sum(), org, sp


@pytest.fixture(scope="module")
def cubic():
    T, org, sp = render_ortho(X0, 0.45, (48, 48, 48), SIG)
    return torch.tensor(T), org, torch.tensor(sp), torch.tensor(X0)


def _make_windowed(T, org, sp, *, n_windows=1, n_directions=32, s_range=None,
                   backend="direct", seed=0, **kw):
    if s_range is None:
        s_range = (S_FLOOR, max(12.0, 8.0 * SIG))
    return WindowedSlicedOT(
        T, org, sp, SIG,
        s_range=s_range,
        n_windows=n_windows,
        n_directions=n_directions,
        backend=backend,
        seed=seed,
        config=SlicedOTConfig(**CFG, backend=backend),
        **kw,
    )


# ------------------------------------------------------------------ 1 wide-window
def test_wide_window_limit(cubic):
    T, org, sp, X = cubic
    global_ot = SlicedOT(T, org, sp, SIG, SlicedOTConfig(**CFG, backend="direct"))
    # Large enough that σ'/σ → 1, β → 1, and G_s is flat over the fragment.
    s_wide = 5000.0 * SIG
    win = _make_windowed(
        T, org, sp, n_windows=1, n_directions=CFG["n_dirs"],
        s_range=(s_wide, s_wide), backend="direct", seed=0,
    )
    a = X.mean(0, keepdim=True)
    s = torch.tensor([s_wide])
    U = global_ot.U.unsqueeze(0)
    windows = (a, s, U)

    # Compare off the floor: at the true pose both scores are ~0 and relative
    # error is meaningless.
    shift = torch.tensor([5.0, 2.0, -3.0])
    Xs = X + shift
    v_g = global_ot(Xs, W, SIG).item()
    v_w = win(Xs, W, SIG, windows=windows).item()
    rel = abs(v_w - v_g) / max(abs(v_g), 1e-30)
    assert rel < 1e-5, f"score rel err {rel:.3e}  global={v_g:.6g} windowed={v_w:.6g}"

    dv_g = global_ot.deformation(Xs, W, SIG)
    dv_w = win.deformation(Xs, W, SIG, windows=windows)
    rel_d = (dv_w - dv_g).norm() / dv_g.norm().clamp_min(1e-30)
    assert rel_d < 1e-5, f"deformation rel err {rel_d:.3e}"


# ------------------------------------------------------------- 2 gauge invariance
def test_gauge_invariance(cubic):
    T, org, sp, X = cubic
    shift = np.array([1.7, -0.4, 2.2])
    win0 = _make_windowed(T, org, sp, n_windows=2, seed=1)
    a, s, U = win0.sample_windows(X)
    # Reset generator counter so the same windows can be reused conceptually;
    # pass windows explicitly for both gauges.
    v0 = win0(X, W, SIG, windows=(a, s, U)).item()
    dv0 = win0.deformation(X, W, SIG, windows=(a, s, U))

    org1 = org + shift
    X1 = X + torch.tensor(shift)
    a1 = a + torch.tensor(shift)
    win1 = _make_windowed(T, org1, sp, n_windows=2, seed=1)
    v1 = win1(X1, W, SIG, windows=(a1, s, U)).item()
    dv1 = win1.deformation(X1, W, SIG, windows=(a1, s, U))

    assert abs(v0 - v1) < 1e-8, (v0, v1)
    assert (dv0 - dv1).abs().max() < 1e-8


# -------------------------------------------------------- 3 finite-difference grad
@pytest.mark.parametrize("backend", ["direct", "grid", "grid_custom"])
@pytest.mark.parametrize("s_val", [S_FLOOR * 1.05, 8.0])
def test_finite_difference_grad(cubic, backend, s_val):
    T, org, sp, X = cubic
    win = _make_windowed(
        T, org, sp, n_windows=1, n_directions=16,
        s_range=(s_val, s_val), backend=backend, seed=2,
    )
    rng = np.random.default_rng(0)
    Y = X.numpy() + 0.2 * rng.standard_normal(X.shape)
    win._gen_count = 0
    a, s, U = win.sample_windows(torch.tensor(Y))
    a, s, U = a[:1], torch.tensor([s_val]), U[:1]
    # Freeze C at the base pose so FD matches the C-detached autograd path
    # (§2.5): black-box FD of the full map would include ∂C/∂r noise.
    C0, _, _, _ = window_atoms(torch.tensor(Y), SIG, a, s)

    y = torch.tensor(Y, requires_grad=True)
    win(y, W, SIG, delta=1e-4, windows=(a, s, U), freeze_C=C0).backward()
    ga = y.grad.detach().clone()

    h = 1e-5
    err = []
    for i in range(0, len(Y), 3):
        for k in range(3):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[i, k] += h
            Ym[i, k] -= h
            vp = win(
                torch.tensor(Yp), W, SIG, delta=1e-4, windows=(a, s, U), freeze_C=C0,
            ).item()
            vm = win(
                torch.tensor(Ym), W, SIG, delta=1e-4, windows=(a, s, U), freeze_C=C0,
            ).item()
            gn = (vp - vm) / (2 * h)
            err.append(abs(ga[i, k].item() - gn) / max(abs(gn), 1e-30))
    med = float(np.median(err))
    assert med < 1e-4, f"backend={backend} s={s_val} fd_med={med:.3e}"


# ----------------------------------------------- 4 windowed deformation oracle
def test_windowed_deformation_oracle(cubic):
    T, org, sp, X = cubic
    shift = torch.tensor([5.0, 2.0, -3.0])
    Xs = X + shift
    # s large enough that a single window covers fragment + shift; the
    # n_windows sweep is what catches sum-instead-of-average (§2.6).
    s_val = 50.0
    mags = []
    for nw in (1, 4, 16):
        win = _make_windowed(
            T, org, sp, n_windows=nw, n_directions=32,
            s_range=(s_val, s_val), backend="direct", seed=3,
        )
        a = Xs.mean(0, keepdim=True).expand(nw, 3) + 1.0 * torch.randn(nw, 3)
        s = torch.full((nw,), s_val)
        U0 = win.sample_windows(Xs)[2][:1]
        U = U0.expand(nw, -1, -1).contiguous()
        dv = win.deformation(Xs, W, SIG, windows=(a, s, U))
        err = ((Xs + dv) - X).norm(dim=-1).max().item()
        assert err < 0.05, f"n_windows={nw} step_err={err:.4f}"
        mags.append(dv.norm(dim=-1).mean().item())
    # Sum-instead-of-average would grow ~linearly with n_windows
    assert mags[2] < 2.0 * mags[0], f"step mag grew with n_windows: {mags}"
    assert abs(mags[2] - mags[0]) / max(mags[0], 1e-30) < 0.05


# ----------------------------------------------------- 5 partition of unity / C
def test_partition_of_unity_and_C_detached(cubic):
    _, _, _, X = cubic
    s = torch.tensor([6.0])
    # Monte Carlo ∫ C da ≈ constant independent of r (uniform a over large box)
    rng = torch.Generator().manual_seed(0)
    Nmc = 4000
    lo = X.min(0).values - 3 * s
    hi = X.max(0).values + 3 * s
    a = lo + (hi - lo) * torch.rand(Nmc, 3, generator=rng)
    vol = (hi - lo).prod()
    C, _, _, _ = window_atoms(X, SIG, a, s.expand(Nmc))
    # C (1, Nmc, N); estimate ∫ C_j da ≈ vol * mean_a C_j
    integ = vol * C[0].mean(0)                                      # (N,)
    assert integ.std() / integ.mean() < 0.15, integ

    # No gradient through C (detached at construction).  β is independent of r
    # when σ is constant; the real Jacobian is through r'.
    x = X.clone().requires_grad_(True)
    C2, _, rp2, _ = window_atoms(x, SIG, a[:4], s.expand(4))
    assert not C2.requires_grad
    assert rp2.requires_grad
    g_rp = torch.autograd.grad(rp2.sum(), x, retain_graph=True)[0]
    g_both = torch.autograd.grad(rp2.sum() + C2.sum(), x)[0]
    assert g_rp is not None and g_rp.abs().sum() > 0
    assert torch.allclose(g_rp, g_both)


# --------------------------------------------------------------- 6 mass identity
def test_mass_identity(cubic):
    T, org, sp, X = cubic
    win = _make_windowed(T, org, sp, n_windows=3, seed=4)
    a, s, U = win.sample_windows(X)
    C, _, r_prime, sigma_prime = window_atoms(X, SIG, a, s)
    Tq = win._target_spectrum_direct(a, s, U)
    q0 = (win.ot.qk == 0).nonzero(as_tuple=True)[0]
    assert q0.numel() > 0
    k0 = int(q0[0])
    for i in range(a.shape[0]):
        w_eff = W * C[0, i]
        m = w_eff.sum()
        Mq = win._model_spectrum(
            r_prime[:, i], w_eff.unsqueeze(0), sigma_prime[:, i], U[i], "direct",
        )
        assert abs(Mq[0, 0, k0].real.item() - m.item()) < 1e-10
        n = win._window_mass_target(a[i], s[i])
        assert abs(Tq[i, 0, k0].real.item() - n.item()) < 1e-10


# -------------------------------------------------- 7 Gabor interpolation accuracy
def test_gabor_interpolation_accuracy():
    """Tiny 2-atom map so the lattice precompute stays cheap on CI."""
    sigma = SIG
    X = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.5, -0.5]])
    w = np.array([0.5, 0.5])
    NG = (20, 20, 20)
    sp = np.ones(3) * 0.5
    org = -0.5 * (np.array(NG) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    Tm = np.zeros(NG)
    for p, wi in zip(X.numpy(), w):
        Tm += wi * np.exp(-((G - p) ** 2).sum(-1) / (2 * sigma * sigma))
    Tm = Tm / Tm.sum()
    T = torch.tensor(Tm)
    sp_t = torch.tensor(sp)
    # Modest qmax keeps the a-field smoother so trilinear hits 1e-3 at s/8 spacing.
    cfg = SlicedOTConfig(
        n_dirs=8, dt=0.4, window=40.0, map_cutoff=1e-7, backend="direct", qmax=0.2,
    )
    win = WindowedSlicedOT(
        T, org, sp_t, sigma, s_range=(6.0, 10.0), n_windows=1, n_directions=8,
        backend="direct", seed=5, config=cfg,
    )
    ot = win.ot
    region_min = X.min(0).values - 2.5
    region_max = X.max(0).values + 2.5
    gt = GaborTarget.precompute(
        T, org, sp_t, ot.centre, ot.qk, ot.keep_idx, ot.P, ot.dt,
        s_range=(6.0, 10.0), n_s=16, n_dirs=8, n_rotations=1,
        region_min=region_min, region_max=region_max,
        lattice_factor=0.06, seed=0,
    )
    rng = torch.Generator().manual_seed(7)
    n_test = 100
    lo = gt.a_lattice.min(0).values + gt.spacing_a
    hi = gt.a_lattice.max(0).values - gt.spacing_a
    a = lo + (hi - lo) * torch.rand(n_test, 3, generator=rng)
    log_s = math.log(6.5) + torch.rand(n_test, generator=rng) * (
        math.log(9.5) - math.log(6.5)
    )
    s = torch.exp(log_s)
    U = gt.U_pool[0]
    T_dir = win._target_spectrum_direct(a, s, U.unsqueeze(0).expand(n_test, -1, -1))
    T_int_u = gt.interpolate(a, s)[:, : U.shape[0]]
    num = (T_int_u - T_dir).reshape(n_test, -1).norm(dim=-1)
    den = T_dir.reshape(n_test, -1).norm(dim=-1).clamp_min(1e-30)
    rel = (num / den).max().item()
    assert rel < 1e-3, f"max frobenius rel err {rel:.3e}"

    gt_coarse = GaborTarget.precompute(
        T, org, sp_t, ot.centre, ot.qk, ot.keep_idx, ot.P, ot.dt,
        s_range=(6.0, 10.0), n_s=16, n_dirs=8, n_rotations=1,
        region_min=region_min, region_max=region_max,
        lattice_factor=1.5, seed=0,
    )
    T_c = gt_coarse.interpolate(a, s)[:, : U.shape[0]]
    num_c = (T_c - T_dir).reshape(n_test, -1).norm(dim=-1)
    rel_c = (num_c / den).max().item()
    assert rel_c > rel, f"coarse err {rel_c:.3e} should exceed fine {rel:.3e}"


# ------------------------------------------------------- 8 constraint enforcement
def test_s_constraint(cubic):
    T, org, sp, X = cubic
    with pytest.raises(ValueError, match="3\\*sigma"):
        _make_windowed(T, org, sp, s_range=(0.5 * SIG, 12.0))
    win = _make_windowed(T, org, sp, s_range=(S_FLOOR, 12.0), n_windows=1)
    a = X.mean(0, keepdim=True)
    s = torch.tensor([2.0 * SIG])  # below 3σ
    U = torch.nn.functional.normalize(torch.randn(1, 8, 3), dim=-1)
    with pytest.raises(ValueError, match="3\\*sigma"):
        win(X, W, SIG, windows=(a, s, U))


# -------------------------------------------------------------- 9 reproducibility
def test_reproducibility(cubic):
    T, org, sp, X = cubic
    w1 = _make_windowed(T, org, sp, n_windows=4, seed=99)
    w2 = _make_windowed(T, org, sp, n_windows=4, seed=99)
    a1, s1, U1 = w1.sample_windows(X)
    a2, s2, U2 = w2.sample_windows(X)
    assert torch.equal(a1, a2) and torch.equal(s1, s2) and torch.equal(U1, U2)
    v1 = w1(X, W, SIG, windows=(a1, s1, U1))
    v2 = w2(X, W, SIG, windows=(a2, s2, U2))
    assert torch.equal(v1, v2)


def test_suggest_L():
    L = suggest_L(6.0, 2.5)
    assert L == math.ceil(math.pi * 4.0 * 6.0 / 2.5)
    with pytest.raises(ValueError):
        suggest_L(-1.0, 2.5)


# ------------------------------------------- Phase 3: grid vs direct agreement
@pytest.mark.parametrize("backend", ["grid", "grid_custom"])
def test_grid_matches_direct(cubic, backend):
    T, org, sp, X = cubic
    # Wide window → σ'≈σ; fine dt / σ_grid so the Agarwal residual is tiny.
    s_val = 50.0
    cfg = {
        "n_dirs": 16, "dt": 0.1, "window": 96.0, "map_cutoff": 1e-7,
        "sigma_grid": 0.2, "n_sigma": 6.0,
    }
    win_d = WindowedSlicedOT(
        T, org, sp, SIG, s_range=(s_val, s_val), n_windows=1, n_directions=16,
        backend="direct", seed=11,
        config=SlicedOTConfig(**cfg, backend="direct"),
    )
    win_g = WindowedSlicedOT(
        T, org, sp, SIG, s_range=(s_val, s_val), n_windows=1, n_directions=16,
        backend=backend, seed=11,
        config=SlicedOTConfig(**cfg, backend=backend),
    )
    a = X.mean(0, keepdim=True)
    s = torch.tensor([s_val])
    U = win_d.sample_windows(X)[2][:1]
    windows = (a, s, U)
    Xs = X + torch.tensor([1.0, -0.5, 0.8])
    # Compare model spectra at the windowed pose (isolates the M backend).
    C, _, rp, sp_ = window_atoms(Xs, SIG, a, s)
    w_eff = (W * C[0, 0]).unsqueeze(0)                              # (1, N)
    Md = win_d._model_spectrum(rp[:, 0], w_eff, sp_[:, 0], U[0], "direct")
    Mg = win_g._model_spectrum(rp[:, 0], w_eff, sp_[:, 0], U[0], backend)
    rel = (Md - Mg).abs().max() / Md.abs().max().clamp_min(1e-30)
    assert rel.item() < 1e-6, f"backend={backend} Mq rel={rel.item():.3e}"
    vd = win_d(Xs, W, SIG, windows=windows)
    vg = win_g(Xs, W, SIG, windows=windows)
    rel_v = (vd - vg).abs() / vd.abs().clamp_min(1e-30)
    assert rel_v.item() < 1e-5, f"backend={backend} score rel={rel_v.item():.3e}"
