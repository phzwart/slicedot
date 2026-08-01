#!/usr/bin/env python3
"""Generate pedagogical figures for the visual guide in docs/paper/guide/.

Writes PNGs (and matching PDFs) into fig/.  Requires the paper extra:

    uv sync --extra paper
    uv run python docs/paper/guide/make_figures.py

Phenol application figures read committed pose caches:

* 24–27  ``fig/cache/phenol_apps.npz``              (extended @ 1.5 Å)
* 28–31  ``fig/cache/phenol_apps_zigzag_3A.npz``    (zigzag @ 3 Å)
* 32     ``fig/32_peptide_AFSSFN_pipeline_panels_3A.png``
         (3-D AFSSFN @ 3 Å; from ``leucine_3d_reach/make_pipeline_panels.py``)

Build or refresh the phenol caches with:

    uv run python docs/paper/guide/build_phenol_apps_cache.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from scipy.optimize import linear_sum_assignment
from scipy.special import erf

FIG_DIR = Path(__file__).resolve().parent / "fig"

# Source / target palette (not purple, not cream/terracotta).
C_MU = "#1F6B8A"       # deep teal-blue — source μ
C_NU = "#C45C26"       # burnt orange — target ν
C_PLAN = "#2F5D3A"     # forest — coupling / transport
C_INK = "#1A1A1A"
C_MUTED = "#6B6B6B"
C_GRID = "#E6E2DA"
C_BG = "#FAF8F4"
C_FILL_MU = "#1F6B8A33"
C_FILL_NU = "#C45C2633"


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": C_BG,
        "axes.edgecolor": C_INK,
        "axes.labelcolor": C_INK,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def _save(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote fig/{stem}.png")


# --------------------------------------------------------------------------- 1D
def _mixtures_1d(x):
    """Two equal-mass 1-D mixtures used throughout the 1-D figures."""
    def g(m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    mu = 0.55 * g(-2.2, 0.55) + 0.45 * g(-0.4, 0.70)
    nu = 0.40 * g(0.8, 0.50) + 0.60 * g(2.6, 0.65)
    # Renormalise to equal mass on the grid (trapezoid).
    dx = x[1] - x[0]
    mu = mu / (mu.sum() * dx)
    nu = nu / (nu.sum() * dx)
    return mu, nu


def fig_1d_densities():
    x = np.linspace(-5, 5, 801)
    mu, nu = _mixtures_1d(x)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.fill_between(x, mu, color=C_MU, alpha=0.28, linewidth=0)
    ax.fill_between(x, nu, color=C_NU, alpha=0.28, linewidth=0)
    ax.plot(x, mu, color=C_MU, lw=2.2, label=r"source $\mu$")
    ax.plot(x, nu, color=C_NU, lw=2.2, label=r"target $\nu$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("density")
    ax.set_title("Two equal-mass densities on the line")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, max(mu.max(), nu.max()) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "01_1d_densities")


def fig_1d_cdf_and_w1():
    x = np.linspace(-5, 5, 801)
    dx = x[1] - x[0]
    mu, nu = _mixtures_1d(x)
    Fmu = np.cumsum(mu) * dx
    Fnu = np.cumsum(nu) * dx
    Fmu /= Fmu[-1]
    Fnu /= Fnu[-1]
    diff = np.abs(Fmu - Fnu)
    w1 = float(diff.sum() * dx)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 0.9], "hspace": 0.18})

    ax = axes[0]
    ax.plot(x, Fmu, color=C_MU, lw=2.2, label=r"$F_\mu$")
    ax.plot(x, Fnu, color=C_NU, lw=2.2, label=r"$F_\nu$")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative distributions")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(-0.02, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.fill_between(x, diff, color=C_PLAN, alpha=0.35, linewidth=0)
    ax.plot(x, diff, color=C_PLAN, lw=2.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|F_\mu - F_\nu|$")
    ax.set_title(rf"$W_1(\mu,\nu)=\int|F_\mu-F_\nu|\,dt$  $\approx$  {w1:.3f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-5, 5)

    _save(fig, "02_1d_cdf_w1")


def fig_1d_monotone_map():
    """Discrete equal-mass particles: sort → match → move."""
    n = 12
    # Sample from the continuous mixtures via inverse-CDF of a grid.
    x = np.linspace(-5, 5, 2001)
    dx = x[1] - x[0]
    mu, nu = _mixtures_1d(x)
    Fmu = np.cumsum(mu) * dx
    Fnu = np.cumsum(nu) * dx
    Fmu /= Fmu[-1]
    Fnu /= Fnu[-1]
    u = (np.arange(n) + 0.5) / n
    xs = np.interp(u, Fmu, x)
    ys = np.interp(u, Fnu, x)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y0, y1 = 0.0, 1.0
    ax.scatter(xs, np.full(n, y0), s=70, c=C_MU, zorder=3, label=r"source samples")
    ax.scatter(ys, np.full(n, y1), s=70, c=C_NU, zorder=3, label=r"target samples")
    for a, b in zip(xs, ys):
        ax.plot([a, b], [y0, y1], color=C_PLAN, lw=1.4, alpha=0.85)
    ax.set_yticks([y0, y1])
    ax.set_yticklabels([r"$\mu$", r"$\nu$"])
    ax.set_xlabel(r"$t$")
    ax.set_title("Monotone rearrangement: sorted mass matches sorted mass")
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.set_xlim(-4.5, 4.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "03_1d_monotone_map")


# ----------------------------------------------------------------------- 2D OT
# Shared Gaussian-mixture components so continuous densities and discrete
# particles are the same object.
_MU_COMPS = [  # (cx, cy, sx, sy, weight)
    (-1.6, -0.4, 0.85, 0.70, 1.0),
    (-0.3, 1.2, 0.55, 0.55, 0.7),
]
_NU_COMPS = [
    (1.5, 0.5, 0.70, 0.70, 1.0),
    (0.5, -1.6, 0.65, 0.55, 1.0),
]


def _mixture_density(xy, comps):
    """Density = the Gaussian mixture itself (no extra blur)."""
    X, Y = np.meshgrid(xy, xy, indexing="xy")
    dens = np.zeros_like(X)
    for cx, cy, sx, sy, a in comps:
        dens += a * np.exp(-0.5 * (((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2))
    dens = np.clip(dens, 0, None)
    dA = (xy[1] - xy[0]) ** 2
    dens /= dens.sum() * dA
    return dens


def _blob_pair(n=160):
    """Two soft 2-D densities of equal mass on a square grid."""
    xy = np.linspace(-4.0, 4.0, n)
    return xy, _mixture_density(xy, _MU_COMPS), _mixture_density(xy, _NU_COMPS)


def _sample_from_density(dens, xy, n, rng):
    """Equal-mass particles by sampling the density grid itself.

    Each draw picks a voxel with probability ∝ dens, then jittered inside
    that voxel — so the discrete cloud is a Monte-Carlo picture of dens.
    """
    p = dens.ravel().astype(float)
    p = p / p.sum()
    idx = rng.choice(p.size, size=n, replace=True, p=p)
    ngrid = dens.shape[0]
    ix = idx % ngrid
    iy = idx // ngrid
    dx = xy[1] - xy[0]
    jitter = rng.uniform(-0.45, 0.45, size=(n, 2)) * dx
    return np.column_stack([xy[ix], xy[iy]]) + jitter


def fig_kantorovich_densities():
    xy, mu, nu = _blob_pair()
    extent = [xy[0], xy[-1], xy[0], xy[-1]]

    cmap_mu = LinearSegmentedColormap.from_list("mu", ["#FAF8F4", C_MU])
    cmap_nu = LinearSegmentedColormap.from_list("nu", ["#FAF8F4", C_NU])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for ax, dens, cmap, title, c in [
        (axes[0], mu, cmap_mu, r"source $\mu(x)$", C_MU),
        (axes[1], nu, cmap_nu, r"target $\nu(y)$", C_NU),
    ]:
        ax.imshow(dens, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        ax.set_title(title, color=c)
        ax.set_xlabel(r"$x_1$" if ax is axes[0] else r"$y_1$")
        ax.set_ylabel(r"$x_2$" if ax is axes[0] else r"$y_2$")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
    fig.suptitle("Kantorovich setting: rearrange $\\mu$ into $\\nu$ in the plane",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, "04_kantorovich_densities")


def fig_kantorovich_coupling():
    """Discretise μ, ν by equal-mass samples, then show the assignment plan."""
    rng = np.random.default_rng(7)
    xy, mu, nu = _blob_pair(n=220)
    n_pts = 56
    xs = _sample_from_density(mu, xy, n_pts, rng)
    ys = _sample_from_density(nu, xy, n_pts, rng)

    C = np.linalg.norm(xs[:, None, :] - ys[None, :, :], axis=-1)
    row, col = linear_sum_assignment(C)
    cost = float(C[row, col].mean())

    extent = [xy[0], xy[-1], xy[0], xy[-1]]
    cmap_mu = LinearSegmentedColormap.from_list("mu", ["#FAF8F4", C_MU])
    cmap_nu = LinearSegmentedColormap.from_list("nu", ["#FAF8F4", C_NU])

    fig = plt.figure(figsize=(8.6, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.22)
    ax_mu = fig.add_subplot(gs[0, 0])
    ax_nu = fig.add_subplot(gs[0, 1])
    ax_plan = fig.add_subplot(gs[1, :])

    def _panel(ax, dens, cmap, pts, color, title):
        ax.imshow(dens, origin="lower", extent=extent, cmap=cmap, aspect="equal",
                  vmin=0, vmax=np.percentile(dens[dens > 0], 99))
        ax.scatter(pts[:, 0], pts[:, 1], s=28, c=color, zorder=3,
                   edgecolors="white", linewidths=0.35)
        ax.set_xlim(-3.8, 3.8)
        ax.set_ylim(-3.8, 3.8)
        ax.set_title(title, color=color, fontsize=11)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])

    _panel(ax_mu, mu, cmap_mu, xs, C_MU,
           r"$\mu$ and its equal-mass samples")
    _panel(ax_nu, nu, cmap_nu, ys, C_NU,
           r"$\nu$ and its equal-mass samples")

    # Assignment: faint densities + particles + matching edges.
    ax_plan.imshow(mu, origin="lower", extent=extent, cmap=cmap_mu, aspect="equal",
                   alpha=0.45, vmin=0, vmax=np.percentile(mu[mu > 0], 99))
    ax_plan.imshow(nu, origin="lower", extent=extent, cmap=cmap_nu, aspect="equal",
                   alpha=0.45, vmin=0, vmax=np.percentile(nu[nu > 0], 99))
    segs = [[xs[i], ys[j]] for i, j in zip(row, col)]
    ax_plan.add_collection(
        LineCollection(segs, colors=C_PLAN, linewidths=0.85, alpha=0.75))
    ax_plan.scatter(xs[:, 0], xs[:, 1], s=28, c=C_MU, zorder=3,
                    label=r"samples of $\mu$", edgecolors="white", linewidths=0.35)
    ax_plan.scatter(ys[:, 0], ys[:, 1], s=28, c=C_NU, zorder=3,
                    label=r"samples of $\nu$", edgecolors="white", linewidths=0.35)
    ax_plan.set_aspect("equal")
    ax_plan.set_xlim(-3.8, 3.8)
    ax_plan.set_ylim(-3.8, 3.8)
    ax_plan.set_xlabel(r"$x_1$")
    ax_plan.set_ylabel(r"$x_2$")
    ax_plan.set_title(
        rf"Optimal assignment  (mean $|x-y|$ $\approx$ {cost:.2f})")
    ax_plan.legend(frameon=False, loc="upper left", fontsize=9)
    ax_plan.spines["top"].set_visible(False)
    ax_plan.spines["right"].set_visible(False)

    fig.suptitle(
        "Discretise each density, then Kantorovich is an assignment problem",
        fontsize=12, y=0.98)
    _save(fig, "05_kantorovich_coupling")


def fig_kantorovich_lp_sketch():
    """Schematic of the coupling π as a joint with fixed marginals."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Main joint square.
    joint = Rectangle((2.2, 1.4), 4.6, 4.6, fill=True, facecolor="#E8EEE8",
                      edgecolor=C_PLAN, linewidth=2.0)
    ax.add_patch(joint)
    ax.text(4.5, 3.7, r"coupling $\pi(x,y)$", ha="center", va="center",
            fontsize=13, color=C_PLAN)

    # Marginal bars.
    ax.add_patch(Rectangle((2.2, 6.15), 4.6, 0.55, facecolor=C_FILL_MU,
                           edgecolor=C_MU, linewidth=1.6))
    ax.text(4.5, 6.42, r"marginal $\mu(x)=\int\pi\,dy$", ha="center", va="center",
            color=C_MU, fontsize=10)

    ax.add_patch(Rectangle((7.0, 1.4), 0.55, 4.6, facecolor=C_FILL_NU,
                           edgecolor=C_NU, linewidth=1.6))
    ax.text(7.28, 3.7, r"$\nu(y)=\int\pi\,dx$", ha="center", va="center",
            color=C_NU, fontsize=10, rotation=90)

    ax.text(5.0, 0.55,
            r"minimise $\iint c(x,y)\,\pi(x,y)\,dx\,dy$  with  $c=\|x-y\|$",
            ha="center", va="center", fontsize=11, color=C_INK)
    ax.set_title("Kantorovich relaxation: optimise over couplings, not maps",
                 fontsize=12, pad=8)
    _save(fig, "06_kantorovich_lp_sketch")


# --------------------------------------------------------------- FFT / slicing
def fig_slice_projection():
    """Project a 2-D density onto a direction u; show the 1-D profile."""
    xy, mu, _ = _blob_pair(n=180)
    # Direction at 35°.
    theta = np.deg2rad(35.0)
    u = np.array([np.cos(theta), np.sin(theta)])

    X, Y = np.meshgrid(xy, xy, indexing="xy")
    t = X * u[0] + Y * u[1]
    # Histogram the projected mass.
    t_bins = np.linspace(-5, 5, 161)
    dt = t_bins[1] - t_bins[0]
    centers = 0.5 * (t_bins[:-1] + t_bins[1:])
    weights = mu.ravel() * (xy[1] - xy[0]) ** 2
    hist, _ = np.histogram(t.ravel(), bins=t_bins, weights=weights)
    profile = hist / dt

    fig = plt.figure(figsize=(8.2, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    extent = [xy[0], xy[-1], xy[0], xy[-1]]
    cmap = LinearSegmentedColormap.from_list("mu", ["#FAF8F4", C_MU])
    ax0.imshow(mu, origin="lower", extent=extent, cmap=cmap, aspect="equal")
    # Projection axis through origin.
    s = np.linspace(-3.6, 3.6, 2)
    ax0.plot(s * u[0], s * u[1], color=C_INK, lw=1.8)
    ax0.annotate("", xy=(2.6 * u[0], 2.6 * u[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.6))
    ax0.text(2.9 * u[0], 2.9 * u[1], r"$u$", fontsize=12, color=C_INK)
    # Faint perpendicular iso-lines (level sets of t = u·x).
    for lev in [-2, -1, 0, 1, 2]:
        # Line: u·x = lev → perpendicular direction v.
        v = np.array([-u[1], u[0]])
        p = lev * u
        ax0.plot([p[0] - 3 * v[0], p[0] + 3 * v[0]],
                 [p[1] - 3 * v[1], p[1] + 3 * v[1]],
                 color=C_MUTED, lw=0.7, alpha=0.55)
    ax0.set_xlim(-3.8, 3.8)
    ax0.set_ylim(-3.8, 3.8)
    ax0.set_title(r"Project $\mu$ onto direction $u$")
    ax0.set_xlabel(r"$x_1$")
    ax0.set_ylabel(r"$x_2$")

    ax1.fill_between(centers, profile, color=C_MU, alpha=0.35, linewidth=0)
    ax1.plot(centers, profile, color=C_MU, lw=2.0)
    ax1.set_xlabel(r"$t = u\cdot x$")
    ax1.set_ylabel(r"$(P_u\mu)(t)$")
    ax1.set_title("One-dimensional projected profile")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(-5, 5)

    fig.suptitle("Slicing turns a 2-D transport problem into many 1-D ones",
                 fontsize=12, y=1.02)
    _save(fig, "07_slice_projection")


def fig_fft_pipeline():
    """Densities → spectra → spectral antiderivative H → |H| cost."""
    # Build two 1-D profiles (projected), then do the FFT route.
    t = np.linspace(-8, 8, 512)
    dt = t[1] - t[0]
    mu, nu = _mixtures_1d(t)
    # Pad already equal length; FFT.
    M = np.fft.fft(mu)
    T = np.fft.fft(nu)
    q = np.fft.fftfreq(t.size, d=dt)
    # Spectral antiderivative of (μ − ν): divide by 2π i q, pin q=0.
    H_hat = np.zeros_like(M)
    mask = np.abs(q) > 1e-12
    H_hat[mask] = (M[mask] - T[mask]) / (2j * np.pi * q[mask])
    H = np.real(np.fft.ifft(H_hat))
    # Pin additive constant so H is zero in the empty left region.
    n_empty = int(0.05 * t.size)
    H = H - H[n_empty]
    # True CDF difference for comparison (integrated densities).
    Fmu = np.cumsum(mu) * dt
    Fnu = np.cumsum(nu) * dt
    Fmu /= Fmu[-1]
    Fnu /= Fnu[-1]
    # H from the spectral route approximates ∫(μ−ν) up to affine; scale to CDF space
    # by noting ∫(μ−ν)= Fμ−Fν after mass normalisation.  For display, overlay
    # Fμ−Fν against H after matching scales at mid-range.
    cdf_diff = Fmu - Fnu

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.30)

    ax = axes[0, 0]
    ax.plot(t, mu, color=C_MU, lw=2.0, label=r"$P_u\mu$")
    ax.plot(t, nu, color=C_NU, lw=2.0, label=r"$P_u\nu$")
    ax.set_title("1. Projected profiles")
    ax.set_xlabel(r"$t$")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-5, 5)

    ax = axes[0, 1]
    # Show |M(q)|, |T(q)| on positive frequencies.
    order = np.argsort(q)
    qp, Mp, Tp = q[order], np.abs(M[order]), np.abs(T[order])
    pos = qp >= 0
    ax.plot(qp[pos], Mp[pos], color=C_MU, lw=1.8, label=r"$|M(q)|$")
    ax.plot(qp[pos], Tp[pos], color=C_NU, lw=1.8, label=r"$|T(q)|$")
    ax.set_title("2. Fourier transforms")
    ax.set_xlabel(r"$q$")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1.2)

    ax = axes[1, 0]
    ax.plot(t, H, color=C_PLAN, lw=2.0, label=r"$H$ (via FFT)")
    ax.plot(t, cdf_diff, color=C_MUTED, lw=1.4, ls="--", label=r"$F_\mu-F_\nu$")
    ax.set_title(r"3. Spectral antiderivative of $M-T$")
    ax.set_xlabel(r"$t$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-5, 5)

    ax = axes[1, 1]
    ax.fill_between(t, np.abs(H), color=C_PLAN, alpha=0.35, linewidth=0)
    ax.plot(t, np.abs(H), color=C_PLAN, lw=2.0)
    w1 = float(np.abs(H).sum() * dt)
    ax.set_title(rf"4. Cost  $\int|H|\,dt$  $\approx$  {w1:.3f}")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|H(t)|$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-5, 5)

    fig.suptitle("Attacking 1-D $W_1$ with FFTs: integration becomes a multiplier",
                 fontsize=12, y=1.01)
    _save(fig, "08_fft_pipeline")


def fig_sliced_average():
    """Several projection directions on a 2-D pair; per-slice W1 bars."""
    xy, mu, nu = _blob_pair(n=160)
    X, Y = np.meshgrid(xy, xy, indexing="xy")
    dA = (xy[1] - xy[0]) ** 2
    L = 8
    thetas = np.linspace(0, np.pi, L, endpoint=False)
    t_bins = np.linspace(-5.5, 5.5, 201)
    dt = t_bins[1] - t_bins[0]
    w1s = []
    for th in thetas:
        u = np.array([np.cos(th), np.sin(th)])
        t_mu = (X * u[0] + Y * u[1]).ravel()
        t_nu = t_mu  # same grid
        h_mu, _ = np.histogram(t_mu, bins=t_bins, weights=mu.ravel() * dA)
        h_nu, _ = np.histogram(t_nu, bins=t_bins, weights=nu.ravel() * dA)
        p_mu = h_mu / dt
        p_nu = h_nu / dt
        Fmu = np.cumsum(p_mu) * dt
        Fnu = np.cumsum(p_nu) * dt
        if Fmu[-1] > 0:
            Fmu /= Fmu[-1]
        if Fnu[-1] > 0:
            Fnu /= Fnu[-1]
        w1s.append(float(np.abs(Fmu - Fnu).sum() * dt))
    w1s = np.asarray(w1s)
    sw1 = float(w1s.mean())

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    ax = axes[0]
    extent = [xy[0], xy[-1], xy[0], xy[-1]]
    ax.contour(xy, xy, mu, levels=6, colors=C_MU, alpha=0.55, linewidths=1.1)
    ax.contour(xy, xy, nu, levels=6, colors=C_NU, alpha=0.55, linewidths=1.1)
    for th in thetas:
        u = np.array([np.cos(th), np.sin(th)])
        s = np.array([-3.2, 3.2])
        ax.plot(s * u[0], s * u[1], color=C_INK, lw=0.9, alpha=0.55)
    ax.set_aspect("equal")
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-3.8, 3.8)
    ax.set_title(rf"$L={L}$ projection directions")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax = axes[1]
    ax.bar(np.arange(L), w1s, color=C_PLAN, alpha=0.85, width=0.7)
    ax.axhline(sw1, color=C_INK, lw=1.4, ls="--",
               label=rf"$\mathcal{{SW}}_1=\mathrm{{mean}}={sw1:.3f}$")
    ax.set_xlabel(r"direction index $\ell$")
    ax.set_ylabel(r"$W_1(P_{u_\ell}\mu,\,P_{u_\ell}\nu)$")
    ax.set_title("Per-slice costs and their average")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.arange(L))

    fig.suptitle(r"Sliced $W_1$: average one-dimensional transport over directions",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "09_sliced_average")


def fig_structure_factor():
    """Schematic: atomic Gaussians → 1-D structure factor along u."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Atoms as Gaussians along a projected axis.
    ax.text(0.3, 4.5, "Model along $u$", fontsize=11, color=C_MU, fontweight="bold")
    xs = np.array([1.2, 2.4, 3.5, 4.3])
    heights = [0.9, 1.3, 0.7, 1.0]
    t = np.linspace(0.5, 5.2, 400)
    profile = np.zeros_like(t)
    for x0, h in zip(xs, heights):
        g = h * np.exp(-0.5 * ((t - x0) / 0.22) ** 2)
        profile += g
        ax.plot(t, 1.2 + g, color=C_MU, lw=1.2, alpha=0.55)
        ax.add_patch(Circle((x0, 1.05), 0.08, color=C_MU, zorder=3))
    ax.plot(t, 1.2 + profile, color=C_MU, lw=2.2)
    ax.text(2.8, 0.55, r"atoms $r_j$  →  projected Gaussians", ha="center",
            fontsize=9, color=C_MUTED)

    # Arrow.
    ax.annotate("", xy=(6.3, 2.5), xytext=(5.4, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.8))
    ax.text(5.85, 2.85, "FFT", ha="center", fontsize=10, color=C_INK)

    # Spectrum panel.
    ax.text(7.0, 4.5, r"Structure factor $M(q)$", fontsize=11, color=C_PLAN,
            fontweight="bold")
    q = np.linspace(0, 4.5, 300)
    # Schematic decaying oscillating spectrum.
    env = np.exp(-0.18 * q ** 2)
    osc = env * np.cos(1.7 * q + 0.4)
    ax.plot(7.0 + q, 2.4 + 1.3 * osc, color=C_PLAN, lw=2.0)
    ax.axhline(2.4, color=C_MUTED, lw=0.7)
    ax.text(9.2, 1.3, r"$M(q)=\sum_j w_j\,e^{-2\pi^2\sigma^2 q^2}\,e^{-2\pi i q\,u\cdot(r_j-c)}$",
            ha="center", fontsize=8.5, color=C_INK)
    ax.text(9.2, 0.55, "form factor  ×  phase", ha="center", fontsize=9, color=C_MUTED)

    ax.set_title("In reciprocal space the projected model is a 1-D structure factor",
                 fontsize=12, pad=6)
    _save(fig, "10_structure_factor")


# --------------------------------------------- dual / reach / force vs length
def _gauss_pdf(t, m, s):
    return np.exp(-0.5 * ((t - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def fig_dual_potential():
    """CDF gap → dual potential f → f' = −sgn(Fμ − Fν)."""
    t = np.linspace(-5, 5, 1201)
    dt = t[1] - t[0]
    # Source left, target right — same 1-D mixtures as §1.
    mu, nu = _mixtures_1d(t)
    Fmu = np.cumsum(mu) * dt
    Fnu = np.cumsum(nu) * dt
    Fmu /= Fmu[-1]
    Fnu /= Fnu[-1]
    gap = Fmu - Fnu
    # Dual: f' = −sgn(gap); integrate with f(−∞)=0.
    fp = -np.sign(gap)
    fp[np.abs(gap) < 1e-12] = 0.0
    f = np.cumsum(fp) * dt
    f -= f[0]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True,
                             gridspec_kw={"hspace": 0.22})

    ax = axes[0]
    ax.plot(t, Fmu, color=C_MU, lw=2.0, label=r"$F_\mu$")
    ax.plot(t, Fnu, color=C_NU, lw=2.0, label=r"$F_\nu$")
    ax.fill_between(t, Fmu, Fnu, color=C_PLAN, alpha=0.18, linewidth=0)
    ax.set_ylabel("CDF")
    ax.set_title(r"Where the CDFs differ, mass still has to move")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(-0.02, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(t, f, color=C_PLAN, lw=2.2)
    ax.set_ylabel(r"$f(t)$")
    ax.set_title(r"Kantorovich dual potential (1-Lipschitz)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.plot(t, fp, color=C_INK, lw=2.0)
    ax.axhline(0, color=C_MUTED, lw=0.7)
    ax.set_ylabel(r"$f'(t)$")
    ax.set_xlabel(r"$t$")
    ax.set_title(r"$f'(t)=-\mathrm{sgn}(F_\mu-F_\nu)$  — a sign, magnitude one")
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlim(-5, 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("The $W_1$ dual: force comes from a sign, not from overlap",
                 fontsize=12, y=1.01)
    _save(fig, "11_dual_potential")


def _two_atom_densities(t, x1, x2, sigma):
    dens = _gauss_pdf(t, x1, sigma) + _gauss_pdf(t, x2, sigma)
    dens = dens / (dens.sum() * (t[1] - t[0]))
    return dens


def _fd_grad_positions(energy_fn, x, eps=1e-4):
    g = np.zeros_like(x)
    e0 = energy_fn(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        g[i] = (energy_fn(xp) - e0) / eps
    return g


def fig_gradient_vs_separation():
    """1-D two-atom toy: |∇E| vs translation for W1, L2, density-at-centre."""
    sigma = 1.0
    spacing = 2.5
    shifts = np.linspace(0.0, 10.0, 61)
    t = np.linspace(-30, 30, 12001)
    dt = t[1] - t[0]
    nu = _two_atom_densities(t, 0.0, spacing, sigma)

    def w1_energy(x):
        mu = _two_atom_densities(t, x[0], x[1], sigma)
        Fmu = np.cumsum(mu) * dt
        Fnu = np.cumsum(nu) * dt
        Fmu /= Fmu[-1]
        Fnu /= Fnu[-1]
        return float(np.abs(Fmu - Fnu).sum() * dt)

    def l2_energy(x):
        mu = _two_atom_densities(t, x[0], x[1], sigma)
        return float(((mu - nu) ** 2).sum() * dt)

    def dac_energy(x):
        # Sum of target density at atomic centres (sign-flipped for minimisation).
        return -float(_gauss_pdf(x, 0.0, sigma).sum()
                      + _gauss_pdf(x, spacing, sigma).sum())

    g_w1, g_l2, g_dac = [], [], []
    for s in shifts:
        x = np.array([s, s + spacing])
        # Project free-atom grad onto the rigid translation (1,1)/√2.
        e = np.array([1.0, 1.0])
        e /= np.linalg.norm(e)
        gw = _fd_grad_positions(w1_energy, x)
        gl = _fd_grad_positions(l2_energy, x)
        gd = _fd_grad_positions(dac_energy, x)
        g_w1.append(abs(float(gw @ e)))
        g_l2.append(abs(float(gl @ e)))
        g_dac.append(abs(float(gd @ e)))

    g_w1 = np.asarray(g_w1)
    g_l2 = np.asarray(g_l2)
    g_dac = np.asarray(g_dac)
    # Normalise each curve by its value at small separation for shape comparison.
    i_ref = int(np.argmin(np.abs(shifts - 1.0)))
    g_w1_n = g_w1 / max(g_w1[i_ref], 1e-30)
    g_l2_n = g_l2 / max(g_l2[i_ref], 1e-30)
    g_dac_n = g_dac / max(g_dac[i_ref], 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))

    ax = axes[0]
    s_show = 6.0
    mu_show = _two_atom_densities(t, s_show, s_show + spacing, sigma)
    ax.fill_between(t, nu, color=C_NU, alpha=0.35, lw=0)
    ax.fill_between(t, mu_show, color=C_MU, alpha=0.35, lw=0)
    ax.plot(t, nu, color=C_NU, lw=2.0, label=r"target $\nu$")
    ax.plot(t, mu_show, color=C_MU, lw=2.0,
            label=rf"model at $s={s_show:.0f}\sigma$")
    ax.set_xlim(-4, 12)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("density")
    ax.set_title("Two-atom model, far from the map")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(shifts / sigma, g_w1_n, color=C_PLAN, lw=2.4, label=r"$W_1$")
    ax.plot(shifts / sigma, g_l2_n, color=C_MUTED, lw=2.0, label=r"$L^2$ residual")
    ax.plot(shifts / sigma, g_dac_n, color=C_NU, lw=2.0, label=r"density at centre")
    ax.set_xlabel(r"separation $s/\sigma$")
    ax.set_ylabel(r"$|\nabla E\cdot e_{\mathrm{trans}}|$  (norm. at $s=\sigma$)")
    ax.set_title("Gradient along the true translation")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.35)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate("plateau", xy=(7.0, 1.0), xytext=(7.8, 0.55),
                fontsize=9, color=C_PLAN,
                arrowprops=dict(arrowstyle="-|>", color=C_PLAN, lw=1.2))

    fig.suptitle(r"$W_1$ keeps a usable gradient when overlap is gone",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "12_gradient_vs_separation")


def fig_force_vs_monge():
    """Force (sign gradient) stays large near the answer; Monge step shrinks."""
    sigma = 1.0
    shifts = np.linspace(0.0, 8.0, 81)
    t = np.linspace(-20, 20, 8001)
    dt = t[1] - t[0]
    nu = _gauss_pdf(t, 0.0, sigma)
    Fnu = np.cumsum(nu) * dt
    Fnu /= Fnu[-1]

    force = []   # |dW1/ds|
    monge = []   # monotone displacement of the model centre
    for s in shifts:
        mu = _gauss_pdf(t, s, sigma)
        Fmu = np.cumsum(mu) * dt
        Fmu /= Fmu[-1]
        w1 = float(np.abs(Fmu - Fnu).sum() * dt)
        eps = 1e-3
        mu_p = _gauss_pdf(t, s + eps, sigma)
        Fmu_p = np.cumsum(mu_p) * dt
        Fmu_p /= Fmu_p[-1]
        force.append(abs((float(np.abs(Fmu_p - Fnu).sum() * dt) - w1) / eps))
        # Monge: map model centre (quantile 0.5 of μ) to the matching ν quantile.
        # For equal-shape Gaussians this is exactly −s.
        q = 0.5
        t_mu = np.interp(q, Fmu, t)
        t_nu = np.interp(q, Fnu, t)
        monge.append(t_nu - t_mu)

    force = np.asarray(force)
    monge = np.asarray(monge)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))

    ax = axes[0]
    ax.plot(shifts / sigma, force, color=C_PLAN, lw=2.4,
            label=r"force $|\partial W_1/\partial s|$")
    ax.plot(shifts / sigma, np.abs(monge) / sigma, color=C_NU, lw=2.4,
            label=r"Monge step $|\delta|/\sigma$")
    ax.set_xlabel(r"separation $s/\sigma$")
    ax.set_ylabel("magnitude")
    ax.set_title("Same plan, two readings")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8.5)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Cartoon: misplaced blob with force arrow (long) vs Monge arrow (exact).
    ax = axes[1]
    ax.set_xlim(-1.5, 7.5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    # Target.
    circ_t = Circle((0.0, 0.0), 0.55, facecolor=C_FILL_NU, edgecolor=C_NU, lw=2.0)
    ax.add_patch(circ_t)
    ax.text(0.0, -0.95, r"target", ha="center", color=C_NU, fontsize=10)
    # Model.
    s0 = 5.0
    circ_m = Circle((s0, 0.0), 0.55, facecolor=C_FILL_MU, edgecolor=C_MU, lw=2.0)
    ax.add_patch(circ_m)
    ax.text(s0, -0.95, r"model", ha="center", color=C_MU, fontsize=10)
    # Force: unit arrow (visualised longer for presence — annotated as sign).
    ax.annotate("", xy=(s0 - 1.6, 0.35), xytext=(s0 + 0.2, 0.35),
                arrowprops=dict(arrowstyle="-|>", color=C_PLAN, lw=2.2))
    ax.text(s0 - 0.7, 0.62, r"force $\sim\mathrm{sgn}$", color=C_PLAN, fontsize=9,
            ha="center")
    # Monge: exact displacement to target.
    ax.annotate("", xy=(0.6, -0.35), xytext=(s0 - 0.6, -0.35),
                arrowprops=dict(arrowstyle="-|>", color=C_NU, lw=2.2))
    ax.text(s0 / 2, -0.68, r"Monge step $=\!-s$", color=C_NU, fontsize=9,
            ha="center")
    ax.set_title("Force points the way; Monge says how far", fontsize=11)

    fig.suptitle("Why search uses the sign, then hands off — or steps by Monge",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "13_force_vs_monge")


def fig_l2_reversal():
    """L2 free-atom grad loses the translation; W1 keeps it.

    Two-atom 1-D toy with a slightly stretched model geometry: once the map
    overlap dies, the L2 self-term (atoms repelling in density space) dominates
    and the cosine against the true displacement collapses / flips sign.
    """
    sigma = 1.0
    target_span = 2.5
    model_span = 3.5  # stretched — self-term never vanishes
    shifts = np.linspace(0.05, 10.0, 60)
    t = np.linspace(-35, 35, 14001)
    dt = t[1] - t[0]
    nu = _two_atom_densities(t, 0.0, target_span, sigma)

    def w1_energy(x):
        mu = _two_atom_densities(t, x[0], x[1], sigma)
        Fmu = np.cumsum(mu) * dt
        Fnu = np.cumsum(nu) * dt
        Fmu /= Fmu[-1]
        Fnu /= Fnu[-1]
        return float(np.abs(Fmu - Fnu).sum() * dt)

    def l2_energy(x):
        mu = _two_atom_densities(t, x[0], x[1], sigma)
        return float(((mu - nu) ** 2).sum() * dt)

    cos_w1, cos_l2 = [], []
    for s in shifts:
        x = np.array([s, s + model_span])
        true = np.array([0.0 - x[0], target_span - x[1]])
        true /= np.linalg.norm(true)
        gw = _fd_grad_positions(w1_energy, x)
        gl = _fd_grad_positions(l2_energy, x)
        dw, dl = -gw, -gl
        cos_w1.append(float(dw @ true / (np.linalg.norm(dw) + 1e-30)))
        cos_l2.append(float(dl @ true / (np.linalg.norm(dl) + 1e-30)))

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.plot(shifts / sigma, cos_w1, color=C_PLAN, lw=2.4, label=r"$W_1$")
    ax.plot(shifts / sigma, cos_l2, color=C_MUTED, lw=2.0, label=r"$L^2$ residual")
    ax.axhline(0, color=C_INK, lw=0.8, ls="--")
    ax.set_xlabel(r"separation $s/\sigma$")
    ax.set_ylabel(r"$\cos(-\nabla E,\;\mathrm{true\ displacement})$")
    ax.set_title(r"$L^2$ loses the direction; $W_1$ keeps it")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.fill_between(shifts / sigma, -1.15, 0, color="#C45C2618", linewidth=0)
    ax.text(7.8, -0.55, "wrong\nhemisphere", ha="center", color=C_NU, fontsize=9)
    _save(fig, "14_l2_reversal")


# --------------------------------------------- Monge step in the sliced setting
def _directions_2d(L: int) -> np.ndarray:
    th = np.arange(L) * np.pi / L + 0.5 * np.pi / L
    return np.stack([np.cos(th), np.sin(th)], axis=1)


def _cdf_gauss_mixture_1d(t, centres, weights, sigma):
    """F_μ(t) for a 1-D Gaussian mixture (analytic erf)."""
    z = (t[None, :] - centres[:, None]) / (sigma * np.sqrt(2.0))
    return (weights[:, None] * 0.5 * (1.0 + erf(z))).sum(0)


def _invert_cdf(F, t_axis, q):
    """Inverse CDF by linear interpolation; q may be scalar or array."""
    F = np.asarray(F, dtype=float)
    # Ensure monotone for interp.
    F = np.maximum.accumulate(F)
    F = F / F[-1]
    return np.interp(np.asarray(q, dtype=float), F, t_axis)


def _sliced_monge_2d(x, w, y_target, w_target, sigma, U):
    """Per-atom Monge displacement in 2-D via slices + M^{-1}.

    x, y_target: (N, 2) atom positions (equal-mass Gaussian mixtures).
    Returns v of shape (N, 2) such that x + v ≈ matching under sliced Monge.
    """
    L = U.shape[0]
    # Build fine 1-D axes covering both clouds.
    span = np.concatenate([x, y_target], axis=0)
    lo = span.min() - 6 * sigma
    hi = span.max() + 6 * sigma
    t_axis = np.linspace(lo, hi, 4001)

    # Per-direction scalar displacements for each source atom.
    d = np.zeros((L, x.shape[0]))
    for ell, u in enumerate(U):
        p = x @ u                         # projected model atoms
        p_tgt = y_target @ u              # projected target atoms
        Fmu_at = _cdf_gauss_mixture_1d(p, p, w, sigma)          # Fμ(p_j)
        Fnu = _cdf_gauss_mixture_1d(t_axis, p_tgt, w_target, sigma)
        t_star = _invert_cdf(Fnu, t_axis, Fmu_at)
        d[ell] = t_star - p

    # Average backprojection, then M^{-1}.
    v = (d[:, :, None] * U[:, None, :]).mean(0)   # (N, 2)
    M = (U[:, :, None] * U[:, None, :]).mean(0)   # (2, 2)
    v = v @ np.linalg.inv(M).T
    return v, d


def fig_monge_quantile_match():
    """One slice: atom → Fμ(p) → invert Fν → displacement δ."""
    sigma = 0.55
    # Source atoms (model) and target atoms along a line for the cartoon.
    # Use a 2-D pair but show the projection along u = e_x.
    x = np.array([[-1.8, 0.3], [-0.6, -0.4], [0.2, 0.5]])
    y = np.array([[1.0, 0.1], [2.0, -0.3], [2.8, 0.4]])
    w = np.ones(3) / 3.0
    u = np.array([1.0, 0.0])
    p = x @ u
    q = y @ u
    t_axis = np.linspace(-4, 5, 2001)
    Fmu = _cdf_gauss_mixture_1d(t_axis, p, w, sigma)
    Fnu = _cdf_gauss_mixture_1d(t_axis, q, w, sigma)
    Fmu_at = _cdf_gauss_mixture_1d(p, p, w, sigma)
    t_star = _invert_cdf(Fnu, t_axis, Fmu_at)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8),
                             gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.28})

    # Top: projected densities + atom ticks.
    ax = axes[0]
    mu = np.zeros_like(t_axis)
    nu = np.zeros_like(t_axis)
    for pj in p:
        mu += w[0] * _gauss_pdf(t_axis, pj, sigma)
    for qj in q:
        nu += w[0] * _gauss_pdf(t_axis, qj, sigma)
    ax.fill_between(t_axis, mu, color=C_MU, alpha=0.30, lw=0)
    ax.fill_between(t_axis, nu, color=C_NU, alpha=0.30, lw=0)
    ax.plot(t_axis, mu, color=C_MU, lw=2.0, label=r"$P_u\mu$")
    ax.plot(t_axis, nu, color=C_NU, lw=2.0, label=r"$P_u\nu$")
    for pj in p:
        ax.axvline(pj, color=C_MU, lw=0.9, alpha=0.7)
    for qj in q:
        ax.axvline(qj, color=C_NU, lw=0.9, alpha=0.7)
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylabel("density")
    ax.set_title(r"One slice: projected model and target")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom: CDF match for the middle atom.
    ax = axes[1]
    j = 1
    ax.plot(t_axis, Fmu, color=C_MU, lw=2.0, label=r"$F_{P_u\mu}$")
    ax.plot(t_axis, Fnu, color=C_NU, lw=2.0, label=r"$F_{P_u\nu}$")
    ax.plot([p[j], p[j]], [0, Fmu_at[j]], color=C_MU, lw=1.2, ls="--")
    ax.plot([-3.5, p[j]], [Fmu_at[j], Fmu_at[j]], color=C_MUTED, lw=1.0, ls=":")
    ax.plot([t_star[j], t_star[j]], [0, Fmu_at[j]], color=C_NU, lw=1.2, ls="--")
    ax.annotate("", xy=(t_star[j], 0.12), xytext=(p[j], 0.12),
                arrowprops=dict(arrowstyle="-|>", color=C_PLAN, lw=2.0))
    ax.text(0.5 * (p[j] + t_star[j]), 0.18,
            rf"$\delta={t_star[j]-p[j]:+.2f}$", ha="center", color=C_PLAN, fontsize=10)
    ax.scatter([p[j]], [Fmu_at[j]], s=50, c=C_MU, zorder=3)
    ax.scatter([t_star[j]], [Fmu_at[j]], s=50, c=C_NU, zorder=3)
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$t=u\cdot x$")
    ax.set_ylabel("CDF")
    ax.set_title(r"Quantile match: $F_{P_u\nu}(p_j+\delta)=F_{P_u\mu}(p_j)$")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Per-slice Monge step: invert the target CDF at the model's quantile",
                 fontsize=12, y=1.01)
    _save(fig, "15_monge_quantile_match")


def fig_monge_backprojection():
    """Several slice displacements → average → M^{-1} recovers a 2-D move."""
    sigma = 0.45
    # Target cloud.
    y = np.array([[0.0, 0.0], [1.2, 0.3], [0.4, 1.1], [-0.5, 0.7]])
    w = np.ones(len(y)) / len(y)
    true_shift = np.array([2.4, -1.1])
    x = y + true_shift  # rigidly misplaced model
    U = _directions_2d(12)
    v, d = _sliced_monge_2d(x, w, y, w, sigma, U)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))

    # Left: geometry with a few slice axes and one atom's per-slice arrows.
    ax = axes[0]
    ax.scatter(y[:, 0], y[:, 1], s=70, c=C_NU, label=r"target", zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.scatter(x[:, 0], x[:, 1], s=70, c=C_MU, label=r"model", zorder=3,
               edgecolors="white", linewidths=0.4)
    j = 0
    for ell in range(0, len(U), 2):
        u = U[ell]
        # Draw direction through the atom.
        sline = np.array([-1.5, 1.5])
        origin = x[j]
        ax.plot(origin[0] + sline * u[0], origin[1] + sline * u[1],
                color=C_MUTED, lw=0.7, alpha=0.55)
        # Per-slice displacement as a segment along u.
        ax.annotate("", xy=origin + d[ell, j] * u, xytext=origin,
                    arrowprops=dict(arrowstyle="-|>", color=C_PLAN, lw=1.3,
                                    alpha=0.85))
    ax.annotate("", xy=x[j] + v[j], xytext=x[j],
                arrowprops=dict(arrowstyle="-|>", color=C_NU, lw=2.4))
    ax.text(*(x[j] + 0.55 * v[j] + np.array([0.15, 0.25])),
            r"$v_j$", color=C_NU, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"Slice displacements $\delta_\ell u_\ell$ and the assembled $v_j$")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: formula schematic + recovery error.
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 9.2, r"Back-projection", ha="center", fontsize=12, color=C_INK,
            fontweight="bold")
    ax.text(5, 7.8,
            r"$\tilde{v}_j = (1/L)\sum_\ell \delta_{\ell j}\, u_\ell$",
            ha="center", fontsize=13)
    ax.text(5, 6.2,
            r"$M=(1/L)\sum_\ell u_\ell u_\ell^T$"
            r"$,\quad v_j = M^{-1}\tilde{v}_j$",
            ha="center", fontsize=13)
    ax.text(5, 4.5,
            r"Rigid translation $t$: each slice sees "
            r"$\delta_\ell = u_\ell\cdot t$",
            ha="center", fontsize=10, color=C_MUTED)
    ax.text(5, 3.5,
            r"$\Rightarrow\ \tilde{v}=Mt\ \Rightarrow\ v=t$",
            ha="center", fontsize=12, color=C_PLAN)
    err = np.linalg.norm(v - (-true_shift)[None, :], axis=1).max()
    ax.text(5, 1.8,
            f"demo: true t=({true_shift[0]:+.1f},{true_shift[1]:+.1f}), "
            f"max atom error {err:.3f}",
            ha="center", fontsize=10, color=C_INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8EEE8",
                      edgecolor=C_PLAN))
    ax.set_title(r"$M^{-1}$ undoes the directional averaging", fontsize=11)

    fig.suptitle("From per-slice lengths to a vector step",
                 fontsize=12, y=1.02)
    _save(fig, "16_monge_backprojection")


def fig_monge_one_step():
    """One Monge step recovers a rigid translation; sign-gradient steps do not."""
    sigma = 0.5
    y = np.array([[0.0, 0.0], [1.4, 0.2], [0.5, 1.2], [-0.6, 0.8], [0.9, -0.7]])
    w = np.ones(len(y)) / len(y)
    true_shift = np.array([3.0, -1.5])
    x0 = y + true_shift
    U = _directions_2d(16)

    # Monge one step.
    v, _ = _sliced_monge_2d(x0, w, y, w, sigma, U)
    x_monge = x0 + v

    # Fake "sign-gradient" steps: unit step along −shift direction, fixed length.
    # (In 1-D W1 the force magnitude is 1; here we take a fixed cap.)
    e = -true_shift / np.linalg.norm(true_shift)
    step = 0.6
    xs_force = [x0.copy()]
    x = x0.copy()
    for _ in range(8):
        x = x + step * e
        xs_force.append(x.copy())

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))

    ax = axes[0]
    ax.scatter(y[:, 0], y[:, 1], s=80, c=C_NU, label="target", zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.scatter(x0[:, 0], x0[:, 1], s=80, c=C_MU, label="start", zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.scatter(x_monge[:, 0], x_monge[:, 1], s=80, facecolors="none",
               edgecolors=C_PLAN, linewidths=1.8, label="after 1 Monge step", zorder=4)
    for a, b in zip(x0, x_monge):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color=C_PLAN, lw=1.4))
    ax.set_aspect("equal")
    ax.set_title("Monge: one step ≈ exact translation")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.scatter(y[:, 0], y[:, 1], s=80, c=C_NU, label="target", zorder=3,
               edgecolors="white", linewidths=0.4)
    # Path of COM under fixed-length force steps.
    coms = np.array([p.mean(0) for p in xs_force])
    ax.plot(coms[:, 0], coms[:, 1], color=C_MUTED, lw=1.5, marker="o", ms=4,
            label="fixed-length sign steps")
    ax.scatter(x0[:, 0], x0[:, 1], s=55, c=C_MU, zorder=3, edgecolors="white",
               linewidths=0.4)
    # Overshoot markers past the target COM.
    ax.scatter([y.mean(0)[0]], [y.mean(0)[1]], s=120, facecolors="none",
               edgecolors=C_NU, linewidths=1.6, linestyle="--", zorder=2)
    ax.set_aspect("equal")
    ax.set_title("Sign force + fixed step: overshoot / hunt")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(r"Step length from Monge — why $M^{-1}$ backprojection exists in code",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "17_monge_one_step")


def fig_monge_vanishes():
    """Monge step length → 0 as the model approaches; force does not."""
    sigma = 0.5
    y = np.array([[0.0, 0.0], [1.2, 0.4], [0.3, 1.0]])
    w = np.ones(len(y)) / len(y)
    U = _directions_2d(16)
    # Approach along a line of translations.
    amounts = np.linspace(0.0, 4.0, 25)
    direction = np.array([1.0, 0.3])
    direction = direction / np.linalg.norm(direction)

    monge_norm = []
    force_proxy = []  # analytic sliced-W1 translation anchor: mean |u·e|
    anchor = float(np.mean(np.abs(U @ direction)))
    for a in amounts:
        x = y + a * direction
        v, _ = _sliced_monge_2d(x, w, y, w, sigma, U)
        monge_norm.append(np.linalg.norm(v.mean(0)))  # COM step length
        force_proxy.append(anchor)  # |∇ SW1| / w  ≈ mean|u·e| for rigid shift

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(amounts, monge_norm, color=C_NU, lw=2.4,
            label=r"Monge COM step $\|v\|$")
    ax.plot(amounts, force_proxy, color=C_PLAN, lw=2.4,
            label=r"force scale $\frac{1}{L}\sum_\ell|u_\ell\cdot e|$")
    ax.set_xlabel(r"rigid separation $\|t\|$")
    ax.set_ylabel("magnitude")
    ax.set_title("Monge vanishes at the solution; the sign force does not")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, max(max(monge_norm), max(force_proxy)) * 1.12)
    _save(fig, "18_monge_vanishes")


# --------------------------------------------- Agarwal / Ten Eyck grid backend
def fig_b_factor_split():
    """Common gridding width + residual B as a reciprocal-space multiplier."""
    q = np.linspace(0, 1.2, 400)
    sigma = 1.0
    sigma_grid = 0.45
    res2 = sigma ** 2 - sigma_grid ** 2
    ff_full = np.exp(-2 * np.pi ** 2 * sigma ** 2 * q ** 2)
    ff_grid = np.exp(-2 * np.pi ** 2 * sigma_grid ** 2 * q ** 2)
    ff_res = np.exp(-2 * np.pi ** 2 * res2 * q ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))

    ax = axes[0]
    ax.plot(q, ff_full, color=C_INK, lw=2.4, label=r"full $e^{-2\pi^2\sigma^2 q^2}$")
    ax.plot(q, ff_grid, color=C_MU, lw=2.0, label=r"grid $e^{-2\pi^2\sigma_g^2 q^2}$")
    ax.plot(q, ff_res, color=C_NU, lw=2.0, label=r"residual $e^{-2\pi^2(\sigma^2-\sigma_g^2)q^2}$")
    ax.plot(q, ff_grid * ff_res, color=C_PLAN, lw=1.6, ls="--",
            label=r"grid $\times$ residual")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel("form factor")
    ax.set_title(r"$B$-factor split")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 9.0, "Why split?", ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 7.6,
            r"Direct SF:  $O(N K)$ phases per direction",
            ha="center", fontsize=10, color=C_MUTED)
    ax.text(5, 6.2,
            r"Grid once at common $\sigma_g$, FFT in $O(P\log P)$",
            ha="center", fontsize=10, color=C_MU)
    ax.text(5, 4.8,
            r"Leftover sharpness: $B_{res}=8\pi^2(\sigma^2-\sigma_g^2)$",
            ha="center", fontsize=10, color=C_NU)
    ax.text(5, 3.2,
            r"$B=8\pi^2\sigma^2$  <=>  $e^{-B q^2/4}=e^{-2\pi^2\sigma^2 q^2}$",
            ha="center", fontsize=10, color=C_INK)
    ax.text(5, 1.6,
            r"Requirement: $\sigma_g \leq \min_j \sigma_j$",
            ha="center", fontsize=10, color=C_PLAN,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8EEE8",
                      edgecolor=C_PLAN))
    ax.set_title("Agarwal / Ten Eyck idea", fontsize=11)

    fig.suptitle("Grid fat, multiply sharp — the crystallographic FFT trick",
                 fontsize=12, y=1.02)
    _save(fig, "19_b_factor_split")


def fig_scatter_fft_pipeline():
    """Atoms → 1-D Gaussian scatter → FFT → residual multiplier → M(q)."""
    dt = 0.25
    P = 256
    sg = 0.5
    sigma = 1.0
    # Phase origin at index 0, matching numpy.fft / the library convention.
    t = np.arange(P) * dt
    # Keep atoms well inside the window so wrap artefacts stay invisible.
    p = np.array([8.0, 12.5, 16.0, 20.5])
    w = np.array([1.0, 1.4, 0.9, 1.1])
    w = w / w.sum()

    A = np.zeros(P)
    Wn = int(np.ceil(4 * sg / dt))
    off = np.arange(-Wn, Wn + 1)
    for pj, wj in zip(p, w):
        i0 = int(np.round(pj / dt))
        idx = (i0 + off) % P
        d = (i0 + off) * dt - pj
        A[idx] += wj * np.exp(-0.5 * (d / sg) ** 2) / (sg * np.sqrt(2 * np.pi)) * dt

    q = np.fft.fftfreq(P, d=dt)
    Mq = np.fft.fft(A)
    res2 = sigma ** 2 - sg ** 2
    ff_res = np.exp(-2 * np.pi ** 2 * res2 * q ** 2)
    Mq_sharp = Mq * ff_res
    Mq_direct = np.zeros(P, dtype=complex)
    for pj, wj in zip(p, w):
        Mq_direct += wj * np.exp(-2 * np.pi ** 2 * sigma ** 2 * q ** 2) \
            * np.exp(-2j * np.pi * q * pj)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.30)
    t_lo, t_hi = 4.0, 26.0

    ax = axes[0, 0]
    ax.vlines(p, 0, w, colors=C_MU, lw=2.0)
    ax.scatter(p, w, s=50, c=C_MU, zorder=3)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(0, w.max() * 1.25)
    ax.set_title(r"1. Projected atoms $p_j=u\cdot(r_j-c)$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("weight")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.fill_between(t, A, color=C_MU, alpha=0.35, lw=0)
    ax.plot(t, A, color=C_MU, lw=2.0)
    ax.set_xlim(t_lo, t_hi)
    ax.set_title(r"2. Scatter Gaussians of width $\sigma_g$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A(t)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    order = np.argsort(q)
    qp = q[order]
    pos = qp >= 0
    ax.plot(qp[pos], np.abs(Mq[order][pos]), color=C_MU, lw=1.8,
            label=r"$|\mathcal{F}A|$")
    ax.plot(qp[pos], ff_res[order][pos], color=C_NU, lw=1.8,
            label=r"$f_{\mathrm{res}}(q)$")
    ax.set_xlim(0, 1.0)
    ax.set_title("3. FFT, then residual multiplier")
    ax.set_xlabel(r"$q$")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    ax.plot(qp[pos], np.abs(Mq_sharp[order][pos]), color=C_PLAN, lw=2.2,
            label=r"gridded $M(q)$")
    ax.plot(qp[pos], np.abs(Mq_direct[order][pos]), color=C_MUTED, lw=1.4,
            ls="--", label=r"direct SF")
    ax.set_xlim(0, 1.0)
    ax.set_title("4. Matches the direct structure factor")
    ax.set_xlabel(r"$q$")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Gridded forward: scatter, FFT, multiply — Agarwal / Ten Eyck",
                 fontsize=12, y=1.01)
    _save(fig, "20_scatter_fft_pipeline")


def fig_gather_kernel():
    """Gather with g itself, not g': 2πiq cancels 1/(2πiq)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))

    # Left: cancellation cartoon in reciprocal space.
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 9.2, "Why the gather kernel is $g$, not $g'$",
            ha="center", fontsize=11, fontweight="bold")
    boxes = [
        (1.0, 7.0, r"phase $\partial_p \sim 2\pi i q$", C_MU),
        (1.0, 5.2, r"spectral $\int$ $=\,1/(2\pi i q)$", C_NU),
        (1.0, 3.4, r"product cancels $\Rightarrow$ no $q$", C_PLAN),
        (1.0, 1.6, r"adjoint of scatter$(g)$ $=$ gather$(g)$", C_INK),
    ]
    for x0, y0, txt, c in boxes:
        ax.add_patch(Rectangle((x0, y0 - 0.55), 8.0, 1.1,
                               facecolor="#FAF8F4", edgecolor=c, lw=1.8))
        ax.text(5, y0, txt, ha="center", va="center", fontsize=10, color=c)
    ax.set_title("Continuum identity", fontsize=11)

    # Right: stencil picture — gather psi with the same Gaussian weights.
    ax = axes[1]
    t = np.linspace(-4, 4, 401)
    sg = 0.7
    g = np.exp(-0.5 * (t / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
    # Fake psi field: smooth sign-like step.
    psi = np.tanh(t / 0.9)
    ax.plot(t, psi, color=C_PLAN, lw=2.2, label=r"$\psi(t)$  (from $\mathrm{sgn}\,H$)")
    ax.fill_between(t, 0, g / g.max() * 0.8, color=C_MU, alpha=0.30, lw=0,
                    label=r"gather weights $\propto g(t-p_j)$")
    ax.axvline(0.0, color=C_MU, lw=1.4, ls="--")
    ax.text(0.15, -0.85, r"$p_j$", color=C_MU, fontsize=11)
    # Indicate the gathered value as a dot.
    wgt = g / g.sum()
    phi = -float((psi * wgt).sum())
    ax.scatter([0], [phi], s=70, c=C_NU, zorder=3)
    ax.text(0.3, phi + 0.15, r"$\phi_j = -(\psi * g)(p_j)$", color=C_NU, fontsize=10)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r"$t$")
    ax.set_title(r"Backward: gather $\psi$ with the atomic Gaussian")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        r"Phase $2\pi i q$ cancels the $1/(2\pi i q)$ from integrating $H$",
        fontsize=12, y=1.02)
    _save(fig, "21_gather_kernel")


def fig_backend_cost():
    """Cost table from the paper: direct vs gridded."""
    N = np.array([23, 200, 2000, 8000])
    direct = np.array([2.6, 22.0, 4594.0, 26079.0])
    grid = np.array([3.9, 5.6, 64.7, 171.4])
    speedup = direct / grid

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))

    ax = axes[0]
    x = np.arange(len(N))
    w = 0.36
    ax.bar(x - w / 2, direct, width=w, color=C_MUTED, label="direct")
    ax.bar(x + w / 2, grid, width=w, color=C_PLAN, label="gridded")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel(r"atom count $N$")
    ax.set_ylabel("time (ms), log scale")
    ax.set_title(r"Value + gradient, $L=24$")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(N, speedup, color=C_NU, lw=2.4, marker="o", ms=7)
    for n, s in zip(N, speedup):
        ax.text(n, s * 1.12, f"{s:.0f}$\\times$", ha="center", fontsize=9,
                color=C_NU)
    ax.set_xscale("log")
    ax.set_xlabel(r"atom count $N$")
    ax.set_ylabel("speedup (direct / grid)")
    ax.set_title("Gridded wins past a few dozen atoms")
    ax.set_ylim(0, max(speedup) * 1.35)
    ax.axhline(1.0, color=C_MUTED, lw=0.8, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Why the grid backend exists in slicedot",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "22_backend_cost")


def fig_algorithm_overview():
    """End-to-end algorithmic overview of SlicedOT."""

    fig, ax = plt.subplots(figsize=(13.2, 9.0))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 94)
    ax.axis("off")

    def tint(color, amount=0.90):
        """Blend a project color toward white while preserving palette coherence."""
        r, g, b = to_rgb(color)
        return (
            r + (1.0 - r) * amount,
            g + (1.0 - g) * amount,
            b + (1.0 - b) * amount,
        )

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------
    def rounded_panel(
        x,
        y,
        w,
        h,
        *,
        fc="white",
        ec=None,
        lw=0.9,
        radius=0.9,
        zorder=1,
    ):
        if ec is None:
            ec = tint(C_MUTED, 0.72)
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=zorder,
        )
        ax.add_patch(patch)
        return patch

    def lane(y, h, label, *, fc, accent):
        # Left gutter for a vertical section title; content cards sit to its right.
        gutter = 4.2
        rounded_panel(
            5,
            y,
            122,
            h,
            fc=fc,
            ec=accent,
            lw=0.8,
            radius=1.1,
            zorder=0,
        )
        # Soft vertical rule separating the title strip from the body.
        ax.plot(
            [5 + gutter, 5 + gutter],
            [y + 1.1, y + h - 1.1],
            color=accent,
            lw=0.8,
            alpha=0.28,
            zorder=1,
        )
        # Font size scales down for long titles so they fit the lane height.
        nchar = len(label.replace("$", "").replace(r"\ell", "l").replace("\\", ""))
        fs = 8.0 if nchar <= 18 else (7.2 if nchar <= 28 else 6.4)
        ax.text(
            5 + 0.55 * gutter,
            y + 0.5 * h,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=fs,
            color=accent,
            fontweight="bold",
            zorder=2,
        )

    def input_card(x, y, w, h, kicker, title, body, *, accent, fc):
        rounded_panel(
            x,
            y,
            w,
            h,
            fc=fc,
            ec=accent,
            lw=1.05,
            radius=0.8,
            zorder=2,
        )
        ax.add_patch(
            Rectangle(
                (x, y + h - 0.52),
                w,
                0.52,
                facecolor=accent,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x + 1.45,
            y + h - 1.45,
            kicker,
            ha="left",
            va="center",
            fontsize=6.6,
            color=accent,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + 1.45,
            y + h - 3.0,
            title,
            ha="left",
            va="center",
            fontsize=8.7,
            color=C_INK,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + 1.45,
            y + 1.25,
            body,
            ha="left",
            va="bottom",
            fontsize=7.55,
            color=C_MUTED,
            linespacing=1.22,
            zorder=4,
        )

    def step_card(
        x,
        y,
        w,
        h,
        number,
        title,
        body,
        *,
        accent,
        fc="white",
        body_fs=7.7,
    ):
        rounded_panel(
            x,
            y,
            w,
            h,
            fc=fc,
            ec=accent,
            lw=1.1,
            radius=0.85,
            zorder=2,
        )
        rounded_panel(
            x + 1.25,
            y + h - 3.1,
            3.0,
            2.05,
            fc=accent,
            ec=accent,
            lw=0.0,
            radius=0.55,
            zorder=3,
        )
        ax.text(
            x + 2.75,
            y + h - 2.08,
            str(number),
            ha="center",
            va="center",
            fontsize=8.2,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + 5.1,
            y + h - 2.05,
            title,
            ha="left",
            va="center",
            fontsize=8.5,
            color=C_INK,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + 1.55,
            y + h - 4.15,
            body,
            ha="left",
            va="top",
            fontsize=body_fs,
            color=C_INK,
            linespacing=1.28,
            zorder=4,
        )

    def strip_card(
        x,
        y,
        w,
        h,
        number,
        title,
        formula,
        *,
        accent,
        fc,
        formula_offset=15.0,
        formula_fs=8.4,
    ):
        rounded_panel(
            x,
            y,
            w,
            h,
            fc=fc,
            ec=accent,
            lw=1.25,
            radius=0.75,
            zorder=2,
        )
        rounded_panel(
            x + 1.0,
            y + 0.72,
            3.0,
            h - 1.44,
            fc=accent,
            ec=accent,
            lw=0.0,
            radius=0.45,
            zorder=3,
        )
        ax.text(
            x + 2.5,
            y + h / 2,
            str(number),
            ha="center",
            va="center",
            fontsize=8.2,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + 5.1,
            y + h / 2,
            title,
            ha="left",
            va="center",
            fontsize=8.3,
            color=accent,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + formula_offset,
            y + h / 2,
            formula,
            ha="left",
            va="center",
            fontsize=formula_fs,
            color=C_INK,
            zorder=4,
        )

    def formula_card(x, y, w, h, number, title, formula, note, *, accent, fc):
        """Two-level card for a long formula plus a short side condition."""
        rounded_panel(
            x,
            y,
            w,
            h,
            fc=fc,
            ec=accent,
            lw=1.25,
            radius=0.78,
            zorder=2,
        )
        rounded_panel(
            x + 1.0,
            y + h - 3.05,
            3.0,
            2.05,
            fc=accent,
            ec=accent,
            lw=0.0,
            radius=0.45,
            zorder=3,
        )
        ax.text(
            x + 2.5,
            y + h - 2.02,
            str(number),
            ha="center",
            va="center",
            fontsize=8.2,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + 5.1,
            y + h - 2.0,
            title,
            ha="left",
            va="center",
            fontsize=8.35,
            color=accent,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + 5.1,
            y + 1.85,
            formula,
            ha="left",
            va="center",
            fontsize=8.35,
            color=C_INK,
            zorder=4,
        )
        ax.text(
            x + w - 1.8,
            y + h - 2.0,
            note,
            ha="right",
            va="center",
            fontsize=7.55,
            color=C_MUTED,
            zorder=4,
        )

    def readout_card(x, y, w, h, title, subtitle, rows, *, accent, fc):
        rounded_panel(
            x,
            y,
            w,
            h,
            fc=fc,
            ec=accent,
            lw=1.25,
            radius=0.9,
            zorder=2,
        )
        ax.add_patch(
            Rectangle(
                (x, y + h - 0.58),
                w,
                0.58,
                facecolor=accent,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x + 1.6,
            y + h - 1.75,
            title,
            ha="left",
            va="center",
            fontsize=9.0,
            color=accent,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + 1.6,
            y + h - 3.1,
            subtitle,
            ha="left",
            va="center",
            fontsize=7.25,
            color=C_MUTED,
            fontweight="bold",
            zorder=4,
        )

        # Distribute rows over the available vertical space. This keeps the
        # three-row Monge card visually balanced against the four-row force card.
        top = y + h - 5.15
        bottom = y + 1.85
        gap = 0.0 if len(rows) == 1 else (top - bottom) / (len(rows) - 1)
        for i, (tag, text, fs) in enumerate(rows):
            yy = top - i * gap
            rounded_panel(
                x + 1.55,
                yy - 0.72,
                3.25,
                1.65,
                fc=accent,
                ec=accent,
                lw=0.0,
                radius=0.45,
                zorder=3,
            )
            ax.text(
                x + 3.17,
                yy + 0.10,
                tag,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
                fontweight="bold",
                zorder=4,
            )
            ax.text(
                x + 5.55,
                yy + 0.08,
                text,
                ha="left",
                va="center",
                fontsize=fs,
                color=C_INK,
                zorder=4,
            )

    def protocol_panel(x, y, w, h):
        rounded_panel(
            x,
            y,
            w,
            h,
            fc="white",
            ec=C_INK,
            lw=1.15,
            radius=0.9,
            zorder=2,
        )
        widths = [37.0, 37.0, w - 74.0]
        starts = [x, x + widths[0], x + widths[0] + widths[1]]
        accents = [C_PLAN, C_NU, C_INK]
        titles = ["REACH", "MOVE", "RESTRAIN + HAND OFF"]
        bodies = [
            r"Use $\nabla E$ to enter the basin"
            + "\n"
            + r"sign force remains $O(1)$",
            r"Use Monge $v_j$"
            + "\n"
            + r"or Adam / ADMM on $\nabla E$",
            r"Apply $P_{\mathrm{restr}}$ to chemistry"
            + "\n"
            + "switch to conventional refinement in-basin",
        ]

        for i, (sx, sw, accent, title, body) in enumerate(
            zip(starts, widths, accents, titles, bodies)
        ):
            if i > 0:
                ax.plot(
                    [sx, sx],
                    [y + 1.0, y + h - 1.0],
                    color=tint(C_MUTED, 0.76),
                    lw=0.9,
                    zorder=3,
                )
            ax.text(
                sx + 2.0,
                y + h - 2.0,
                title,
                ha="left",
                va="center",
                fontsize=7.4,
                color=accent,
                fontweight="bold",
                zorder=4,
            )
            ax.text(
                sx + 2.0,
                y + h - 4.0,
                body,
                ha="left",
                va="top",
                fontsize=7.25,
                color=C_INK,
                linespacing=1.28,
                zorder=4,
            )

        for xx in [x + widths[0], x + widths[0] + widths[1]]:
            ax.text(
                xx,
                y + h / 2,
                "›",
                ha="center",
                va="center",
                fontsize=18,
                color=tint(C_MUTED, 0.58),
                zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.0),
            )

    def arrow(x1, y1, x2, y2, *, c=C_INK, lw=1.35, ms=12):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=ms,
                color=c,
                lw=lw,
                shrinkA=1.0,
                shrinkB=1.0,
                zorder=6,
            )
        )

    def elbow_arrow(points, *, c=C_INK, lw=1.25, ms=11):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=ms,
                color=c,
                lw=lw,
                zorder=6,
            )
        )

    def pill(x, y, w, text, *, fc, tc):
        rounded_panel(
            x,
            y,
            w,
            2.55,
            fc=fc,
            ec=fc,
            lw=0.0,
            radius=1.25,
            zorder=2,
        )
        ax.text(
            x + w / 2,
            y + 1.28,
            text,
            ha="center",
            va="center",
            fontsize=7.15,
            color=tc,
            fontweight="bold",
            zorder=3,
        )

    # ------------------------------------------------------------------
    # Title and high-level visual key
    # ------------------------------------------------------------------
    ax.text(
        66,
        91.0,
        "Algorithmic overview: sliced $W_1$ as a structure-factor computation",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=C_INK,
    )
    pill(30.5, 86.6, 21.5, "FORCE  ·  reach", fc=tint(C_PLAN, 0.88), tc=C_PLAN)
    pill(55.2, 86.6, 21.5, "MONGE  ·  length", fc=tint(C_NU, 0.88), tc=C_NU)
    pill(79.9, 86.6, 21.5, "FFT  ·  evaluate", fc=tint(C_MU, 0.88), tc=C_MU)

    # ------------------------------------------------------------------
    # Lanes
    # ------------------------------------------------------------------
    lane(70.5, 14.2, "INPUTS", fc=tint(C_MUTED, 0.94), accent=C_MUTED)
    lane(42.0, 26.4, "FORWARD COMPUTATION", fc=tint(C_MU, 0.94), accent=C_MU)
    lane(
        20.0,
        20.0,
        "STEPS",
        fc=tint(C_NU, 0.95),
        accent=C_NU,
    )
    lane(2.5, 15.0, "UPDATES", fc=tint(C_PLAN, 0.94), accent=C_PLAN)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    input_card(
        11.5,
        72.1,
        24.5,
        9.6,
        "TARGET",
        "Density map",
        r"$\rho_T$  ·  origin  ·  spacing" + "\n" + r"$\sigma_{\mathrm{data}}$",
        accent=C_NU,
        fc=tint(C_NU, 0.95),
    )
    input_card(
        39.5,
        72.1,
        24.5,
        9.6,
        "MODEL",
        "Atomic model",
        r"$r_j\in\mathbb{R}^3$  ·  $w_j$" + "\n" + r"stage width $\sigma$",
        accent=C_MU,
        fc=tint(C_MU, 0.95),
    )
    input_card(
        67.5,
        72.1,
        24.5,
        9.6,
        "SAMPLING",
        r"Directions $u_\ell$",
        "Fibonacci / semicircle" + "\n" + r"count $L$",
        accent=C_MUTED,
        fc="white",
    )
    input_card(
        95.5,
        72.1,
        24.5,
        9.6,
        "EXECUTION",
        "Backend",
        "direct  |  grid" + "\n" + r"$\sigma_g,\;\Delta t,\;P$",
        accent=C_MUTED,
        fc="white",
    )

    # ------------------------------------------------------------------
    # Forward computation
    # ------------------------------------------------------------------
    step_card(
        11.5,
        55.0,
        33,
        10.4,
        1,
        "TARGET SLICES  ·  precompute",
        r"$T_\ell(q)=\sum_v m_v e^{-2\pi i q\,u_\ell\cdot(r_v-c)}$"
        + "\n"
        + r"apply stage blur as a $q$-space multiplier",
        accent=C_NU,
        body_fs=7.45,
    )
    step_card(
        49,
        55.0,
        28,
        10.4,
        2,
        "PROJECT ATOMS",
        r"$p_{\ell j}=u_\ell\cdot(r_j-c)$"
        + "\n"
        + "shared by the spectral and Monge paths",
        accent=C_MU,
        body_fs=7.55,
    )
    step_card(
        82,
        55.0,
        40,
        10.4,
        3,
        r"MODEL SPECTRUM  $M_\ell(q)$",
        "direct structure factors"
        + "\n"
        + r"or scatter $g_{\sigma_g}$  $\rightarrow$  FFT  $\times$  $f_{\mathrm{res}}(q)$",
        accent=C_PLAN,
        body_fs=7.45,
    )

    formula_card(
        22,
        47.1,
        88,
        6.0,
        4,
        "SPECTRAL ANTIDERIVATIVE",
        r"$H_\ell=\mathcal{F}^{-1}[(M_\ell-T_\ell)/(2\pi i q)]$",
        r"pin:  $H\leftarrow H-H[n_{\mathrm{empty}}]$",
        accent=C_INK,
        fc="white",
    )

    strip_card(
        37,
        42.7,
        58,
        3.65,
        5,
        "SCORE",
        r"$E=\frac{1}{L}\sum_\ell\int |H_\ell|\,dt$   (or log-cosh)",
        accent=C_PLAN,
        fc=tint(C_PLAN, 0.89),
    )

    # Inputs -> forward computation.
    arrow(22.5, 72.1, 22.5, 65.6, c=C_NU)
    elbow_arrow(
        [(51.5, 72.1), (51.5, 69.6), (57.0, 69.6), (57.0, 65.6)],
        c=C_MU,
    )
    elbow_arrow(
        [(80.5, 72.1), (80.5, 68.9), (69.0, 68.9), (69.0, 65.6)],
        c=C_MUTED,
    )
    elbow_arrow(
        [(109.5, 72.1), (109.5, 68.9), (105.0, 68.9), (105.0, 65.6)],
        c=C_MUTED,
    )

    # Internal forward dependencies.
    arrow(77.0, 60.2, 81.8, 60.2, c=C_MU)
    elbow_arrow(
        [(27.0, 55.0), (27.0, 53.8), (43.0, 53.8), (43.0, 53.25)],
        c=C_NU,
    )
    elbow_arrow(
        [(102.0, 55.0), (102.0, 53.8), (91.0, 53.8), (91.0, 53.25)],
        c=C_PLAN,
    )
    arrow(66.0, 47.1, 66.0, 46.5, c=C_INK)

    # ------------------------------------------------------------------
    # Two readings of H_l
    # ------------------------------------------------------------------
    force_rows = [
        ("6a", r"$s_\ell=\mathrm{sgn}(H_\ell)$   (or $\tanh$)", 7.55),
        (
            "6b",
            r"build $\psi_\ell$ on the $e^{-2\pi iqp}$ branch $\times f_{\mathrm{res}}$",
            7.25,
        ),
        ("6c", r"gather with $g$ (not $g'$) $\rightarrow \phi_{\ell j}$", 7.45),
        (
            "6d",
            r"$\nabla_{r_j}E=\frac{1}{L}\sum_\ell w_j\phi_{\ell j}u_\ell$",
            7.55,
        ),
    ]
    length_rows = [
        (
            "7a",
            r"$\delta_{\ell j}=F^{-1}_{P_{u_\ell}\nu}(F_{P_{u_\ell}\mu}(p_{\ell j}))-p_{\ell j}$",
            7.05,
        ),
        ("7b", r"$\tilde v_j=\frac{1}{L}\sum_\ell \delta_{\ell j}u_\ell$", 7.55),
        (
            "7c",
            r"$M=\frac{1}{L}\sum_\ell u_\ell u_\ell^T$,   $v_j=M^{-1}\tilde v_j$",
            7.35,
        ),
    ]

    readout_card(
        11.5,
        22.0,
        50.5,
        14.8,
        "FORCE",
        "dual / search gradient",
        force_rows,
        accent=C_PLAN,
        fc="white",
    )
    readout_card(
        70,
        22.0,
        52,
        14.8,
        "LENGTH",
        "sliced Monge step",
        length_rows,
        accent=C_NU,
        fc="white",
    )

    # The same H feeds both interpretations.
    elbow_arrow(
        [(52.0, 42.7), (52.0, 39.2), (36.0, 39.2), (36.0, 36.95)],
        c=C_PLAN,
    )
    elbow_arrow(
        [(80.0, 42.7), (80.0, 39.2), (96.0, 39.2), (96.0, 36.95)],
        c=C_NU,
    )

    # ------------------------------------------------------------------
    # Update protocol
    # ------------------------------------------------------------------
    protocol_panel(11.5, 4.1, 110.5, 9.5)
    elbow_arrow(
        [(36.0, 22.0), (36.0, 18.65), (28.5, 18.65), (28.5, 13.8)],
        c=C_PLAN,
        lw=1.55,
        ms=12,
    )
    elbow_arrow(
        [(96.0, 22.0), (96.0, 18.65), (65.5, 18.65), (65.5, 13.8)],
        c=C_NU,
        lw=1.55,
        ms=12,
    )

    _save(fig, "23_algorithm_overview")
    return fig


# ------------------------------------------------------------- phenol apps (§9)
PHENOL_CACHE = FIG_DIR / "cache" / "phenol_apps.npz"
PHENOL_CACHE_ZIGZAG_3A = FIG_DIR / "cache" / "phenol_apps_zigzag_3A.npz"


def _load_phenol_apps(path: Path | None = None):
    path = PHENOL_CACHE if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Build it with:\n"
            "  uv run python docs/paper/guide/build_phenol_apps_cache.py"
        )
    return np.load(path, allow_pickle=True)


def _phenol_arrow_tol(data) -> float:
    """Match tolerance scales with stage width (looser at low resolution)."""
    sigma = float(np.asarray(data["sigma"]))
    return float(max(0.35, 0.55 * sigma))


def _phenol_extent_limits(data, *Xs, pad=1.4):
    """Tight ROI around the given poses, clipped to the map extent."""
    pts = np.concatenate([np.asarray(X, dtype=np.float64) for X in Xs], axis=0)
    xmin = float(pts[:, 0].min() - pad)
    xmax = float(pts[:, 0].max() + pad)
    ymin = float(pts[:, 1].min() - pad)
    ymax = float(pts[:, 1].max() + pad)
    ex = np.asarray(data["extent"], dtype=np.float64)
    return (
        (max(xmin, ex[0]), min(xmax, ex[1])),
        (max(ymin, ex[2]), min(ymax, ex[3])),
    )


def _draw_density(ax, data, xlim, ylim):
    rho = np.asarray(data["rhoT"], dtype=np.float64)
    extent = list(np.asarray(data["extent"], dtype=np.float64))
    cmap = LinearSegmentedColormap.from_list(
        "dens", ["#FAF8F4", "#E8D5B5", "#C45C26"],
    )
    ax.imshow(
        rho, origin="lower", extent=extent, cmap=cmap,
        vmin=0.0, vmax=float(rho.max()), interpolation="nearest",
        aspect="equal", zorder=0,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def _draw_bonds(ax, X, bonds, *, color=C_MUTED, lw=1.15, alpha=0.85, zorder=2):
    X = np.asarray(X, dtype=np.float64)
    segs = [[X[i], X[j]] for i, j in np.asarray(bonds, dtype=int)]
    if not segs:
        return
    ax.add_collection(LineCollection(
        segs, colors=color, linewidths=lw, alpha=alpha, zorder=zorder,
    ))


def _draw_atoms(ax, X, *, filled=True, color=C_MU, ms=6.0, zorder=4, mew=1.1):
    X = np.asarray(X, dtype=np.float64)
    if filled:
        ax.plot(
            X[:, 0], X[:, 1], "o", ms=ms, mfc=color, mec=color,
            zorder=zorder, clip_on=False,
        )
    else:
        ax.plot(
            X[:, 0], X[:, 1], "o", ms=ms, mfc="none", mec=color,
            mew=mew, zorder=zorder, clip_on=False,
        )


def _label_arrows(ax, X_true, X_cur, *, tol=0.35, zorder=5):
    """Arrows from each true labelled site to the current carrier of that label."""
    X_true = np.asarray(X_true, dtype=np.float64)
    X_cur = np.asarray(X_cur, dtype=np.float64)
    for i in range(len(X_true)):
        a, b = X_true[i], X_cur[i]
        dist = float(np.linalg.norm(b - a))
        ok = dist <= tol
        color = C_PLAN if ok else C_NU
        if dist < 0.05:
            ax.plot(
                [a[0]], [a[1]], "o", ms=3.2, mfc=color, mec=color, zorder=zorder,
            )
            continue
        ax.annotate(
            "",
            xy=(b[0], b[1]),
            xytext=(a[0], a[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.15,
                mutation_scale=9,
                shrinkA=3,
                shrinkB=3,
            ),
            zorder=zorder,
        )


def _highlight_names(ax, X, names, which=("O", "Ce"), *, color=C_INK):
    names = [str(n) for n in np.asarray(names).tolist()]
    X = np.asarray(X, dtype=np.float64)
    for tag in which:
        if tag not in names:
            continue
        i = names.index(tag)
        ax.text(
            X[i, 0] + 0.22, X[i, 1] + 0.22, tag,
            fontsize=7.5, color=color, fontweight="bold", zorder=6,
        )


def _phenol_scene_tag(data) -> str:
    res = float(np.asarray(data["resolution"]))
    chain = (
        str(data["chain"])
        if "chain" in data.files
        else "extended"
    )
    return f"{chain} @ {res:g} Å"


def fig_phenol_density_fit(
    cache: Path | None = None,
    stem: str = "24_phenol_density_fit",
):
    """What a density-solved phenol pose looks like."""
    data = _load_phenol_apps(cache)
    X_true = data["X_true"]
    bonds = data["bonds"]
    names = data["names"]
    tag = _phenol_scene_tag(data)
    xlim, ylim = _phenol_extent_limits(data, X_true, pad=1.6)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    _draw_density(ax, data, xlim, ylim)
    _draw_bonds(ax, X_true, bonds, color=C_INK, lw=1.35, alpha=0.9)
    _draw_atoms(ax, X_true, filled=True, color=C_MU, ms=7.0)
    _highlight_names(ax, X_true, names)
    ax.set_xlabel(r"$x$ (Å)")
    ax.set_ylabel(r"$y$ (Å)")
    ax.set_title("Target density with the true named model")
    ax.text(
        0.02, 0.02,
        r"solved for density $=$ atoms sit in $\rho_T$"
        "\n(names / restraints are not part of this picture)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2,
        color=C_INK,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_GRID, alpha=0.92),
    )
    fig.suptitle(
        f"Phenol ({tag}): density fit is a transport problem",
        fontsize=12, y=1.01,
    )
    _save(fig, stem)


def fig_phenol_rot180_names(
    cache: Path | None = None,
    stem: str = "25_phenol_rot180_names",
):
    """180° start → free OT: density OK, labels scrambled (arrows)."""
    data = _load_phenol_apps(cache)
    X_true = data["X_true"]
    X0 = data["X_start_180"]
    X1 = data["X_free_180"]
    bonds = data["bonds"]
    names = data["names"]
    tag = _phenol_scene_tag(data)
    tol = _phenol_arrow_tol(data)
    xlim, ylim = _phenol_extent_limits(data, X_true, X0, X1, pad=1.5)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), constrained_layout=True)
    for ax, X, title in (
        (
            axes[0], X0,
            f"start: 180°, COM-matched\n"
            f"NN-RMSD {float(data['nn0_180']):.2f} Å · "
            f"label {float(data['label0_180']):.2f} Å",
        ),
        (
            axes[1], X1,
            f"free OT (no names, no $P_{{\\mathrm{{restr}}}}$)\n"
            f"NN-RMSD {float(data['nn_180']):.2f} Å · "
            f"label {float(data['label_180']):.2f} Å",
        ),
    ):
        _draw_density(ax, data, xlim, ylim)
        _draw_bonds(ax, X_true, bonds, color="0.55", lw=0.9, alpha=0.45)
        _draw_atoms(ax, X_true, filled=False, color="0.35", ms=5.5)
        _draw_bonds(ax, X, bonds, color=C_MU, lw=1.25, alpha=0.9)
        _draw_atoms(ax, X, filled=True, color=C_MU, ms=6.0)
        if ax is axes[1]:
            _label_arrows(ax, X_true, X, tol=tol)
            _highlight_names(ax, X, names, which=("O", "Ce"))
            _highlight_names(ax, X_true, names, which=("O",), color=C_MUTED)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r"$x$ (Å)")
    axes[0].set_ylabel(r"$y$ (Å)")
    fig.suptitle(
        f"Massive rotation ({tag}): free OT fills the map, names stay wrong",
        fontsize=12,
    )
    fig.text(
        0.5, -0.02,
        "Arrows: each true labelled site → current carrier of that same label "
        "(orange = mismatch, forest = already correct).",
        ha="center", va="top", fontsize=8.2, color=C_MUTED,
    )
    _save(fig, stem)


def fig_phenol_random_landing(
    cache: Path | None = None,
    stem: str = "26_phenol_random_landing",
):
    """Random scatter → free OT: same density solution, same name problem."""
    data = _load_phenol_apps(cache)
    X_true = data["X_true"]
    X0 = data["X_start_rand"]
    X1 = data["X_free_rand"]
    bonds = data["bonds"]
    names = data["names"]
    tag = _phenol_scene_tag(data)
    tol = _phenol_arrow_tol(data)
    xlim, ylim = _phenol_extent_limits(data, X_true, X0, X1, pad=1.8)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), constrained_layout=True)
    for ax, X, title, show_bonds_cur in (
        (
            axes[0], X0,
            f"random scatter (seed {int(data['seed'])})\n"
            f"NN-RMSD {float(data['nn0_rand']):.2f} Å · "
            f"label {float(data['label0_rand']):.2f} Å",
            False,
        ),
        (
            axes[1], X1,
            f"free OT landing\n"
            f"NN-RMSD {float(data['nn_rand']):.2f} Å · "
            f"label {float(data['label_rand']):.2f} Å",
            True,
        ),
    ):
        _draw_density(ax, data, xlim, ylim)
        _draw_bonds(ax, X_true, bonds, color="0.55", lw=0.9, alpha=0.45)
        _draw_atoms(ax, X_true, filled=False, color="0.35", ms=5.5)
        if show_bonds_cur:
            _draw_bonds(ax, X, bonds, color=C_MU, lw=1.25, alpha=0.9)
        _draw_atoms(ax, X, filled=True, color=C_MU, ms=6.0)
        if ax is axes[1]:
            _label_arrows(ax, X_true, X, tol=tol)
            _highlight_names(ax, X, names, which=("O", "Ce"))
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r"$x$ (Å)")
    axes[0].set_ylabel(r"$y$ (Å)")
    fig.suptitle(
        f"Random start ({tag}): same density occupancy (start-agnostic)",
        fontsize=12,
    )
    fig.text(
        0.5, -0.02,
        "Label arrows again: density agreement does not imply chemical naming.",
        ha="center", va="top", fontsize=8.2, color=C_MUTED,
    )
    _save(fig, stem)


def fig_phenol_namer_rescue(
    cache: Path | None = None,
    stem: str = "27_phenol_namer_rescue",
):
    """Unlabelled free cloud → Namer → ADMM cleanup."""
    data = _load_phenol_apps(cache)
    X_true = data["X_true"]
    X_free = data["X_shuffled"]
    X_named = data["X_named"]
    X_admm = data["X_admm"]
    bonds = data["bonds"]
    names = data["names"]
    tag = _phenol_scene_tag(data)
    tol = _phenol_arrow_tol(data)
    xlim, ylim = _phenol_extent_limits(
        data, X_true, X_free, X_named, X_admm, pad=1.5,
    )

    panels = [
        (
            X_free, False,
            "unlabelled free cloud\n"
            f"(shuffled indices after OT)",
            False,
        ),
        (
            X_named, True,
            f"after Namer.assign\n"
            f"label-RMSD {float(data['named_label_rmsd']):.2f} Å · "
            f"restr {float(data['named_restraint_rms']):.2f}",
            True,
        ),
        (
            X_admm, True,
            f"ADMM OT+L1+$P_{{\\mathrm{{restr}}}}$\n"
            f"RMSD {float(data['admm_rmsd']):.3f} Å",
            True,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2), constrained_layout=True)
    for ax, (X, draw_bonds_cur, title, arrows) in zip(axes, panels):
        _draw_density(ax, data, xlim, ylim)
        _draw_bonds(ax, X_true, bonds, color="0.55", lw=0.85, alpha=0.4)
        _draw_atoms(ax, X_true, filled=False, color="0.35", ms=5.0)
        if draw_bonds_cur:
            _draw_bonds(ax, X, bonds, color=C_MU, lw=1.2, alpha=0.9)
        _draw_atoms(ax, X, filled=True, color=C_MU, ms=5.8)
        if arrows:
            _label_arrows(ax, X_true, X, tol=tol)
            _highlight_names(ax, X, names, which=("O", "Ce"))
        ax.set_title(title, fontsize=9.2)
        ax.set_xlabel(r"$x$ (Å)")
    axes[0].set_ylabel(r"$y$ (Å)")
    fig.suptitle(
        f"Post-hoc naming ({tag}) when no labelled start was given",
        fontsize=12,
    )
    fig.text(
        0.5, -0.03,
        r"Namer uses the CIF restraint dictionary (bonds / angles / plane), "
        r"not the OT dual. Arrows turn forest when labels match.",
        ha="center", va="top", fontsize=8.2, color=C_MUTED,
    )
    _save(fig, stem)


def fig_phenol_zigzag_3A_suite():
    """Same four-panel story for zigzag chain at 3 Å (stems 28–31)."""
    cache = PHENOL_CACHE_ZIGZAG_3A
    fig_phenol_density_fit(cache, "28_phenol_zigzag3A_density_fit")
    fig_phenol_rot180_names(cache, "29_phenol_zigzag3A_rot180_names")
    fig_phenol_random_landing(cache, "30_phenol_zigzag3A_random_landing")
    fig_phenol_namer_rescue(cache, "31_phenol_zigzag3A_namer_rescue")


def main():
    _style()
    print("Generating guide figures →", FIG_DIR)
    fig_1d_densities()
    fig_1d_cdf_and_w1()
    fig_1d_monotone_map()
    fig_kantorovich_densities()
    fig_kantorovich_coupling()
    fig_kantorovich_lp_sketch()
    fig_slice_projection()
    fig_fft_pipeline()
    fig_sliced_average()
    fig_structure_factor()
    fig_dual_potential()
    fig_gradient_vs_separation()
    fig_force_vs_monge()
    fig_l2_reversal()
    fig_monge_quantile_match()
    fig_monge_backprojection()
    fig_monge_one_step()
    fig_monge_vanishes()
    fig_b_factor_split()
    fig_scatter_fft_pipeline()
    fig_gather_kernel()
    fig_backend_cost()
    fig_algorithm_overview()
    fig_phenol_density_fit()
    fig_phenol_rot180_names()
    fig_phenol_random_landing()
    fig_phenol_namer_rescue()
    fig_phenol_zigzag_3A_suite()
    print("done.")


if __name__ == "__main__":
    main()
