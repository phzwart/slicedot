"""Gabor-windowed sliced W1 -- locally weighted projections via Gaussian windows.

The global sliced path integrates out both transverse dimensions, so two atoms at
the same projected coordinate but far apart in the perpendicular plane are
indistinguishable within a slice.  Multiplying model and map by a Gaussian window
G_s(r - a) before slicing restores locality; a Gaussian times a Gaussian atom is
still a Gaussian atom, so the Agarwal / Ten Eyck backend survives with modified
weights, positions and width.

Traps (do not rediscover)
-------------------------
  * Monge displacements must be C-weighted *averages* over windows, not sums.
    Forces add; lengths average.  A sum-instead-of-average bug overshoots as K×
    with n_windows.
  * Detach C_j.  Under uniform π_a, ∫ C_j da is independent of r_j, so the
    gradient through C is pure variance (and under non-uniform π_a it is a bias
    toward the sampling density).  The β_j Jacobian through r' is a different
    thing and must be kept.
  * Never re-centre the phase origin c on the model centroid -- that annihilates
    the translation signal.
  * Per-window mass balance does not hold.  Rescale T̃ = (m/n) T for the
    transport term; carry λ_mass (m - n)² separately.
  * The FFT-branch and pinning issues documented in core.py apply unchanged.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn

from slicedot.core import (
    TWOPI,
    SlicedOT,
    SlicedOTConfig,
    _score_from_spectra,
    fibonacci_directions,
)

__all__ = [
    "GaborTarget",
    "WindowedSlicedOT",
    "suggest_L",
    "window_atoms",
]

_GB = 1024.0 ** 3


# --------------------------------------------------------------------------- helpers
def suggest_L(s: float, d: float) -> int:
    """Direction count for a window of width ``s`` at resolution ``d``.

    L ≳ π D_box / d with D_box ≈ 4 s (the region the window selects).
    """
    if s <= 0.0 or d <= 0.0:
        raise ValueError(f"s and d must be positive, got s={s!r}, d={d!r}")
    return max(3, math.ceil(math.pi * 4.0 * s / d))


def _resolution_from_sigma(sigma: float) -> float:
    return float(sigma) * 2.3548


def _random_rotation(dtype, device, generator=None) -> torch.Tensor:
    """Haar-random SO(3) via QR of a Gaussian matrix."""
    A = torch.randn(3, 3, dtype=dtype, device=device, generator=generator)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.diag(R).sign().unsqueeze(0)
    if torch.det(Q) < 0:
        Q = Q.clone()
        Q[:, 0] = -Q[:, 0]
    return Q


def window_atoms(
    x: torch.Tensor,
    sigma: torch.Tensor,
    a: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form Gaussian-window × Gaussian-atom product.

    Parameters
    ----------
    x : (B, N, 3) or (N, 3)
    sigma : scalar, (N,), or (B, N)
    a : (A, 3) window centres
    s : (A,) window widths

    Returns
    -------
    C : (B, A, N) importance weights -- already detached
    beta : (B, A, N)  s² / (σ² + s²)
    r_prime : (B, A, N, 3)
    sigma_prime : (B, A, N)
    """
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    B, N, _ = x.shape
    A = a.shape[0]
    device, dtype = x.device, x.dtype

    if not torch.is_tensor(sigma):
        sigma = torch.full((B, N), float(sigma), dtype=dtype, device=device)
    else:
        sigma = sigma.to(dtype=dtype, device=device)
        if sigma.ndim == 0:
            sigma = torch.full((B, N), float(sigma), dtype=dtype, device=device)
        elif sigma.ndim == 1:
            sigma = sigma.view(1, N).expand(B, N)
    s = s.to(dtype=dtype, device=device).view(A)
    a = a.to(dtype=dtype, device=device)

    sig2 = sigma ** 2                                          # (B, N)
    s2 = s ** 2                                                # (A,)
    denom = sig2[:, None, :] + s2[None, :, None]               # (B, A, N)
    # C_j = (2π(σ²+s²))^{-3/2} exp(-||a-r||² / 2(σ²+s²))
    # Detach: ∫ C da is independent of r under uniform π_a (pure variance).
    d2 = ((a[None, :, None, :] - x[:, None, :, :]) ** 2).sum(-1)  # (B,A,N)
    pref = (TWOPI * denom).pow(-1.5)
    C = (pref * torch.exp(-0.5 * d2 / denom)).detach()
    beta = s2[None, :, None] / denom
    r_prime = (1.0 - beta)[..., None] * a[None, :, None, :] + beta[..., None] * x[:, None, :, :]
    sigma_prime = (sig2[:, None, :] * s2[None, :, None] / denom).sqrt()
    return C, beta, r_prime, sigma_prime


def _structure_factor_fast(vox_proj, mv, qk, cdtype, chunk: int = 4096):
    """Batched structure factor.  vox_proj (M,L) -> (L,K)."""
    M, L = vox_proj.shape
    K = qk.numel()
    out = vox_proj.new_zeros((L, K), dtype=cdtype)
    mv_c = mv.to(cdtype)
    for s0 in range(0, M, chunk):
        e = min(s0 + chunk, M)
        p = vox_proj[s0:e]                                          # (c, L)
        ph = torch.exp(-1j * TWOPI * qk.view(1, 1, K) * p[..., None])
        out = out + (ph * mv_c[s0:e, None, None]).sum(0)
    return out


def _model_spectrum_direct(
    r_prime: torch.Tensor,
    w_eff: torch.Tensor,
    sigma_prime: torch.Tensor,
    U: torch.Tensor,
    centre: torch.Tensor,
    qk: torch.Tensor,
    cdtype,
) -> torch.Tensor:
    """Windowed model structure factor.  r_prime (B,N,3), w_eff (B,N), σ' (B,N)
    or broadcastable; U (L,3).  Returns Mq (B,L,K)."""
    p = ((r_prime - centre) @ U.T).transpose(1, 2).contiguous()   # (B,L,N)
    B, L, N = p.shape
    K = qk.numel()
    if sigma_prime.ndim == 1:
        sigma_prime = sigma_prime.view(1, N).expand(B, N)
    # Per-atom form factor: ff (B,N,K)
    ff = torch.exp(
        -2 * math.pi ** 2 * sigma_prime[:, :, None] ** 2 * qk.view(1, 1, K) ** 2
    )
    out = p.new_zeros((B, L, K), dtype=cdtype)
    step = 256
    for s0 in range(0, N, step):
        e = min(s0 + step, N)
        ph = torch.exp(-1j * TWOPI * qk.view(1, 1, 1, K) * p[..., s0:e, None])
        amp = (w_eff[:, s0:e, None] * ff[:, s0:e, :]).to(cdtype)   # (B,n,K)
        out = out + (ph * amp[:, None, :, :]).sum(dim=2)
    return out


def _quantile_match_displacements(
    Fmu: torch.Tensor,
    Fnu: torch.Tensor,
    ts: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """1-D Monge displacements.  Fmu (B,L,N), Fnu (L,P), ts (P,), p (B,L,N)."""
    L = Fmu.shape[1]
    P = ts.numel()
    d = torch.empty_like(Fmu)
    for l in range(L):
        j = torch.searchsorted(Fnu[l].contiguous(), Fmu[:, l].contiguous())
        j = j.clamp(1, P - 1)
        f0, f1 = Fnu[l][j - 1], Fnu[l][j]
        t0, t1 = ts[j - 1], ts[j]
        frac = (Fmu[:, l] - f0) / torch.clamp(f1 - f0, min=1e-30)
        d[:, l] = t0 + frac * (t1 - t0) - p[:, l]
    return d


# ===================================================================== GaborTarget
class GaborTarget:
    """Precomputed windowed target spectra on a lattice of centres and s levels.

    Directions are drawn from a fixed pool of rotated Fibonacci sets frozen at
    precompute time.  Runtime ``π_u`` therefore reuses that pool rather than
    freshly randomising each call -- a deliberate trade-off so interpolated T
    shares the same u-grid as the stored tensor.

    Stored layout: ``T[a_lattice, s_level, u, q]`` complex.
    """

    def __init__(
        self,
        T: torch.Tensor,
        a_lattice: torch.Tensor,
        s_levels: torch.Tensor,
        U_pool: torch.Tensor,
        qk: torch.Tensor,
        centre: torch.Tensor,
        spacing_a: float,
        axes: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        warn_gb: float = 2.0,
    ):
        self.T = T
        self.a_lattice = a_lattice
        self.s_levels = s_levels
        self.U_pool = U_pool          # (n_pool, L, 3)
        self.qk = qk
        self.centre = centre
        self.spacing_a = float(spacing_a)
        if axes is None:
            pts = a_lattice.reshape(-1, 3)
            axes = (
                torch.unique(pts[:, 0].round(decimals=6)).sort().values,
                torch.unique(pts[:, 1].round(decimals=6)).sort().values,
                torch.unique(pts[:, 2].round(decimals=6)).sort().values,
            )
        self.axes = axes
        self.a_shape = (int(axes[0].numel()), int(axes[1].numel()), int(axes[2].numel()))
        nbytes = T.element_size() * T.nelement()
        gb = nbytes / _GB
        print(f"GaborTarget: {gb:.3f} GB  shape={tuple(T.shape)}")
        if gb > warn_gb:
            warnings.warn(
                f"GaborTarget footprint {gb:.2f} GB exceeds threshold {warn_gb} GB",
                stacklevel=2,
            )

    @classmethod
    def precompute(
        cls,
        target_map: torch.Tensor,
        origin: Sequence[float],
        spacing,
        centre: torch.Tensor,
        qk: torch.Tensor,
        keep_idx: torch.Tensor,
        P: int,
        dt: float,
        s_range: tuple[float, float],
        n_s: int = 4,
        n_dirs: int = 48,
        n_rotations: int = 4,
        region_min: torch.Tensor | None = None,
        region_max: torch.Tensor | None = None,
        lattice_factor: float = 0.5,
        warn_gb: float = 2.0,
        dtype=torch.float64,
        device=None,
        seed: int = 0,
    ) -> GaborTarget:
        """Build T[a,s,u,q] by windowing the map, 3-D FFT of local subbox, slice.

        Lattice spacing is ``lattice_factor * s_min`` (default s/2).  ``s`` levels
        are a geometric ladder over ``s_range``.
        """
        m = target_map.to(dtype=dtype, device=device)
        m = m / m.sum()
        sp = torch.as_tensor(spacing, dtype=dtype, device=m.device)
        if sp.ndim == 0:
            sp = sp.repeat(3)
        org = torch.as_tensor(origin, dtype=dtype, device=m.device)
        Nz, Ny, Nx = m.shape
        ax = [torch.arange(int(n), dtype=dtype, device=m.device) * sp[i]
              for i, n in enumerate(m.shape)]
        gz, gy, gx = torch.meshgrid(*ax, indexing="ij")
        V = torch.stack([gz, gy, gx], dim=-1) + org             # absolute
        centre = centre.to(dtype=dtype, device=m.device)

        s_min, s_max = float(s_range[0]), float(s_range[1])
        s_levels = torch.exp(
            torch.linspace(math.log(s_min), math.log(s_max), n_s, dtype=dtype, device=m.device)
        )
        spacing_a = lattice_factor * s_min
        if region_min is None or region_max is None:
            # Cover the map plus 2 s_max margin
            region_min = org - 2.0 * s_max
            region_max = org + (torch.tensor([Nz, Ny, Nx], dtype=dtype, device=m.device) - 1) * sp + 2.0 * s_max
        region_min = region_min.to(dtype=dtype, device=m.device)
        region_max = region_max.to(dtype=dtype, device=m.device)

        grids = [
            torch.arange(float(region_min[i]), float(region_max[i]) + 0.5 * spacing_a,
                         spacing_a, dtype=dtype, device=m.device)
            for i in range(3)
        ]
        # arange uses z,y,x order matching meshgrid ij on (z,y,x) map axes... use x,y,z Cartesian
        gx_a, gy_a, gz_a = torch.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
        a_lattice = torch.stack([gx_a, gy_a, gz_a], dim=-1)     # (nx,ny,nz,3)
        a_flat = a_lattice.reshape(-1, 3)
        n_a = a_flat.shape[0]

        gen = torch.Generator(device=m.device)
        gen.manual_seed(int(seed))
        U0 = fibonacci_directions(n_dirs, dtype=dtype).to(m.device)
        U_pool = []
        for _ in range(n_rotations):
            R = _random_rotation(dtype, m.device, gen)
            U_pool.append(U0 @ R.T)
        U_pool_t = torch.stack(U_pool, dim=0)                   # (n_rot, L, 3)
        n_u = n_rotations * n_dirs
        U_all = U_pool_t.reshape(n_u, 3)

        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        K = qk.numel()
        T = torch.empty((n_a, n_s, n_u, K), dtype=cdtype, device=m.device)

        # Precompute voxel coords relative to centre for projection
        V_rel = (V - centre).reshape(-1, 3)
        m_flat = m.reshape(-1)

        V_abs = V_rel + centre
        keep = m_flat > 1e-12 * m_flat.max()
        V_abs_k = V_abs[keep]
        V_rel_k = V_rel[keep]
        m_k = m_flat[keep]
        vox_proj_all = V_rel_k @ U_all.T                              # (M', n_u)
        M = V_abs_k.shape[0]
        # Batch lattice points: T[a,u,k] = sum_m mw[a,m] exp(-2πi qk * proj[m,u])
        a_batch = 64
        for si, s in enumerate(s_levels):
            s2 = float(s) ** 2
            pref = (TWOPI * s2) ** (-1.5)
            for a0 in range(0, n_a, a_batch):
                a1 = min(a0 + a_batch, n_a)
                aa = a_flat[a0:a1]                                    # (b, 3)
                # Gs[b,m]
                d2 = ((V_abs_k[None, :, :] - aa[:, None, :]) ** 2).sum(-1)
                mw = m_k[None, :] * (pref * torch.exp(-0.5 * d2 / s2))
                # SF for all b windows sharing projections
                # out[b,u,k] = sum_m mw[b,m] * exp(-2πi q[k] proj[m,u])
                bsz = a1 - a0
                out = T.new_zeros((bsz, n_u, K))
                chunk = 2048
                for m0 in range(0, M, chunk):
                    m1 = min(m0 + chunk, M)
                    p = vox_proj_all[m0:m1]                            # (c, n_u)
                    ph = torch.exp(
                        -1j * TWOPI * qk.view(1, 1, K) * p[..., None]
                    )                                                 # (c, n_u, K)
                    out = out + torch.einsum(
                        "bc,cuk->buk", mw[:, m0:m1].to(cdtype), ph,
                    )
                T[a0:a1, si] = out

        return cls(
            T=T, a_lattice=a_flat, s_levels=s_levels, U_pool=U_pool_t,
            qk=qk, centre=centre, spacing_a=spacing_a,
            axes=(grids[0], grids[1], grids[2]), warn_gb=warn_gb,
        )

    def interpolate(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        u_index: torch.Tensor | None = None,
        scheme: str = "trilinear",
    ) -> torch.Tensor:
        """Interpolate T in a and log s.  Returns T (A, n_u, K).

        ``scheme`` is swappable; currently ``trilinear`` (complex) in a and
        linear in log s.  Lattice spacing ≤ s/2 is the Nyquist guideline for
        the a-dependence bandlimited to |k − qu| ≲ 1/s.
        """
        if scheme != "trilinear":
            raise ValueError(f"scheme must be 'trilinear', got {scheme!r}")
        T = self.T                                  # (n_a, n_s, n_u, K)
        xs, ys, zs = self.axes
        nx, ny, nz = self.a_shape
        sp_a = self.spacing_a

        def _frac(pts, origin, n):
            if n <= 1:
                return pts.new_zeros(pts.shape[0])
            return ((pts - origin) / sp_a).clamp(0, n - 1.001)

        fx = _frac(a[:, 0], xs[0], nx)
        fy = _frac(a[:, 1], ys[0], ny)
        fz = _frac(a[:, 2], zs[0], nz)
        ix0 = fx.long().clamp(0, max(nx - 1, 0))
        iy0 = fy.long().clamp(0, max(ny - 1, 0))
        iz0 = fz.long().clamp(0, max(nz - 1, 0))
        ix1 = (ix0 + 1).clamp(0, max(nx - 1, 0))
        iy1 = (iy0 + 1).clamp(0, max(ny - 1, 0))
        iz1 = (iz0 + 1).clamp(0, max(nz - 1, 0))
        wx = (fx - ix0.to(fx.dtype)).clamp(0, 1)
        wy = (fy - iy0.to(fy.dtype)).clamp(0, 1)
        wz = (fz - iz0.to(fz.dtype)).clamp(0, 1)

        def flat(ix, iy, iz):
            return iz + nz * (iy + ny * ix)

        log_s = torch.log(s.clamp_min(1e-30))
        log_levels = torch.log(self.s_levels)
        n_s = log_levels.numel()
        if n_s == 1:
            si0 = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
            si1 = si0
            ws = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
        else:
            lo, hi = log_levels[0], log_levels[-1]
            fs = ((log_s - lo) / (hi - lo) * (n_s - 1)).clamp(0, n_s - 1.001)
            si0 = fs.long().clamp(0, n_s - 1)
            si1 = (si0 + 1).clamp(0, n_s - 1)
            ws = (fs - si0.to(fs.dtype)).clamp(0, 1)

        A = a.shape[0]
        n_u, K = T.shape[2], T.shape[3]
        out = T.new_zeros((A, n_u, K))
        corners = [
            (ix0, iy0, iz0, (1 - wx) * (1 - wy) * (1 - wz)),
            (ix1, iy0, iz0, wx * (1 - wy) * (1 - wz)),
            (ix0, iy1, iz0, (1 - wx) * wy * (1 - wz)),
            (ix1, iy1, iz0, wx * wy * (1 - wz)),
            (ix0, iy0, iz1, (1 - wx) * (1 - wy) * wz),
            (ix1, iy0, iz1, wx * (1 - wy) * wz),
            (ix0, iy1, iz1, (1 - wx) * wy * wz),
            (ix1, iy1, iz1, wx * wy * wz),
        ]
        for ix, iy, iz, w_a in corners:
            idx = flat(ix, iy, iz)
            T0 = T[idx, si0]
            T1 = T[idx, si1]
            Ts = (1 - ws)[:, None, None] * T0 + ws[:, None, None] * T1
            out = out + w_a[:, None, None].to(out.dtype) * Ts
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save(
            {
                "T": self.T,
                "a_lattice": self.a_lattice,
                "s_levels": self.s_levels,
                "U_pool": self.U_pool,
                "qk": self.qk,
                "centre": self.centre,
                "spacing_a": self.spacing_a,
                "axes": self.axes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, warn_gb: float = 2.0) -> GaborTarget:
        data = torch.load(Path(path), weights_only=False)
        return cls(
            T=data["T"], a_lattice=data["a_lattice"], s_levels=data["s_levels"],
            U_pool=data["U_pool"], qk=data["qk"], centre=data["centre"],
            spacing_a=data["spacing_a"], axes=data.get("axes"), warn_gb=warn_gb,
        )


# ================================================================= WindowedSlicedOT
class WindowedSlicedOT(nn.Module):
    """Locally windowed sliced W1.  Composes an internal ``SlicedOT`` for buffers.

    Monte Carlo over window centres a, widths s and directions U.  Per-sample
    contributions are gathered then averaged -- never update between samples.

    Parameters
    ----------
    s_range : (s_min, s_max) in Å.  Requires s_min ≥ 3 σ_max.
    n_windows : window centres sampled per call
    n_directions : None → suggest_L(s_max, d)
    pi_a : ``"uniform"`` | ``"model"`` | callable(x, n, generator) -> (A,3)
    lambda_mass : weight on (m - n)² mass residual (default 0)
    backend : ``"direct"`` | ``"grid"`` | ``"grid_custom"`` | ``"auto"``
    gabor_target : optional Phase-2 precompute; when set, T is interpolated
    """

    def __init__(
        self,
        target_map: torch.Tensor,
        origin: Sequence[float],
        spacing,
        sigma_data: float,
        s_range: tuple[float, float] | None = None,
        n_windows: int = 8,
        n_directions: int | None = None,
        pi_a: str | Callable = "uniform",
        lambda_mass: float = 0.0,
        backend: str = "auto",
        seed: int | None = None,
        config: SlicedOTConfig | None = None,
        dtype=torch.float64,
        device=None,
        gabor_target: GaborTarget | None = None,
        memory_warn_gb: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        from dataclasses import fields
        cfg = SlicedOTConfig() if config is None else config
        if kwargs:
            valid = {f.name for f in fields(SlicedOTConfig)}
            updates = {k: v for k, v in kwargs.items() if k in valid}
            unknown = set(kwargs) - valid
            if unknown:
                raise ValueError(f"unknown kwargs: {sorted(unknown)!r}")
            if updates:
                cfg = SlicedOTConfig(**{**{f.name: getattr(cfg, f.name) for f in fields(SlicedOTConfig)},
                                        **updates})
        self.ot = SlicedOT(
            target_map, origin, spacing, sigma_data, config=cfg,
            dtype=dtype, device=device,
        )
        self.sigma_data = float(sigma_data)
        s_floor = 3.0 * self.sigma_data
        if s_range is None:
            s_range = (s_floor, max(12.0, 8.0 * self.sigma_data))
        self.s_min = float(s_range[0])
        self.s_max = float(s_range[1])
        if self.s_min > self.s_max:
            raise ValueError(f"s_range must be (s_min, s_max), got {s_range!r}")
        if self.s_min < s_floor - 1e-12:
            raise ValueError(
                f"s_range[0] must be >= 3*sigma_max ({s_floor:.4g}), "
                f"got {self.s_min:.4g}"
            )
        self.n_windows = int(n_windows)
        self.lambda_mass = float(lambda_mass)
        self.backend = backend
        self.pi_a = pi_a
        self.gabor_target = gabor_target
        self.memory_warn_gb = float(memory_warn_gb)
        d_res = _resolution_from_sigma(self.sigma_data)
        self.n_directions = (
            int(n_directions) if n_directions is not None
            else suggest_L(self.s_max, d_res)
        )
        # Full map for Phase-1 target windowing (not cutoff voxels)
        m = target_map.to(dtype=dtype, device=self.ot.centre.device)
        m = m / m.sum()
        self.register_buffer("map_full", m)
        sp = self.ot.spacing_vec
        org = torch.as_tensor(origin, dtype=dtype, device=m.device)
        self.register_buffer("origin", org)
        ax = [torch.arange(int(n), dtype=dtype, device=m.device) * sp[i]
              for i, n in enumerate(m.shape)]
        gz, gy, gx = torch.meshgrid(*ax, indexing="ij")
        V_abs = torch.stack([gz, gy, gx], dim=-1) + org
        self.register_buffer("V_abs", V_abs.reshape(-1, 3))
        self.register_buffer("m_flat", m.reshape(-1))
        self._seed = seed
        self._gen_count = 0

    # -------------------------------------------------------------- sampling
    def _generator(self, device) -> torch.Generator:
        gen = torch.Generator(device=device if device.type == "cpu" else "cpu")
        if self._seed is not None:
            # Advance by call count so successive sample_windows calls differ,
            # but the same seed + call index is reproducible.
            gen.manual_seed(int(self._seed) + self._gen_count)
        else:
            gen.manual_seed(torch.seed() % (2 ** 31 - 1))
        return gen

    def sample_windows(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw (a, s, U) with a (A,3), s (A,), U (A,L,3).  Seedable."""
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        device, dtype = x.device, x.dtype
        gen = self._generator(device)
        self._gen_count += 1
        A = self.n_windows
        L = self.n_directions

        # π_s: log-uniform
        u = torch.rand(A, dtype=dtype, generator=gen)
        log_s = math.log(self.s_min) + u * (math.log(self.s_max) - math.log(self.s_min))
        s = torch.exp(log_s).to(device)

        # Enforce s ≥ 3 σ_max (σ_max from current model widths approximated by sigma_data
        # at sample time; forward re-checks against actual sigma).
        s_floor = 3.0 * self.sigma_data
        if float(s.min()) < s_floor - 1e-12:
            raise ValueError(
                f"sampled s must be >= 3*sigma_max ({s_floor:.4g}), "
                f"got min s={float(s.min()):.4g}"
            )

        # π_a
        if callable(self.pi_a):
            a = self.pi_a(x, A, gen)
        elif self.pi_a == "uniform":
            lo = x.amin(dim=(0, 1)) - 2.0 * self.s_max
            hi = x.amax(dim=(0, 1)) + 2.0 * self.s_max
            a = lo + (hi - lo) * torch.rand(A, 3, dtype=dtype, generator=gen).to(device)
        elif self.pi_a == "model":
            # Sample centres near randomly chosen atoms
            B, N, _ = x.shape
            idx = torch.randint(0, N, (A,), generator=gen)
            b = torch.randint(0, B, (A,), generator=gen)
            a = x[b, idx] + 0.5 * s[:, None] * torch.randn(A, 3, dtype=dtype, generator=gen).to(device)
        else:
            raise ValueError(
                f"pi_a must be 'uniform', 'model', or callable, got {self.pi_a!r}"
            )

        # π_u: randomly rotated Fibonacci (or from GaborTarget pool)
        U0 = fibonacci_directions(L, dtype=dtype)
        if self.gabor_target is not None:
            pool = self.gabor_target.U_pool.to(device=device, dtype=dtype)
            # pool (n_rot, Lp, 3); if Lp != L, fall back to fresh rotations
            if pool.shape[1] == L:
                n_rot = pool.shape[0]
                choice = torch.randint(0, n_rot, (A,), generator=gen)
                U = pool[choice]
            else:
                U = torch.stack(
                    [U0 @ _random_rotation(dtype, "cpu", gen).T for _ in range(A)],
                    dim=0,
                ).to(device)
        else:
            U = torch.stack(
                [U0 @ _random_rotation(dtype, "cpu", gen).T for _ in range(A)],
                dim=0,
            ).to(device)
        # Monte Carlo samples must not attach to x: otherwise autograd walks
        # the full-map Gabor (V_abs ~ grid³) through a and OOMs.
        return a.detach().to(device), s.detach(), U.detach()

    # ----------------------------------------------------------- spectra
    def _check_s(self, s: torch.Tensor, sigma_max: float) -> None:
        floor = 3.0 * sigma_max
        if float(s.min()) < floor - 1e-12:
            raise ValueError(
                f"s must be >= 3*sigma_max ({floor:.4g}), got min s={float(s.min()):.4g}"
            )

    def _target_spectrum_direct(
        self, a: torch.Tensor, s: torch.Tensor, U: torch.Tensor,
    ) -> torch.Tensor:
        """Window full map, project, structure factor.  Returns Tq (A,L,K)."""
        device = a.device
        dtype = a.dtype
        V = self.V_abs.to(device=device, dtype=dtype)
        mv = self.m_flat.to(device=device, dtype=dtype)
        centre = self.ot.centre.to(device=device, dtype=dtype)
        qk = self.ot.qk.to(device=device, dtype=dtype)
        cdtype = self.ot.cdtype
        Awin = a.shape[0]
        L = U.shape[1]
        K = qk.numel()
        V_rel = V - centre
        Tq = torch.empty((Awin, L, K), dtype=cdtype, device=device)
        for i in range(Awin):
            s2 = s[i] ** 2
            pref = (TWOPI * s2) ** (-1.5)
            Gs = pref * torch.exp(-0.5 * ((V - a[i]) ** 2).sum(-1) / s2)
            mw = mv * Gs
            vox_proj = V_rel @ U[i].T                                 # (M, L)
            Tq[i] = _structure_factor_fast(vox_proj, mw, qk, cdtype)
        return Tq

    def _target_spectrum(
        self, a: torch.Tensor, s: torch.Tensor, U: torch.Tensor,
    ) -> torch.Tensor:
        if self.gabor_target is None:
            return self._target_spectrum_direct(a, s, U)
        # Interpolate (A, n_u, K); map each sample's U to nearest pool direction
        T_all = self.gabor_target.interpolate(a, s)             # (A, n_u, K)
        pool = self.gabor_target.U_pool.reshape(-1, 3).to(a.device, a.dtype)
        A, L, _ = U.shape
        K = T_all.shape[-1]
        Tq = T_all.new_zeros((A, L, K))
        for i in range(A):
            # Nearest pool direction for each of the L directions
            dots = U[i] @ pool.T                                # (L, n_u)
            idx = dots.argmax(dim=-1)
            Tq[i] = T_all[i, idx]
        return Tq

    def _model_spectrum(
        self,
        r_prime: torch.Tensor,
        w_eff: torch.Tensor,
        sigma_prime: torch.Tensor,
        U: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        """Mq (B,L,K).  r_prime/w_eff/sigma_prime are for a single window: (B,N,*)."""
        ot = self.ot
        centre = ot.centre
        # Common σ'?  grid backends require it
        sp = sigma_prime
        if sp.ndim == 1:
            common = bool(torch.allclose(sp, sp[:1].expand_as(sp)))
            sig_val = float(sp[0])
        else:
            common = bool(torch.allclose(sp, sp[:, :1].expand_as(sp)))
            sig_val = float(sp[0, 0])

        if backend == "auto":
            backend = "grid" if r_prime.shape[1] >= 64 and common else "direct"

        if backend in ("grid", "grid_custom") and not common:
            backend = "direct"

        if backend == "direct":
            return _model_spectrum_direct(
                r_prime, w_eff, sp if sp.ndim == 2 else sp.unsqueeze(0).expand(r_prime.shape[0], -1),
                U, centre, ot.qk, ot.cdtype,
            )

        # Grid path: reuse SlicedOT._model_grid with substituted inputs
        p = ((r_prime - centre) @ U.T).transpose(1, 2).contiguous()
        if sig_val < ot.sigma_grid:
            raise ValueError("sigma_grid must not exceed the smallest atomic sigma")
        res2 = max(sig_val ** 2 - ot.sigma_grid ** 2, 0.0)
        ff_res = torch.exp(-2 * math.pi ** 2 * res2 * ot.qk ** 2)
        if backend == "grid_custom":
            # Score path needs Tq; here we only build Mq -- use grid autograd path
            return ot._model_grid(p, w_eff, ff_res)
        return ot._model_grid(p, w_eff, ff_res)

    def _resolve_sigma(self, x, sigma) -> torch.Tensor:
        B, N, _ = x.shape
        device, dtype = x.device, x.dtype
        if sigma is None:
            return torch.full((B, N), self.sigma_data, dtype=dtype, device=device)
        if not torch.is_tensor(sigma):
            return torch.full((B, N), float(sigma), dtype=dtype, device=device)
        sigma = sigma.to(dtype=dtype, device=device)
        if sigma.ndim == 0:
            return torch.full((B, N), float(sigma), dtype=dtype, device=device)
        if sigma.ndim == 1:
            return sigma.view(1, N).expand(B, N).clone()
        return sigma

    # -------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        sigma=None,
        delta: float = 0.0,
        windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        freeze_C: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scalar (or (B,)) windowed sliced-W1.  Differentiable in x (via r', not C).

        ``freeze_C`` (B,A,N), if given, replaces the computed C.  Used by FD tests so
        both autograd and finite differences differentiate the same C-detached objective.
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        if w.ndim == 1:
            w = w.expand(x.shape[0], -1)
        sigma_t = self._resolve_sigma(x, sigma)
        sigma_max = float(sigma_t.max())

        if windows is None:
            a, s, U = self.sample_windows(x)
        else:
            a, s, U = windows
            # Caller-supplied windows: still treat as MC samples (no ∂/∂a).
            a, s, U = a.detach(), s.detach(), U.detach()
        self._check_s(s, sigma_max)

        C, _, r_prime, sigma_prime = window_atoms(x, sigma_t, a, s)
        if freeze_C is not None:
            C = freeze_C.detach()
        Awin = a.shape[0]
        B, N, _ = x.shape
        # Target spectra depend only on the map + (a,s,U); keep them off the graph.
        with torch.no_grad():
            Tq_all = self._target_spectrum(a, s, U)             # (A,L,K)
        backend = self.backend
        # Batch windows into the leading dimension when (for grid) σ' is common.
        L = U.shape[1]
        sp_flat = sigma_prime.reshape(B * Awin, N)
        common_sp = bool(torch.allclose(sp_flat, sp_flat[:1].expand_as(sp_flat)))
        if backend == "direct" or common_sp:
            # Fuse (B, A) -> BA
            rp = r_prime.reshape(B * Awin, N, 3)
            w_eff = (w[:, None, :] * C).reshape(B * Awin, N)
            sp = sigma_prime.reshape(B * Awin, N)
            # Per-window U: model spectrum needs one U; loop U but batch B.
            # When all U identical, one call; else stack per window.
            U0 = U[0]
            if Awin == 1 or torch.allclose(U, U0.expand_as(U)):
                Mq = self._model_spectrum(rp, w_eff, sp, U0, backend)  # (BA,L,K)
                Mq = Mq.view(B, Awin, L, -1)
            else:
                Mqs = []
                for i in range(Awin):
                    Mqs.append(self._model_spectrum(
                        r_prime[:, i], w[:, :] * C[:, i, :],
                        sigma_prime[:, i], U[i], backend,
                    ))
                Mq = torch.stack(Mqs, dim=1)                          # (B,A,L,K)
            m = w_eff.view(B, Awin, N).sum(-1).clamp_min(1e-30)       # (B,A)
            n = torch.stack(
                [self._window_mass_target(a[i], s[i]) for i in range(Awin)]
            ).clamp_min(1e-30)                                        # (A,)
            scale = m / n.view(1, Awin)
            T_tilde = Tq_all.unsqueeze(0) * scale[:, :, None, None].to(Tq_all.dtype)
            # Score each window: flatten BA
            Mq_f = Mq.reshape(B * Awin, L, -1)
            T_f = T_tilde.reshape(B * Awin, L, -1)
            val_f = _score_from_spectra(
                Mq_f, T_f, self.ot.qk, self.ot.keep_idx, self.ot.dt,
                self.ot.P, self.ot.n_empty, self.ot.cdtype, delta,
            ).view(B, Awin) / m
            out = val_f.mean(dim=1)
            if self.lambda_mass != 0.0:
                out = out + self.lambda_mass * ((m - n.view(1, Awin)) ** 2).mean(1)
            return out[0] if squeeze else out

        scores = []
        mass_pen = []
        for i in range(Awin):
            w_eff = w * C[:, i, :]
            Mq = self._model_spectrum(
                r_prime[:, i], w_eff, sigma_prime[:, i], U[i], backend,
            )
            m = w_eff.sum(-1).clamp_min(1e-30)
            n = self._window_mass_target(a[i], s[i]).clamp_min(1e-30)
            T_tilde = Tq_all[i].unsqueeze(0) * (m / n)[:, None, None].to(Tq_all.dtype)
            val = _score_from_spectra(
                Mq, T_tilde, self.ot.qk, self.ot.keep_idx, self.ot.dt,
                self.ot.P, self.ot.n_empty, self.ot.cdtype, delta,
            ) / m
            scores.append(val)
            if self.lambda_mass != 0.0:
                mass_pen.append(self.lambda_mass * (m - n) ** 2)
        out = torch.stack(scores, dim=0).mean(dim=0)
        if self.lambda_mass != 0.0 and mass_pen:
            out = out + torch.stack(mass_pen, dim=0).mean(dim=0)
        return out[0] if squeeze else out

    def _window_mass_target(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s2 = s ** 2
        pref = (TWOPI * s2) ** (-1.5)
        Gs = pref * torch.exp(-0.5 * ((self.V_abs - a) ** 2).sum(-1) / s2)
        return (self.m_flat * Gs).sum()

    # ---------------------------------------------------------- deformation
    @torch.no_grad()
    def deformation(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        sigma=None,
        windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """C-weighted average of per-window Monge steps, with Δr = Δr'/β.

        Forces add; displacements average.  Normalise so per-atom C-weights sum to 1.
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        if w.ndim == 1:
            w = w.expand(x.shape[0], -1)
        sigma_t = self._resolve_sigma(x, sigma)
        sigma_max = float(sigma_t.max())
        if windows is None:
            a, s, U = self.sample_windows(x)
        else:
            a, s, U = windows
        self._check_s(s, sigma_max)

        C, beta, r_prime, sigma_prime = window_atoms(x, sigma_t, a, s)
        A = a.shape[0]
        B, N, _ = x.shape
        num = x.new_zeros((B, N, 3))
        den = x.new_zeros((B, N))
        Tq_all = self._target_spectrum(a, s, U)
        ot = self.ot
        t_axis = torch.arange(ot.P, dtype=x.dtype, device=x.device) * ot.dt
        t_axis = torch.where(t_axis > ot.Lt / 2, t_axis - ot.Lt, t_axis)
        order = torch.argsort(t_axis)
        ts = t_axis[order]

        for i in range(A):
            w_eff = w * C[:, i, :]
            rp = r_prime[:, i]
            sp = sigma_prime[:, i]
            Ui = U[i]
            L = Ui.shape[0]
            m = w_eff.sum(-1).clamp_min(1e-30)                        # (B,)
            Tq = Tq_all[i]
            q0 = (ot.qk == 0).nonzero(as_tuple=True)[0]
            if q0.numel():
                n = Tq[:, int(q0[0])].real.mean().clamp_min(1e-30)
            else:
                n = self._window_mass_target(a[i], s[i]).clamp_min(1e-30)
            # Rescale target so masses match for quantile matching
            T_tilde = Tq * (m.mean() / n).to(Tq.dtype)

            A_spec = Tq.new_zeros((L, ot.P))
            A_spec[:, ot.keep_idx] = T_tilde
            prof = torch.fft.ifft(A_spec, dim=-1).real[:, order]
            # Renormalise profile to mass m (per batch mean) for CDF
            mass_prof = prof.sum(-1).clamp_min(1e-30)                 # (L,)
            Fnu = torch.cumsum(prof, dim=-1) - 0.5 * prof
            # Scale Fnu so Fnu[..., -1] ≈ m (use first batch element for profile)
            # Actually profile from T_tilde already has mass ≈ m; build CDF in [0, mass]
            # Quantile match: Fmu uses absolute masses; searchsorted needs same scale.
            p = ((rp - ot.centre) @ Ui.T).transpose(1, 2)             # (B,L,N)
            # Gaussian-mixture CDF with per-atom σ'
            # z_{jk} = (p_j - p_k) / (σ'_k √2)
            Fmu = p.new_zeros((B, L, N))
            for l in range(L):
                pk = p[:, l, :]                                       # (B,N)
                # Broadcast: for each atom j, sum_k w_k * 0.5(1+erf((p_j-p_k)/(σ_k√2)))
                # σ' may vary per atom
                sig = sp
                if sig.ndim == 1:
                    sig = sig.view(1, N).expand(B, N)
                z = (pk[:, :, None] - pk[:, None, :]) / (
                    sig[:, None, :] * math.sqrt(2.0) + 1e-30
                )
                Fmu[:, l] = (w_eff[:, None, :] * 0.5 * (1.0 + torch.erf(z))).sum(-1)

            # Scale Fnu to match total model mass per direction
            for l in range(L):
                Fnu[l] = Fnu[l] * (m.mean() / mass_prof[l])

            d = _quantile_match_displacements(Fmu, Fnu, ts, p)        # (B,L,N)
            v_prime = torch.einsum("bln,lk->bnk", d, Ui) / L
            Mmat = torch.einsum("li,lj->ij", Ui, Ui) / L
            v_prime = v_prime @ torch.linalg.inv(Mmat).T
            # Convert r' displacement back: Δr = Δr' / β
            b = beta[:, i, :].clamp_min(1e-30)                        # (B,N)
            v = v_prime / b[..., None]
            cw = C[:, i, :]                                           # (B,N)
            num = num + cw[..., None] * v
            den = den + cw

        v_out = num / den.clamp_min(1e-30)[..., None]
        return v_out[0] if squeeze else v_out
