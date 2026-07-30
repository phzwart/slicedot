"""slicedot -- sliced Wasserstein-1 against a sampled density map, in reciprocal space.

A reference PyTorch implementation of the formulation validated in the NumPy prototypes
(fourier_ot2.py, fourier_ot_fast.py, deform.py).  Designed to be imported as a module or
wrapped in a service: no globals, no file I/O in the core, batched, device- and
dtype-agnostic, and with the expensive target precompute separable and serialisable.

Why PyTorch makes this SHORTER, not just faster
-----------------------------------------------
The NumPy version hand-derives the adjoint, and two of the three defects that cost the most
time in development lived there:

  * the gradient needs the exp(-2 pi i q p) branch; gathering the ifft branch instead is
    wrong by a factor of order 2 and passes the floor check, the anchor check and the
    mixed-form-factor check.  Only finite differences catch it.
  * pinning the additive constant of H is a linear functional of H, so it contributes a
    rank-one term to dH/dx.  Omitting it leaves the k = 0 component of Phi' wrong by
    s_0/P -- up to 0.6 against |Phi'| <= 1 -- of which all but 4e-5 cancels.

Under autograd neither can occur: the forward pass is written once and the adjoint is
derived.  The same applies to the gridded backend, where the adjoint of a scatter is a
gather with the same kernel -- by construction rather than by argument.

Backends
--------
  direct   structure factors by explicit summation.  O(B L N K).  Exact.  Best for N < ~50.
  grid     atoms gridded as Gaussians of a common width, residual B applied per type as a
           reciprocal-space multiplier (the Agarwal / Ten Eyck route).  O(B L (N W + P log P)).
           Measured 152x faster than direct at N = 8000 in the NumPy prototype.

Conventions (stated, because they are the dominant source of error here)
------------------------------------------------------------------------
  * phase origin at the map centre, c = origin + (S-1)/2 * spacing
  * profile transform  T_u(q) = sum_v m_v exp(-2 pi i q u.(r_v - c))
  * form factor exp(-2 pi^2 sigma^2 q^2) = exp(-B q^2 / 4)
  * integration is division by 2 pi i q; the q = 0 coefficient is the free additive
    constant, pinned by H -> H - H[n_empty]
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import math
import torch
import torch.nn as nn

__all__ = [
    "SlicedOTConfig",
    "SlicedOT",
    "CrystalSlicedOT",
    "fibonacci_directions",
    "sigma_from_resolution",
    "form_factor_qmax",
    "b_from_sigma",
    "orthogonalization_matrix",
]

TWOPI = 2.0 * math.pi


# --------------------------------------------------------------------------- helpers
def sigma_from_resolution(d: float) -> float:
    """the report's convention: a Gaussian whose FWHM equals the resolution"""
    return d / 2.3548


def form_factor_qmax(sigma: float, eps: float = 1e-9) -> float:
    """the q beyond which the atomic form factor is below eps.

    The sampling Nyquist is necessary but not sufficient: on a heavily skewed lattice the
    interplanar spacings can be half the step lengths, so the correct Nyquist admits
    frequencies where exp(-2 pi^2 sigma^2 q^2) has long since died but the map's voxel-sum
    representation still carries discretisation structure.  Measured cost of not capping,
    on a cell with alpha/beta/gamma = 58/63/112: floor 3.5e-4 instead of ~5e-7.
    """
    return math.sqrt(-math.log(eps) / (2 * math.pi ** 2)) / sigma


def b_from_sigma(sigma: float) -> float:
    return 8.0 * math.pi ** 2 * sigma ** 2


def fibonacci_directions(n: int, dtype=torch.float64) -> torch.Tensor:
    """n deterministic Fibonacci-sphere directions.  Random directions are measurably
    worse: in 2-D a random set is +19.9% off the analytic anchor at L = 16 and does not
    reliably improve with L."""
    i = torch.arange(n, dtype=dtype)
    z = 1.0 - (2.0 * i + 1.0) / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1)


@dataclass
class SlicedOTConfig:
    n_dirs: int = 48
    dt: float = 0.25                    # sampling of the H axis, A
    window: Optional[float] = None      # real-space window, A.  None -> 3x map diagonal
    qmax: Optional[float] = None        # bandwidth, A^-1.  None -> map Nyquist
    backend: str = "auto"               # "direct" | "grid" | "auto"
    sigma_grid: Optional[float] = None  # gridding width for the grid backend
    n_sigma: float = 4.0                # kernel truncation, in sigma_grid
    atom_chunk: int = 512               # cap on materialised phase array
    map_cutoff: float = 1e-6            # drop target voxels below this fraction of the max

    def as_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------- module
class SlicedOT(nn.Module):
    """Sliced W1 between a static sampled map and a Gaussian-mixture model.

    The map is consumed once at construction and reduced to L one-dimensional slices;
    the map itself is not retained.  ``state_dict`` therefore carries everything needed
    to score a model, which is what makes this servable.

    forward(x, w, sigma) -> (B,) losses.  Differentiable in x and w.
    """

    def __init__(self, target_map: torch.Tensor, origin: Sequence[float], spacing: float,
                 sigma_data: float, config: SlicedOTConfig = SlicedOTConfig(),
                 dtype=torch.float64, device=None):
        super().__init__()
        self.cfg = config
        self.sigma_data = float(sigma_data)
        cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.cdtype = cdt

        m = target_map.to(dtype=dtype, device=device)
        if m.ndim != 3:
            raise ValueError("target_map must be 3-D")
        Nv = torch.tensor(m.shape, dtype=dtype, device=m.device)
        sp = torch.as_tensor(spacing, dtype=dtype, device=m.device)
        if sp.ndim == 0:
            sp = sp.repeat(3)
        self.spacing_vec = sp
        self.spacing = float(sp.max())
        org = torch.as_tensor(origin, dtype=dtype, device=m.device)
        centre = org + 0.5 * (Nv - 1) * sp

        diag = float(((Nv - 1) * sp).norm())
        Lt = 3.0 * diag if config.window is None else float(config.window)
        P = int(2 ** math.ceil(math.log2(Lt / config.dt)))
        dt = Lt / P
        q = torch.fft.fftfreq(P, d=dt, dtype=dtype, device=m.device)
        qn = (min(0.5 / self.spacing, form_factor_qmax(sigma_data))
              if config.qmax is None else float(config.qmax))
        keep = q.abs() < min(qn, 0.5 / dt)

        U = fibonacci_directions(config.n_dirs, dtype=dtype).to(m.device)

        # --- target slices, exact: a structure factor over the significant voxels
        m = m / m.sum()
        sel = m > config.map_cutoff * m.max()
        ax = [torch.arange(int(n), dtype=dtype, device=m.device) * sp[i]
              for i, n in enumerate(m.shape)]
        gz, gy, gx = torch.meshgrid(*ax, indexing="ij")
        V = torch.stack([gz, gy, gx], dim=-1).reshape(-1, 3) + org - centre
        V = V[sel.reshape(-1)]
        mv = m.reshape(-1)[sel.reshape(-1)]
        mv = mv / mv.sum()
        qk = q[keep]
        Tq = torch.empty((config.n_dirs, qk.numel()), dtype=cdt, device=m.device)
        for l in range(config.n_dirs):                       # one direction at a time
            a = V @ U[l]
            Tq[l] = torch.exp(-1j * TWOPI * qk[:, None] * a[None, :]).to(cdt) @ mv.to(cdt)

        sg = config.sigma_grid if config.sigma_grid is not None else 2.0 * dt
        self.register_buffer("q", q)
        self.register_buffer("qk", qk)
        self.register_buffer("keep_idx", torch.nonzero(keep, as_tuple=True)[0])
        self.register_buffer("U", U)
        self.register_buffer("Tq", Tq)
        self.register_buffer("centre", centre)
        self.P, self.dt, self.Lt = P, dt, Lt
        self.n_empty = P // 2
        self.sigma_grid = float(sg)
        self.n_dirs = config.n_dirs
        self.q_nyquist = float(qn)

    # ------------------------------------------------------------------- staging
    def stage(self, sigma: float) -> torch.Tensor:
        """target spectrum blurred from sigma_data up to a stage width -- a multiplier"""
        extra2 = max(sigma ** 2 - self.sigma_data ** 2, 0.0)
        if extra2 <= 0.0:
            return self.Tq
        return self.Tq * torch.exp(-2 * math.pi ** 2 * extra2 * self.qk ** 2).to(self.cdtype)

    # ------------------------------------------------------------- model spectrum
    def _model_direct(self, p: torch.Tensor, w: torch.Tensor,
                      ff: torch.Tensor) -> torch.Tensor:
        """p (B,L,N), w (B,N) -> (B,L,K).  Chunked over atoms to cap memory."""
        B, L, N = p.shape
        K = self.qk.numel()
        out = p.new_zeros((B, L, K), dtype=self.cdtype)
        step = max(1, self.cfg.atom_chunk)
        for s in range(0, N, step):
            e = min(s + step, N)
            ph = torch.exp(-1j * TWOPI * self.qk.view(1, 1, 1, K) * p[..., s:e, None])
            out = out + (ph * w[:, None, s:e, None].to(self.cdtype)).sum(dim=2)
        return out * ff.view(1, 1, K).to(self.cdtype)

    def _model_grid(self, p: torch.Tensor, w: torch.Tensor,
                    ff_res: torch.Tensor) -> torch.Tensor:
        """Gaussian scatter + FFT + residual multiplier.  The adjoint of the scatter is a
        gather with the same kernel; autograd supplies it, so it cannot be got wrong."""
        B, L, N = p.shape
        sg, dt, P = self.sigma_grid, self.dt, self.P
        Wn = int(math.ceil(self.cfg.n_sigma * sg / dt))
        off = torch.arange(-Wn, Wn + 1, device=p.device)
        i0 = torch.round(p / dt)
        idx = (i0[..., None] + off).long() % P                       # (B,L,N,2W+1)
        d = (i0[..., None] + off) * dt - p[..., None]
        wgt = torch.exp(-0.5 * (d / sg) ** 2) / (sg * math.sqrt(TWOPI)) * dt
        wgt = wgt * w[:, None, :, None]
        A = p.new_zeros((B, L, P))
        A.scatter_add_(2, idx.reshape(B, L, -1), wgt.reshape(B, L, -1))
        return torch.fft.fft(A.to(self.cdtype), dim=-1)[..., self.keep_idx] \
            * ff_res.view(1, 1, -1).to(self.cdtype)

    # ------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, w: torch.Tensor, sigma: Optional[float] = None,
                Tq: Optional[torch.Tensor] = None,
                delta: float = 0.0) -> torch.Tensor:
        """x (N,3) or (B,N,3); w (N,) or (B,N).  Returns (B,) sliced-W1 values.

        delta > 0 substitutes the log-cosh surrogate for |H|, which removes the kinks at
        the zero crossings of H.  Those kinks are why finite-difference checks on the
        exact objective show occasional 1e-5 excursions against a 1e-10 median.
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        if w.ndim == 1:
            w = w.expand(x.shape[0], -1)
        sigma = self.sigma_data if sigma is None else float(sigma)
        Tq = self.stage(sigma) if Tq is None else Tq

        p = (x - self.centre) @ self.U.T                       # (B,N,L)
        p = p.transpose(1, 2).contiguous()                     # (B,L,N)
        ff = torch.exp(-2 * math.pi ** 2 * sigma ** 2 * self.qk ** 2)

        backend = self.cfg.backend
        if backend == "auto":
            backend = "grid" if x.shape[1] >= 64 else "direct"
        if backend == "grid_custom":
            res2 = max(sigma ** 2 - self.sigma_grid ** 2, 0.0)
            return _GridOT.apply(
                p, w, self.qk, self.keep_idx,
                torch.exp(-2 * math.pi ** 2 * res2 * self.qk ** 2), Tq, self.dt,
                self.P, self.n_empty, self.sigma_grid, self.cfg.n_sigma,
                self.cdtype, delta)[0 if squeeze else slice(None)]
        if backend == "direct":
            Mq = self._model_direct(p, w, ff)
        else:
            res2 = max(sigma ** 2 - self.sigma_grid ** 2, 0.0)
            if sigma < self.sigma_grid:
                raise ValueError("sigma_grid must not exceed the smallest atomic sigma")
            Mq = self._model_grid(
                p, w, torch.exp(-2 * math.pi ** 2 * res2 * self.qk ** 2))

        dif = Mq - Tq.unsqueeze(0)
        denom = 1j * TWOPI * self.qk * self.dt
        nz = self.qk != 0
        A = dif.new_zeros((dif.shape[0], self.n_dirs, self.P))
        A[..., self.keep_idx[nz]] = dif[..., nz] / denom[nz].to(self.cdtype)
        H = torch.fft.ifft(A, dim=-1).real
        H = H - H[..., self.n_empty : self.n_empty + 1]

        if delta > 0.0:
            pen = delta * (torch.logaddexp(H / delta, -H / delta)
                           - math.log(2.0))
        else:
            pen = H.abs()
        val = pen.sum(-1).mean(-1) * self.dt
        return val[0] if squeeze else val

    # -------------------------------------------------------- deformation oracle
    @torch.no_grad()
    def deformation(self, x: torch.Tensor, w: torch.Tensor,
                    sigma: Optional[float] = None) -> torch.Tensor:
        """Backprojected Monge displacement, preconditioned by M^{-1} -> d*I.

        This is a LENGTH, not a gradient.  The W1 gradient is a smoothed sign: bounded
        away from zero, so it has reach but carries no step size.  Measured on a leucine
        fragment at 2.5 A, steps taken from this oracle reach 0.010 A final RMSD against
        0.133 A for a fixed step cap, and remove the end-of-run oscillation entirely.
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        if w.ndim == 1:
            w = w.expand(x.shape[0], -1)
        sigma = self.sigma_data if sigma is None else float(sigma)
        Tq = self.stage(sigma)

        A = self.Tq.new_zeros((self.n_dirs, self.P))
        A[:, self.keep_idx] = Tq
        prof = torch.fft.ifft(A, dim=-1).real
        t = torch.arange(self.P, dtype=prof.dtype, device=prof.device) * self.dt
        t = torch.where(t > self.Lt / 2, t - self.Lt, t)
        order = torch.argsort(t)
        ts, prof = t[order], prof[:, order]
        Fnu = torch.cumsum(prof, dim=-1) - 0.5 * prof                # (L,P)

        p = ((x - self.centre) @ self.U.T).transpose(1, 2)           # (B,L,N)
        z = (p[..., :, None] - p[..., None, :]) / (sigma * math.sqrt(2.0))
        Fmu = (w[:, None, None, :] * 0.5 * (1.0 + torch.erf(z))).sum(-1)   # (B,L,N)

        d = torch.empty_like(Fmu)
        for l in range(self.n_dirs):
            j = torch.searchsorted(Fnu[l].contiguous(), Fmu[:, l].contiguous())
            j = j.clamp(1, self.P - 1)
            f0, f1 = Fnu[l][j - 1], Fnu[l][j]
            t0, t1 = ts[j - 1], ts[j]
            frac = (Fmu[:, l] - f0) / torch.clamp(f1 - f0, min=1e-30)
            d[:, l] = t0 + frac * (t1 - t0) - p[:, l]

        v = torch.einsum("bln,lk->bnk", d, self.U) / self.n_dirs
        M = torch.einsum("li,lj->ij", self.U, self.U) / self.n_dirs
        v = v @ torch.linalg.inv(M).T
        return v[0] if squeeze else v


# ============================================================================ custom
class _GridOT(torch.autograd.Function):
    """Gridded forward with the hand-derived backward: the Agarwal gradient-map recipe.

    Autograd's backward through ``scatter_add_`` is the exact adjoint, but it must keep the
    stencil tensors alive -- idx, d and wgt are each (B, L, N, 2W+1), so at N = 8000,
    L = 24, W = 8 that is ~80 MB of saved activations before the FFTs.  The hand-derived
    route saves only p (B,L,N) and H (B,L,P) and rebuilds the stencil in backward, which
    is ~50x less.

    The backward gathers the psi-field with the atomic Gaussian ITSELF, not its derivative.
    The exact adjoint of the scatter needs dg/dt applied to the field carrying a 1/(2 pi i q);
    integrating by parts moves the derivative onto the field and cancels the 1/q, leaving a
    gather with g against the field from s' * ff_res.  Those are identical in the continuum
    and differ on the grid only by band-limiting -- which is measured, not assumed.
    """

    @staticmethod
    def forward(ctx, p, w, qk, keep_idx, ff_res, Tq, dt, P, n_empty, sg, nsig,
                cdtype, delta):
        B, L, N = p.shape
        Wn = int(math.ceil(nsig * sg / dt))
        off = torch.arange(-Wn, Wn + 1, device=p.device)
        i0 = torch.round(p / dt)
        idx = ((i0[..., None] + off).long() % P).reshape(B, L, -1)
        d = (i0[..., None] + off) * dt - p[..., None]
        wgt = (torch.exp(-0.5 * (d / sg) ** 2) / (sg * math.sqrt(TWOPI)) * dt
               * w[:, None, :, None]).reshape(B, L, -1)
        A = p.new_zeros((B, L, P)).scatter_add_(2, idx, wgt)
        Mq = torch.fft.fft(A.to(cdtype), dim=-1)[..., keep_idx] * ff_res.to(cdtype)
        dif = Mq - Tq.unsqueeze(0)
        nz = qk != 0
        Aq = dif.new_zeros((B, L, P))
        Aq[..., keep_idx[nz]] = dif[..., nz] / (1j * TWOPI * qk[nz] * dt).to(cdtype)
        H = torch.fft.ifft(Aq, dim=-1).real
        H = H - H[..., n_empty : n_empty + 1]
        if delta > 0.0:
            val = (delta * (torch.logaddexp(H / delta, -H / delta) - math.log(2.0))
                   ).sum(-1).mean(-1) * dt
        else:
            val = H.abs().sum(-1).mean(-1) * dt
        ctx.save_for_backward(p, w, H, qk, keep_idx, ff_res)
        ctx.meta = (dt, P, n_empty, sg, nsig, cdtype, delta, L)
        return val

    @staticmethod
    def backward(ctx, gout):
        p, w, H, qk, keep_idx, ff_res = ctx.saved_tensors
        dt, P, n_empty, sg, nsig, cdtype, delta, L = ctx.meta
        B, _, N = p.shape
        s = torch.tanh(H / delta) if delta > 0.0 else torch.sign(H)
        shat = torch.fft.fft(s.to(cdtype), dim=-1)
        s0 = shat[..., 0].real
        alt = torch.where(keep_idx % 2 == 0, 1.0, -1.0).to(p.dtype)
        sp = torch.conj(shat[..., keep_idx]) - s0[..., None] * alt
        S = sp.new_zeros((B, L, P))
        S[..., keep_idx] = sp * ff_res.to(cdtype)
        psi = torch.fft.fft(S, dim=-1).real / P                 # NOT ifft: the e^{-iqp} branch
        Wn = int(math.ceil(nsig * sg / dt))
        off = torch.arange(-Wn, Wn + 1, device=p.device)
        i0 = torch.round(p / dt)
        idx = ((i0[..., None] + off).long() % P)
        d = (i0[..., None] + off) * dt - p[..., None]
        g = torch.exp(-0.5 * (d / sg) ** 2) / (sg * math.sqrt(TWOPI)) * dt
        phi = -(torch.gather(psi, 2, idx.reshape(B, L, -1)).reshape(d.shape)
                * g).sum(-1)                                    # (B,L,N)
        # 1/L: the value averages over directions, so the backward must too
        gp = gout[:, None, None] * w[:, None, :] * phi / L
        return (gp, None, None, None, None, None, None, None, None, None, None,
                None, None)


# ====================================================================== crystal grids
def orthogonalization_matrix(a, b, c, alpha, beta, gamma, dtype=torch.float64):
    """fractional -> Cartesian, PDB/CCP4 convention (a along x, b in the xy plane).

    Angles in degrees.  Returns O with r_cart = O @ f_frac.
    """
    al, be, ga = (math.radians(t) for t in (alpha, beta, gamma))
    ca, cb, cg, sg = math.cos(al), math.cos(be), math.cos(ga), math.sin(ga)
    V = a * b * c * math.sqrt(max(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 1e-12))
    return torch.tensor([[a, b * cg, c * cb],
                         [0.0, b * sg, c * (ca - cb * cg) / sg],
                         [0.0, 0.0, V / (a * b * sg)]], dtype=dtype)


class CrystalSlicedOT(SlicedOT):
    """SlicedOT built from a crystallographic map: (n_a, n_b, n_c) on a general cell.

    Real maps are neither cubic nor equispaced nor orthogonal.  The only place the grid
    geometry enters is the Cartesian position of each voxel, because the target slices are
    a structure factor over voxels rather than an interpolated central slice.  So the
    generalisation is exact rather than approximate: replace r_v = origin + n*spacing by
    r_v = O @ (n / N) and nothing else changes.

    Bandwidth is set from the COARSEST grid direction, since a projection is only sampled
    as finely as the worst axis allows.
    """

    def __init__(self, target_map: torch.Tensor, cell, sigma_data: float,
                 config: SlicedOTConfig = SlicedOTConfig(), origin_frac=(0.0, 0.0, 0.0),
                 dtype=torch.float64, device=None):
        nn.Module.__init__(self)
        self.cfg = config
        self.sigma_data = float(sigma_data)
        cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.cdtype = cdt

        m = target_map.to(dtype=dtype, device=device)
        if m.ndim != 3:
            raise ValueError("target_map must be 3-D")
        N = torch.tensor(m.shape, dtype=dtype, device=m.device)
        O = orthogonalization_matrix(*cell, dtype=dtype).to(m.device)
        # sampling-lattice basis (columns) and its reciprocal basis
        Smat = torch.stack([O[:, i] / N[i] for i in range(3)], dim=1)  # columns = steps
        Gmat = torch.linalg.inv(Smat).T                                # columns = g_i
        g_len = Gmat.norm(dim=0)
        # Nyquist: half the shortest reciprocal vector of the SAMPLING lattice.  For a
        # skewed lattice the interplanar spacing d_i = 1/|g_i| is SMALLER than the step
        # length |s_i|, so using max|s_i| would be conservative and waste bandwidth.
        self.q_nyquist = float(0.5 * g_len.min())
        self.spacing = float(0.5 / self.q_nyquist)                     # effective d_max

        m = m / m.sum()
        sel = m > config.map_cutoff * m.max()
        ii = torch.nonzero(sel, as_tuple=False).to(dtype)             # (M,3) grid indices
        frac = ii / N + torch.tensor(origin_frac, dtype=dtype, device=m.device)
        V = frac @ O.T                                                # (M,3) Cartesian
        mv = m[sel]
        mv = mv / mv.sum()
        centre = (V * mv[:, None]).sum(0)                             # density centroid
        V = V - centre

        extent = float((V.norm(dim=1)).max())
        Lt = 6.0 * extent if config.window is None else float(config.window)
        P = int(2 ** math.ceil(math.log2(Lt / config.dt)))
        dt = Lt / P
        q = torch.fft.fftfreq(P, d=dt, dtype=dtype, device=m.device)
        qn = (min(self.q_nyquist, form_factor_qmax(sigma_data))
              if config.qmax is None else float(config.qmax))
        keep = q.abs() < min(qn, 0.5 / dt)
        qk = q[keep]

        U = fibonacci_directions(config.n_dirs, dtype=dtype).to(m.device)
        Tq = torch.empty((config.n_dirs, qk.numel()), dtype=cdt, device=m.device)
        for l in range(config.n_dirs):
            a = V @ U[l]
            Tq[l] = torch.exp(-1j * TWOPI * qk[:, None] * a[None, :]).to(cdt) @ mv.to(cdt)

        self.register_buffer("q", q)
        self.register_buffer("qk", qk)
        self.register_buffer("keep_idx", torch.nonzero(keep, as_tuple=True)[0])
        self.register_buffer("U", U)
        self.register_buffer("Tq", Tq)
        self.register_buffer("centre", centre)
        self.register_buffer("O", O)
        self.P, self.dt, self.Lt = P, dt, Lt
        self.n_empty = P // 2
        self.sigma_grid = (config.sigma_grid if config.sigma_grid is not None
                           else 2.0 * dt)
        self.n_dirs = config.n_dirs
        self.q_nyquist = float(qn)
