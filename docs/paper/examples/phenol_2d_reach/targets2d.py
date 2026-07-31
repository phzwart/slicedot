"""2-D refinement targets for the phenol reach figure.

Coordinates are Cartesian (x, y) in Å.  The density map is stored as
``rho[iy, ix]`` on a regular grid with origin at the centre of voxel (0, 0)
and isotropic spacing ``dx``.

Atomic densities are isotropic Gaussians **clamped to zero beyond**
``GAUSS_TRUNC`` σ (default 3). Target and model share this kernel so L1 / OT
stay operator-consistent.

Targets
-------
  DensityAtCentre   E = -Σ_i w_i ρ_T(x_i)
  L1Diff            E = Σ_v |ρ_T(v) - ρ_M(v)|   (unit-mass densities)
  ConsistentSlicedW1  operator-consistent sliced W₁ (grid-render + project)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d, map_coordinates

# Hard support of each atomic Gaussian: g = 0 for r > GAUSS_TRUNC * σ.
GAUSS_TRUNC = 3.0


def sigma_from_resolution(d: float) -> float:
    """Gaussian sigma whose FWHM equals resolution (paper convention)."""
    return float(d) / 2.3548


def truncated_gauss(d2: np.ndarray, sigma: float,
                    n_sigma: float = GAUSS_TRUNC) -> np.ndarray:
    """Gaussian exp(-r²/(2σ²)) with hard zero outside ``n_sigma`` σ."""
    s2 = float(sigma) * float(sigma)
    g = np.exp(-d2 / (2.0 * s2))
    return np.where(d2 <= (n_sigma * float(sigma)) ** 2, g, 0.0)


def directions_2d(L: int) -> np.ndarray:
    """Equally spaced directions on the semicircle (exact 2-D design)."""
    th = np.arange(L) * np.pi / L + 0.5 * np.pi / L
    return np.stack([np.cos(th), np.sin(th)], axis=1)


def make_grid(shape: tuple[int, int], dx: float, origin: np.ndarray | None = None):
    """Return voxel-centre coordinates (Ny*Nx, 2) as (x, y) in Å, plus origin."""
    Ny, Nx = shape
    if origin is None:
        origin = np.array([-(Nx - 1) * dx / 2.0, -(Ny - 1) * dx / 2.0], dtype=np.float64)
    xs = origin[0] + np.arange(Nx) * dx
    ys = origin[1] + np.arange(Ny) * dx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    V = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return V, origin.astype(np.float64)


def render(coords: np.ndarray, w: np.ndarray, sigma: float, V: np.ndarray,
           shape: tuple[int, int], n_sigma: float = GAUSS_TRUNC) -> np.ndarray:
    """Unit-mass truncated-Gaussian mixture on the grid (zero beyond n_sigma σ)."""
    d2 = ((V[None, :, :] - coords[:, None, :]) ** 2).sum(-1)  # (N, V)
    M = (w[:, None] * truncated_gauss(d2, sigma, n_sigma=n_sigma)).sum(0)
    M = M / M.sum()
    return M.reshape(shape)


def _to_index(x_xy: np.ndarray, origin: np.ndarray, dx: float) -> np.ndarray:
    """(N, 2) Å (x,y) -> map_coordinates order (y_index, x_index)."""
    ix = (x_xy[:, 0] - origin[0]) / dx
    iy = (x_xy[:, 1] - origin[1]) / dx
    return np.stack([iy, ix], axis=0)


class DensityAtCentre:
    """E = -Σ w_i ρ_T(x_i); gradient from the precomputed density gradient map."""

    def __init__(self, rhoT: np.ndarray, origin: np.ndarray, dx: float):
        self.rho = rhoT / rhoT.max()
        # np.gradient along axis 0 (y) then axis 1 (x); convert to Å^-1
        gy, gx = np.gradient(self.rho, dx, dx)
        self.g = np.stack([gy, gx])  # [0]=∂/∂y, [1]=∂/∂x in index order
        self.origin = np.asarray(origin, dtype=np.float64)
        self.dx = float(dx)

    def value_grad(self, x: np.ndarray, w: np.ndarray):
        c = _to_index(x, self.origin, self.dx)
        v = map_coordinates(self.rho, c, order=1, mode="constant", cval=0.0)
        # map_coordinates of gy, gx -> convert to (∂/∂x, ∂/∂y) for (x,y) coords
        gy = map_coordinates(self.g[0], c, order=1, mode="constant", cval=0.0)
        gx = map_coordinates(self.g[1], c, order=1, mode="constant", cval=0.0)
        G = np.stack([gx, gy], axis=1)
        return float(-(w * v).sum()), -(w[:, None] * G)


class L1Diff:
    """E = Σ_v |ρ_T - ρ_M| with both densities unit-mass normalized."""

    def __init__(self, rhoT: np.ndarray, V: np.ndarray, sigma: float):
        self.T = rhoT.ravel() / rhoT.sum()
        self.V = V
        self.sigma = float(sigma)
        self.shape = rhoT.shape

    def value_grad(self, x: np.ndarray, w: np.ndarray):
        s2 = self.sigma ** 2
        d = self.V[None, :, :] - x[:, None, :]          # (N, V, 2)
        d2 = (d ** 2).sum(-1)
        g = truncated_gauss(d2, self.sigma)             # (N, V), 0 outside 3σ
        raw = (w[:, None] * g).sum(0)
        Z = raw.sum()
        M = raw / Z
        resid = M - self.T
        val = float(np.abs(resid).sum())
        s = np.sign(resid)
        # dE/dM_v = sign(M-T); through unit-mass normalisation:
        # dM_v/draw_u = (δ_vu - M_v) / Z
        # Interior gradient of truncated Gaussian matches the full Gaussian;
        # outside the support g=0 so those voxels contribute nothing.
        s_c = s - (s * M).sum()
        coef = (w[:, None] * g) * s_c[None, :] / Z
        G = (coef[:, :, None] * d).sum(1) / s2
        return val, G


def _gauss_kernel(sigma: float, dx: float, trunc: float = GAUSS_TRUNC) -> np.ndarray:
    r = int(max(1, np.ceil(trunc * sigma / dx)))
    t = np.arange(-r, r + 1) * dx
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k = np.where(np.abs(t) <= trunc * sigma, k, 0.0)
    return k / k.sum()


class ConsistentSlicedW1:
    """Operator-consistent sliced W₁: model and target share render + binning."""

    def __init__(self, rhoT: np.ndarray, V: np.ndarray, U: np.ndarray,
                 nbins: int = 260, pad: float = 10.0):
        self.V = V
        self.U = U
        self.L = U.shape[0]
        self.shape = rhoT.shape
        P = V @ U.T
        self.edges = np.linspace(P.min() - pad, P.max() + pad, nbins + 1)
        self.grid = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.dx = float(self.grid[1] - self.grid[0])
        self.nb = nbins
        self.idx = np.clip(
            ((P - self.edges[0]) / self.dx).astype(int), 0, nbins - 1
        )
        self.tgt = self._project(rhoT.ravel() / rhoT.sum())
        self.Fnu = np.cumsum(self.tgt, 1) - 0.5 * self.tgt

    def _project(self, dens: np.ndarray) -> np.ndarray:
        out = np.empty((self.L, self.nb))
        for l in range(self.L):
            out[l] = np.bincount(self.idx[:, l], weights=dens, minlength=self.nb)
        return out / out.sum(1, keepdims=True)

    def stage(self, sig_data: float, sig: float) -> np.ndarray:
        extra = np.sqrt(max(sig ** 2 - sig_data ** 2, 0.0))
        h = self.tgt
        if extra > 1e-6:
            h = convolve1d(h, _gauss_kernel(extra, self.dx), axis=1, mode="nearest")
            h = h / h.sum(1, keepdims=True)
        return np.cumsum(h, 1) - 0.5 * h

    def value_grad(self, x: np.ndarray, w: np.ndarray, sigma: float,
                   Fnu: np.ndarray | None = None, delta: float = 0.0):
        Fnu = self.Fnu if Fnu is None else Fnu
        d2 = ((self.V[None, :, :] - x[:, None, :]) ** 2).sum(-1)
        g = truncated_gauss(d2, sigma)
        raw = (w[:, None] * g).sum(0)
        Z = raw.sum()
        M = raw / Z
        proj = self._project(M)
        H = (np.cumsum(proj, 1) - 0.5 * proj) - Fnu
        if delta > 0:
            val = float(
                (delta * np.logaddexp(H / delta, -H / delta) - delta * np.log(2.0))
                .sum(1)
                .mean()
                * self.dx
            )
            s = np.tanh(H / delta)
        else:
            val = float(np.abs(H).sum(1).mean() * self.dx)
            s = np.sign(H)
        gl = (self.dx / self.L) * (s.sum(1, keepdims=True) - np.cumsum(s, 1) + 0.5 * s)
        off = (gl * proj).sum(1)
        Psi = np.zeros(self.V.shape[0])
        for l in range(self.L):
            Psi += gl[l][self.idx[:, l]] - off[l]
        Psi_c = Psi - (Psi * M).sum()
        coef = (w[:, None] * g) * Psi_c[None, :] / Z
        G = (coef[:, :, None] * (self.V[None, :, :] - x[:, None, :])).sum(1) / (
            sigma * sigma
        )
        return val, G
