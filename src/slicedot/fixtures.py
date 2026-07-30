"""Small synthetic molecules used by tests and examples.

The capped leucine fragment is an idealized ACE-LEU-NME heavy-atom geometry
(13 atoms). Coordinates are approximate stereochemistry, sufficient for the
mathematical validation protocol (floor, analytic translation anchor, FD
gradients, deformation oracle). They are not a crystallographic deposition.
"""
from __future__ import annotations

import numpy as np

# Atom order: ACE C, ACE O, ACE CH3, N, CA, C, O, CB, CG, CD1, CD2, NME N, NME CH3
_ATOM_NAMES = (
    "ACE_C",
    "ACE_O",
    "ACE_CH3",
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CD1",
    "CD2",
    "NME_N",
    "NME_CH3",
)

# Electron counts for heavy atoms (C=6, N=7, O=8)
_Z = np.array([6, 8, 6, 7, 6, 6, 8, 6, 6, 6, 6, 7, 6], dtype=np.float64)

_X0 = np.array(
    [
        [-2.3231, -1.0000, -0.1769],
        [-2.8231, 0.1000, -0.1769],
        [-3.2231, -2.2000, -0.1769],
        [-1.0231, -1.4000, -0.1769],
        [-0.1231, -0.5000, -0.1769],
        [1.2769, -1.0000, -0.1769],
        [1.6769, -2.1000, -0.1769],
        [-0.5231, 0.9000, 0.4231],
        [0.2769, 2.1000, 0.0231],
        [-0.4231, 3.4000, 0.5231],
        [1.6769, 2.0000, 0.6231],
        [2.0769, 0.0000, -0.1769],
        [3.4769, -0.3000, -0.1769],
    ],
    dtype=np.float64,
)


def sigma_of(resolution: float) -> float:
    """Gaussian sigma whose FWHM equals ``resolution`` (paper convention)."""
    return float(resolution) / 2.3548


# Public aliases matching the historical ``leu3d`` prototype module.
X0 = _X0.copy()
W = (_Z / _Z.sum()).copy()
names = _ATOM_NAMES
Z = _Z.copy()


__all__ = ["X0", "W", "Z", "names", "sigma_of"]
