"""Planar ortho-alkyl phenol for the 2-D reach figure.

Phenol heavy atoms plus a 5-carbon floppy chain on the ortho carbon (C2).
Idealized lengths: aromatic C–C ≈ 1.39 Å, C–O ≈ 1.36 Å, aliphatic C–C ≈ 1.54 Å.

Coordinates are (N, 2) in Å with columns (x, y), centred on the COM.
Atom order: C1..C6, O, Ca..Ce (chain).

Naming uses the CIF restraint dictionary ``phenol_restraints.cif`` (ring
angles harmonic; floppy chain angles flat in [108°, 180°]).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_PHENOL_CIF = Path(__file__).resolve().parent / "phenol_restraints.cif"

# Atom order: C1 (ipso), C2 (ortho+chain), C3, C4, C5, C6, O, Ca, Cb, Cc, Cd, Ce
NAMES = ("C1", "C2", "C3", "C4", "C5", "C6", "O", "Ca", "Cb", "Cc", "Cd", "Ce")
N_RING = 7  # C1..C6 + O
N_CHAIN = 5
Z = np.array(
    [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 8.0] + [6.0] * N_CHAIN,
    dtype=np.float64,
)

# Ring + hydroxyl (rigid aromatic / planar).
_RING_BONDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)]
# Ortho (C2) → Ca → Cb → Cc → Cd → Ce
_CHAIN_BONDS = [(1, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
BONDS = _RING_BONDS + _CHAIN_BONDS
# Floppy: no 1–4 across these, so the tail can fold in-plane.
ROTATABLE_BONDS = list(_CHAIN_BONDS)
# Only the phenol ring + O stay planar; the chain is free of planarity restraints.
PLANAR_GROUPS = [list(range(N_RING))]


def _build_chain(
    c2: np.ndarray,
    u: np.ndarray,
    n_chain: int,
    c_ali: float,
    chain_style: str,
    zigzag_angle_deg: float,
) -> np.ndarray:
    """Place ``n_chain`` carbons starting from C2 along direction ``u``."""
    style = str(chain_style).lower()
    if style == "extended":
        return np.stack(
            [c2 + (k + 1) * c_ali * u for k in range(n_chain)], axis=0,
        )
    if style in ("zigzag", "zig-zag", "zipzag", "zip-zag"):
        # Planar all-trans zig-zag: alternate ±(180° − bond angle) turns.
        turn = np.pi - np.deg2rad(float(zigzag_angle_deg))
        direction = np.asarray(u, dtype=np.float64).copy()
        direction /= np.linalg.norm(direction)
        pos = np.asarray(c2, dtype=np.float64)
        atoms = np.empty((n_chain, 2), dtype=np.float64)
        for k in range(n_chain):
            if k > 0:
                sign = 1.0 if (k % 2 == 1) else -1.0
                ca, sa = np.cos(sign * turn), np.sin(sign * turn)
                direction = np.array(
                    [ca * direction[0] - sa * direction[1],
                     sa * direction[0] + ca * direction[1]],
                    dtype=np.float64,
                )
            pos = pos + c_ali * direction
            atoms[k] = pos
        return atoms
    raise ValueError(
        f"unknown chain_style {chain_style!r}; expected 'extended' or 'zigzag'"
    )


def build_phenol(
    cc: float = 1.39,
    co: float = 1.36,
    c_ali: float = 1.54,
    n_chain: int = N_CHAIN,
    chain_style: str = "extended",
    zigzag_angle_deg: float = 109.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords, weights) with weights = Z / sum(Z), COM at the origin.

    ``chain_style``:
      * ``"extended"`` — fully extended along the C2 radial direction
      * ``"zigzag"`` — planar all-trans zig-zag with bond angle
        ``zigzag_angle_deg`` (default tetrahedral 109.5°)
    """
    if n_chain != N_CHAIN:
        raise ValueError(f"this module is wired for n_chain={N_CHAIN}, got {n_chain}")
    r = cc  # distance from ring centre to each carbon in a regular hexagon
    angles = np.arange(6) * (np.pi / 3.0)
    ring = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    # Oxygen along the C1 radial direction
    c1 = ring[0]
    oxygen = c1 + co * (c1 / np.linalg.norm(c1))
    # Ortho chain on C2
    c2 = ring[1]
    u = c2 / np.linalg.norm(c2)
    chain = _build_chain(c2, u, n_chain, c_ali, chain_style, zigzag_angle_deg)
    X = np.vstack([ring, oxygen[None, :], chain])
    X = X - X.mean(axis=0)
    w = Z / Z.sum()
    return X.astype(np.float64), w


def phenol_geometry(X_ref_2d: np.ndarray | None = None):
    """Build a 3-D ``Geometry`` (z = 0 embedding) with a floppy ortho chain."""
    from slicedot import Geometry

    if X_ref_2d is None:
        X_ref_2d, _ = build_phenol()
    X3 = np.column_stack([X_ref_2d, np.zeros(len(X_ref_2d))])
    return Geometry(
        X3,
        BONDS,
        rotatable_bonds=ROTATABLE_BONDS,
        chiral_centres=(),
        planar_groups=PLANAR_GROUPS,
        antibump=True,
        antibump_r0=2.8,
    )


def phenol_restraint_set():
    """Load the phenol naming dictionary (CIF)."""
    from slicedot import load_restraint_cif

    return load_restraint_cif(_PHENOL_CIF)


def phenol_namer(X_ref_2d: np.ndarray | None = None):
    """``Namer`` for the planar phenol using ``phenol_restraints.cif``."""
    from slicedot import Namer

    if X_ref_2d is None:
        X_ref_2d, _ = build_phenol()
    X3 = np.column_stack([np.asarray(X_ref_2d, dtype=np.float64),
                          np.zeros(len(X_ref_2d))])
    rs = phenol_restraint_set()
    # CIF atom order must match NAMES / coordinate order.
    if tuple(rs.atom_ids) != NAMES:
        raise ValueError(
            f"phenol CIF atom order {rs.atom_ids} != NAMES {NAMES}"
        )
    return Namer(
        X3,
        restraint_set=rs,
        rotatable_bonds=ROTATABLE_BONDS,
        chiral_centres=(),
        planar_groups=PLANAR_GROUPS,
    )


def embed3(X2: np.ndarray) -> np.ndarray:
    """(N, 2) → (N, 3) with z = 0."""
    X2 = np.asarray(X2, dtype=np.float64)
    return np.column_stack([X2, np.zeros(len(X2))])


def project_2d(
    geom,
    X2: np.ndarray,
    tol: float = 1e-3,
    max_iter: int = 80,
    slack: float | None = None,
):
    """P_restr on a 2-D pose via z=0 embedding; return xy only."""
    X3 = np.column_stack([X2, np.zeros(len(X2))])
    Xp, wrms, nfev = geom.project(X3, tol=tol, max_iter=max_iter, slack=slack)
    return Xp[:, :2].copy(), wrms, nfev


def geom_rms_2d(geom, X2: np.ndarray) -> float:
    """Weighted restraint RMS of a 2-D pose (z=0 embedding); no projection."""
    X3 = np.column_stack([X2, np.zeros(len(X2))])
    return float(geom.residual(X3)["weighted_rms"])


def molecular_radius(X: np.ndarray) -> float:
    """Max distance from the COM to any atom."""
    return float(np.linalg.norm(X - X.mean(axis=0), axis=1).max())


def rotate(X: np.ndarray, angle_rad: float, centre: np.ndarray | None = None) -> np.ndarray:
    c = X.mean(0) if centre is None else np.asarray(centre, dtype=np.float64)
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    return (X - c) @ R.T + c
