"""Small synthetic molecules used by tests and examples.

Capped leucine (ACE-LEU-NME, 13 heavy atoms) with tetrahedral L-Cα,
planar peptides, and an extended side chain so topo>3 pairs clear ~3 Å.
"""
from __future__ import annotations

import numpy as np

_ATOM_NAMES = (
    "ACE_C", "ACE_O", "ACE_CH3", "N", "CA", "C", "O",
    "CB", "CG", "CD1", "CD2", "NME_N", "NME_CH3",
)

ACE_C, ACE_O, ACE_CH3, N, CA, C, O, CB, CG, CD1, CD2, NME_N, NME_CH3 = range(13)

_Z = np.array([6, 8, 6, 7, 6, 6, 8, 6, 6, 6, 6, 7, 6], dtype=np.float64)

_BONDS = [
    (ACE_CH3, ACE_C), (ACE_C, ACE_O), (ACE_C, N),
    (N, CA), (CA, C), (C, O), (C, NME_N), (NME_N, NME_CH3),
    (CA, CB), (CB, CG), (CG, CD1), (CG, CD2),
]
_ROTATABLE = [(CA, CB), (CB, CG)]
_CHIRAL = [(CA, N, C, CB)]
_PLANAR = [
    [ACE_CH3, ACE_C, ACE_O, N],
    [C, O, NME_N, NME_CH3],
]


def _u(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v


def _build_leucine_coords() -> np.ndarray:
    """Hand-built extended conformer (Å)."""
    X = np.zeros((13, 3), dtype=np.float64)

    # Backbone in xy, Cα at origin; CB above plane for L chirality
    X[N] = np.array([-1.45, 0.20, 0.00])
    X[CA] = np.array([0.00, 0.00, 0.00])
    X[C] = np.array([1.52, 0.30, 0.00])
    X[CB] = np.array([-0.40, -0.80, 1.35])  # out of plane → V ≠ 0

    # ACE peptide plane (z=0): CH3–C(=O)–N
    X[ACE_C] = np.array([-2.50, 0.90, 0.00])
    X[ACE_O] = np.array([-2.45, 2.12, 0.00])
    X[ACE_CH3] = np.array([-3.85, 0.25, 0.00])

    # LEU–NME peptide plane (z=0)
    X[O] = np.array([2.20, -0.75, 0.00])
    X[NME_N] = np.array([2.30, 1.45, 0.00])
    X[NME_CH3] = np.array([3.70, 1.80, 0.00])

    # Extended side chain along +y,+z away from ACE/NME
    X[CG] = np.array([-1.10, -2.00, 1.80])
    X[CD1] = np.array([-2.50, -2.20, 2.40])
    X[CD2] = np.array([-0.40, -3.30, 2.20])

    X -= X.mean(0)
    return X


def _idealize(X: np.ndarray) -> np.ndarray:
    """Snap approximate coords onto distance/chiral/planar manifold (no antibump)."""
    from slicedot.geometry import Geometry

    g = Geometry(
        X, _BONDS, _ROTATABLE, _CHIRAL, _PLANAR,
        antibump=False,
        weights={"bond": 0.02, "angle": 0.04, "torsion14": 0.05,
                 "chiral": 0.05, "planar": 0.01, "bump": 0.3},
    )
    Xp, _, _ = g.project(X, tol=1e-8, max_iter=800)
    return Xp


_X0 = _idealize(_build_leucine_coords())


def sigma_of(resolution: float) -> float:
    return float(resolution) / 2.3548


def leucine_topology():
    return {
        "X_ref": _X0.copy(),
        "names": _ATOM_NAMES,
        "elements": _Z.astype(np.int64).copy(),
        "bonds": list(_BONDS),
        "rotatable_bonds": list(_ROTATABLE),
        "chiral_centres": list(_CHIRAL),
        "planar_groups": [list(g) for g in _PLANAR],
        "idx": {
            "ACE_C": ACE_C, "ACE_O": ACE_O, "ACE_CH3": ACE_CH3,
            "N": N, "CA": CA, "C": C, "O": O,
            "CB": CB, "CG": CG, "CD1": CD1, "CD2": CD2,
            "NME_N": NME_N, "NME_CH3": NME_CH3,
        },
        # CD1↔CD2 is a graph automorphism
        "automorphism_generators": [
            np.arange(13, dtype=np.int64),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12], dtype=np.int64),
        ],
    }


X0 = _X0.copy()
W = (_Z / _Z.sum()).copy()
names = _ATOM_NAMES
Z = _Z.copy()
bonds = list(_BONDS)
rotatable_bonds = list(_ROTATABLE)
chiral_centres = list(_CHIRAL)
planar_groups = [list(g) for g in _PLANAR]

def leucine_elements():
    """Atomic numbers for the capped leucine fixture."""
    return _Z.astype(np.int64).copy()


__all__ = [
    "X0", "W", "Z", "names", "sigma_of",
    "bonds", "rotatable_bonds", "chiral_centres", "planar_groups",
    "leucine_topology", "leucine_elements",
]

# Re-export oligopeptide builder for convenience.
from slicedot.fixtures_peptide import lrp_pdb_topology, oligopeptide_topology  # noqa: E402

__all__.extend(["oligopeptide_topology", "lrp_pdb_topology"])
