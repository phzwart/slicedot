"""Coordinate perturbations for tests and peptide demos.

These are generators of *starting* poses, not part of P_restr.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "rotation_matrix",
    "set_torsion",
    "backrub",
    "downstream_atoms",
    "dihedral",
]


def dihedral(X: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Signed dihedral angle i-j-k-l in radians, range (-π, π]."""
    b0 = X[j] - X[i]
    b1 = X[k] - X[j]
    b2 = X[l] - X[k]
    b1u = b1 / (np.linalg.norm(b1) + 1e-30)
    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u
    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)
    return float(np.arctan2(y, x))


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation by ``angle`` (radians) about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-15:
        return np.eye(3)
    axis = axis / n
    K = np.array(
        [[0, -axis[2], axis[1]],
         [axis[2], 0, -axis[0]],
         [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def downstream_atoms(bonds, n: int, a: int, b: int) -> list[int]:
    """Atoms on the ``b`` side of bond a–b (excluding a), via BFS with a blocked."""
    adj = [[] for _ in range(n)]
    for i, j in bonds:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    seen = {b}
    stack = [b]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v == a or v in seen:
                continue
            seen.add(v)
            stack.append(v)
    return sorted(seen)


def set_torsion(
    X: np.ndarray,
    a: int,
    b: int,
    downstream: Optional[Sequence[int]],
    angle: float,
    bonds=None,
) -> np.ndarray:
    """Rotate ``downstream`` about bond a→b by ``angle`` (radians).

    If ``downstream`` is None, infer it from ``bonds`` as the b-side of a–b.
    """
    X = np.asarray(X, dtype=np.float64).copy()
    if downstream is None:
        if bonds is None:
            raise ValueError("bonds required when downstream is None")
        downstream = downstream_atoms(bonds, len(X), a, b)
    R = rotation_matrix(X[b] - X[a], angle)
    pivot = X[b]
    for i in downstream:
        if i == a:
            continue
        X[i] = (X[i] - pivot) @ R.T + pivot
    return X


def backrub(
    X: np.ndarray,
    ca_im1: int,
    ca_ip1: int,
    peptide_atoms: Sequence[int],
    angle: float,
) -> np.ndarray:
    """Backrub: rotate ``peptide_atoms`` about the Cα(i−1)–Cα(i+1) axis.

    A full production backrub also counter-rotates the flanking peptide groups;
    this helper applies the primary axis rotation used to generate perturbed
    starts for demos.
    """
    X = np.asarray(X, dtype=np.float64).copy()
    axis = X[ca_ip1] - X[ca_im1]
    R = rotation_matrix(axis, angle)
    pivot = X[ca_im1]
    for i in peptide_atoms:
        X[i] = (X[i] - pivot) @ R.T + pivot
    return X
