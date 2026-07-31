"""Geometry idealisation operator (P_restr) for alternating-projection refinement.

Builds distance / chiral / planar / antibump residuals from a reference structure
and drives an input pose onto that restraint manifold with nonlinear least
squares (SciPy trust-region reflective / Levenberg--Marquardt).

Distance, planar, and antibump terms use a ReLU flat-bottom: the residual is
zero while the violation is within ``slack``, and grows only outside that dead
zone (two-sided for distances / planarity; one-sided for antibump contacts).
Callers can anneal ``slack`` from loose → tight over Stage A so early OT steps
have room to move before chemistry is sharpened.

This is an *approximate* projection: it lands on (near) the restraint manifold
but is not guaranteed to be the nearest point (that would need Dykstra).  It is
rotation- and translation-invariant because all restraints are relative.

1-4 distance restraints are omitted across rotatable bonds so rotamer changes
remain reachable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "Geometry",
    "DEFAULT_WEIGHTS",
    "topo_distance_matrix",
    "build_distance_pairs",
    "chiral_volume",
    "shortest_path",
]

DEFAULT_WEIGHTS = {
    "bond": 0.02,       # 1-2, Å
    "angle": 0.04,      # 1-3 as distance, Å
    "torsion14": 0.04,  # 1-4 across non-rotatable bonds, Å
    "chiral": 0.1,      # Å³
    "planar": 0.02,     # Å
    "bump": 0.3,        # Å
}

INF = 10**6


def topo_distance_matrix(bonds: Iterable[tuple[int, int]], n: int) -> np.ndarray:
    """Floyd–Warshall topological distances on an undirected bond graph."""
    D = np.full((n, n), INF, dtype=np.int32)
    np.fill_diagonal(D, 0)
    for i, j in bonds:
        i, j = int(i), int(j)
        D[i, j] = D[j, i] = 1
    for k in range(n):
        D = np.minimum(D, D[:, k][:, None] + D[k][None, :])
    return D


def shortest_path(bonds: Iterable[tuple[int, int]], n: int, src: int,
                  dst: int) -> list[int]:
    """BFS shortest path from src to dst (node list including endpoints)."""
    adj = [[] for _ in range(n)]
    for i, j in bonds:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    parent = {src: -1}
    queue = [src]
    for u in queue:
        if u == dst:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if dst not in parent:
        return []
    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _norm_bond(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def build_distance_pairs(
    X_ref: np.ndarray,
    bonds: Iterable[tuple[int, int]],
    rotatable_bonds: Iterable[tuple[int, int]],
    maxsep: int = 3,
) -> list[tuple[int, int, float, str]]:
    """Return (i, j, d0, kind) with kind in {'bond','angle','torsion14'}.

    1-2 and 1-3 always; 1-4 only if the central bond of the unique shortest
    path is *not* rotatable.
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    n = X_ref.shape[0]
    bond_list = [(_norm_bond(int(i), int(j))) for i, j in bonds]
    rot = {_norm_bond(int(i), int(j)) for i, j in rotatable_bonds}
    D = topo_distance_matrix(bond_list, n)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dtop = int(D[i, j])
            if dtop < 1 or dtop > maxsep:
                continue
            d0 = float(np.linalg.norm(X_ref[j] - X_ref[i]))
            if dtop == 1:
                pairs.append((i, j, d0, "bond"))
            elif dtop == 2:
                pairs.append((i, j, d0, "angle"))
            else:  # dtop == 3
                path = shortest_path(bond_list, n, i, j)
                if len(path) != 4:
                    continue
                mid = _norm_bond(path[1], path[2])
                if mid in rot:
                    continue
                pairs.append((i, j, d0, "torsion14"))
    return pairs


def chiral_volume(X: np.ndarray, c: int, a: int, b: int, d: int) -> float:
    """Signed volume V = (a-c) · ((b-c) × (d-c))."""
    va = X[a] - X[c]
    vb = X[b] - X[c]
    vd = X[d] - X[c]
    return float(np.dot(va, np.cross(vb, vd)))


@dataclass
class _PlaneState:
    indices: np.ndarray
    # no reference offsets: residual drives signed height to 0


def _relu_flat(dev: float, slack: float) -> float:
    """Signed ReLU flat-bottom: 0 if |dev| ≤ slack, else sign(dev)·(|dev|-slack)."""
    a = abs(float(dev))
    s = float(slack)
    if a <= s:
        return 0.0
    return math.copysign(a - s, dev)


class Geometry:
    """Idealise coordinates against a reference geometry (P_restr)."""

    def __init__(
        self,
        X_ref: np.ndarray,
        bonds: Iterable[tuple[int, int]],
        rotatable_bonds: Iterable[tuple[int, int]] = (),
        chiral_centres: Iterable[tuple[int, int, int, int]] = (),
        planar_groups: Iterable[Sequence[int]] = (),
        weights: Optional[dict] = None,
        antibump: bool = True,
        antibump_r0: float = 2.8,
        slack: float = 0.0,
    ):
        self.X_ref = np.asarray(X_ref, dtype=np.float64).copy()
        if self.X_ref.ndim != 2 or self.X_ref.shape[1] != 3:
            raise ValueError("X_ref must be (N, 3)")
        self.n = int(self.X_ref.shape[0])
        self.bonds = [_norm_bond(int(i), int(j)) for i, j in bonds]
        self.rotatable_bonds = {_norm_bond(int(i), int(j)) for i, j in rotatable_bonds}
        self.chiral_centres = [
            tuple(int(x) for x in t) for t in chiral_centres
        ]
        self.planar_groups = [np.asarray(g, dtype=np.int64) for g in planar_groups]
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.antibump = bool(antibump)
        self.antibump_r0 = float(antibump_r0)
        # Dead-zone width (Å) for distance / planar / antibump ReLU flat-bottoms.
        self.slack = float(slack)

        self.D = topo_distance_matrix(self.bonds, self.n)
        self.pairs = build_distance_pairs(
            self.X_ref, self.bonds, self.rotatable_bonds, maxsep=3,
        )
        self.V_ref = [
            chiral_volume(self.X_ref, c, a, b, d)
            for c, a, b, d in self.chiral_centres
        ]
        if any(abs(v) < 1e-6 for v in self.V_ref):
            raise ValueError(
                "chiral volume ~0 in X_ref; need a non-planar tetrahedral centre"
            )
        # candidate antibump pairs: topo distance > 3
        self.bump_pairs = [
            (i, j)
            for i in range(self.n)
            for j in range(i + 1, self.n)
            if self.D[i, j] > 3 and self.D[i, j] < INF
        ]

    # ---------------------------------------------------------------- residuals
    def _distance_residuals(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Weighted ReLU flat-bottom residuals: 0 while |L−d0| ≤ slack."""
        slack = self.slack
        vals, kinds = [], []
        for i, j, d0, kind in self.pairs:
            L = float(np.linalg.norm(X[j] - X[i]))
            sigma = self.w["bond" if kind == "bond" else
                           "angle" if kind == "angle" else "torsion14"]
            vals.append(_relu_flat(L - d0, slack) / sigma)
            kinds.append(kind)
        return np.asarray(vals, dtype=np.float64), kinds

    def _distance_errors_A(self, X: np.ndarray) -> dict[str, list[float]]:
        """True |L−d0| in Å (ignores slack); for diagnostics only."""
        by: dict[str, list[float]] = {"bond": [], "angle": [], "torsion14": []}
        for i, j, d0, kind in self.pairs:
            L = float(np.linalg.norm(X[j] - X[i]))
            by[kind].append(abs(L - d0))
        return by

    def _chiral_residuals(self, X: np.ndarray) -> np.ndarray:
        sig = self.w["chiral"]
        return np.asarray(
            [(chiral_volume(X, c, a, b, d) - V0) / sig
             for (c, a, b, d), V0 in zip(self.chiral_centres, self.V_ref)],
            dtype=np.float64,
        )

    def _planar_residuals(self, X: np.ndarray) -> np.ndarray:
        """Signed out-of-plane heights with the same ReLU slack (Å)."""
        sig = self.w["planar"]
        slack = self.slack
        out = []
        for idxs in self.planar_groups:
            for h in self._planar_group_heights(X, idxs):
                out.append(_relu_flat(float(h), slack) / sig)
        return np.asarray(out, dtype=np.float64)

    def _bump_residuals(self, X: np.ndarray) -> tuple[np.ndarray, int]:
        """One-sided ReLU: 0 while L ≥ r0 − slack, else (r0 − slack − L)/σ."""
        if not self.antibump:
            return np.zeros(0, dtype=np.float64), 0
        sig = self.w["bump"]
        r0 = self.antibump_r0
        slack = self.slack
        vals = []
        active = 0
        for i, j in self.bump_pairs:
            L = float(np.linalg.norm(X[j] - X[i]))
            # Annealable dead zone: allow up to ``slack`` Å of contact before
            # the antibump residual turns on (one-sided; never attract).
            pen = max(0.0, r0 - L - slack)
            if pen > 0.0:
                vals.append(pen / sig)
                active += 1
            else:
                vals.append(0.0)
        return np.asarray(vals, dtype=np.float64), active

    def _bump_errors_A(self, X: np.ndarray) -> list[float]:
        """True penetrations max(0, r0−L) in Å (ignores slack); diagnostics only."""
        if not self.antibump:
            return []
        r0 = self.antibump_r0
        return [
            max(0.0, r0 - float(np.linalg.norm(X[j] - X[i])))
            for i, j in self.bump_pairs
        ]

    def _pack(self, X: np.ndarray) -> np.ndarray:
        d, _ = self._distance_residuals(X)
        parts = [d, self._chiral_residuals(X), self._planar_residuals(X)]
        bump, _ = self._bump_residuals(X)
        parts.append(bump)
        return np.concatenate(parts) if parts else np.zeros(0)

    def _jac(self, X: np.ndarray) -> np.ndarray:
        """Analytic Jacobian for distances / chiral / bump; FD for planarity."""
        n3 = 3 * self.n
        slack = self.slack
        rows = []
        # distances — zero Jacobian inside the flat-bottom dead zone
        for i, j, d0, kind in self.pairs:
            sigma = self.w["bond" if kind == "bond" else
                           "angle" if kind == "angle" else "torsion14"]
            v = X[j] - X[i]
            L = float(np.linalg.norm(v))
            row = np.zeros(n3)
            if L > 1e-12 and abs(L - d0) > slack:
                u = v / (L * sigma)
                row[3 * i:3 * i + 3] = -u
                row[3 * j:3 * j + 3] = u
            rows.append(row)
        # chiral
        sigc = self.w["chiral"]
        for (c, a, b, d), _V0 in zip(self.chiral_centres, self.V_ref):
            ga = np.cross(X[b] - X[c], X[d] - X[c]) / sigc
            gb = np.cross(X[d] - X[c], X[a] - X[c]) / sigc
            gd = np.cross(X[a] - X[c], X[b] - X[c]) / sigc
            gc = -(ga + gb + gd)
            row = np.zeros(n3)
            row[3 * a:3 * a + 3] = ga
            row[3 * b:3 * b + 3] = gb
            row[3 * d:3 * d + 3] = gd
            row[3 * c:3 * c + 3] = gc
            rows.append(row)
        # planar — forward FD on the small group (respects slack)
        eps = 1e-6
        sigp = self.w["planar"]
        for idxs in self.planar_groups:
            base = np.asarray(
                [_relu_flat(float(h), slack) / sigp
                 for h in self._planar_group_heights(X, idxs)],
                dtype=np.float64,
            )
            for h_i in range(len(idxs)):
                row = np.zeros(n3)
                for atom in idxs:
                    for k in range(3):
                        Xp = X.copy()
                        Xp[atom, k] += eps
                        hp = _relu_flat(
                            float(self._planar_group_heights(Xp, idxs)[h_i]),
                            slack,
                        ) / sigp
                        row[3 * atom + k] = (hp - base[h_i]) / eps
                rows.append(row)
        # bump — zero Jacobian inside the (r0 − slack) dead zone
        if self.antibump:
            sigb = self.w["bump"]
            r0 = self.antibump_r0
            r_on = r0 - slack
            for i, j in self.bump_pairs:
                v = X[j] - X[i]
                L = float(np.linalg.norm(v))
                row = np.zeros(n3)
                if L < r_on and L > 1e-12:
                    u = v / (L * sigb)
                    # r = (r0 - slack - L)/sig → dr/dxi = +u, dr/dxj = -u
                    row[3 * i:3 * i + 3] = u
                    row[3 * j:3 * j + 3] = -u
                rows.append(row)
        return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, n3))

    @staticmethod
    def _planar_group_heights(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
        P = X[idxs]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        return (P - c) @ vt[-1]

    def residual(self, X: np.ndarray) -> dict:
        """Per-class residual diagnostics (unweighted Å / Å³ where noted).

        Distance / planar / bump diagnostics report the true deviation (Å), not
        the slack-clipped ReLU residual used by the optimiser.
        """
        X = np.asarray(X, dtype=np.float64)
        by = self._distance_errors_A(X)
        bump_raw = self._bump_errors_A(X)
        _, n_active = self._bump_residuals(X)
        chiral = self._chiral_residuals(X)
        # raw planar heights (Å), not slack-clipped
        planar_raw = []
        for idxs in self.planar_groups:
            planar_raw.extend(np.abs(self._planar_group_heights(X, idxs)).tolist())
        planar_raw = np.asarray(planar_raw, dtype=np.float64)

        def _stats(arr_A):
            a = np.asarray(arr_A, dtype=np.float64)
            if a.size == 0:
                return {"max": 0.0, "rms": 0.0, "n": 0}
            return {"max": float(a.max()), "rms": float(np.sqrt((a ** 2).mean())),
                    "n": int(a.size)}

        packed = self._pack(X)
        return {
            "bond": _stats(by["bond"]),
            "angle": _stats(by["angle"]),
            "torsion14": _stats(by["torsion14"]),
            "chiral": {
                "max": float(np.max(np.abs(chiral)) * self.w["chiral"]) if chiral.size else 0.0,
                "rms": float(np.sqrt((chiral ** 2).mean()) * self.w["chiral"]) if chiral.size else 0.0,
                "n": int(chiral.size),
                "signs": [
                    int(np.sign(chiral_volume(X, c, a, b, d)))
                    for c, a, b, d in self.chiral_centres
                ],
            },
            "planar": _stats(planar_raw),
            "bump": {**_stats(bump_raw), "active": int(n_active)},
            "weighted_rms": float(np.sqrt((packed ** 2).mean())) if packed.size else 0.0,
            "slack": float(self.slack),
            "distance_max_A": float(max(
                _stats(by["bond"])["max"],
                _stats(by["angle"])["max"],
                _stats(by["torsion14"])["max"],
            )),
        }

    @staticmethod
    def _align_to(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Rigidly align P onto Q (Kabsch).  Picks the SE(3) gauge nearest the input."""
        pc, qc = P.mean(0), Q.mean(0)
        A, B = P - pc, Q - qc
        U, _, Vt = np.linalg.svd(A.T @ B)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt = Vt.copy()
            Vt[-1] *= -1
            R = Vt.T @ U.T
        return (A @ R) + qc

    def _gauss_seidel(self, X: np.ndarray, sweeps: int = 40) -> np.ndarray:
        """SHAKE-style warm start on distance restraints only."""
        X = X.copy()
        slack = self.slack
        for _ in range(sweeps):
            for i, j, d0, _kind in self.pairs:
                v = X[j] - X[i]
                L = float(np.linalg.norm(v))
                if L < 1e-12:
                    continue
                # Pull only to the edge of the flat-bottom dead zone.
                if abs(L - d0) <= slack:
                    continue
                target = d0 + math.copysign(slack, L - d0)
                c = 0.5 * 1.5 * (L - target) / L * v
                X[i] += c
                X[j] -= c
            # chiral volume Newton-ish correction
            for (c, a, b, d), V0 in zip(self.chiral_centres, self.V_ref):
                V = chiral_volume(X, c, a, b, d)
                # dV/d(x_a) = (b-c)×(d-c), etc.; move a,b,d equally
                ga = np.cross(X[b] - X[c], X[d] - X[c])
                gb = np.cross(X[d] - X[c], X[a] - X[c])
                gd = np.cross(X[a] - X[c], X[b] - X[c])
                gnorm2 = (ga * ga).sum() + (gb * gb).sum() + (gd * gd).sum()
                if gnorm2 < 1e-18:
                    continue
                step = 0.25 * (V - V0) / gnorm2
                X[a] -= step * ga
                X[b] -= step * gb
                X[d] -= step * gd
        return X

    # ---------------------------------------------------------------- project
    def project(
        self,
        X: np.ndarray,
        tol: float = 1e-4,
        max_iter: int = 200,
        slack: Optional[float] = None,
    ) -> tuple[np.ndarray, float, int]:
        """Idealise ``X`` onto the restraint manifold.

        Parameters
        ----------
        slack : float, optional
            Temporary ReLU flat-bottom width (Å) for distance / planar /
            antibump terms.  ``None`` keeps ``self.slack``.  Restored after
            the call.

        Returns
        -------
        X_proj : (N, 3)
        residual_rms : float
            RMS of the *weighted* residual vector (dimensionless).
        n_iter : int
            Number of residual evaluations reported by the solver.
        """
        X0 = np.asarray(X, dtype=np.float64).copy()
        if X0.shape != self.X_ref.shape:
            raise ValueError(f"X shape {X0.shape} != ref {self.X_ref.shape}")

        prev_slack = self.slack
        if slack is not None:
            self.slack = float(slack)
        try:
            packed0 = self._pack(X0)
            rms0 = float(np.sqrt((packed0 ** 2).mean())) if packed0.size else 0.0
            if rms0 <= tol:
                return X0, rms0, 0

            # Warm-start only when far from the manifold (GS can jostle near-solutions).
            if rms0 > 10 * tol:
                X_warm = self._gauss_seidel(X0, sweeps=min(80, max(20, max_iter // 2)))
                warm_cost = 80
            else:
                X_warm = X0
                warm_cost = 0

            def fun(v):
                return self._pack(v.reshape(self.n, 3))

            def jac(v):
                return self._jac(v.reshape(self.n, 3))

            m0 = fun(X_warm.ravel())
            method = "lm" if m0.size >= X0.size else "trf"
            X_proj = X_warm
            nfev = warm_cost
            rms = float(np.sqrt((m0 ** 2).mean())) if m0.size else 0.0
            # Satisfied once true errors sit inside the flat-bottom (+ tol).
            dist_ok = tol + self.slack
            for _ in range(3):
                info = self.residual(X_proj)
                if (
                    info["distance_max_A"] <= dist_ok
                    and info["planar"]["max"] <= 5 * tol + self.slack
                    and info["chiral"]["max"] <= 5 * tol
                    and info["bump"]["max"] <= tol + self.slack
                ):
                    break
                res = least_squares(
                    fun,
                    X_proj.ravel(),
                    jac=jac,
                    method=method,
                    ftol=1e-12,
                    xtol=1e-10,
                    gtol=1e-10,
                    max_nfev=max(max_iter * 2, 100),
                    verbose=0,
                )
                X_proj = res.x.reshape(self.n, 3)
                nfev += int(res.nfev)
                packed = self._pack(X_proj)
                rms = float(np.sqrt((packed ** 2).mean())) if packed.size else 0.0
            # Distance restraints leave an SE(3) gauge; align back to the input pose
            # so the operator is idempotent and equivariant under rigid motions.
            X_proj = self._align_to(X_proj, X0)
            packed = self._pack(X_proj)
            rms = float(np.sqrt((packed ** 2).mean())) if packed.size else 0.0
            return X_proj, rms, nfev
        finally:
            self.slack = prev_slack
