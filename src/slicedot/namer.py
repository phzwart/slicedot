"""Recover atom labels for a free-atom (unlabelled) coordinate cloud.

Pipeline
--------
0. Element blocking (hard)
1. Hungarian assignment on unary distance to ``X_prior``
2. Score restraint residual; flag outliers / near-degenerate swaps
3. Local repair (seed-and-extend + small-cluster enumeration)
4. Chirality sign repair
5. Quotient by graph automorphisms; report factorised ambiguous groups

This is *not* a global quadratic assignment solve — the molecular graph keeps
clusters tiny.
"""
from __future__ import annotations

import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from slicedot.geometry import (
    Geometry,
    chiral_volume,
    topo_distance_matrix,
)
from slicedot.restraints import (
    PairRestraint,
    RestraintSet,
    pair_dev,
    restraint_set_from_geometry,
)

__all__ = ["Assignment", "Namer"]

_INF = 1e12

# ΔZ below this is not treated as weight-discriminable (C/N/O all fail).
MIN_WEIGHT_DZ = 10

# Naming uses looser σ than P_restr chemistry weights: free-atom clouds sit at
# ~0.3–0.7 Å RMSD, so bond/angle targets cannot be enforced at 0.02 Å.
# Used only by the geometry-compat RestraintSet builder.
DEFAULT_NAMING_WEIGHTS = {
    "bond": 0.35,       # Å
    "angle": 0.45,      # Å
    "torsion14": 0.55,  # Å
    "chiral": 0.5,      # Å³ (unused in E; chirality is sign-only)
    "planar": 0.3,
    "bump": 0.5,
}

# Back-compat aliases
_Rest = PairRestraint
_pair_dev = pair_dev


def _weight_discriminable_groups(z_vals: Sequence[int],
                                 min_dz: int = MIN_WEIGHT_DZ) -> list[list[int]]:
    """Connected components of Z under |Zi−Zj| < min_dz (indiscriminable).

    Distinct components differ by ≥ ``min_dz`` electrons and may be separated
    by scattering weights; within a component, geometry must type atoms.
    """
    z_vals = sorted(int(z) for z in z_vals)
    parent = {z: z for z in z_vals}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(z_vals):
        for b in z_vals[i + 1:]:
            if abs(a - b) < int(min_dz):
                union(a, b)
    groups: dict[int, list[int]] = {}
    for z in z_vals:
        groups.setdefault(find(z), []).append(z)
    return [sorted(g) for g in groups.values()]


@dataclass
class Assignment:
    """Result of ``Namer.assign``."""

    perm: np.ndarray  # label i -> position index in Y
    Y_named: np.ndarray  # Y[perm]
    restraint_rms: float
    unary_rms: float
    n_repaired: int
    chirality_repaired: int
    ambiguous_groups: list = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _norm_bond(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _as_elements(elements) -> np.ndarray:
    return np.asarray(elements, dtype=np.int64).ravel()


def _adj_list(bonds: Iterable[tuple[int, int]], n: int) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for i, j in bonds:
        i, j = int(i), int(j)
        adj[i].append(j)
        adj[j].append(i)
    return adj


# ---------------------------------------------------------------------------
# Automorphism search (coloured graph, colour refinement + backtrack)
# ---------------------------------------------------------------------------

def _colour_refine(adj: list[list[int]], colours: np.ndarray) -> np.ndarray:
    """1-dimensional Weisfeiler–Lehman colour refinement."""
    n = len(adj)
    colours = np.asarray(colours, dtype=np.int64).copy()
    while True:
        signatures = []
        for i in range(n):
            neigh = tuple(sorted(int(colours[j]) for j in adj[i]))
            signatures.append((int(colours[i]), neigh))
        # Rank unique signatures
        ranking = {sig: k for k, sig in enumerate(sorted(set(signatures)))}
        new = np.array([ranking[s] for s in signatures], dtype=np.int64)
        if np.array_equal(new, colours):
            return new
        colours = new


def _automorphism_group(
    adj: list[list[int]],
    colours: np.ndarray,
    max_group: int = 4096,
) -> list[np.ndarray]:
    """Return Aut(G) as permutations (including identity), capped at max_group.

    Backtracking search with colour-refinement pruning.  Adequate for peptide
    graphs whose automorphism group is a product of small local swaps.
    """
    n = len(adj)
    colours = _colour_refine(adj, colours)
    # Cells: colour -> list of vertices
    cells: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(colours):
        cells[int(c)].append(i)
    # Only cells with size > 1 can move
    free_cells = [sorted(v) for v in cells.values() if len(v) > 1]
    if not free_cells:
        return [np.arange(n, dtype=np.int64)]

    # Precompute adjacency matrix for fast checks
    A = np.zeros((n, n), dtype=np.int8)
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            A[i, j] = 1

    group: list[np.ndarray] = []

    def consistent(perm_partial: dict[int, int], src: int, dst: int) -> bool:
        if colours[src] != colours[dst]:
            return False
        for u, v in perm_partial.items():
            if A[src, u] != A[dst, v]:
                return False
        return True

    # Flatten free vertices in cell order
    free = [v for cell in free_cells for v in cell]
    fixed = [i for i in range(n) if i not in set(free)]

    def search(perm: dict[int, int], remaining: list[int]):
        if len(group) >= max_group:
            return
        if not remaining:
            p = np.empty(n, dtype=np.int64)
            for i in fixed:
                p[i] = i
            for s, d in perm.items():
                p[s] = d
            # Fixed points that weren't in free must map to themselves — already.
            # Verify full adjacency preservation on free set.
            ok = True
            for s, d in perm.items():
                for u, v in perm.items():
                    if A[s, u] != A[d, v]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                group.append(p)
            return
        src = remaining[0]
        # Candidate images: unused vertices in same colour cell
        cell = [v for v in cells[int(colours[src])] if v not in perm.values()]
        for dst in cell:
            if not consistent(perm, src, dst):
                continue
            perm[src] = dst
            search(perm, remaining[1:])
            del perm[src]
            if len(group) >= max_group:
                return

    # Seed with identity mapping on fixed + search free
    search({}, free)
    if not group:
        group.append(np.arange(n, dtype=np.int64))
    # Ensure identity present
    ident = np.arange(n, dtype=np.int64)
    if not any(np.array_equal(p, ident) for p in group):
        group.insert(0, ident)
    return group


def _compose_aut_generators(gens: Sequence[np.ndarray], n: int,
                            max_group: int = 4096) -> list[np.ndarray]:
    """Close a set of permutation generators under composition."""
    ident = np.arange(n, dtype=np.int64)
    group = {tuple(ident.tolist())}
    frontier = [ident]
    gen_list = [np.asarray(g, dtype=np.int64) for g in gens]
    while frontier and len(group) < max_group:
        p = frontier.pop()
        for g in gen_list:
            q = g[p]  # apply g after p: i -> g[p[i]]
            key = tuple(q.tolist())
            if key not in group:
                group.add(key)
                frontier.append(q)
    return [np.array(k, dtype=np.int64) for k in group]


# ---------------------------------------------------------------------------
# Namer
# ---------------------------------------------------------------------------

class Namer:
    """Assign labels to an unlabelled free-atom cloud using a restraint CIF prior.

    Pairwise geometry comes from ``restraint_set``: 1–2 bonds, 1–3 angles,
    ring/plane planarity, and selective plane-scoped 1–4 distances.  ``X_ref``
    is only a coordinate scaffold for chirality volumes and, when no
    ``restraint_set`` is given, for the geometry-compat dictionary builder.
    Unary matching still uses the caller-supplied ``X_prior``.
    """

    def __init__(
        self,
        X_ref: np.ndarray,
        elements=None,
        bonds: Iterable[tuple[int, int]] = (),
        *,
        restraint_set: Optional[RestraintSet] = None,
        rotatable_bonds: Iterable[tuple[int, int]] = (),
        chiral_centres: Iterable[tuple[int, int, int, int]] = (),
        planar_groups: Iterable[Sequence[int]] = (),
        weights: Optional[dict] = None,
        atom_ids: Optional[Sequence[str]] = None,
        automorphisms: Optional[Sequence[np.ndarray]] = None,
        flag_k_sigma: float = 3.0,
        swap_tau: Optional[float] = None,
        residual_flag: float = 0.5,
        beam_width: int = 8,
        max_enum: int = 8,
        cluster_cap: int = 10,
        torsion14: str = "planar",
    ):
        self.X_ref = np.asarray(X_ref, dtype=np.float64)
        if self.X_ref.ndim != 2 or self.X_ref.shape[1] != 3:
            raise ValueError("X_ref must be (N, 3)")
        self.n = int(self.X_ref.shape[0])
        plane_groups_in = [list(map(int, g)) for g in planar_groups]

        if restraint_set is not None:
            self.restraint_set = restraint_set
            if int(restraint_set.n) != self.n:
                raise ValueError("restraint_set size must match X_ref")
            self.elements = np.asarray(restraint_set.elements, dtype=np.int64)
            self.bonds = restraint_set.bond_indices()
            if not plane_groups_in and restraint_set.planes:
                plane_groups_in = restraint_set.plane_index_groups()
        else:
            if elements is None:
                raise ValueError("elements required when restraint_set is omitted")
            self.elements = _as_elements(elements)
            if self.elements.shape[0] != self.n:
                raise ValueError("elements length must match X_ref")
            bond_list = [_norm_bond(int(i), int(j)) for i, j in bonds]
            self.restraint_set = restraint_set_from_geometry(
                self.X_ref,
                self.elements,
                bond_list,
                rotatable_bonds=rotatable_bonds,
                planar_groups=plane_groups_in,
                atom_ids=atom_ids,
                weights={**DEFAULT_NAMING_WEIGHTS, **(weights or {})},
                torsion14=torsion14,
            )
            self.bonds = bond_list
            if not plane_groups_in and self.restraint_set.planes:
                plane_groups_in = self.restraint_set.plane_index_groups()

        self.rotatable_bonds = {
            _norm_bond(int(i), int(j)) for i, j in rotatable_bonds
        }
        self.chiral_centres = [
            tuple(int(x) for x in t) for t in chiral_centres
        ]
        self.planar_groups = plane_groups_in
        self._plane_sig = (
            self.restraint_set.plane_esds()
            if self.restraint_set.planes
            else [float(DEFAULT_NAMING_WEIGHTS["planar"])] * len(self.planar_groups)
        )
        if len(self._plane_sig) < len(self.planar_groups):
            self._plane_sig = list(self._plane_sig) + [
                float(DEFAULT_NAMING_WEIGHTS["planar"])
            ] * (len(self.planar_groups) - len(self._plane_sig))
        self.w = {**DEFAULT_NAMING_WEIGHTS, **(weights or {})}
        self.flag_k_sigma = float(flag_k_sigma)
        self.residual_flag = float(residual_flag)
        self.beam_width = int(beam_width)
        self.max_enum = int(max_enum)
        self.cluster_cap = int(cluster_cap)

        self.D = topo_distance_matrix(self.bonds, self.n)
        self.adj = _adj_list(self.bonds, self.n)
        self.restraints: list[PairRestraint] = self.restraint_set.to_naming_pairs()

        # Min bond length at each atom (for unary flag threshold)
        self.min_bond = np.full(self.n, 1.4, dtype=np.float64)
        for r in self.restraints:
            if r.kind == "bond":
                d0 = 0.5 * (r.d_lo + r.d_hi)
                self.min_bond[r.i] = min(self.min_bond[r.i], d0)
                self.min_bond[r.j] = min(self.min_bond[r.j], d0)

        self.V_ref = [
            chiral_volume(self.X_ref, c, a, b, d)
            for c, a, b, d in self.chiral_centres
        ]

        # Median pairwise σ² → default temperature / swap tau
        sigs = np.array([r.sig for r in self.restraints], dtype=np.float64)
        med_sig2 = float(np.median(sigs ** 2)) if sigs.size else 1e-3
        self.swap_tau = float(swap_tau) if swap_tau is not None else 4.0 * med_sig2
        self.T = med_sig2

        # Automorphisms
        if automorphisms is not None:
            gens = [np.asarray(p, dtype=np.int64) for p in automorphisms]
            self.automorphisms = _compose_aut_generators(gens, self.n)
        else:
            self.automorphisms = _automorphism_group(
                self.adj, self.elements.copy(),
            )

        # Element -> label indices
        self._elem_labels: dict[int, np.ndarray] = {}
        for z in np.unique(self.elements):
            self._elem_labels[int(z)] = np.where(self.elements == z)[0]

    # ---------------------------------------------------------------- energy
    def _planar_energy(self, Yn: np.ndarray,
                       atoms: Optional[set[int]] = None) -> float:
        e = 0.0
        for idxs, sig in zip(self.planar_groups, self._plane_sig):
            if atoms is not None and not atoms.intersection(idxs):
                continue
            if len(idxs) < 3:
                continue
            hs = Geometry._planar_group_heights(Yn, np.asarray(idxs, dtype=np.int64))
            s = max(float(sig), 1e-3)
            e += float(((hs / s) ** 2).sum())
        return e

    def energy(self, Y: np.ndarray, perm: np.ndarray, X_prior: np.ndarray,
               unary_weight: float = 1.0) -> float:
        """Full naming energy E(π)."""
        Y = np.asarray(Y, dtype=np.float64)
        Xp = np.asarray(X_prior, dtype=np.float64)
        perm = np.asarray(perm, dtype=np.int64)
        Yn = Y[perm]
        e_u = float(((Yn - Xp) ** 2).sum())
        e_p = 0.0
        for r in self.restraints:
            L = float(np.linalg.norm(Yn[r.i] - Yn[r.j]))
            e_p += (_pair_dev(L, r.d_lo, r.d_hi) / r.sig) ** 2
        e_p += self._planar_energy(Yn)
        return unary_weight * e_u + e_p

    def _pairwise_energy(self, Y: np.ndarray, perm: np.ndarray,
                         atoms: Optional[Sequence[int]] = None) -> float:
        Yn = Y[perm]
        atom_set = None if atoms is None else set(int(a) for a in atoms)
        e = 0.0
        for r in self.restraints:
            if atom_set is not None and r.i not in atom_set and r.j not in atom_set:
                continue
            L = float(np.linalg.norm(Yn[r.i] - Yn[r.j]))
            e += (_pair_dev(L, r.d_lo, r.d_hi) / r.sig) ** 2
        e += self._planar_energy(Yn, atom_set)
        return e

    def restraint_rms(self, Y: np.ndarray, perm: np.ndarray) -> float:
        Yn = Y[np.asarray(perm, dtype=np.int64)]
        errs = [
            _pair_dev(float(np.linalg.norm(Yn[r.i] - Yn[r.j])), r.d_lo, r.d_hi)
            for r in self.restraints
        ]
        for idxs in self.planar_groups:
            if len(idxs) < 3:
                continue
            hs = Geometry._planar_group_heights(Yn, np.asarray(idxs, dtype=np.int64))
            errs.extend(np.abs(hs).tolist())
        if not errs:
            return 0.0
        a = np.asarray(errs, dtype=np.float64)
        return float(np.sqrt((a ** 2).mean()))

    def unary_rms(self, Y: np.ndarray, perm: np.ndarray,
                  X_prior: np.ndarray) -> float:
        Yn = Y[np.asarray(perm, dtype=np.int64)]
        d2 = ((Yn - X_prior) ** 2).sum(-1)
        return float(np.sqrt(d2.mean()))

    # ---------------------------------------------------------------- stage 0/1
    def _hungarian(self, Y: np.ndarray, X_prior: np.ndarray,
                   y_elements: Optional[np.ndarray] = None) -> np.ndarray:
        """Element-blocked Hungarian on unary squared distances.

        Returns perm: label -> Y index.
        """
        Y = np.asarray(Y, dtype=np.float64)
        Xp = np.asarray(X_prior, dtype=np.float64)
        n = self.n
        if Y.shape != (n, 3) or Xp.shape != (n, 3):
            raise ValueError(f"Y/X_prior must be ({n}, 3)")

        if y_elements is None:
            # Positions typed only through assignment within label element blocks:
            # both sides of each block are the same-Z labels; Y slots are shared
            # globally but we assign each Z-block of labels to a Z-block of
            # *candidate slots*.  Without typed Y, use all slots for the first
            # pass is wrong — we need |block| slots.  Convention: Y is a
            # permutation of labelled atoms (same composition), so we can type
            # Y slots by a provisional nearest-label Z, or treat Y as untyped
            # and assign each element block to the globally remaining slots
            # greedily by Hungarian within expanding pools.
            #
            # Standard approach when composition matches: partition *labels* by
            # Z and partition *Y indices* by inferred Z from weights, else run
            # one global Hungarian with infinite cross-element costs after
            # typing Y as unknown → use label Z only by matching equal-sized
            # blocks against all Y (sequential claim).
            y_elements = self._infer_y_elements(Y, Xp, weights=None)

        perm = np.full(n, -1, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        for z, labels in self._elem_labels.items():
            slots = np.where(y_elements == z)[0]
            if slots.size != labels.size:
                # Fallback: take unused slots closest in count
                free = np.where(~used)[0]
                if free.size < labels.size:
                    raise ValueError(
                        f"element Z={z}: {labels.size} labels but "
                        f"{slots.size} typed / {free.size} free slots"
                    )
                # Prefer correctly typed; fill from free if needed
                if slots.size < labels.size:
                    extra = [i for i in free if i not in set(slots.tolist())]
                    slots = np.concatenate([
                        slots, np.asarray(extra[: labels.size - slots.size]),
                    ])
                else:
                    slots = slots[: labels.size]
            # Cost: labels × slots
            cost = np.empty((labels.size, slots.size), dtype=np.float64)
            for a, li in enumerate(labels):
                d = Y[slots] - Xp[li]
                cost[a] = (d ** 2).sum(-1)
            ri, cj = linear_sum_assignment(cost)
            for a, b in zip(ri, cj):
                perm[labels[a]] = int(slots[b])
                used[slots[b]] = True
        if np.any(perm < 0):
            raise RuntimeError("Hungarian left unassigned labels")
        return perm

    def _geometric_type_slots(
        self,
        Y: np.ndarray,
        X_prior: np.ndarray,
        z_allowed: Optional[Sequence[int]] = None,
        slot_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Capacity-constrained geometric typing on a subset of Z / slots.

        Returns a length-``n`` array; slots outside ``slot_mask`` are -1.
        """
        n = self.n
        z_vals = sorted(
            int(z) for z in (z_allowed if z_allowed is not None
                             else self._elem_labels.keys())
        )
        if slot_mask is None:
            slots = np.arange(n, dtype=np.int64)
        else:
            slots = np.where(slot_mask)[0].astype(np.int64)
        tokens: list[int] = []
        for z in z_vals:
            tokens.extend([z] * int(self._elem_labels[z].size))
        if len(tokens) != slots.size:
            raise RuntimeError(
                f"geometric typing size mismatch: {len(tokens)} tokens vs "
                f"{slots.size} slots"
            )
        tokens_a = np.asarray(tokens, dtype=np.int64)
        m = slots.size
        cost = np.empty((m, m), dtype=np.float64)
        Ys = Y[slots]
        for z in z_vals:
            labs = self._elem_labels[z]
            d2 = ((Ys[:, None, :] - X_prior[labs][None, :, :]) ** 2).sum(-1)
            min_d = d2.min(axis=1)
            cols = np.where(tokens_a == z)[0]
            cost[:, cols] = min_d[:, None]
        ri, cj = linear_sum_assignment(cost)
        y_el = np.full(n, -1, dtype=np.int64)
        for a, b in zip(ri, cj):
            y_el[int(slots[a])] = int(tokens_a[b])
        return y_el

    def _infer_y_elements(
        self,
        Y: np.ndarray,
        X_prior: np.ndarray,
        weights: Optional[np.ndarray],
        min_dz: int = MIN_WEIGHT_DZ,
    ) -> np.ndarray:
        """Type each Y slot by element (composition-preserving).

        Scattering weights only separate species whose Z differs by ≥
        ``min_dz`` (default 10 e⁻).  C/N/O (ΔZ ≤ 2) are never split by
        weight — geometry types within each indiscriminable group.

        If weights are absent or all Z lie in one group, falls back to
        capacity-constrained geometric typing against ``X_prior``.
        """
        n = self.n
        z_vals = sorted(self._elem_labels.keys())
        groups = _weight_discriminable_groups(z_vals, min_dz=min_dz)

        # Single indiscriminable group (e.g. organic C/N/O only) → geometry.
        if (
            weights is None
            or len(groups) <= 1
            or np.ptp(np.asarray(weights, dtype=np.float64)) <= 1e-15
        ):
            return self._geometric_type_slots(Y, X_prior)

        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != n:
            raise ValueError("weights length must match N")

        # Coarse tokens: one per label, valued by mean Z of its weight-group.
        z_to_group = {}
        group_mean: list[float] = []
        for gi, g in enumerate(groups):
            group_mean.append(float(np.mean(g)))
            for z in g:
                z_to_group[z] = gi
        # Slot → group via Hungarian on |w_scaled - group_mean|
        scale = float(np.mean(group_mean) / (w.mean() + 1e-30))
        tokens_g: list[int] = []
        for z in z_vals:
            tokens_g.extend([z_to_group[z]] * int(self._elem_labels[z].size))
        tokens_g_a = np.asarray(tokens_g, dtype=np.int64)
        means = np.asarray(group_mean, dtype=np.float64)
        cost = np.abs(w[:, None] * scale - means[tokens_g_a][None, :])
        ri, cj = linear_sum_assignment(cost)
        slot_group = np.empty(n, dtype=np.int64)
        for s, t in zip(ri, cj):
            slot_group[s] = int(tokens_g_a[t])

        # Within each group, geometry splits the Z's that weight cannot.
        y_el = np.full(n, -1, dtype=np.int64)
        for gi, g in enumerate(groups):
            mask = slot_group == gi
            if not np.any(mask):
                continue
            if len(g) == 1:
                y_el[mask] = int(g[0])
            else:
                part = self._geometric_type_slots(
                    Y, X_prior, z_allowed=g, slot_mask=mask,
                )
                y_el[mask] = part[mask]

        if np.any(y_el < 0):
            return self._geometric_type_slots(Y, X_prior)
        ok = all(
            int((y_el == z).sum()) == labs.size
            for z, labs in self._elem_labels.items()
        )
        if not ok:
            return self._geometric_type_slots(Y, X_prior)
        return y_el

    def _distance_only_perm(self, Y: np.ndarray, X_prior: np.ndarray) -> np.ndarray:
        """Global Hungarian on ||Y[j]-X_prior[i]||^2 (no element block)."""
        cost = ((Y[:, None, :] - X_prior[None, :, :]) ** 2).sum(-1).T  # (label, slot)
        ri, cj = linear_sum_assignment(cost)
        perm = np.empty(self.n, dtype=np.int64)
        perm[ri] = cj
        return perm

    def _bond_fingerprint(self, coords: np.ndarray, i: int) -> np.ndarray:
        """Sorted bond distances at labelled atom ``i`` (pad/truncate to 4)."""
        nbrs = self.adj[i]
        if not nbrs:
            return np.zeros(4, dtype=np.float64)
        d = sorted(float(np.linalg.norm(coords[i] - coords[j])) for j in nbrs)
        out = np.zeros(4, dtype=np.float64)
        out[: min(4, len(d))] = d[:4]
        return out

    def _spatial_fingerprint(
        self, coords: np.ndarray, i: int,
        rmin: float = 1.15, rmax: float = 1.85,
    ) -> np.ndarray:
        """Sorted near-neighbour distances — valid on an unlabelled cloud."""
        d = np.linalg.norm(coords - coords[i], axis=1)
        mask = (d >= rmin) & (d <= rmax)
        mask[i] = False
        dists = sorted(float(x) for x in d[mask])
        out = np.zeros(4, dtype=np.float64)
        out[: min(4, len(dists))] = dists[:4]
        return out

    def _fingerprint_hungarian(
        self, Y: np.ndarray, y_elements: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Rotation-invariant init: match spatial bond-length fingerprints.

        Uses distance-shell neighbours so ``Y`` need not be in label order
        (the covalent ``adj`` graph only applies to labelled ``X_ref``).
        """
        del y_elements
        fp_ref = np.stack(
            [self._spatial_fingerprint(self.X_ref, i) for i in range(self.n)]
        )
        fp_y = np.stack(
            [self._spatial_fingerprint(Y, j) for j in range(self.n)]
        )
        cost = ((fp_ref[:, None, :] - fp_y[None, :, :]) ** 2).sum(-1)
        ri, cj = linear_sum_assignment(cost)
        perm = np.empty(self.n, dtype=np.int64)
        perm[ri] = cj
        return perm

    # ---------------------------------------------------------------- stage 2
    def _flag_atoms(self, Y: np.ndarray, perm: np.ndarray,
                    X_prior: np.ndarray) -> set[int]:
        Yn = Y[perm]
        flagged: set[int] = set()
        # Unary
        for i in range(self.n):
            d = float(np.linalg.norm(Yn[i] - X_prior[i]))
            thr = max(0.7, 0.5 * float(self.min_bond[i]))
            if d > thr:
                flagged.add(i)
        # Pairwise — use a soft threshold so geometry errors surface under noise
        k = self.flag_k_sigma
        for r in self.restraints:
            L = float(np.linalg.norm(Yn[r.i] - Yn[r.j]))
            # Absolute floor 0.35 Å catches bond/angle breaks at ~0.5 Å RMSD
            if _pair_dev(L, r.d_lo, r.d_hi) > max(k * r.sig, 0.35):
                flagged.add(r.i)
                flagged.add(r.j)
        # Expand to bonded neighbours, then trial improving / near-degenerate swaps.
        seed = set(flagged)
        for i in list(seed):
            for j in self.adj[i]:
                seed.add(j)
        atoms = sorted(seed)
        e0 = self.energy(Y, perm, X_prior)
        for a, b in itertools.combinations(atoms, 2):
            if self.elements[a] != self.elements[b]:
                continue
            if float(np.linalg.norm(Yn[a] - Yn[b])) > 3.5:
                continue
            p2 = perm.copy()
            p2[a], p2[b] = p2[b], p2[a]
            e1 = self.energy(Y, p2, X_prior)
            if e1 < e0 - 1e-9 or abs(e1 - e0) < self.swap_tau:
                flagged.add(a)
                flagged.add(b)
        # One global pass: any same-Z pair within 3 Å whose swap lowers E.
        for r in self.restraints:
            i, j = r.i, r.j
            if self.elements[i] != self.elements[j]:
                continue
            if float(np.linalg.norm(Yn[i] - Yn[j])) > 3.0:
                continue
            p2 = perm.copy()
            p2[i], p2[j] = p2[j], p2[i]
            if self.energy(Y, p2, X_prior) < e0 - 1e-9:
                flagged.add(i)
                flagged.add(j)
        return flagged

    def _clusters(self, flagged: set[int]) -> list[list[int]]:
        """Connected components of flagged ∪ restraint-neighbours, capped."""
        if not flagged:
            return []
        # Expand one hop along bonds
        nodes = set(flagged)
        for i in list(flagged):
            for j in self.adj[i]:
                nodes.add(j)
        # Also link through restraint edges inside nodes
        node_list = sorted(nodes)
        node_set = set(node_list)
        adj_f: dict[int, list[int]] = {i: [] for i in node_list}
        for i in node_list:
            for j in self.adj[i]:
                if j in node_set:
                    adj_f[i].append(j)
        for r in self.restraints:
            if r.i in node_set and r.j in node_set:
                adj_f[r.i].append(r.j)
                adj_f[r.j].append(r.i)

        seen = set()
        comps: list[list[int]] = []
        for s in node_list:
            if s in seen:
                continue
            comp = []
            dq = deque([s])
            seen.add(s)
            while dq:
                u = dq.popleft()
                comp.append(u)
                for v in adj_f[u]:
                    if v not in seen:
                        seen.add(v)
                        dq.append(v)
            # Cap: if too big, keep flagged core only
            if len(comp) > self.cluster_cap:
                core = sorted(set(comp) & flagged)
                if not core:
                    core = sorted(comp)[: self.cluster_cap]
                elif len(core) > self.cluster_cap:
                    core = core[: self.cluster_cap]
                comps.append(core)
            else:
                comps.append(sorted(comp))
        return comps

    # ---------------------------------------------------------------- stage 3
    def _cluster_energy(self, Y, perm, X_prior, cluster: Sequence[int]) -> float:
        """Pairwise restraints touching cluster + weak unary tie-break."""
        Yn = Y[perm]
        e = 0.0
        cset = set(cluster)
        for i in cluster:
            e += 0.05 * float(((Yn[i] - X_prior[i]) ** 2).sum())
        for r in self.restraints:
            if r.i in cset or r.j in cset:
                L = float(np.linalg.norm(Yn[r.i] - Yn[r.j]))
                e += (_pair_dev(L, r.d_lo, r.d_hi) / r.sig) ** 2
        return e

    def _enumerate_cluster(
        self, Y, perm, X_prior, cluster: Sequence[int],
    ) -> np.ndarray:
        cluster = list(cluster)
        m = len(cluster)
        if m == 0:
            return perm
        # Group by element within cluster
        by_z: dict[int, list[int]] = defaultdict(list)
        for i in cluster:
            by_z[int(self.elements[i])].append(i)
        # Positions currently assigned to cluster labels
        slots_by_z = {
            z: [int(perm[i]) for i in labs] for z, labs in by_z.items()
        }
        best_perm = perm.copy()
        best_e = self._cluster_energy(Y, best_perm, X_prior, cluster)

        # Cartesian product of per-element permutations of slots
        z_keys = list(by_z.keys())
        slot_perms = []
        for z in z_keys:
            labs = by_z[z]
            slots = slots_by_z[z]
            slot_perms.append(list(itertools.permutations(slots)))
        # Bound explosion
        total = 1
        for sp in slot_perms:
            total *= max(len(sp), 1)
        if total > 40320:  # 8!
            # Fallback: pairwise 2-opt
            return self._two_opt_cluster(Y, perm, X_prior, cluster)

        for choice in itertools.product(*slot_perms):
            p = perm.copy()
            for z, slots_perm in zip(z_keys, choice):
                for lab, slot in zip(by_z[z], slots_perm):
                    p[lab] = int(slot)
            e = self._cluster_energy(Y, p, X_prior, cluster)
            if e < best_e - 1e-12:
                best_e = e
                best_perm = p
        return best_perm

    def _two_opt_cluster(self, Y, perm, X_prior, cluster) -> np.ndarray:
        p = perm.copy()
        improved = True
        while improved:
            improved = False
            e0 = self._cluster_energy(Y, p, X_prior, cluster)
            for a, b in itertools.combinations(cluster, 2):
                if self.elements[a] != self.elements[b]:
                    continue
                p[a], p[b] = p[b], p[a]
                e1 = self._cluster_energy(Y, p, X_prior, cluster)
                if e1 < e0 - 1e-12:
                    e0 = e1
                    improved = True
                else:
                    p[a], p[b] = p[b], p[a]
        return p

    def _seed_and_extend(
        self, Y, perm, X_prior, cluster: Sequence[int],
    ) -> Optional[np.ndarray]:
        """Grow assignment from best-determined atom; return new perm or None."""
        cluster = list(cluster)
        if not cluster:
            return perm
        Yn0 = Y[perm]
        # Seed: lowest unary among cluster, unique by margin
        unaries = {
            i: float(np.linalg.norm(Yn0[i] - X_prior[i])) for i in cluster
        }
        ordered = sorted(cluster, key=lambda i: unaries[i])
        seed = ordered[0]
        if len(ordered) > 1 and unaries[ordered[1]] - unaries[seed] < 0.05:
            # Ambiguous seed — let enumeration handle it
            return None

        # Pool of Y slots belonging to cluster labels
        pool = {int(perm[i]) for i in cluster}
        # Beam: list of (partial map label->slot, used slots)
        # Start by assigning seed to its current slot (or best same-Z in pool)
        z_seed = int(self.elements[seed])
        cand_slots = [s for s in pool if True]  # typed below
        # Prefer slots near X_prior[seed]
        scored = []
        for s in pool:
            # Only same element — we don't have Y elements here; use all pool
            # and filter by checking that some cluster label of that Z owns it
            # under π₀ … actually pool slots can be reassigned among same Z.
            scored.append((float(np.linalg.norm(Y[s] - X_prior[seed])), s))
        scored.sort()
        # Restrict seed candidates to slots currently held by same-Z labels
        same_z_slots = [int(perm[i]) for i in cluster if self.elements[i] == z_seed]
        seed_cands = [s for _, s in scored if s in same_z_slots][: self.beam_width]
        if not seed_cands:
            return None

        ksig = self.flag_k_sigma
        # Restraints inside / touching cluster for pruning
        local_rest = [
            r for r in self.restraints if r.i in cluster or r.j in cluster
        ]

        beam: list[dict[int, int]] = [{seed: s} for s in seed_cands]

        placed_order = [seed]
        remaining = [i for i in cluster if i != seed]
        # Extend along bond graph: prefer neighbours of placed
        while remaining and beam:
            # Pick next atom: bonded to placed if possible
            next_atom = None
            for i in remaining:
                if any(j in placed_order for j in self.adj[i]):
                    next_atom = i
                    break
            if next_atom is None:
                next_atom = remaining[0]
            z = int(self.elements[next_atom])
            avail_slots_template = [
                int(perm[i]) for i in cluster if self.elements[i] == z
            ]

            new_beam: list[tuple[float, dict[int, int]]] = []
            for partial in beam:
                used = set(partial.values())
                for s in avail_slots_template:
                    if s in used:
                        continue
                    # Check restraints against already placed
                    ok = True
                    for r in local_rest:
                        if r.i == next_atom and r.j in partial:
                            L = float(np.linalg.norm(Y[s] - Y[partial[r.j]]))
                            if _pair_dev(L, r.d_lo, r.d_hi) > ksig * r.sig * 2.0:
                                ok = False
                                break
                        elif r.j == next_atom and r.i in partial:
                            L = float(np.linalg.norm(Y[partial[r.i]] - Y[s]))
                            if _pair_dev(L, r.d_lo, r.d_hi) > ksig * r.sig * 2.0:
                                ok = False
                                break
                    if not ok:
                        continue
                    p2 = dict(partial)
                    p2[next_atom] = s
                    # Score: unary + satisfied pairwise
                    e = float(np.linalg.norm(Y[s] - X_prior[next_atom]) ** 2)
                    for r in local_rest:
                        if r.i in p2 and r.j in p2:
                            L = float(np.linalg.norm(Y[p2[r.i]] - Y[p2[r.j]]))
                            e += (_pair_dev(L, r.d_lo, r.d_hi) / r.sig) ** 2
                    new_beam.append((e, p2))
            new_beam.sort(key=lambda t: t[0])
            beam = [p for _, p in new_beam[: self.beam_width]]
            placed_order.append(next_atom)
            remaining.remove(next_atom)

        if not beam or len(beam[0]) != len(cluster):
            return None
        # Pick best complete partial
        best_partial = min(
            beam,
            key=lambda p: self._cluster_energy(
                Y, self._apply_partial(perm, p), X_prior, cluster,
            ),
        )
        return self._apply_partial(perm, best_partial)

    @staticmethod
    def _apply_partial(perm: np.ndarray, partial: dict[int, int]) -> np.ndarray:
        p = perm.copy()
        # Reassign carefully: partial maps label->slot; may need swaps
        # Build inverse of current perm for labels in partial
        labels = list(partial.keys())
        # Target slots
        for lab, slot in partial.items():
            p[lab] = slot
        # Fix duplicates: if two labels share a slot, restore from unused
        # Within cluster, partial should be a bijection onto its slot set.
        return p

    def _unary_margin(self, Y, perm, X_prior, label: int) -> float:
        """Gap between assigned unary cost and best alternate same-Z slot."""
        z = int(self.elements[label])
        labs = self._elem_labels[z]
        slots = [int(perm[i]) for i in labs]
        d_asg = float(np.linalg.norm(Y[perm[label]] - X_prior[label]))
        best_alt = _INF
        for s in slots:
            if s == int(perm[label]):
                continue
            best_alt = min(
                best_alt, float(np.linalg.norm(Y[s] - X_prior[label])),
            )
        return best_alt - d_asg

    def _repair(self, Y, perm, X_prior, flagged: set[int]) -> tuple[np.ndarray, int]:
        """Local repair on uncertain atoms (small unary margin or pair breaks)."""
        p = perm.copy()
        n_repaired = 0

        def score(pp: np.ndarray) -> float:
            return (
                self._pairwise_energy(Y, pp)
                + 0.25 * float(((Y[pp] - X_prior) ** 2).sum())
            )

        # Only touch atoms that are uncertain under the prior or pairwise-broken.
        uncertain = {
            i for i in flagged
            if self._unary_margin(Y, p, X_prior, i) < 0.5
        }
        # Always keep pairwise-flagged atoms (margin ignored)
        Yn = Y[p]
        for r in self.restraints:
            L = float(np.linalg.norm(Yn[r.i] - Yn[r.j]))
            if _pair_dev(L, r.d_lo, r.d_hi) > max(self.flag_k_sigma * r.sig, 0.35):
                uncertain.add(r.i)
                uncertain.add(r.j)
        if not uncertain:
            return p, 0

        # 2-opt over uncertain same-Z pairs
        improved = True
        guard = 0
        while improved and guard < 50:
            improved = False
            guard += 1
            e0 = score(p)
            Yn = Y[p]
            atoms = sorted(uncertain)
            for a, b in itertools.combinations(atoms, 2):
                if self.elements[a] != self.elements[b]:
                    continue
                if float(np.linalg.norm(Yn[a] - Yn[b])) > 3.5:
                    continue
                # Refuse to swap two high-confidence atoms
                if (
                    self._unary_margin(Y, p, X_prior, a) > 0.8
                    and self._unary_margin(Y, p, X_prior, b) > 0.8
                ):
                    continue
                p[a], p[b] = p[b], p[a]
                e1 = score(p)
                if e1 < e0 - 1e-9:
                    e0 = e1
                    n_repaired += 1
                    improved = True
                    uncertain.add(a)
                    uncertain.add(b)
                else:
                    p[a], p[b] = p[b], p[a]

        for cluster in self._clusters(uncertain):
            if len(cluster) <= 1:
                continue
            # Skip clusters where every atom is unary-certain
            if all(self._unary_margin(Y, p, X_prior, i) > 0.8 for i in cluster):
                continue
            p_before = p.copy()
            if len(cluster) <= self.max_enum:
                p_try = self._enumerate_cluster(Y, p, X_prior, cluster)
            else:
                p_try = self._seed_and_extend(Y, p, X_prior, cluster)
                if p_try is None:
                    Yn = Y[p]
                    core = sorted(
                        cluster,
                        key=lambda i: -float(np.linalg.norm(Yn[i] - X_prior[i])),
                    )[: self.max_enum]
                    p_try = self._enumerate_cluster(Y, p, X_prior, core)
            if p_try is not None and score(p_try) <= score(p) + 1e-12:
                n_repaired += int(
                    np.sum(p_try[list(cluster)] != p_before[list(cluster)])
                ) // 2
                p = p_try
        return p, n_repaired

    # ---------------------------------------------------------------- stage 4
    def _repair_chirality(
        self, Y, perm, X_prior,
    ) -> tuple[np.ndarray, int, list[str]]:
        p = perm.copy()
        n_fix = 0
        flags: list[str] = []
        for (c, a, b, d), V0 in zip(self.chiral_centres, self.V_ref):
            if abs(V0) < 1e-8:
                continue
            Yn = Y[p]
            V = chiral_volume(Yn, c, a, b, d)
            if np.sign(V) == np.sign(V0):
                continue
            # Try swapping pairs among substituents {a,b,d} with same Z
            subs = [a, b, d]
            fixed = False
            best = None
            best_e = self.energy(Y, p, X_prior)
            for i, j in itertools.combinations(subs, 2):
                if self.elements[i] != self.elements[j]:
                    continue
                p2 = p.copy()
                p2[i], p2[j] = p2[j], p2[i]
                Yn2 = Y[p2]
                V2 = chiral_volume(Yn2, c, a, b, d)
                if np.sign(V2) != np.sign(V0):
                    continue
                e2 = self.energy(Y, p2, X_prior)
                if e2 < best_e + self.swap_tau:  # allow mild energy rise
                    best_e = e2
                    best = p2
                    fixed = True
            if fixed and best is not None:
                p = best
                n_fix += 1
            else:
                flags.append(f"chiral_inversion:{c}")
        return p, n_fix, flags

    # ---------------------------------------------------------------- stage 5
    def _ambiguous_groups(
        self, Y, perm, X_prior,
    ) -> list:
        """Independent local swap groups with Boltzmann weights."""
        groups = []
        # Aut orbits of size > 1 among same-Z that are near-degenerate under E
        # Collect transposition-like generators from automorphism group
        seen_pairs: set[tuple[int, int]] = set()
        e0 = self.energy(Y, perm, X_prior)
        for alpha in self.automorphisms:
            # Find moved points
            moved = np.where(alpha != np.arange(self.n))[0]
            if moved.size == 0:
                continue
            # Partition moved into cycles; keep 2-cycles
            visited = set()
            for s in moved:
                if s in visited:
                    continue
                cycle = []
                x = int(s)
                while x not in visited:
                    visited.add(x)
                    cycle.append(x)
                    x = int(alpha[x])
                if len(cycle) == 2:
                    a, b = sorted(cycle)
                    if (a, b) in seen_pairs:
                        continue
                    seen_pairs.add((a, b))
                    # Evaluate identity vs swap on these labels' slots…
                    # Ambiguity of *labelling*: α acts on labels.
                    # Equivalent assignments: perm' = perm ∘ α means
                    # label i gets slot that α⁻¹(i) had… 
                    # Report local alternatives on labels (a,b):
                    p_swap = perm.copy()
                    # Applying α to labels: new_perm[i] = perm[α[i]]
                    # For transposition (a b): swap slots assigned to a and b
                    p_swap[a], p_swap[b] = perm[b], perm[a]
                    e1 = self.energy(Y, p_swap, X_prior)
                    if abs(e1 - e0) < self.swap_tau * 4:
                        # Boltzmann weights
                        logits = np.array([0.0, -(e1 - e0) / max(self.T, 1e-12)])
                        logits -= logits.max()
                        w = np.exp(logits)
                        w = w / w.sum()
                        groups.append((
                            [a, b],
                            [
                                (np.array([0, 1], dtype=np.int64), float(w[0])),
                                (np.array([1, 0], dtype=np.int64), float(w[1])),
                            ],
                        ))
        return groups

    def equivalent_to(self, perm: np.ndarray, truth: np.ndarray) -> bool:
        """True if ``perm`` equals ``truth ∘ α`` for some automorphism α."""
        perm = np.asarray(perm, dtype=np.int64)
        truth = np.asarray(truth, dtype=np.int64)
        for alpha in self.automorphisms:
            # perm[i] == truth[alpha[i]]  ∀i
            if np.array_equal(perm, truth[alpha]):
                return True
        return False

    # ---------------------------------------------------------------- assign
    def assign(
        self,
        Y: np.ndarray,
        X_prior: np.ndarray,
        weights: Optional[np.ndarray] = None,
        *,
        distance_only: bool = False,
        element_blocked_only: bool = False,
    ) -> Assignment:
        """Return an ``Assignment`` of labels to positions in ``Y``."""
        Y = np.asarray(Y, dtype=np.float64)
        X_prior = np.asarray(X_prior, dtype=np.float64)
        flags: list[str] = []

        if distance_only:
            perm = self._distance_only_perm(Y, X_prior)
            return Assignment(
                perm=perm,
                Y_named=Y[perm].copy(),
                restraint_rms=self.restraint_rms(Y, perm),
                unary_rms=self.unary_rms(Y, perm, X_prior),
                n_repaired=0,
                chirality_repaired=0,
                ambiguous_groups=[],
                flags=flags,
            )

        y_el = self._infer_y_elements(Y, X_prior, weights)
        # When scattering weights cannot split species (e.g. organic C/N/O),
        # a geometric Z pre-type is brittle under OT landing noise and then
        # traps the assignment: repair only swaps same-Z labels.  Prefer a
        # global unary Hungarian in that case; keep element-blocked init when
        # weights actually separate groups (ΔZ ≥ MIN_WEIGHT_DZ).
        z_vals = sorted(self._elem_labels.keys())
        weight_groups = _weight_discriminable_groups(z_vals, min_dz=MIN_WEIGHT_DZ)
        if element_blocked_only or len(weight_groups) > 1:
            perm0 = self._hungarian(Y, X_prior, y_elements=y_el)
        else:
            perm0 = self._distance_only_perm(Y, X_prior)
        perm = perm0.copy()
        n_repaired = 0
        chirality_repaired = 0

        # If the prior is far (crossing / wrong pose), unary init is unreliable —
        # fall back to rotation-invariant bond fingerprints from X_ref.
        if self.unary_rms(Y, perm, X_prior) > 2.0:
            perm_fp = self._fingerprint_hungarian(Y, y_el)
            if self._pairwise_energy(Y, perm_fp) < self._pairwise_energy(Y, perm):
                perm = perm_fp
                flags.append("fingerprint_init")

        if not element_blocked_only:
            flagged = self._flag_atoms(Y, perm, X_prior)
            if flagged:
                perm, n_repaired = self._repair(Y, perm, X_prior, flagged)
            # Chirality only if restraint residual is already sane — otherwise
            # substituent swaps on a mis-labelled centre make things worse.
            if self.restraint_rms(Y, perm) <= self.residual_flag * 2:
                perm, chirality_repaired, cflags = self._repair_chirality(
                    Y, perm, X_prior,
                )
                flags.extend(cflags)
            else:
                # Still report inversions without attempting repair.
                Yn = Y[perm]
                for (c, a, b, d), V0 in zip(self.chiral_centres, self.V_ref):
                    if abs(V0) < 1e-8:
                        continue
                    if np.sign(chiral_volume(Yn, c, a, b, d)) != np.sign(V0):
                        flags.append(f"chiral_inversion:{c}")

        r_rms = self.restraint_rms(Y, perm)
        u_rms = self.unary_rms(Y, perm, X_prior)
        if r_rms > self.residual_flag:
            flags.append("high_restraint_residual")

        amb = []
        if not element_blocked_only and not distance_only:
            amb = self._ambiguous_groups(Y, perm, X_prior)

        return Assignment(
            perm=perm,
            Y_named=Y[perm].copy(),
            restraint_rms=r_rms,
            unary_rms=u_rms,
            n_repaired=int(n_repaired),
            chirality_repaired=int(chirality_repaired),
            ambiguous_groups=amb,
            flags=flags,
        )
