"""CIF-shaped geometric restraint dictionaries for naming (and later P_restr).

Naming dictionaries use standard chem_comp content only:
  * 1–2 bonds
  * 1–3 angles
  * planarity / ring planes
  * a small set of specific 1–4 distances (ring / plane diagonals)

Bulk conformer-measured 1–4 torsions across every rigid bond are *not* part of
the naming prior.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from slicedot.geometry import build_distance_pairs, shortest_path, topo_distance_matrix

__all__ = [
    "AngleRestraint",
    "BondRestraint",
    "PairRestraint",
    "PlaneRestraint",
    "RestraintSet",
    "load_restraint_cif",
    "pair_dev",
    "restraint_set_from_geometry",
    "write_restraint_cif",
]

_ELEMENT_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "CL": 17, "BR": 35, "I": 53, "SE": 34, "FE": 26, "ZN": 30,
}


@dataclass(frozen=True)
class PairRestraint:
    """Pairwise distance restraint; harmonic iff ``d_lo == d_hi``."""

    i: int
    j: int
    d_lo: float
    d_hi: float
    sig: float
    kind: str = "bond"


@dataclass(frozen=True)
class BondRestraint:
    atom_id_1: str
    atom_id_2: str
    value: float
    esd: float


@dataclass(frozen=True)
class AngleRestraint:
    """Angle at ``atom_id_2``. Optional ``value_min``/``value_max`` → flat bottom."""

    atom_id_1: str
    atom_id_2: str
    atom_id_3: str
    value: float
    esd: float
    value_min: Optional[float] = None
    value_max: Optional[float] = None


@dataclass(frozen=True)
class PlaneRestraint:
    """Named planar / ring group (``chem_comp_plane``)."""

    plane_id: str
    atom_ids: tuple[str, ...]
    esd: float = 0.30  # Å, out-of-plane height σ for naming


def pair_dev(L: float, d_lo: float, d_hi: float) -> float:
    """Unsigned deviation outside ``[d_lo, d_hi]`` (0 inside)."""
    if L < d_lo:
        return d_lo - L
    if L > d_hi:
        return L - d_hi
    return 0.0


def dist_for_angle(L1: float, L2: float, theta_deg: float) -> float:
    """1–3 distance for bond lengths L1, L2 and angle θ (degrees)."""
    ca = math.cos(math.radians(float(theta_deg)))
    return float(math.sqrt(max(L1 * L1 + L2 * L2 - 2.0 * L1 * L2 * ca, 0.0)))


def _angle_to_dist_sigma(L1: float, L2: float, theta_deg: float, esd_deg: float,
                         d0: float) -> float:
    """Propagate angle esd (degrees) to a 1–3 distance σ."""
    th = math.radians(float(theta_deg))
    esd = math.radians(float(esd_deg))
    # d² = L1²+L2²−2 L1 L2 cosθ  →  ∂d/∂θ = (L1 L2 sinθ) / d
    if d0 < 1e-8:
        return max(0.1, abs(L1 * L2 * esd))
    dd = abs(L1 * L2 * math.sin(th) / d0) * esd
    return float(max(dd, 0.1))


@dataclass
class RestraintSet:
    """Named-atom restraint dictionary (CIF chem_comp style)."""

    comp_id: str
    atom_ids: list[str]
    elements: np.ndarray  # atomic numbers, parallel to atom_ids
    bonds: list[BondRestraint] = field(default_factory=list)
    angles: list[AngleRestraint] = field(default_factory=list)
    planes: list[PlaneRestraint] = field(default_factory=list)
    # Specific 1–4 (ring/plane diagonals); not bulk rigid-torsion inventory
    extra_distances: list[PairRestraint] = field(default_factory=list)

    def __post_init__(self):
        self.atom_ids = [str(a) for a in self.atom_ids]
        self.elements = np.asarray(self.elements, dtype=np.int64).ravel()
        if len(self.atom_ids) != int(self.elements.shape[0]):
            raise ValueError("atom_ids and elements length mismatch")
        if len(set(self.atom_ids)) != len(self.atom_ids):
            raise ValueError("atom_ids must be unique")
        self._index = {a: i for i, a in enumerate(self.atom_ids)}

    @property
    def n(self) -> int:
        return len(self.atom_ids)

    def index_of(self, name: str) -> int:
        try:
            return self._index[str(name)]
        except KeyError as e:
            raise KeyError(f"unknown atom_id {name!r}") from e

    def bond_indices(self) -> list[tuple[int, int]]:
        out = []
        for b in self.bonds:
            i, j = self.index_of(b.atom_id_1), self.index_of(b.atom_id_2)
            out.append((i, j) if i < j else (j, i))
        return out

    def bond_length(self, a: str, b: str) -> float:
        """Ideal bond length between two atom ids (order-independent)."""
        a, b = str(a), str(b)
        for br in self.bonds:
            if {br.atom_id_1, br.atom_id_2} == {a, b}:
                return float(br.value)
        raise KeyError(f"no bond restraint for {a!r}–{b!r}")

    def to_naming_pairs(self) -> list[PairRestraint]:
        """Convert bond/angle table to pairwise distance restraints."""
        out: list[PairRestraint] = []
        for br in self.bonds:
            i = self.index_of(br.atom_id_1)
            j = self.index_of(br.atom_id_2)
            d0 = float(br.value)
            sig = max(float(br.esd), 1e-3)
            out.append(PairRestraint(i, j, d0, d0, sig, kind="bond"))

        for ar in self.angles:
            i = self.index_of(ar.atom_id_1)
            m = self.index_of(ar.atom_id_2)
            j = self.index_of(ar.atom_id_3)
            L1 = self.bond_length(ar.atom_id_2, ar.atom_id_1)
            L2 = self.bond_length(ar.atom_id_2, ar.atom_id_3)
            d0 = dist_for_angle(L1, L2, ar.value)
            sig = _angle_to_dist_sigma(L1, L2, ar.value, ar.esd, d0)
            if ar.value_min is not None or ar.value_max is not None:
                lo = float(ar.value_min if ar.value_min is not None else ar.value)
                hi = float(ar.value_max if ar.value_max is not None else ar.value)
                ds = [d0, dist_for_angle(L1, L2, lo), dist_for_angle(L1, L2, hi)]
                d_lo, d_hi = min(ds), max(ds)
            else:
                d_lo = d_hi = d0
            a, b = (i, j) if i < j else (j, i)
            out.append(PairRestraint(a, b, d_lo, d_hi, sig, kind="angle"))

        out.extend(self.extra_distances)
        return out

    def plane_index_groups(self) -> list[list[int]]:
        """Plane atom indices (for Namer planar residuals)."""
        return [[self.index_of(a) for a in p.atom_ids] for p in self.planes]

    def plane_esds(self) -> list[float]:
        return [float(p.esd) for p in self.planes]


# Default distance-space σ used when synthesising a RestraintSet from coords.
_DEFAULT_GEOM_SIG = {
    "bond": 0.35,
    "angle": 0.45,
    "torsion14": 0.55,
    "planar": 0.30,
}


def _planar_scoped_14(
    X_ref: np.ndarray,
    bond_list: list[tuple[int, int]],
    planar_groups: Sequence[Sequence[int]],
    sig: float,
) -> list[PairRestraint]:
    """1–4 distances whose both ends lie in the same planar / ring group."""
    n = int(X_ref.shape[0])
    D = topo_distance_matrix(bond_list, n)
    seen: set[tuple[int, int]] = set()
    out: list[PairRestraint] = []
    for group in planar_groups:
        idxs = sorted({int(i) for i in group})
        if len(idxs) < 4:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if int(D[i, j]) != 3:
                    continue
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                seen.add(key)
                d0 = float(np.linalg.norm(X_ref[j] - X_ref[i]))
                out.append(
                    PairRestraint(key[0], key[1], d0, d0, float(sig), kind="torsion14")
                )
    return out


def restraint_set_from_geometry(
    X_ref: np.ndarray,
    elements,
    bonds: Iterable[tuple[int, int]],
    *,
    rotatable_bonds: Iterable[tuple[int, int]] = (),
    planar_groups: Iterable[Sequence[int]] = (),
    atom_ids: Optional[Sequence[str]] = None,
    weights: Optional[dict] = None,
    comp_id: str = "LIG",
    torsion14: str = "planar",
) -> RestraintSet:
    """Build a naming ``RestraintSet`` from coordinates + topology (compat).

    Parameters
    ----------
    torsion14
        ``\"planar\"`` (default): only 1–4 pairs inside planar/ring groups.
        ``\"none\"``: bonds + angles + planes only.
        ``\"all\"``: legacy — every non-rotatable 1–4 (not recommended for naming).
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    n = int(X_ref.shape[0])
    elements = np.asarray(elements, dtype=np.int64).ravel()
    if atom_ids is None:
        atom_ids = [f"A{i}" for i in range(n)]
    else:
        atom_ids = [str(a) for a in atom_ids]
    w = {**_DEFAULT_GEOM_SIG, **(weights or {})}
    bond_list = [(int(i), int(j)) for i, j in bonds]
    plane_groups = [list(map(int, g)) for g in planar_groups]
    # Always harvest 1–2 / 1–3; optionally all rigid 1–4 for legacy.
    maxsep = 3 if torsion14 == "all" else 2
    pairs = build_distance_pairs(
        X_ref, bond_list, rotatable_bonds, maxsep=maxsep,
    )

    bond_rest: list[BondRestraint] = []
    angle_rest: list[AngleRestraint] = []
    extra: list[PairRestraint] = []
    for i, j, d0, kind in pairs:
        d0 = float(d0)
        if kind == "bond":
            bond_rest.append(
                BondRestraint(atom_ids[i], atom_ids[j], d0, float(w["bond"]))
            )
        elif kind == "angle":
            path = shortest_path(bond_list, n, int(i), int(j))
            if len(path) != 3:
                continue
            a, m, b = path
            v1 = X_ref[a] - X_ref[m]
            v2 = X_ref[b] - X_ref[m]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            ca = float(np.clip(np.dot(v1, v2) / max(n1 * n2, 1e-30), -1.0, 1.0))
            ang = math.degrees(math.acos(ca))
            sig_d = float(w["angle"])
            th = math.radians(ang)
            denom = abs(n1 * n2 * math.sin(th) / max(d0, 1e-8))
            esd_deg = math.degrees(sig_d / max(denom, 1e-8)) if denom > 1e-12 else 15.0
            angle_rest.append(
                AngleRestraint(
                    atom_ids[a], atom_ids[m], atom_ids[b],
                    ang, max(esd_deg, 1.0),
                )
            )
        else:  # torsion14 from maxsep=3 / torsion14=="all"
            extra.append(
                PairRestraint(int(i), int(j), d0, d0, float(w["torsion14"]),
                              kind="torsion14")
            )

    if torsion14 == "planar":
        extra.extend(
            _planar_scoped_14(X_ref, bond_list, plane_groups, float(w["torsion14"]))
        )
    elif torsion14 not in ("none", "all", "planar"):
        raise ValueError(
            f"torsion14 must be 'planar', 'none', or 'all'; got {torsion14!r}"
        )

    planes: list[PlaneRestraint] = []
    for k, group in enumerate(plane_groups):
        if len(group) < 3:
            continue
        planes.append(
            PlaneRestraint(
                plane_id=f"P{k}",
                atom_ids=tuple(atom_ids[i] for i in group),
                esd=float(w["planar"]),
            )
        )

    return RestraintSet(
        comp_id=comp_id,
        atom_ids=list(atom_ids),
        elements=elements,
        bonds=bond_rest,
        angles=angle_rest,
        planes=planes,
        extra_distances=extra,
    )


# ---------------------------------------------------------------------------
# Minimal chem_comp CIF I/O
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""(?:"([^"]*)")|(?:'([^']*)')|([^\s]+)""",
)


def _tokenize(line: str) -> list[str]:
    """Tokenise a CIF data line (no semicolon text fields)."""
    out = []
    for m in _TOKEN_RE.finditer(line.strip()):
        out.append(m.group(1) or m.group(2) or m.group(3))
    return out


def _z_from_symbol(sym: str) -> int:
    key = str(sym).strip().upper()
    if key in _ELEMENT_Z:
        return _ELEMENT_Z[key]
    # Element may be given as "C" / "C1" style — take leading letters
    m = re.match(r"([A-Za-z]{1,2})", key)
    if m and m.group(1) in _ELEMENT_Z:
        return _ELEMENT_Z[m.group(1)]
    raise ValueError(f"unknown element symbol {sym!r}")


def load_restraint_cif(path: Union[str, Path]) -> RestraintSet:
    """Load bond/angle/plane restraints from a minimal chem_comp CIF."""
    text = Path(path).read_text()
    lines = text.splitlines()

    comp_id = "LIG"
    atoms: list[tuple[str, int]] = []
    bonds: list[BondRestraint] = []
    angles: list[AngleRestraint] = []
    plane_atoms: dict[str, list[str]] = {}
    plane_esd: dict[str, float] = {}
    extra: list[PairRestraint] = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.startswith("data_"):
            comp_id = line[5:].strip() or comp_id
            i += 1
            continue
        if line.startswith("_chem_comp.id"):
            toks = _tokenize(line)
            if len(toks) >= 2:
                comp_id = toks[1]
            i += 1
            continue
        if line == "loop_":
            i += 1
            tags: list[str] = []
            while i < len(lines):
                t = lines[i].strip()
                if t.startswith("_"):
                    tags.append(t.split()[0])
                    i += 1
                else:
                    break
            rows: list[list[str]] = []
            while i < len(lines):
                t = lines[i].strip()
                if not t or t.startswith("#"):
                    i += 1
                    continue
                if t == "loop_" or t.startswith("data_") or t.startswith("_"):
                    break
                rows.append(_tokenize(t))
                i += 1

            def col(name: str) -> Optional[int]:
                for k, t in enumerate(tags):
                    if t.split(".")[-1] == name:
                        return k
                return None

            kind = tags[0] if tags else ""
            if "chem_comp_atom" in kind and "plane" not in kind:
                i_id, i_sym = col("atom_id"), col("type_symbol")
                if i_id is None or i_sym is None:
                    continue
                for row in rows:
                    if max(i_id, i_sym) >= len(row):
                        continue
                    atoms.append((row[i_id], _z_from_symbol(row[i_sym])))
            elif "chem_comp_bond" in kind:
                i1, i2 = col("atom_id_1"), col("atom_id_2")
                iv, ie = col("value_dist"), col("value_dist_esd")
                if None in (i1, i2, iv, ie):
                    continue
                for row in rows:
                    if max(i1, i2, iv, ie) >= len(row):
                        continue
                    bonds.append(
                        BondRestraint(
                            row[i1], row[i2],
                            float(row[iv]), float(row[ie]),
                        )
                    )
            elif "chem_comp_angle" in kind:
                i1, i2, i3 = col("atom_id_1"), col("atom_id_2"), col("atom_id_3")
                iv, ie = col("value_angle"), col("value_angle_esd")
                imin, imax = col("value_angle_min"), col("value_angle_max")
                if None in (i1, i2, i3, iv, ie):
                    continue
                for row in rows:
                    if max(i1, i2, i3, iv, ie) >= len(row):
                        continue
                    vmin = (
                        float(row[imin])
                        if imin is not None and imin < len(row)
                        and row[imin] not in (".", "?")
                        else None
                    )
                    vmax = (
                        float(row[imax])
                        if imax is not None and imax < len(row)
                        and row[imax] not in (".", "?")
                        else None
                    )
                    angles.append(
                        AngleRestraint(
                            row[i1], row[i2], row[i3],
                            float(row[iv]), float(row[ie]),
                            value_min=vmin, value_max=vmax,
                        )
                    )
            elif "chem_comp_plane" in kind:
                ip, ia = col("plane_id"), col("atom_id")
                ie = col("dist_esd")
                if ip is None or ia is None:
                    continue
                for row in rows:
                    if max(ip, ia) >= len(row):
                        continue
                    pid, aid = row[ip], row[ia]
                    plane_atoms.setdefault(pid, []).append(aid)
                    if ie is not None and ie < len(row) and row[ie] not in (".", "?"):
                        plane_esd[pid] = float(row[ie])
            elif "chem_comp_dist" in kind:
                # Optional explicit 1–4 (etc.) distance restraints.
                i1, i2 = col("atom_id_1"), col("atom_id_2")
                iv, ie = col("value_dist"), col("value_dist_esd")
                if None in (i1, i2, iv, ie) or not atoms:
                    continue
                id_to_i = {a: k for k, (a, _) in enumerate(atoms)}
                for row in rows:
                    if max(i1, i2, iv, ie) >= len(row):
                        continue
                    a, b = row[i1], row[i2]
                    if a not in id_to_i or b not in id_to_i:
                        continue
                    ia_, ib_ = id_to_i[a], id_to_i[b]
                    if ia_ > ib_:
                        ia_, ib_ = ib_, ia_
                    d0 = float(row[iv])
                    extra.append(
                        PairRestraint(
                            ia_, ib_, d0, d0, float(row[ie]), kind="torsion14",
                        )
                    )
            continue
        i += 1

    if not atoms:
        raise ValueError(f"no _chem_comp_atom loop in {path}")
    atom_ids = [a for a, _ in atoms]
    elements = np.array([z for _, z in atoms], dtype=np.int64)
    planes = [
        PlaneRestraint(
            plane_id=pid,
            atom_ids=tuple(aids),
            esd=float(plane_esd.get(pid, 0.30)),
        )
        for pid, aids in plane_atoms.items()
        if len(aids) >= 3
    ]
    return RestraintSet(
        comp_id=comp_id,
        atom_ids=atom_ids,
        elements=elements,
        bonds=bonds,
        angles=angles,
        planes=planes,
        extra_distances=extra,
    )


def write_restraint_cif(path: Union[str, Path], rs: RestraintSet) -> None:
    """Write a minimal chem_comp restraint CIF."""
    lines = [
        f"data_{rs.comp_id}",
        "# slicedot naming / geometry restraint dictionary",
        f"_chem_comp.id {rs.comp_id}",
        "loop_",
        "_chem_comp_atom.comp_id",
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
    ]
    rev_z = {v: k for k, v in _ELEMENT_Z.items()}
    for name, z in zip(rs.atom_ids, rs.elements):
        sym = rev_z.get(int(z), f"X{int(z)}")
        # Prefer single-letter for C/N/O/H
        lines.append(f"{rs.comp_id} {name} {sym}")

    if rs.bonds:
        lines += [
            "loop_",
            "_chem_comp_bond.comp_id",
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_dist",
            "_chem_comp_bond.value_dist_esd",
        ]
        for b in rs.bonds:
            lines.append(
                f"{rs.comp_id} {b.atom_id_1} {b.atom_id_2} "
                f"{b.value:.5f} {b.esd:.5f}"
            )

    if rs.angles:
        has_bounds = any(
            a.value_min is not None or a.value_max is not None for a in rs.angles
        )
        lines += [
            "loop_",
            "_chem_comp_angle.comp_id",
            "_chem_comp_angle.atom_id_1",
            "_chem_comp_angle.atom_id_2",
            "_chem_comp_angle.atom_id_3",
            "_chem_comp_angle.value_angle",
            "_chem_comp_angle.value_angle_esd",
        ]
        if has_bounds:
            lines += [
                "_chem_comp_angle.value_angle_min",
                "_chem_comp_angle.value_angle_max",
            ]
        for a in rs.angles:
            row = (
                f"{rs.comp_id} {a.atom_id_1} {a.atom_id_2} {a.atom_id_3} "
                f"{a.value:.3f} {a.esd:.3f}"
            )
            if has_bounds:
                vmin = f"{a.value_min:.3f}" if a.value_min is not None else "."
                vmax = f"{a.value_max:.3f}" if a.value_max is not None else "."
                row += f" {vmin} {vmax}"
            lines.append(row)

    if rs.planes:
        lines += [
            "loop_",
            "_chem_comp_plane_atom.comp_id",
            "_chem_comp_plane_atom.plane_id",
            "_chem_comp_plane_atom.atom_id",
            "_chem_comp_plane_atom.dist_esd",
        ]
        for p in rs.planes:
            for aid in p.atom_ids:
                lines.append(
                    f"{rs.comp_id} {p.plane_id} {aid} {p.esd:.5f}"
                )

    if rs.extra_distances:
        # Explicit selective 1–4 (ring/plane diagonals).
        lines += [
            "loop_",
            "_chem_comp_dist.comp_id",
            "_chem_comp_dist.atom_id_1",
            "_chem_comp_dist.atom_id_2",
            "_chem_comp_dist.value_dist",
            "_chem_comp_dist.value_dist_esd",
        ]
        for r in rs.extra_distances:
            a1 = rs.atom_ids[r.i]
            a2 = rs.atom_ids[r.j]
            d0 = 0.5 * (r.d_lo + r.d_hi)
            lines.append(
                f"{rs.comp_id} {a1} {a2} {d0:.5f} {r.sig:.5f}"
            )

    Path(path).write_text("\n".join(lines) + "\n")
