"""Synthetic capped oligopeptide (~100 heavy atoms) for naming tests.

ACE–(Leu/Ala mix)–NME built from idealised residue geometry.  Leucine CD1/CD2
pairs supply known automorphism generators for oracle tests.

Also ships a crystallographic ACE–LRP–NME reference extracted from PDB 4D5M
(triptorelin, 0.85 Å) for figures that need a chemically realistic pose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from slicedot.geometry import Geometry
from slicedot.perturb import rotation_matrix

__all__ = [
    "oligopeptide_topology",
    "lrp_pdb_topology",
    "OLIGO_N_RES",
    "OLIGO_SEQUENCE",
]

_LRP_PDB_NPZ = Path(__file__).resolve().parent / "data" / "4D5M_ACE_LRP_NME.npz"

# Target ~100 heavy atoms: ACE(3) + residues + NME(2).
# Ala=5, Leu=8 → with sequence LALALALALA L (11 residues):
#   6 Leu + 5 Ala = 48 + 25 = 73; +ACE+NME = 78 — a bit short.
# Use 9 Leu + 4 Ala = 72 + 20 = 92; +5 = 97.
OLIGO_SEQUENCE = ("L", "A", "L", "A", "L", "A", "L", "A", "L", "L", "L", "L", "A")
OLIGO_N_RES = len(OLIGO_SEQUENCE)  # 13 → 9L*8 + 4A*5 + 5 = 72+20+5 = 97


def _idealize(X, bonds, rotatable, chiral, planar) -> np.ndarray:
    g = Geometry(
        X, bonds, rotatable, chiral, planar,
        antibump=False,
        weights={
            "bond": 0.02, "angle": 0.04, "torsion14": 0.05,
            "chiral": 0.05, "planar": 0.01, "bump": 0.3,
        },
    )
    Xp, _, _ = g.project(X, tol=1e-6, max_iter=400)
    return Xp


def _build_residue_local(kind: str) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    """Local heavy-atom coords for one residue in a CA-centred frame.

    Backbone in xy; CB above plane for L chirality.  Peptide N is at the
    N-terminus of the residue; C/O at the C-terminus.
    """
    kind = kind.upper()
    N = np.array([-1.45, 0.20, 0.00])
    CA = np.array([0.00, 0.00, 0.00])
    C = np.array([1.52, 0.30, 0.00])
    O = np.array([2.20, -0.75, 0.00])
    CB = np.array([-0.40, -0.80, 1.35])
    atoms = {"N": N, "CA": CA, "C": C, "O": O, "CB": CB}
    bonds = [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")]
    if kind == "A":
        return atoms, bonds
    if kind == "L":
        CG = np.array([-1.10, -2.00, 1.80])
        CD1 = np.array([-2.50, -2.20, 2.40])
        CD2 = np.array([-0.40, -3.30, 2.20])
        atoms.update({"CG": CG, "CD1": CD1, "CD2": CD2})
        bonds.extend([("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")])
        return atoms, bonds
    if kind == "R":
        # Extended arginine side chain + planar guanidinium.
        CG = np.array([-1.10, -2.00, 1.80])
        CD = np.array([-1.80, -3.20, 2.40])
        NE = np.array([-2.50, -4.20, 3.00])
        CZ = np.array([-3.20, -5.30, 3.50])
        NH1 = np.array([-4.00, -5.20, 4.50])
        NH2 = np.array([-3.10, -6.40, 2.80])
        atoms.update({
            "CG": CG, "CD": CD, "NE": NE, "CZ": CZ, "NH1": NH1, "NH2": NH2,
        })
        bonds.extend([
            ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
            ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"),
        ])
        return atoms, bonds
    if kind == "P":
        # L-proline ring (envelope-ish); CD bonded back to N.
        CB = np.array([-0.50, -1.20, 1.10])
        CG = np.array([-1.70, -1.80, 0.40])
        CD = np.array([-2.20, -0.70, -0.40])
        atoms = {"N": N, "CA": CA, "C": C, "O": O, "CB": CB, "CG": CG, "CD": CD}
        bonds = [
            ("N", "CA"), ("CA", "C"), ("C", "O"),
            ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N"),
        ]
        return atoms, bonds
    raise ValueError(f"unknown residue kind {kind!r}; expected A/L/R/P")


def _build_oligopeptide(
    sequence: Sequence[str] = OLIGO_SEQUENCE,
) -> dict:
    """Assemble ACE–sequence–NME with extended backbone along +x."""
    sequence = tuple(s.upper() for s in sequence)
    names: list[str] = []
    Z_list: list[float] = []
    coords: list[np.ndarray] = []
    bonds: list[tuple[int, int]] = []
    rotatable: list[tuple[int, int]] = []
    chiral: list[tuple[int, int, int, int]] = []
    planar: list[list[int]] = []
    idx: dict[str, int] = {}
    # Known automorphism generators: identity + each symmetric heavy-atom swap.
    cd_swaps: list[tuple[int, int]] = []
    nh_swaps: list[tuple[int, int]] = []
    _NITROGEN = {"N", "NE", "NH1", "NH2"}

    def add(name: str, xyz: np.ndarray, z: float) -> int:
        i = len(names)
        names.append(name)
        Z_list.append(float(z))
        coords.append(np.asarray(xyz, dtype=np.float64).copy())
        idx[name] = i
        return i

    # --- ACE ---
    # Place ACE so its C bonds to residue-0 N.  Residue 0 CA will sit near origin
    # of its local frame; we place residues with CA spacing ~3.8 Å along +x.
    ca_spacing = 3.80
    # Precompute residue local geometries and placement transforms.
    res_locals = [_build_residue_local(k) for k in sequence]

    # First place all residue atoms in a straight extended chain.
    res_atom_maps: list[dict[str, int]] = []
    for r, (kind, (latoms, lbonds)) in enumerate(zip(sequence, res_locals)):
        # Rigid place: translate local frame so CA -> (r * ca_spacing, 0, 0)
        # and rotate slightly about x so consecutive peptides don't stack.
        origin = np.array([r * ca_spacing, 0.0, 0.0])
        # Mild twist keeps side chains from clashing before idealisation.
        R = rotation_matrix(np.array([1.0, 0.0, 0.0]), r * 0.35)
        amap: dict[str, int] = {}
        for aname, xyz in latoms.items():
            full = f"R{r}_{aname}"
            z = 7.0 if aname in _NITROGEN else (8.0 if aname == "O" else 6.0)
            i = add(full, origin + R @ xyz, z)
            amap[aname] = i
        res_atom_maps.append(amap)
        # Intra-residue bonds
        for a, b in lbonds:
            bonds.append((amap[a], amap[b]))
        # Rotatable χ (Pro ring is rigid — no rotatable bonds).
        if kind == "L":
            rotatable.append((amap["CA"], amap["CB"]))
            rotatable.append((amap["CB"], amap["CG"]))
            cd_swaps.append((amap["CD1"], amap["CD2"]))
        elif kind == "A":
            rotatable.append((amap["CA"], amap["CB"]))
        elif kind == "R":
            rotatable.append((amap["CA"], amap["CB"]))
            rotatable.append((amap["CB"], amap["CG"]))
            rotatable.append((amap["CG"], amap["CD"]))
            rotatable.append((amap["CD"], amap["NE"]))
            nh_swaps.append((amap["NH1"], amap["NH2"]))
            planar.append([amap["NE"], amap["CZ"], amap["NH1"], amap["NH2"]])
        elif kind == "P":
            pass
        # Chiral Cα: (CA, N, C, CB)
        chiral.append((amap["CA"], amap["N"], amap["C"], amap["CB"]))

    # Peptide bonds between residues: C(r)–N(r+1)
    for r in range(len(sequence) - 1):
        bonds.append((res_atom_maps[r]["C"], res_atom_maps[r + 1]["N"]))
        # Peptide plane: CA(r), C(r), O(r), N(r+1)
        planar.append([
            res_atom_maps[r]["CA"],
            res_atom_maps[r]["C"],
            res_atom_maps[r]["O"],
            res_atom_maps[r + 1]["N"],
        ])

    # --- ACE attached to first N ---
    # ACE_C near first N, along -x from R0_N.
    n0 = res_atom_maps[0]["N"]
    ca0 = res_atom_maps[0]["CA"]
    u = coords[n0] - coords[ca0]
    u = u / (np.linalg.norm(u) + 1e-30)
    ace_c = add("ACE_C", coords[n0] + 1.33 * u, 6.0)
    # Perpendicular for O / CH3 in a peptide-ish plane.
    tmp = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(tmp, u)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, tmp)
    v = v / (np.linalg.norm(v) + 1e-30)
    w = np.cross(u, v)
    ace_o = add("ACE_O", coords[ace_c] + 1.24 * v, 8.0)
    ace_ch3 = add("ACE_CH3", coords[ace_c] + 1.50 * (-u * 0.3 + w), 6.0)
    bonds.extend([(ace_ch3, ace_c), (ace_c, ace_o), (ace_c, n0)])
    planar.append([ace_ch3, ace_c, ace_o, n0])

    # --- NME attached to last C ---
    last = res_atom_maps[-1]
    c_last = last["C"]
    ca_last = last["CA"]
    u2 = coords[c_last] - coords[ca_last]
    u2 = u2 / (np.linalg.norm(u2) + 1e-30)
    nme_n = add("NME_N", coords[c_last] + 1.33 * u2, 7.0)
    tmp = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(tmp, u2)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    v2 = np.cross(u2, tmp)
    v2 = v2 / (np.linalg.norm(v2) + 1e-30)
    nme_ch3 = add("NME_CH3", coords[nme_n] + 1.45 * (u2 * 0.5 + v2), 6.0)
    bonds.extend([(c_last, nme_n), (nme_n, nme_ch3)])
    planar.append([last["CA"], c_last, last["O"], nme_n])

    X = np.stack(coords, axis=0)
    X = X - X.mean(0)
    n = X.shape[0]
    Z = np.asarray(Z_list, dtype=np.float64)

    # Idealise
    X = _idealize(X, bonds, rotatable, chiral, planar)

    # Automorphism generators: identity + each CD1↔CD2 / NH1↔NH2 transposition.
    identity = np.arange(n, dtype=np.int64)
    aut_gens: list[np.ndarray] = [identity.copy()]
    for a, b in list(cd_swaps) + list(nh_swaps):
        p = identity.copy()
        p[a], p[b] = b, a
        aut_gens.append(p)

    return {
        "X_ref": X.astype(np.float64),
        "elements": Z.astype(np.int64),
        "Z": Z.copy(),
        "W": (Z / Z.sum()).copy(),
        "names": tuple(names),
        "bonds": [(int(i), int(j)) for i, j in bonds],
        "rotatable_bonds": [(int(i), int(j)) for i, j in rotatable],
        "chiral_centres": [tuple(int(x) for x in t) for t in chiral],
        "planar_groups": [[int(x) for x in g] for g in planar],
        "idx": dict(idx),
        "sequence": sequence,
        "cd_swaps": [(int(a), int(b)) for a, b in cd_swaps],
        "nh_swaps": [(int(a), int(b)) for a, b in nh_swaps],
        "automorphism_generators": aut_gens,
        "n": int(n),
    }


_OLIGO = None


def oligopeptide_topology(sequence: Sequence[str] | None = None) -> dict:
    """Return (and cache default) oligopeptide topology dict."""
    global _OLIGO
    if sequence is None:
        if _OLIGO is None:
            _OLIGO = _build_oligopeptide(OLIGO_SEQUENCE)
        # Fresh copies of mutable arrays
        d = dict(_OLIGO)
        d["X_ref"] = d["X_ref"].copy()
        d["elements"] = d["elements"].copy()
        d["Z"] = d["Z"].copy()
        d["W"] = d["W"].copy()
        d["automorphism_generators"] = [p.copy() for p in d["automorphism_generators"]]
        return d
    return _build_oligopeptide(sequence)


def _topology_from_named_coords(
    names: Sequence[str],
    X: np.ndarray,
    sequence: Sequence[str],
) -> dict:
    """Build bond / restraint topology for a named ACE–seq–NME coordinate set.

    Coordinates are taken as-is (no ``Geometry.project`` idealisation).
    """
    sequence = tuple(s.upper() for s in sequence)
    names = tuple(names)
    X = np.asarray(X, dtype=np.float64)
    if X.shape != (len(names), 3):
        raise ValueError(f"X shape {X.shape} != ({len(names)}, 3)")
    idx = {n: i for i, n in enumerate(names)}

    def z_of(name: str) -> float:
        if name.startswith(("ACE_", "NME_")):
            leaf = name.split("_", 1)[1]
        elif name.startswith("R") and "_" in name:
            leaf = name.split("_", 1)[1]
        else:
            leaf = name
        if leaf in {"N", "NE", "NH1", "NH2"}:
            return 7.0
        if leaf == "O":
            return 8.0
        return 6.0

    Z = np.array([z_of(n) for n in names], dtype=np.float64)
    bonds: list[tuple[int, int]] = []
    rotatable: list[tuple[int, int]] = []
    chiral: list[tuple[int, int, int, int]] = []
    planar: list[list[int]] = []
    cd_swaps: list[tuple[int, int]] = []
    nh_swaps: list[tuple[int, int]] = []

    # Caps
    bonds.extend([
        (idx["ACE_CH3"], idx["ACE_C"]),
        (idx["ACE_C"], idx["ACE_O"]),
        (idx["ACE_C"], idx["R0_N"]),
        (idx[f"R{len(sequence) - 1}_C"], idx["NME_N"]),
        (idx["NME_N"], idx["NME_CH3"]),
    ])
    planar.append([idx["ACE_CH3"], idx["ACE_C"], idx["ACE_O"], idx["R0_N"]])

    res_bonds = {
        "L": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
              ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
        "A": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
        "R": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
              ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
              ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
        "P": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
              ("CB", "CG"), ("CG", "CD"), ("CD", "N")],
    }
    for r, kind in enumerate(sequence):
        pref = f"R{r}_"
        for a, b in res_bonds[kind]:
            bonds.append((idx[pref + a], idx[pref + b]))
        chiral.append((
            idx[pref + "CA"], idx[pref + "N"],
            idx[pref + "C"], idx[pref + "CB"],
        ))
        if kind == "L":
            rotatable.append((idx[pref + "CA"], idx[pref + "CB"]))
            rotatable.append((idx[pref + "CB"], idx[pref + "CG"]))
            cd_swaps.append((idx[pref + "CD1"], idx[pref + "CD2"]))
        elif kind == "A":
            rotatable.append((idx[pref + "CA"], idx[pref + "CB"]))
        elif kind == "R":
            rotatable.extend([
                (idx[pref + "CA"], idx[pref + "CB"]),
                (idx[pref + "CB"], idx[pref + "CG"]),
                (idx[pref + "CG"], idx[pref + "CD"]),
                (idx[pref + "CD"], idx[pref + "NE"]),
            ])
            nh_swaps.append((idx[pref + "NH1"], idx[pref + "NH2"]))
            planar.append([
                idx[pref + "NE"], idx[pref + "CZ"],
                idx[pref + "NH1"], idx[pref + "NH2"],
            ])
        if r + 1 < len(sequence):
            bonds.append((idx[pref + "C"], idx[f"R{r + 1}_N"]))
            planar.append([
                idx[pref + "CA"], idx[pref + "C"],
                idx[pref + "O"], idx[f"R{r + 1}_N"],
            ])
    last = len(sequence) - 1
    planar.append([
        idx[f"R{last}_CA"], idx[f"R{last}_C"],
        idx[f"R{last}_O"], idx["NME_N"],
    ])

    n = len(names)
    identity = np.arange(n, dtype=np.int64)
    aut_gens: list[np.ndarray] = [identity.copy()]
    for a, b in list(cd_swaps) + list(nh_swaps):
        p = identity.copy()
        p[a], p[b] = b, a
        aut_gens.append(p)

    Xc = X - X.mean(0)
    return {
        "X_ref": Xc.astype(np.float64),
        "elements": Z.astype(np.int64),
        "Z": Z.copy(),
        "W": (Z / Z.sum()).copy(),
        "names": names,
        "bonds": [(int(i), int(j)) for i, j in bonds],
        "rotatable_bonds": [(int(i), int(j)) for i, j in rotatable],
        "chiral_centres": [tuple(int(x) for x in t) for t in chiral],
        "planar_groups": [[int(x) for x in g] for g in planar],
        "idx": dict(idx),
        "sequence": sequence,
        "cd_swaps": [(int(a), int(b)) for a, b in cd_swaps],
        "nh_swaps": [(int(a), int(b)) for a, b in nh_swaps],
        "automorphism_generators": aut_gens,
        "n": int(n),
    }


def lrp_pdb_topology(path: str | Path | None = None) -> dict:
    """ACE–Leu–Arg–Pro–NME from PDB 4D5M (0.85 Å triptorelin).

    Heavy atoms for LEU7–ARG8–PRO9 (chain A); NME from GLY10 N/CA; ACE
    placed at ideal peptide geometry on the leucine N.  No geometry
    idealisation — crystallographic side-chain / proline ring geometry is kept.
    """
    path = Path(path) if path is not None else _LRP_PDB_NPZ
    if not path.is_file():
        raise FileNotFoundError(
            f"LRP PDB reference not found at {path}. "
            "Expected packaged 4D5M_ACE_LRP_NME.npz."
        )
    data = np.load(path, allow_pickle=False)
    names = [str(n) for n in data["names"]]
    X = np.asarray(data["X"], dtype=np.float64)
    topo = _topology_from_named_coords(names, X, ("L", "R", "P"))
    topo["source"] = {
        "pdb_id": str(data["pdb_id"]) if "pdb_id" in data.files else "4D5M",
        "chain": str(data["chain"]) if "chain" in data.files else "A",
        "note": str(data["note"]) if "note" in data.files else "",
        "path": str(path),
    }
    return topo
