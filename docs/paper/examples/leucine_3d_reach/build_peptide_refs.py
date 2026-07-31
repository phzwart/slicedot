#!/usr/bin/env python3
"""Extract uncapped oligopeptide references from high-resolution PDB entries.

Downloads mmCIF, pulls contiguous heavy-atom stretches (no ACE/NME caps),
and writes ``data/peptides/*.npz`` plus ``data/peptides/index.json``.

Usage
-----
  PYTHONPATH=../../../src python build_peptide_refs.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "peptides"
CIF_CACHE = ROOT / "data" / "cif_cache"

# One-letter → expected heavy atoms + chemistry.
AA: dict[str, dict] = {
    "A": {
        "resn": "ALA",
        "atoms": ["N", "CA", "C", "O", "CB"],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
        "rotatable": [("CA", "CB")],
        "chiral": True,
    },
    "C": {
        "resn": "CYS",
        "atoms": ["N", "CA", "C", "O", "CB", "SG"],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "SG")],
        "rotatable": [("CA", "CB")],
        "chiral": True,
    },
    "D": {
        "resn": "ASP",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG")],
        "chiral": True,
        "planar": [("CB", "CG", "OD1", "OD2")],
        "od_swap": ("OD1", "OD2"),
    },
    "E": {
        "resn": "GLU",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
        "chiral": True,
        "planar": [("CG", "CD", "OE1", "OE2")],
        "oe_swap": ("OE1", "OE2"),
    },
    "F": {
        "resn": "PHE",
        "atoms": [
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ",
        ],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"),
            ("CE1", "CZ"), ("CE2", "CZ"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG")],
        "chiral": True,
        "planar": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
        "ring_swap": (("CD1", "CD2"), ("CE1", "CE2")),
    },
    "G": {
        "resn": "GLY",
        "atoms": ["N", "CA", "C", "O"],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O")],
        "rotatable": [],
        "chiral": False,
    },
    "H": {
        "resn": "HIS",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2"),
            ("CE1", "NE2"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG")],
        "chiral": True,
        "planar": [("CG", "ND1", "CD2", "CE1", "NE2")],
    },
    "I": {
        "resn": "ILE",
        "atoms": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG1")],
        "chiral": True,
    },
    "K": {
        "resn": "LYS",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ"),
        ],
        "rotatable": [
            ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"),
        ],
        "chiral": True,
    },
    "L": {
        "resn": "LEU",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG")],
        "chiral": True,
        "cd_swap": ("CD1", "CD2"),
    },
    "M": {
        "resn": "MET",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "SD"), ("SD", "CE"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG"), ("CG", "SD")],
        "chiral": True,
    },
    "N": {
        "resn": "ASN",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG")],
        "chiral": True,
        "planar": [("CB", "CG", "OD1", "ND2")],
    },
    "P": {
        "resn": "PRO",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "CD"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "CD"), ("CD", "N"),
        ],
        "rotatable": [],
        "chiral": True,
    },
    "Q": {
        "resn": "GLN",
        "atoms": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
        "chiral": True,
        "planar": [("CG", "CD", "OE1", "NE2")],
    },
    "R": {
        "resn": "ARG",
        "atoms": [
            "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2",
        ],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
            ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"),
        ],
        "rotatable": [
            ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
        ],
        "chiral": True,
        "planar": [("NE", "CZ", "NH1", "NH2")],
        "nh_swap": ("NH1", "NH2"),
    },
    "S": {
        "resn": "SER",
        "atoms": ["N", "CA", "C", "O", "CB", "OG"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG"),
        ],
        "rotatable": [("CA", "CB")],
        "chiral": True,
    },
    "T": {
        "resn": "THR",
        "atoms": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "OG1"), ("CB", "CG2"),
        ],
        "rotatable": [("CA", "CB")],
        "chiral": True,
    },
    "V": {
        "resn": "VAL",
        "atoms": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
            ("CB", "CG1"), ("CB", "CG2"),
        ],
        "rotatable": [("CA", "CB")],
        "chiral": True,
        "cg_swap": ("CG1", "CG2"),
    },
    "W": {
        "resn": "TRP",
        "atoms": [
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2",
            "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
        ],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("CD2", "CE2"),
            ("CD2", "CE3"), ("NE1", "CE2"), ("CE2", "CZ2"), ("CE3", "CZ3"),
            ("CZ2", "CH2"), ("CZ3", "CH2"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG")],
        "chiral": True,
        "planar": [
            ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        ],
    },
    "Y": {
        "resn": "TYR",
        "atoms": [
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2",
            "CE1", "CE2", "CZ", "OH",
        ],
        "bonds": [
            ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
            ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"),
            ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH"),
        ],
        "rotatable": [("CA", "CB"), ("CB", "CG")],
        "chiral": True,
        "planar": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
        "ring_swap": (("CD1", "CD2"), ("CE1", "CE2")),
    },
}

RESN_TO_AA = {v["resn"]: k for k, v in AA.items()}

# (pdb_id, preferred_chains, start_seq, end_seq, expected_sequence)
# Preferred (pdb, chains, start, end, sequence).  If start/end numbering
# mismatches the deposit, the extractor falls back to a sequence scan.
TARGETS = [
    ("1FN8", "AB", 1, 3, "GAR"),
    ("1FY5", "AB", 1, 3, "GAK"),
    ("5TDA", "AB", 1, 4, "RLWS"),
    ("7ETN", "A", 1, 4, "PFLI"),
    ("8DTS", "A", 1, 6, "AFSSFN"),
    ("4TUT", "A", 126, 131, "GGYMLG"),
    ("6KJ3", "A", 37, 42, "SYSGYS"),
    ("4D5M", "A", 7, 9, "LRP"),
    ("2OL9", "A", 1, 6, "SNQNNF"),
]


def tokenize_cif(s: str) -> list[str]:
    toks: list[str] = []
    j = 0
    while j < len(s):
        if s[j].isspace():
            j += 1
            continue
        if s[j] in "'\"":
            q = s[j]
            j += 1
            start = j
            while j < len(s) and s[j] != q:
                j += 1
            toks.append(s[start:j])
            j += 1
        else:
            start = j
            while j < len(s) and not s[j].isspace():
                j += 1
            toks.append(s[start:j])
    return toks


def parse_atom_site(cif_text: str) -> list[dict]:
    lines = cif_text.splitlines()
    atoms: list[dict] = []
    i = 0
    while i < len(lines):
        if (
            lines[i].strip() == "loop_"
            and i + 1 < len(lines)
            and lines[i + 1].startswith("_atom_site.")
        ):
            tags: list[str] = []
            i += 1
            while i < len(lines) and lines[i].startswith("_atom_site."):
                tags.append(lines[i].strip().split(".")[1])
                i += 1
            while i < len(lines):
                line = lines[i].strip()
                if (
                    not line
                    or line.startswith("#")
                    or line.startswith("loop_")
                    or line.startswith("_")
                ):
                    break
                toks = tokenize_cif(lines[i])
                if len(toks) >= len(tags):
                    atoms.append(dict(zip(tags, toks)))
                i += 1
            break
        i += 1
    return atoms


def fetch_cif(pdb_id: str) -> str:
    CIF_CACHE.mkdir(parents=True, exist_ok=True)
    path = CIF_CACHE / f"{pdb_id.upper()}.cif"
    if not path.is_file():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        print(f"  downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_text()


def pick_xyz(
    atoms: list[dict],
    chain: str,
    seq: int,
    resn: str,
    atom: str,
) -> np.ndarray | None:
    cands = [
        a
        for a in atoms
        if a.get("group_PDB") == "ATOM"
        and a.get("auth_asym_id") == chain
        and a.get("auth_comp_id") == resn
        and a.get("label_atom_id") == atom
        and a.get("type_symbol") != "H"
        and int(a["auth_seq_id"]) == seq
        and a.get("label_alt_id", ".") in (".", "A", "")
    ]
    if not cands:
        return None
    cands.sort(
        key=lambda a: (
            0 if a.get("label_alt_id", ".") in (".", "") else 1,
            -float(a.get("occupancy", "1") or 1),
        )
    )
    a = cands[0]
    return np.array(
        [float(a["Cartn_x"]), float(a["Cartn_y"]), float(a["Cartn_z"])],
        dtype=np.float64,
    )


def chain_residues(atoms: list[dict], chain: str) -> list[tuple[int, str]]:
    """Ordered unique (seq, resn) for standard AA on a chain."""
    seen: list[tuple[int, str]] = []
    for a in atoms:
        if a.get("group_PDB") != "ATOM":
            continue
        if a.get("auth_asym_id") != chain:
            continue
        if a.get("label_alt_id", ".") not in (".", "A", ""):
            continue
        resn = a.get("auth_comp_id", "")
        if resn not in RESN_TO_AA:
            continue
        key = (int(a["auth_seq_id"]), resn)
        if not seen or seen[-1] != key:
            seen.append(key)
    return seen


def z_of_leaf(leaf: str) -> float:
    if leaf in {"N", "NE", "NE1", "NE2", "ND1", "ND2", "NH1", "NH2", "NZ"}:
        return 7.0
    if leaf in {"O", "OG", "OG1", "OD1", "OD2", "OE1", "OE2", "OH"}:
        return 8.0
    if leaf in {"SG", "SD"}:
        return 16.0
    return 6.0


def extract_peptide(
    atoms: list[dict],
    chain: str,
    start: int,
    end: int,
    expected: str,
) -> dict | None:
    residues = [(s, r) for s, r in chain_residues(atoms, chain) if start <= s <= end]
    if len(residues) != end - start + 1:
        return None
    seq_letters = "".join(RESN_TO_AA[r] for _, r in residues)
    if seq_letters != expected:
        return None

    names: list[str] = []
    coords: list[np.ndarray] = []
    elements: list[float] = []
    idx: dict[str, int] = {}

    def add(name: str, xyz: np.ndarray, z: float) -> int:
        i = len(names)
        names.append(name)
        coords.append(xyz.copy())
        elements.append(z)
        idx[name] = i
        return i

    for r_i, (seq_id, resn) in enumerate(residues):
        aa = RESN_TO_AA[resn]
        for atom in AA[aa]["atoms"]:
            xyz = pick_xyz(atoms, chain, seq_id, resn, atom)
            if xyz is None:
                return None
            add(f"R{r_i}_{atom}", xyz, z_of_leaf(atom))

    X = np.stack(coords, axis=0)
    X = X - X.mean(0)
    Z = np.asarray(elements, dtype=np.float64)

    # Topology arrays for downstream Namer / Geometry.
    bonds: list[tuple[int, int]] = []
    rotatable: list[tuple[int, int]] = []
    chiral: list[tuple[int, int, int, int]] = []
    planar: list[list[int]] = []
    swaps: list[tuple[int, int]] = []

    for r_i, (_, resn) in enumerate(residues):
        aa = RESN_TO_AA[resn]
        chem = AA[aa]
        pref = f"R{r_i}_"
        for a, b in chem["bonds"]:
            bonds.append((idx[pref + a], idx[pref + b]))
        for a, b in chem["rotatable"]:
            rotatable.append((idx[pref + a], idx[pref + b]))
        if chem["chiral"]:
            chiral.append((
                idx[pref + "CA"], idx[pref + "N"],
                idx[pref + "C"], idx[pref + "CB"],
            ))
        for group in chem.get("planar", ()):
            planar.append([idx[pref + a] for a in group])
        for key in ("cd_swap", "nh_swap", "od_swap", "oe_swap", "cg_swap"):
            if key in chem:
                a, b = chem[key]
                swaps.append((idx[pref + a], idx[pref + b]))
        for pair in chem.get("ring_swap", ()):
            a, b = pair
            swaps.append((idx[pref + a], idx[pref + b]))
        if r_i + 1 < len(residues):
            bonds.append((idx[pref + "C"], idx[f"R{r_i + 1}_N"]))
            planar.append([
                idx[pref + "CA"], idx[pref + "C"],
                idx[pref + "O"], idx[f"R{r_i + 1}_N"],
            ])

    # Bond-length sanity (Å)
    for a, b in bonds:
        d = float(np.linalg.norm(X[a] - X[b]))
        if d < 0.9 or d > 2.2:
            return None

    return {
        "names": np.array(names),
        "X": X.astype(np.float64),
        "Z": Z.astype(np.float64),
        "sequence": np.array(seq_letters),
        "bonds": np.asarray(bonds, dtype=np.int64),
        "rotatable_bonds": np.asarray(rotatable, dtype=np.int64),
        "chiral_centres": np.asarray(chiral, dtype=np.int64),
        # ragged planar → object array of variable-length int arrays
        "planar_groups": np.array(
            [np.asarray(g, dtype=np.int64) for g in planar], dtype=object
        ),
        "swaps": np.asarray(swaps, dtype=np.int64).reshape(-1, 2)
        if swaps
        else np.zeros((0, 2), dtype=np.int64),
        "chain": np.array(chain),
        "start_seq": np.array(start),
        "end_seq": np.array(end),
    }


def try_extract(pdb_id: str, chains: str, start: int, end: int, expected: str):
    atoms = parse_atom_site(fetch_cif(pdb_id))
    for ch in chains:
        got = extract_peptide(atoms, ch, start, end, expected)
        if got is not None:
            got["pdb_id"] = np.array(pdb_id.upper())
            return got, ch
    # Fallback: scan all chains for the expected sequence stretch.
    chain_ids = sorted({
        a.get("auth_asym_id")
        for a in atoms
        if a.get("group_PDB") == "ATOM" and a.get("auth_asym_id")
    })
    for ch in chain_ids:
        res = chain_residues(atoms, ch)
        letters = "".join(RESN_TO_AA[r] for _, r in res)
        pos = letters.find(expected)
        if pos < 0:
            continue
        s0 = res[pos][0]
        s1 = res[pos + len(expected) - 1][0]
        got = extract_peptide(atoms, ch, s0, s1, expected)
        if got is not None:
            got["pdb_id"] = np.array(pdb_id.upper())
            return got, ch
    return None, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for pdb_id, chains, start, end, expected in TARGETS:
        print(f"{pdb_id} {expected} …", flush=True)
        got, chain = try_extract(pdb_id, chains, start, end, expected)
        if got is None:
            print(f"  FAILED", flush=True)
            continue
        stem = f"{pdb_id.upper()}_{expected}"
        path = OUT / f"{stem}.npz"
        # np.savez doesn't like object arrays of planar groups well with allow_pickle=False;
        # store planar as JSON sidecar fields in index, and bonds etc in npz.
        planar = got.pop("planar_groups")
        np.savez_compressed(path, **got)
        n_atoms = int(got["X"].shape[0])
        entry = {
            "id": stem,
            "file": path.name,
            "pdb_id": pdb_id.upper(),
            "chain": chain,
            "sequence": expected,
            "n_residues": len(expected),
            "n_atoms": n_atoms,
            "start_seq": int(got["start_seq"]),
            "end_seq": int(got["end_seq"]),
            "n_bonds": int(got["bonds"].shape[0]),
            "planar_groups": [g.tolist() for g in planar],
        }
        # also write planar next to npz for loader
        (OUT / f"{stem}.planar.json").write_text(json.dumps(entry["planar_groups"]) + "\n")
        # Naming CIF: 1–2 / 1–3 / planes / plane-scoped 1–4 only.
        from slicedot.restraints import restraint_set_from_geometry, write_restraint_cif

        names = [str(n) for n in got["names"]]
        rs = restraint_set_from_geometry(
            got["X"],
            got["Z"],
            [(int(a), int(b)) for a, b in got["bonds"]],
            rotatable_bonds=[(int(a), int(b)) for a, b in got["rotatable_bonds"]],
            planar_groups=entry["planar_groups"],
            atom_ids=names,
            comp_id=stem.replace("-", "_")[:20],
            torsion14="planar",
        )
        cif_path = OUT / f"{stem}.cif"
        write_restraint_cif(cif_path, rs)
        entry["restraint_cif"] = cif_path.name
        entry["n_angles"] = len(rs.angles)
        entry["n_planes"] = len(rs.planes)
        entry["n_14"] = len(rs.extra_distances)
        index.append(entry)
        print(
            f"  OK chain={chain} atoms={n_atoms} bonds={entry['n_bonds']} "
            f"angles={entry['n_angles']} planes={entry['n_planes']} "
            f"1-4={entry['n_14']} → {path.name}",
            flush=True,
        )

    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {len(index)} peptides → {OUT / 'index.json'}")


if __name__ == "__main__":
    main()
