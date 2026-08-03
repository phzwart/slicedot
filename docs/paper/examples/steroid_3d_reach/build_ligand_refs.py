#!/usr/bin/env python3
"""Download steroid references (CCD / PubChem) into per-ligand subdirectories.

For each entry in ``ligands.json`` writes::

  ligands/<slug>/
    meta.json
    source.cif | source.sdf
    topology.npz
    restraints.cif
    out/          (empty, for resolution sweeps)

Usage
-----
  uv run python build_ligand_refs.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from slicedot.restraints import (
    restraint_set_from_geometry,
    write_restraint_cif,
)

ROOT = Path(__file__).resolve().parent
LIGANDS_JSON = ROOT / "ligands.json"
LIGANDS_DIR = ROOT / "ligands"
CCD_CACHE = ROOT / "data" / "ccd_cache"
CCD_URL = "https://files.rcsb.org/ligands/download/{code}.cif"
PUBCHEM_SDF_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
    "?record_type=3d"
)

Z_MAP = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "CL": 17, "BR": 35, "I": 53, "SI": 14, "B": 5,
}


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    print(f"  download {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def _parse_ccd_cif(path: Path) -> dict:
    """Heavy-atom coords (ideal preferred) + bonds from a CCD component CIF."""
    text = path.read_text(errors="replace").splitlines()
    atoms: list[dict] = []
    bonds: list[tuple[str, str]] = []
    stereo: dict[str, str] = {}
    meta: dict[str, str] = {}
    mode = None
    atom_headers: list[str] = []
    bond_headers: list[str] = []

    def flush_headers(kind: str, headers: list[str]):
        nonlocal mode, atom_headers, bond_headers
        mode = kind
        if kind == "atom":
            atom_headers = headers[:]
        elif kind == "bond":
            bond_headers = headers[:]

    i = 0
    while i < len(text):
        line = text[i]
        if line.startswith("_chem_comp.id"):
            parts = line.split(None, 1)
            meta["id"] = parts[1].strip() if len(parts) > 1 else ""
        elif line.startswith("_chem_comp.name"):
            parts = line.split(None, 1)
            meta["name"] = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
        elif line.startswith("_chem_comp.formula_weight"):
            toks = line.split()
            meta["fw"] = toks[-1] if len(toks) > 1 else ""
        elif line.startswith("_chem_comp.pdbx_synonyms"):
            parts = line.split(None, 1)
            meta["synonyms"] = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
        elif line.startswith("_chem_comp.formula "):
            parts = line.split(None, 1)
            meta["formula"] = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
        elif line.startswith("loop_"):
            headers: list[str] = []
            i += 1
            while i < len(text) and text[i].startswith("_"):
                headers.append(text[i].strip())
                i += 1
            if any("chem_comp_atom." in h for h in headers) and not any(
                "plane" in h for h in headers
            ):
                flush_headers("atom", headers)
            elif any("chem_comp_bond." in h for h in headers):
                flush_headers("bond", headers)
            else:
                mode = None
            continue
        elif mode == "atom" and line.strip() and not line.startswith("#") and not line.startswith("_") and not line.startswith("loop_"):
            # CIF values may be quoted; split carefully.
            vals = _cif_split(line)
            if len(vals) < len(atom_headers):
                i += 1
                continue
            row = dict(zip(atom_headers, vals[: len(atom_headers)]))
            sym = row.get("_chem_comp_atom.type_symbol", "?").upper()
            aid = row["_chem_comp_atom.atom_id"]
            stereo[aid] = row.get("_chem_comp_atom.pdbx_stereo_config", "N")
            if sym == "H":
                i += 1
                continue
            # Prefer ideal coordinates.
            def _f(key_ideal, key_model):
                v = row.get(key_ideal, "?")
                if v not in ("?", ".", ""):
                    return float(v)
                return float(row[key_model])

            atoms.append({
                "id": aid,
                "elem": sym,
                "xyz": np.array([
                    _f("_chem_comp_atom.pdbx_model_Cartn_x_ideal",
                       "_chem_comp_atom.model_Cartn_x"),
                    _f("_chem_comp_atom.pdbx_model_Cartn_y_ideal",
                       "_chem_comp_atom.model_Cartn_y"),
                    _f("_chem_comp_atom.pdbx_model_Cartn_z_ideal",
                       "_chem_comp_atom.model_Cartn_z"),
                ], dtype=np.float64),
            })
        elif mode == "bond" and line.strip() and not line.startswith("#") and not line.startswith("_") and not line.startswith("loop_"):
            vals = _cif_split(line)
            if len(vals) < len(bond_headers):
                i += 1
                continue
            row = dict(zip(bond_headers, vals[: len(bond_headers)]))
            a1 = row["_chem_comp_bond.atom_id_1"]
            a2 = row["_chem_comp_bond.atom_id_2"]
            bonds.append((a1, a2))
        elif line.startswith("#") or line.strip() == "":
            pass
        elif line.startswith("data_") or line.startswith("loop_") or line.startswith("_"):
            mode = None
        i += 1

    return {"meta": meta, "atoms": atoms, "bonds": bonds, "stereo": stereo}


def _cif_split(line: str) -> list[str]:
    """Split a CIF data line respecting simple quotes."""
    out: list[str] = []
    buf: list[str] = []
    quote = None
    for ch in line.strip():
        if quote:
            if ch == quote:
                quote = None
            else:
                buf.append(ch)
        elif ch in ("'", '"'):
            quote = ch
        elif ch.isspace():
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _parse_sdf(path: Path) -> dict:
    """Parse a single-molecule SDF (V2000) into heavy atoms + bonds."""
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError(f"empty SDF {path}")
    counts = lines[3]
    n_atoms = int(counts[0:3])
    n_bonds = int(counts[3:6])
    atoms_raw = []
    for k in range(n_atoms):
        line = lines[4 + k]
        x, y, z = float(line[0:10]), float(line[10:20]), float(line[20:30])
        elem = line[31:34].strip().upper()
        atoms_raw.append((elem, np.array([x, y, z], dtype=np.float64)))
    bonds_raw = []
    for k in range(n_bonds):
        line = lines[4 + n_atoms + k]
        a = int(line[0:3]) - 1
        b = int(line[3:6]) - 1
        order = int(line[6:9])
        bonds_raw.append((a, b, order))

    # Keep heavy atoms; remap bonds.
    keep = [i for i, (e, _) in enumerate(atoms_raw) if e != "H"]
    old_to_new = {old: new for new, old in enumerate(keep)}
    atoms = []
    for i in keep:
        elem, xyz = atoms_raw[i]
        atoms.append({"id": f"{elem}{len(atoms)+1}", "elem": elem, "xyz": xyz})
    # Prefer element+serial naming that matches common steroid atom ids when possible.
    # Rebuild with sequential element counts.
    counts_e: dict[str, int] = {}
    for a in atoms:
        counts_e[a["elem"]] = counts_e.get(a["elem"], 0) + 1
        a["id"] = f"{a['elem']}{counts_e[a['elem']]}"

    bonds = []
    for a, b, _order in bonds_raw:
        if a in old_to_new and b in old_to_new:
            bonds.append((atoms[old_to_new[a]]["id"], atoms[old_to_new[b]]["id"]))
    return {
        "meta": {"id": path.stem, "name": path.stem, "source": "pubchem_sdf"},
        "atoms": atoms,
        "bonds": bonds,
        "stereo": {},
    }


def _ring_bonds(n: int, bonds: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Undirected edges that lie on any cycle (edge is non-bridge)."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in bonds:
        adj[a].add(b)
        adj[b].add(a)

    def connected(u: int, v: int, blocked: tuple[int, int]) -> bool:
        ban = {blocked, (blocked[1], blocked[0])}
        seen = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            if x == v:
                return True
            for y in adj[x]:
                if (x, y) in ban or y in seen:
                    continue
                seen.add(y)
                stack.append(y)
        return False

    on_cycle: set[tuple[int, int]] = set()
    for a, b in bonds:
        e = (a, b) if a < b else (b, a)
        if connected(a, b, (a, b)):
            on_cycle.add(e)
    return on_cycle


def topology_from_parsed(parsed: dict, *, label: str, slug: str) -> dict:
    atoms = parsed["atoms"]
    names = tuple(a["id"] for a in atoms)
    idx = {n: i for i, n in enumerate(names)}
    X = np.stack([a["xyz"] for a in atoms], axis=0)
    X = X - X.mean(0)
    Z = np.array([float(Z_MAP.get(a["elem"], 6)) for a in atoms], dtype=np.float64)
    bonds_idx: list[tuple[int, int]] = []
    for a, b in parsed["bonds"]:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            bonds_idx.append((i, j) if i < j else (j, i))
    bonds_idx = sorted(set(bonds_idx))
    cycle = _ring_bonds(len(names), bonds_idx)
    rotatable = [
        (a, b) for a, b in bonds_idx
        if (a, b) not in cycle and (b, a) not in cycle
    ]
    # Chirality from CCD stereo flags (R/S) when present: centre + 3 neighbours.
    chiral: list[tuple[int, int, int, int]] = []
    stereo = parsed.get("stereo") or {}
    for name, conf in stereo.items():
        if conf not in ("R", "S") or name not in idx:
            continue
        c = idx[name]
        neigh = sorted({a if b == c else b for a, b in bonds_idx if c in (a, b)})
        if len(neigh) >= 3:
            chiral.append((c, neigh[0], neigh[1], neigh[2]))

    # A-ring enone-ish: largest set of C/O atoms with ≥2 neighbours each, loosely planar.
    planar: list[list[int]] = []
    # Mark fused steroid A-ring carbons if present by name conventions (C1–C5, O1).
    a_ring = [n for n in ("C1", "C2", "C3", "C4", "C5", "C10", "O1") if n in idx]
    if len(a_ring) >= 5:
        planar.append([idx[n] for n in a_ring])

    n = len(names)
    identity = np.arange(n, dtype=np.int64)
    rs = restraint_set_from_geometry(
        X, Z.astype(np.int64), bonds_idx,
        rotatable_bonds=rotatable,
        planar_groups=planar,
        atom_ids=names,
        comp_id=slug.upper()[:8],
        torsion14="planar",
    )
    return {
        "X_ref": X.astype(np.float64),
        "elements": Z.astype(np.int64),
        "Z": Z.copy(),
        "W": (Z / Z.sum()).copy(),
        "names": names,
        "bonds": bonds_idx,
        "rotatable_bonds": rotatable,
        "chiral_centres": chiral,
        "planar_groups": planar,
        "idx": idx,
        "automorphism_generators": [identity.copy()],
        "n": int(n),
        "label": label,
        "restraint_set": rs,
        "meta": parsed.get("meta", {}),
    }


def write_ligand_dir(entry: dict) -> Path:
    slug = entry["slug"]
    out = LIGANDS_DIR / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(exist_ok=True)

    if "ccd" in entry:
        code = entry["ccd"]
        src = _fetch(CCD_URL.format(code=code), CCD_CACHE / f"{code}.cif")
        dest_src = out / "source.cif"
        dest_src.write_bytes(src.read_bytes())
        parsed = _parse_ccd_cif(dest_src)
    elif "pubchem_cid" in entry:
        cid = int(entry["pubchem_cid"])
        src = _fetch(PUBCHEM_SDF_URL.format(cid=cid), out / "source.sdf")
        parsed = _parse_sdf(src)
    else:
        raise ValueError(f"entry {slug} needs ccd or pubchem_cid")

    topo = topology_from_parsed(parsed, label=entry["label"], slug=slug)
    np.savez_compressed(
        out / "topology.npz",
        X=topo["X_ref"],
        Z=topo["Z"],
        names=np.array(topo["names"]),
        bonds=np.asarray(topo["bonds"], dtype=np.int64),
        rotatable_bonds=np.asarray(topo["rotatable_bonds"], dtype=np.int64).reshape(-1, 2),
        chiral_centres=(
            np.asarray(topo["chiral_centres"], dtype=np.int64)
            if topo["chiral_centres"] else np.zeros((0, 4), dtype=np.int64)
        ),
    )
    planar_path = out / "planar.json"
    planar_path.write_text(json.dumps(topo["planar_groups"]))
    write_restraint_cif(out / "restraints.cif", topo["restraint_set"])

    meta = {
        **entry,
        "n_atoms": topo["n"],
        "n_bonds": len(topo["bonds"]),
        "n_rotatable": len(topo["rotatable_bonds"]),
        "n_chiral": len(topo["chiral_centres"]),
        "source_meta": topo["meta"],
        "atom_names": list(topo["names"]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"  {slug:28s}  N={topo['n']:2d}  bonds={len(topo['bonds']):2d}  "
        f"rot={len(topo['rotatable_bonds']):2d}  "
        f"R≈{float(np.linalg.norm(topo['X_ref'], axis=1).max()):.1f} Å",
        flush=True,
    )
    return out


def main():
    entries = json.loads(LIGANDS_JSON.read_text())
    LIGANDS_DIR.mkdir(parents=True, exist_ok=True)
    CCD_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"building {len(entries)} steroid ligand refs …", flush=True)
    for e in entries:
        write_ligand_dir(e)
    print(f"done → {LIGANDS_DIR}", flush=True)


if __name__ == "__main__":
    main()
