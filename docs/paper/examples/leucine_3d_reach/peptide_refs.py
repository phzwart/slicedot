"""Load crystallographic peptide references from ``data/peptides/``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "peptides"


def list_peptide_refs() -> list[dict]:
    index = DATA / "index.json"
    if not index.is_file():
        raise FileNotFoundError(
            f"No peptide index at {index}. Run build_peptide_refs.py first."
        )
    return json.loads(index.read_text())


def load_peptide_ref(ref_id: str | None = None, *, sequence: str | None = None) -> dict:
    """Return a topology dict compatible with the leucine / LRP ensemble scripts.

    Identify by ``ref_id`` (e.g. ``\"4D5M_LRP\"``) or by exact ``sequence``
    (e.g. ``\"LRP\"``). If neither is given, returns the first entry.
    """
    index = list_peptide_refs()
    entry = None
    if ref_id is not None:
        for e in index:
            if e["id"] == ref_id:
                entry = e
                break
        if entry is None:
            raise KeyError(f"unknown peptide ref {ref_id!r}; have {[e['id'] for e in index]}")
    elif sequence is not None:
        seq = sequence.upper()
        for e in index:
            if e["sequence"] == seq:
                entry = e
                break
        if entry is None:
            raise KeyError(f"no peptide with sequence {seq!r}")
    else:
        entry = index[0]

    data = np.load(DATA / entry["file"], allow_pickle=False)
    names = tuple(str(n) for n in data["names"])
    X = np.asarray(data["X"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    bonds = [(int(a), int(b)) for a, b in np.asarray(data["bonds"])]
    rotatable = [(int(a), int(b)) for a, b in np.asarray(data["rotatable_bonds"])]
    chiral_raw = np.asarray(data["chiral_centres"])
    chiral = [tuple(int(x) for x in row) for row in chiral_raw] if chiral_raw.size else []
    planar_path = DATA / f"{entry['id']}.planar.json"
    if planar_path.is_file():
        planar = json.loads(planar_path.read_text())
    else:
        planar = entry.get("planar_groups", [])
    swaps = [(int(a), int(b)) for a, b in np.asarray(data["swaps"]).reshape(-1, 2)]
    n = X.shape[0]
    identity = np.arange(n, dtype=np.int64)
    aut_gens = [identity.copy()]
    for a, b in swaps:
        p = identity.copy()
        p[a], p[b] = b, a
        aut_gens.append(p)

    seq = tuple(str(entry["sequence"]))
    cif_name = entry.get("restraint_cif", f"{entry['id']}.cif")
    cif_path = DATA / cif_name
    out = {
        "X_ref": X.copy(),
        "elements": Z.astype(np.int64),
        "Z": Z.copy(),
        "W": (Z / Z.sum()).copy(),
        "names": names,
        "bonds": bonds,
        "rotatable_bonds": rotatable,
        "chiral_centres": chiral,
        "planar_groups": planar,
        "idx": {n: i for i, n in enumerate(names)},
        "sequence": seq,
        "automorphism_generators": aut_gens,
        "n": int(n),
        "source": {
            "id": entry["id"],
            "pdb_id": entry["pdb_id"],
            "chain": entry["chain"],
            "start_seq": entry["start_seq"],
            "end_seq": entry["end_seq"],
            "file": entry["file"],
            "restraint_cif": cif_name if cif_path.is_file() else None,
        },
        "label": f"{'–'.join(seq)} ({entry['pdb_id']})",
    }
    if cif_path.is_file():
        from slicedot.restraints import load_restraint_cif

        out["restraint_set"] = load_restraint_cif(cif_path)
    return out
