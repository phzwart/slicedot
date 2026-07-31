#!/usr/bin/env python3
"""Export peptide structure + density for the web viewer.

Writes into ``viewer/data/``:
  * ``structure.json`` — atoms, chemical bonds, true coords, optional ensemble
  * ``density.cube`` — Gaussian cube of the rendered map
  * ``meta.json`` — resolution, isovalue hints, sequence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from slicedot import sigma_from_resolution
from slicedot.fixtures_peptide import lrp_pdb_topology, oligopeptide_topology

try:
    from peptide_refs import load_peptide_ref
except ImportError:
    load_peptide_ref = None

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "viewer" / "data"
Z_TO_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


def render_ortho(X, sp, NG, sigma, weights):
    sp = np.atleast_1d(sp) * np.ones(3)
    org = -0.5 * (np.asarray(NG, dtype=np.float64) - 1) * sp
    ax = [np.arange(n) * sp[i] for i, n in enumerate(NG)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1) + org
    T = np.zeros(NG, dtype=np.float64)
    for p, w in zip(X, weights):
        T += w * np.exp(-((G - p) ** 2).sum(-1) / (2.0 * sigma * sigma))
    return T / T.sum(), org, sp


_AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def _res_info(name: str, sequence: tuple[str, ...] = ()) -> tuple[str, str, int]:
    """Return (atom_name, resname, resid) for display."""
    if name.startswith("R") and "_" in name:
        head, atom = name.split("_", 1)
        try:
            r = int(head[1:])
        except ValueError:
            return name, "UNK", 1
        resn = _AA3.get(sequence[r], "UNK") if r < len(sequence) else "UNK"
        return atom, resn, r + 1
    return name, "UNK", 1


def write_cube(path: Path, T: np.ndarray, origin: np.ndarray, spacing, title="density"):
    """Gaussian cube: outer loop x, then y, then z (BOHR units for headers)."""
    bohr = 1.8897259886
    sp = np.atleast_1d(spacing) * np.ones(3)
    nx, ny, nz = T.shape
    org_b = np.asarray(origin, dtype=np.float64) * bohr
    sp_b = sp * bohr
    lines = [
        title,
        "slicedot rendered ortho density",
        f"  1 {org_b[0]:12.6f} {org_b[1]:12.6f} {org_b[2]:12.6f}",
        f"{nx:5d} {sp_b[0]:12.6f} {0.0:12.6f} {0.0:12.6f}",
        f"{ny:5d} {0.0:12.6f} {sp_b[1]:12.6f} {0.0:12.6f}",
        f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {sp_b[2]:12.6f}",
        f"  6 {0.0:12.6f} {org_b[0]:12.6f} {org_b[1]:12.6f} {org_b[2]:12.6f}",
    ]
    flat = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                flat.append(f"{T[ix, iy, iz]:13.5E}")
                if len(flat) == 6:
                    lines.append(" ".join(flat))
                    flat = []
    if flat:
        lines.append(" ".join(flat))
    path.write_text("\n".join(lines) + "\n")


def main(resolution: float = 3.0, sequence: str = "LRP", spacing: float = 0.45):
    seq = tuple(sequence.upper())
    seq_str = "".join(seq)
    topo = None
    if load_peptide_ref is not None:
        try:
            topo = load_peptide_ref(sequence=seq_str)
        except KeyError:
            topo = None
    if topo is None and seq == ("L", "R", "P"):
        topo = lrp_pdb_topology()
        topo["label"] = f"L–R–P ({topo.get('source', {}).get('pdb_id', 'PDB')})"
    if topo is None:
        topo = oligopeptide_topology(seq)
        topo["label"] = "–".join(seq)
    label = topo.get("label", "–".join(seq))
    X = np.asarray(topo["X_ref"], dtype=np.float64)
    w = topo["W"]
    sig = float(sigma_from_resolution(resolution))
    R = float(np.linalg.norm(X - X.mean(0), axis=1).max())
    half = R + 4.0 * sig + 3.0
    n = int(np.ceil(2.0 * half / spacing))
    if n % 2 == 0:
        n += 1
    n = min(n, 81)
    T, org, sp = render_ortho(X, spacing, (n, n, n), sig, w)

    DATA.mkdir(parents=True, exist_ok=True)

    atoms = []
    for i, (xyz, name, z) in enumerate(zip(X, topo["names"], topo["elements"])):
        aname, resn, resid = _res_info(str(name), seq)
        atoms.append({
            "i": i,
            "name": aname,
            "full": str(name),
            "elem": Z_TO_ELEM.get(int(z), "C"),
            "resn": resn,
            "resi": resid,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        })
    # Chemical bonds only (topology), undirected unique pairs.
    bonds = sorted({
        (int(a), int(b)) if a < b else (int(b), int(a))
        for a, b in topo["bonds"]
    })

    tag = f"{float(resolution):g}".replace(".", "p")
    npz = ROOT / "out" / f"trajectory_peptide_{''.join(seq)}_ot_name_refine_{tag}A_n10.npz"
    ensemble = []
    if npz.is_file():
        z = np.load(npz)
        for P in z["cleaned_poses"]:
            ensemble.append(np.asarray(P, dtype=np.float64).tolist())

    structure = {
        "atoms": atoms,
        "bonds": [list(p) for p in bonds],
        "true_coords": X.tolist(),
        "ensemble_coords": ensemble,
        "source": topo.get("source"),
    }
    (DATA / "structure.json").write_text(json.dumps(structure) + "\n")

    write_cube(
        DATA / "density.cube", T, org, sp,
        title=f"ACE-{'-'.join(seq)}-NME @{resolution:g}A",
    )

    vals = np.sort(T.ravel())[::-1]
    csum = np.cumsum(vals)
    iso = float(vals[int(np.searchsorted(csum, 0.35 * csum[-1]))])
    meta = {
        "sequence": list(seq),
        "label": label,
        "source": topo.get("source"),
        "resolution": float(resolution),
        "sigma": sig,
        "n_atoms": int(topo["n"]),
        "n_bonds": len(bonds),
        "grid": list(T.shape),
        "spacing": float(spacing),
        "iso_default": iso,
        "iso_high": float(iso * 2.5),
        "iso_low": float(iso * 0.4),
        "n_ensemble": len(ensemble),
        "density_max": float(T.max()),
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {DATA / 'structure.json'}  "
          f"({len(atoms)} atoms, {len(bonds)} chemical bonds, "
          f"{len(ensemble)} ensemble frames)")
    print(f"wrote {DATA / 'density.cube'}  shape={T.shape} iso≈{iso:.3e}")
    print(f"wrote {DATA / 'meta.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=float, default=3.0)
    ap.add_argument("--sequence", type=str, default="LRP")
    ap.add_argument("--spacing", type=float, default=0.45)
    args = ap.parse_args()
    main(resolution=args.resolution, sequence=args.sequence, spacing=args.spacing)
