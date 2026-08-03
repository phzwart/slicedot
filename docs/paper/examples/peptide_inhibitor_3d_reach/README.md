# Peptide-inhibitor 3-D reach

Marketed peptidomimetic protease inhibitors (HIV / HCV / SARS-CoV-2). More
flexible than the steroid set (many rotatable bonds).

## Ligands

See `ligands.json`. Each slug has `ligands/<slug>/` with CCD source,
`topology.npz`, `restraints.cif`, and an `out/` for products.

## Build refs

```bash
cd docs/paper/examples/peptide_inhibitor_3d_reach
uv run python build_ligand_refs.py
```

## 3 Å pipeline (free-OT×2, then torsion screen)

```bash
uv run python run_pipeline_3A.py
```

1. **Free OT ×2** @ 3 Å → ghost prune → Namer → standard ADMM → L1+geom polish  
   (`run_resolution_sweep.py --atom-factor 2 --resolutions 3`)
2. **Torsion search** → PCA+OT → L1 top-10 → same ADMM + L1+geom polish  
   (`screen_rdkit_conformers.py`)

Logs: `out/free_ot_x2_3A.log`, `out/rdkit_screen.log`.

## RDKit-target protocol (new map/truth; does not overwrite)

Uses an RDKit torsion+MMFF conformation as the density target (not the CCD
ideal). Outputs go under `ligands/<slug>/out/rdkit_tgt_s<seed>/`.

```bash
uv run python run_rdkit_target_protocol.py --seed 42
```

Per ligand: free OT×2 and torsion screen; from each start both
**OT+ADMM → L1+geom** and **L1+geom alone**. Aggregate:
`out/rdkit_tgt_s42_summary.json`.

## 50k torsion / L1-top-1k (RDKit target)

Same RDKit targets; new folder `rdkit_tgt_s42_n50k` (does not overwrite):

```bash
uv run python run_rdkit_target_torsion_50k.py
```

50 000 torsion confs → L1 rank → **retain top 1000** → dual-refine **best 10**
(`--refine-top`). Aggregate: `out/rdkit_tgt_s42_n50k_summary.json`.
