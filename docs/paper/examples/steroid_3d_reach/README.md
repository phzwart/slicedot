# Steroid 3-D reach

Marketed corticosteroid ligands (properly sized small molecules) run through
free-atom OT → `Namer` → ADMM OT+L1+`P_restr` against rendered ortho maps.

## Ligands

See `ligands.json`. Each slug has `ligands/<slug>/` with CCD/PubChem source,
`topology.npz`, `restraints.cif`, and an `out/` for sweep products.

## Build refs

```bash
cd docs/paper/examples/steroid_3d_reach
uv run python build_ligand_refs.py
```

## Resolution sweep

Default resolutions: 1.5, 2.0, 2.5, 3.0, 3.5 Å (one seed each).

```bash
uv run python run_resolution_sweep.py
uv run python run_resolution_sweep.py --ligands dexamethasone,prednisone --resolutions 2,3
uv run python run_resolution_sweep.py --skip-existing
```

Writes per ligand×resolution under `ligands/<slug>/out/` and an aggregate
`out/sweep_summary.json`.
