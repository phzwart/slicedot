# Paper examples

Regenerable figures and numerical demos for the manuscript in `docs/paper/`.

## Setup

```bash
uv sync --extra paper
```

## Examples

### `phenol_2d_reach/`

2-D ortho-pentyl phenol (ring + OH + 5-carbon floppy chain) at 1.5 Å: 90° orientation mismatch, translated three molecular radii from the true density. Headline search is consensus ADMM (inexact OT/L1 prox + annealed-slack `P_restr`). Movie panels: OT | L1 | DAC | ADMM.

```bash
uv run python docs/paper/examples/phenol_2d_reach/make_figure.py
uv run python docs/paper/examples/phenol_2d_reach/make_ot_unrestrained.py
uv run python docs/paper/examples/phenol_2d_reach/make_ot_name_refine.py
uv run python docs/paper/examples/phenol_2d_reach/make_movie.py --resolution 3
uv run python docs/paper/examples/phenol_2d_reach/make_movie.py --reuse
```

Writes `phenol_2d_reach/out/phenol_2d_reach.pdf`, `.png`, `phenol_ot_unrestrained*.png` (Adam OT only, NN-matched RMSD), `phenol_ot_name_refine_*.png` (free OT → Namer → ADMM cleanup), and `phenol_2d_reach_movie_<res>.gif`.

The visual guide (`docs/paper/guide/overview.md` §9) embeds regenerable 2-D panels from these runs (and a zigzag @ 3 Å suite).

### `leucine_3d_reach/`

3-D free-atom OT → `Namer` → ADMM OT+L1+`P_restr` on leucine / short peptides (including AFSSFN) against rendered maps. Pipeline panel figure for AFSSFN at 3 Å:

```bash
cd docs/paper/examples/leucine_3d_reach
uv run --extra paper python make_ot_name_refine_ensemble.py --sequence AFSSFN --resolution 3
uv run --extra paper python make_pipeline_panels.py \
    --path out/path_AFSSFN_3A_seed0.npz --sequence AFSSFN
```

Writes `out/peptide_AFSSFN_pipeline_panels_3A.png` (and PDF). A copy lives in the
guide collection as [`docs/paper/guide/fig/32_peptide_AFSSFN_pipeline_panels_3A.png`](../guide/fig/32_peptide_AFSSFN_pipeline_panels_3A.png)
and is shown in `docs/paper/guide/overview.md` §9.6.
