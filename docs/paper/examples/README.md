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
uv run python docs/paper/examples/phenol_2d_reach/make_movie.py --resolution 3
uv run python docs/paper/examples/phenol_2d_reach/make_movie.py --reuse
```

Writes `phenol_2d_reach/out/phenol_2d_reach.pdf`, `.png`, and `phenol_2d_reach_movie_<res>.gif`.
