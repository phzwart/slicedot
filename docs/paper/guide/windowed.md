# Gabor-windowed sliced $`W_1`$

The global sliced path of the [visual guide](overview.md) projects the entire
model and map onto each direction $`u`$:

$$
P_u\mu(t)=\int_{u\cdot r=t}\mu.
$$

That integrates out both transverse dimensions, so two atoms at the same
projected coordinate but far apart in the perpendicular plane are
indistinguishable within a slice. Locality is recovered only by intersecting
many directions — a tomography problem with $`L\gtrsim\pi D/d`$ for a region of
diameter $`D`$ at resolution $`d`$.

This note records the locally weighted variant implemented in
`src/slicedot/windowed.py`. Multiply both model and map by a Gaussian window
before slicing, and take the objective as an expectation over window centres,
widths, and directions. The key algebraic fact: a Gaussian window times a
Gaussian atom is still a Gaussian atom, so the Agarwal / Ten Eyck backend
survives with modified weights, positions, and width.

---

## 1. Window and the model side

The window is a unit-normalised isotropic Gaussian of width $`s`$ centred at
$`a`$:

$$
w_{a,s}(r)=G_s(r-a).
$$

For a model atom $`j`$ with weight $`w_j`$, position $`r_j`$, and width
$`\sigma_j`$,

$$
G_s(r-a)\,G_{\sigma_j}(r-r_j)
=C_j(a,s)\,G_{\sigma_j'}(r-r_j'),
$$

with

$$
C_j(a,s)
=(2\pi(\sigma_j^2+s^2))^{-3/2}
\exp\!\Bigl(-\frac{\lVert a-r_j\rVert^2}{2(\sigma_j^2+s^2)}\Bigr),
$$

$$
\sigma_j'^2=\frac{\sigma_j^2 s^2}{\sigma_j^2+s^2},\qquad
\beta_j=\frac{s^2}{\sigma_j^2+s^2},\qquad
r_j'=(1-\beta_j)\,a+\beta_j\,r_j.
$$

$`C_j`$ is the per-atom importance weight (Gaussian overlap of the atom with
the window). For $`s\gg\sigma_j`$ it reduces to the window evaluated at the atom
centre, $`\beta_j\to 1`$, and $`\sigma_j'\to\sigma_j`$.

Projected coordinate and windowed structure factor along $`u`$, with the usual
phase origin $`c`$:

$$
p_j'=u\cdot(r_j'-c),\qquad
M_{a,s}(q;u)
=\sum_j w_j\,C_j(a,s)\,
e^{-2\pi^2\sigma_j'^2 q^2}\,
e^{-2\pi i q p_j'}.
$$

This is structurally identical to the global $`M(q)`$ of §3.3 of the guide.
When all $`\sigma_j`$ are equal, all $`\sigma_j'`$ are equal, so the grid backend
still uses one common gridding width and one residual multiplier.

---

## 2. Target side

$$
T_{a,s}(q;u)
=\int G_s(r-a)\,\rho(r)\,e^{-2\pi i q\,u\cdot(r-c)}\,\mathrm{d}r.
$$

This is the Gabor transform of the map at spatial position $`a`$ and frequency
$`q u`$. Equivalently, in reciprocal space,

$$
T_{a,s}(q;u)
=\int\hat\rho(k)\,\hat G_s(k-qu)\,e^{2\pi i(k-qu)\cdot a}\,\mathrm{d}k.
$$

The Phase-1 oracle evaluates the real-space form by windowing the map on the
grid, projecting, and taking a 1-D FFT. Phase 2 precomputes $`T`$ on a lattice
of centres (spacing $`\le s/2`$) and a geometric $`s`$-ladder, then interpolates
trilinearly in $`a`$ and linearly in $`\log s`$. The $`a`$-dependence is
bandlimited to $`\lvert k-qu\rvert\lesssim 1/s`$ by construction, so that lattice
spacing sits at the Nyquist guideline.

---

## 3. Normalisation and the mass term

Local windows do not conserve mass:

$$
m_{a,s}=M_{a,s}(0)=\sum_j w_j C_j(a,s),\qquad
n_{a,s}=T_{a,s}(0).
$$

Split the two effects rather than burying one in the other:

1. **Transport.** Rescale $`\tilde T=(m/n)\,T`$ so that
   $`M(0)=\tilde T(0)`$ exactly. Then proceed as in §3.2 of the guide:

   $$
   H=\mathcal{F}^{-1}\!\left(\frac{M-\tilde T}{2\pi i q}\right),\qquad
   H\leftarrow H-H[n_{\mathrm{empty}}],\qquad
   \text{score}=\int\lvert H\rvert.
   $$

   Because $`M(0)=\tilde T(0)`$ after rescaling, the numerator of
   $`(M-\tilde T)/(2\pi i q)`$ genuinely vanishes at the origin, so the
   division is finite rather than conventionally pinned. Because the window
   drives the profile to zero at the edges, $`n_{\mathrm{empty}}`$ pinning is
   well defined rather than assumed.

   The implementation reports the *unit-mass* sliced $`W_1`$ (divide the
   mass-$`m`$ score by $`m`$), so the wide-window limit reproduces global
   `SlicedOT`.

2. **Mass residual.** Carry $`\lambda_{\mathrm{mass}}(m_{a,s}-n_{a,s})^2`$ as a
   separately weighted term (default $`\lambda_{\mathrm{mass}}=0`$).

---

## 4. Objective and sampling

$$
E
=\mathbb{E}_{a\sim\pi_a,\,s\sim\pi_s,\,u\sim\pi_u}
\bigl[E_{a,s}(u)\bigr].
$$

Estimate by Monte Carlo: draw $`n_{\mathrm{windows}}`$ triples per call, score
each, then take a single mean. Never update between samples.

Uniform $`\pi_a`$ gives an exact partition of unity for free, since
$`\int\mathrm{d}a\,G_s(r-a)=1`$ identically — no seams, no lattice artifacts.
Defaults: $`\pi_a`$ uniform over the model bounding box plus a margin of
$`2 s_{\max}`$; $`\pi_s`$ log-uniform on $`[s_{\min},s_{\max}]`$; $`\pi_u`$ a
randomly rotated Fibonacci set (Phase 1) or a fixed pool frozen with the
`GaborTarget` precompute (Phase 2).

Hard constraint: $`s\ge 3\,\sigma_{\max}`$. Below that, $`\sigma'`$ differs
materially from $`\sigma`$ and the window deforms atomic form factors rather
than selecting a region. Direction count defaults to
$`L=\lceil\pi\cdot 4s/d\rceil`$ (`suggest_L`).

---

## 5. Gradients and the Monge step

$`\nabla_{r_j}E`$ has three potential paths:

1. **Through $`p_j'`$.** Keep. The chain rule picks up
   $`\partial r_j'/\partial r_j=\beta_j`$.
2. **Through $`C_j`$.** Detach. Under uniform $`\pi_a`$,
   $`\int\mathrm{d}a\,C_j`$ is independent of $`r_j`$ — pure variance. Under
   non-uniform $`\pi_a`$ it is a bias toward the sampling density.
3. **Through $`\sigma_j'`$.** No $`r_j`$ dependence.

The Monge / deformation path quantile-matches on the windowed CDFs exactly as
`SlicedOT.deformation` does, giving per-slice displacements $`\delta_{\ell j}`$,
then $`\tilde v_j=(1/L)\sum_\ell\delta_{\ell j}u_\ell`$ and
$`v_j=M^{-1}\tilde v_j`$. Two windowed corrections:

- The displacement lives in $`r'`$ coordinates: $`\Delta r_j=\Delta r_j'/\beta_j`$.
- Forces add; displacements average. Gather the Monge path as a
  $`C_j`$-weighted average over windows (weights sum to 1 per atom). Summing
  instead of averaging overshoots as $`K\times`$ with $`n_{\mathrm{windows}}`$.

---

## 6. Implementation map

| Piece | Module |
|-------|--------|
| Oracle (Phase 1) | `WindowedSlicedOT` with `backend="direct"` |
| Gabor precompute (Phase 2) | `GaborTarget` |
| Grid model path (Phase 3) | `backend="grid"` / `"grid_custom"` when $`\sigma'`$ is common |
| Direction helper | `suggest_L(s, d)` |
| Tests | `tests/test_windowed.py` |
