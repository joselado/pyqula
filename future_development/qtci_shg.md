# The triangle method and QTCI for the second-harmonic conductivity

This note records how the triangle integration of the linear optical
conductivity (`qtci_optical_response.md`) was carried over to the
second-harmonic conductivity, and what it was measured to give. The outcome
is in the code as `integration="triangle"` and `integration="qtci"` of
`h.get_shg_conductivity()` (`conductivitytk/shgsimplex.py`, tests in
`tests/conductivity/test_shg_simplex.py`). What is left open is at the end.

## What has to be interpolated, and what is integrated exactly

We write out the velocity-gauge recursion of `shg.py`. With z = w + i delta,
the k-integrand is -q^3/z^2 times five families of terms. Two are single
resonances, 1/(z - E_mn) and 1/(2z - E_mn), with weights h^ab h^c f and
h^a h^bc f. One has no resonance (h^abc f). Two are products of a resonance
at w and one at 2w, 1/((z - E_ln)(2z - E_mn)), with the three-band loop
h^a_nm h^b_ml h^c_ln times an occupation difference as weight. The 1/z^2 in
front does not depend on k. Every weight is a closed loop of band indices,
so it is gauge invariant and smooth. The weights are averaged over
degenerate multiplets and over the vertices of each triangle, as in the
linear case. The band energies are interpolated linearly, so every energy
difference is linear on a triangle.

The single resonances are integrated exactly by Hermite-Genocchi, as in the
linear case. The 2w shell is the same formula with the nodes E/2, and it
carries the broadening 2 delta, as in `shg.py`.

The products are the new part. When m = n the 2w factor is the constant
1/(2z). When the 2w and the w factor belong to the same pair of bands, a
partial fraction in z gives (1/z)(1/(z - E) - 1/(2z - E)). Both have
constant coefficients, so they are exact single resonances. In a two-band
model these are all the products there are, so for two bands nothing beyond
the linear interpolation of the bands is approximated. A spin copy falls in
the same case, because the case is decided per triangle from the vertex
energies.

For three different bands, the partial fraction gives the coefficient
1/(2E_l - E_m - E_n). It diverges on the line where the w shell of one pair
crosses the 2w shell of another (a double resonance), so it cannot be
interpolated as a weight. Instead we write the product as a divided
difference, (1/2) g[A, C] with g(x) = 1/(z - x), A = E_ln and C = E_mn/2,
and this as an integral over s in [0,1] of g'((1-s)A + sC). The triangle
times [0,1] is a prism, and on it the argument is bilinear in (k, s), not
linear. We cut the prism into three tetrahedra, interpolate the argument
linearly on each from its exact values at the six vertices, and integrate g'
exactly on each tetrahedron, again by Hermite-Genocchi (a third divided
difference of the second antiderivative). The bottom and the top of the
prism are exact, so both shells sit where the linear bands put them, and the
double resonance is only a confluent case of the divided difference.

The interior of the prism is approximated, at O(h |grad(A - C)|) with h the
mesh spacing, and this does not grow as delta goes to zero. With one
triangulation the sum converges only at first order in 1/nk. The
triangulation depends on the order of the three vertices. Averaging over
the six orderings (twelve distinct tetrahedra) removes the first-order part,
and the measured convergence is then of second order. Cutting the prism into
slabs in s does not do this, and it was dropped. An exact formula exists in
principle, a dilogarithm, but it divides by the Jacobian between A and B,
which vanishes when their gradients are parallel; it is not built.

On the three-band MoS2 model (`three_band_tmdc`, spinless, yyy, w from 0.8
to 3.5 eV), the error against the converged grid, as max|s - ref|/max|ref|
over the window:

| delta | nk | one ordering | six orderings |
|---|---|---|---|
| 0.1 | 24 | 2.5e-2 | 1.8e-2 |
| 0.1 | 48 | 1.4e-2 | 4.5e-3 |
| 0.1 | 96 | 7.1e-3 | 1.1e-3 |
| 0.1 | 192 | 3.6e-3 | 2.7e-4 |
| 0.05 | 24 | 4.3e-2 | 3.8e-2 |
| 0.05 | 48 | 3.0e-2 | 8.6e-3 |
| 0.05 | 96 | 1.6e-2 | 2.0e-3 |
| 0.05 | 192 | 7.8e-3 | 5.1e-4 |

The references are the grid at nk = 1024 (delta = 0.1) and nk = 1536
(delta = 0.05). The same second order holds on the three-sublattice
triangular lattice of the tests: 9.9e-3, 2.5e-3, 6.2e-4 at nk = 32, 64, 128,
delta = 0.15.

What is not cured: the 1/z^2 prefactor. Its divergence cancels between the
five families only after the k-integral. The grid keeps this cancellation to
exponential accuracy, the triangle sum only to O(1/nk^2), so the residual is
amplified by 1/|z|^2 at small w.

## Large broadening: the check of the formula

At a broadening the grid resolves, the triangle result must approach the
converged grid. On the gapped honeycomb lattice (sublattice imbalance 0.3,
gap 0.6 t, xxx, w from 0.3 to 3 t), with the grid at nk = 1024 (delta = 0.1)
and 1536 (delta = 0.05) as reference:

| nk | delta = 0.1 | delta = 0.05 |
|---|---|---|
| 24 | 2.3e-1 | 3.2e-1 |
| 48 | 6.2e-2 | 9.3e-2 |
| 96 | 1.6e-2 | 2.4e-2 |
| 192 | 4.0e-3 | 6.0e-3 |
| 384 | 1.0e-3 | 1.5e-3 |

This is second order. It is also a loss against the grid at this
broadening: the grid at nk = 96 is already converged to 5e-4 at delta = 0.1,
while the triangle needs nk = 384 to reach 1e-3. For a broadening that the
mesh resolves, the grid converges exponentially and is the right choice. The
largest errors sit at the two absorption edges, w = 0.3 t (the 2w edge) and
w = 0.6 t (the w edge).

## Small broadening: what the triangle method buys

At delta = 1e-3 the grid needs a mesh whose energy spacing is below delta,
so this is where the triangle method should win. The reference is the
triangle result itself at the finest mesh. The error is max|s - ref| over
the window divided by max|ref| over the same window. This is not the
pointwise relative error of the linear note, because the SHG changes sign.
We give two windows: the whole one, which starts at the 2w edge, and the part
above the w edge.

The gapped honeycomb lattice (two bands, every term exact on the
interpolated bands), xxx, T = 1e-3 t, 55 frequencies, reference the triangle
at nk = 2048. Its own error, from the second order of the 1024 to 2048 step,
is about 9e-4 over the whole window and 1e-4 above w = 0.65 t, so the
smallest entries are not better than that.

| method | diagonalizations | error, w in [0.3, 3] t | error, w in [0.65, 3] t | wall time |
|---|---|---|---|---|
| grid nk = 512 | 262k | 8.3e-2 | 2.8e-1 | 20 s |
| grid nk = 1024 | 1.0M | 3.3e-2 | 1.1e-1 | 48 s |
| grid nk = 2048 | 4.2M | 7.0e-3 | 2.4e-2 | 203 s |
| grid nk = 4096 | 16.8M | 1.3e-3 | 4.6e-3 | 856 s |
| triangle nk = 128 | 16k | 1.9e-1 | 3.3e-2 | 2.0 s |
| triangle nk = 256 | 65k | 5.4e-2 | 5.1e-3 | 6.4 s |
| triangle nk = 512 | 262k | 1.3e-2 | 1.4e-3 | 24 s |
| triangle nk = 1024 | 1.0M | 2.6e-3 | 3.2e-4 | 55 s |
| qtci R = 8 + triangle | 56.5k | 5.4e-2 | 5.1e-3 | 22 s |
| qtci R = 9 + triangle | 134k | 1.3e-2 | 1.4e-3 | 31 s |
| qtci R = 10 + triangle | 201k | 2.7e-3 | 4.0e-4 | 52 s |
| qtci R = 11 + triangle | 238k | 3.1e-4 (*) | 1.6e-4 (*) | 126 s |

(*) R = 11 is the mesh of the reference, so this is only the interpolation
error; the true error is that of the reference, about 9e-4 and 1e-4.

The wall times of the triangle and qtci rows are those of the source kernel
(see "Cost of the simplex sum" below), measured on 4 cores of a workstation
that other jobs kept busy. With the direct sum over simplices they were 3,
9, 34 and 124 s for the triangle at nk = 128 to 1024, and 19, 53, 161 and
600 s for qtci at R = 8 to 11. The diagonalization counts and the errors are
those of the first runs: the qtci counts of the new run differ by up to 4%
(56.9k, 130k, 194k and 230k). The source kernel does not touch the
interpolation, and the reason for the difference was not traced.

Above the w edge the triangle method converges at second order (3.3e-2,
5.1e-3, 1.4e-3, 3.2e-4) and nk = 512 is three times better than the grid at
nk = 4096, with 64 times fewer diagonalizations and 36 times less wall time.
Over the whole window it converges more slowly at first, because the error
sits exactly at the two edges, w = 0.3 t and w = 0.6 t. There the response
at delta = 1e-3 is sharply peaked, and the linear interpolation moves the
band minimum at K by O(1/nk^2). The error falls by four per doubling only
once that shift is below delta, which here is from nk = 256 to 512 on.

The three-band MoS2 model (products of three different bands present), yyy,
T = 1e-3, 28 frequencies, reference the triangle at nk = 1024 (own error
about 4e-4 over the whole window):

| method | diagonalizations | error, w in [0.8, 3.5] eV | error, w in [1.9, 3.5] eV | wall time |
|---|---|---|---|---|
| grid nk = 1024 | 1.0M | 1.2e-2 | 7.6e-3 | 47 s |
| grid nk = 2048 | 4.2M | 5.5e-3 | 1.1e-3 | 164 s |
| grid nk = 4096 | 16.8M | 4.0e-4 | 2.1e-4 | 652 s |
| triangle nk = 64 | 4k | 5.6e-2 | 5.6e-2 | 0.4 s |
| triangle nk = 128 | 16k | 1.9e-2 | 1.3e-2 | 2.0 s |
| triangle nk = 256 | 65k | 5.5e-3 | 2.8e-3 | 8.2 s |
| triangle nk = 512 | 262k | 1.3e-3 | 5.5e-4 | 33 s |
| triangle nk = 1024 | 1.0M | (reference) | (reference) | 184 s |
| qtci R = 8 + triangle | 65.5k (of 65.5k) | 5.5e-3 | 2.8e-3 | 71 s |
| qtci R = 9 + triangle | 249k (of 262k) | 1.3e-3 | 5.6e-4 | 178 s |

The triangle result converges at second order here too (5.6e-2, 1.9e-2,
5.5e-3, 1.3e-3). The wall times are again those of the source kernel. With
the direct sum they were 15, 39, 103, 401 and about 1800 s for nk = 64 to
1024, and qtci took 138 and 451 s, because each product of three bands
cost twelve third divided differences per triangle and per frequency. Then
the grid won on wall time on this model. With the source kernel the triangle
wins on wall time as well. The triangle at nk = 256 equals the grid at
nk = 2048 in error, with 64 times fewer diagonalizations and 20 times less
wall time (8.2 s against 164 s). The triangle at nk = 512 gives 1.3e-3 in
33 s; the grid needs nk = 4096 and 652 s for an error below that, and this
error, 4e-4, is already the one of the reference. So on MoS2 the triangle
is now 5 to 20 times faster than the grid at equal error.

## QTCI of the smooth fields: a gain in diagonalizations for two bands only

The qtci path interpolates the band energies and every term weight (real
and imaginary part) on a quantics grid, sharing one cache of
diagonalizations, and then does the same triangle sum on the 2^R mesh.

- The first version diagonalized every point of the mesh. The imaginary
  parts of the product weights are real numbers up to rounding (1e-15), and
  the cross interpolation, whose tolerance is relative to each field,
  resolved that noise at full rank. A part below 1e-10 of its term is now
  set to zero. Fields equal or opposite on the coarse scan, a symmetry of the
  model, are interpolated once: on the honeycomb lattice this brings 15
  interpolations down to 7.
- On the honeycomb lattice the count then saturates, as for the linear
  conductivity: 56.5k, 134k, 201k and 238k diagonalizations at R = 8 to 11,
  with ranks 64 to 67. At R = 10 this is five times fewer than the triangle
  at nk = 1024 for the same error, and at R = 11 eighteen times fewer than
  nk = 2048.
- On MoS2 it does not saturate at the meshes we ran: R = 8 needs 65.5k of
  65.5k points and R = 9 needs 249k of 262k, with ranks about 120, twice
  those of the honeycomb lattice. The weights are products of three velocity
  elements, and there are many of them: 25 terms kept for three bands (37
  before pruning), each with a real and an imaginary part. The union of the
  pivots of all these interpolations covers almost the whole mesh.
- In wall time qtci does not win for these small models. With the direct
  sum the simplex sum was the dominant cost (600 s at R = 11 against 594 s
  for the triangle at nk = 2048 on the honeycomb lattice). With the source
  kernel the sum is cheap, and on the honeycomb lattice qtci at R = 10 and
  the triangle at nk = 1024 take the same time (52 s and 55 s). On MoS2
  the interpolation of 25 terms is the cost: 71 s against 8.2 s at R = 8,
  178 s against 33 s at R = 9. So qtci only pays where a diagonalization is
  expensive.

So for the SHG the triangle method is the gain. QTCI on top of it saves
diagonalizations for a two-band model, by about the same factor as for the
linear response, and saves nothing for the three-band model up to R = 9.

The interpolation itself was a private copy in `shgsimplex.py`. Since 7
November 2026 it calls the shared `qtcitk/meshfields.qtci_fields`, one
interpolation per field, as the DOS and the optical conductivity did. The
two filters of the copy stay in `shgsimplex.sigma_qtci`: a part of a weight
below 1e-10 of its term is set to zero, and a field equal or opposite to
another one on the coarse scan is interpolated once. The spectrum moves only
within the interpolation tolerance. Against the old copy, the largest change
is 7e-6 of the largest entry at tolerance 1e-6 and 1.8e-8 at 1e-8 on the
gapped honeycomb lattice at nk = 128, and 6e-7 and 7e-12 on MoS2 at
nk = 64, with the same number of diagonalizations. At these meshes the
interpolation visits every k-point, so this checks the filters and the
assembly of the fields, not the sharing of pivots.

The qtci optical conductivity and the qtci LDOS moved on the same day to
one interpolation of all the fields at once (`qtci_stacked_fields`), which
needs 1.6 to 3.7 times fewer diagonalizations when there are a few fields,
at the same error, while the qtci DOS kept one interpolation per field,
because on a 32-band supercell the stacked form was about twice as slow
(the tables are in `qtci_optical_response.md` and `qtci_dos_green.md`). For
the SHG this was not tried. With 25 terms on MoS2,
each with a real and an imaginary part, the stacked rank would be large,
and on the 80 fields of a supercell in the linear case it needed a rank cap
of 800 instead of 400.

## What is left open

- **Cost of the simplex sum (built, 6 November 2026).** The direct sum ran
  frequency by frequency, and for a three-band model the products
  dominated: 1768 s at nk = 1024 for 28 frequencies on four threads. Now the
  sum is built once, as for the linear conductivity
  (`qtci_optical_response.md`). Every resonant factor is a divided
  difference of psi_d over nodes that do not depend on w: the vertex values
  of E_m - E_n for a single resonance at w, their halves for one at 2w, and
  the bottom and top nodes of the prism tetrahedra for a product of three
  bands. The branches (recursion or Taylor series of a cluster of close
  nodes, and the exact cases B = 0 and B = A) do not depend on w either,
  once the Taylor test compares the span of a cluster with delta and not
  with its distance to the pole. This change of the test moves the direct
  sum only by rounding (2.8e-10 of the largest entry on MoS2 at nk = 256),
  and it makes the two kernels take the same branches (no longer: the direct sum decides
  per frequency again since the seventh bug audit, see its repairs below). So we run the
  recursion of the divided differences backwards once, and the whole sum
  becomes a list of sources in x = e - z at a single z: psi_d at the nodes,
  indexed by k-point and node type and merged by energy; pairs of nodes
  closer than delta/4, in closed form; and a few Taylor clusters of three or
  more close nodes. The w and 2w families share z, because the 2w nodes are
  halved energies. The terms of the cases B = 0 and B = A carry 1/z and are
  kept as a second family of coefficients. The build runs in parallel over
  rows of the mesh, in three sets of rows that share no vertex, and it gives
  the same sources bit for bit on 1 and on 4 threads (checked at odd and
  even nk). The direct sum stays as
  `_sigma_from_fields(..., kernel="simplices")` for the test.

  Wall times of the sum alone, both kernels on the same fields and back to
  back in one process, on 4 cores of a workstation that other jobs kept
  busy (load 12 to 19 on 16 threads), delta = 1e-3. The new time includes
  the build of the sources:

  | model | nk | frequencies | old (s) | new (s) | build (s) | max difference / max abs(sigma) |
  |---|---|---|---|---|---|---|
  | gapped honeycomb, xxx | 256 | 55 | 9.0 | 0.65 | 0.43 | 2.5e-12 |
  | gapped honeycomb, xxx | 512 | 55 | 39.1 | 3.4 | 1.7 | 2.1e-12 |
  | gapped honeycomb, xxx | 1024 | 55 | 160 | 16.2 | 7.4 | 1.9e-12 |
  | MoS2, yyy | 128 | 28 | 17.9 | 1.16 | 1.16 | 5.7e-11 |
  | MoS2, yyy | 256 | 28 | 78.3 | 5.6 | 2.6 | 8.5e-10 |
  | MoS2, yyy | 512 | 28 | 314 | 23.6 | 10.8 | 4.8e-10 |
  | MoS2, yyy | 1024 | 28 | 1659 | 112 | 55 | 2.1e-9 |

  The gain is 10 to 15. The fields (diagonalization, velocities and term
  weights) take 2.7, 11.5 and 42 s on the honeycomb lattice and 1.1, 4.8,
  13 and 54 s on MoS2 at the same meshes, so for MoS2 the sum is still the
  larger part. The full spectra with the public call, against the earlier
  runs of the direct sum on the same machine, are in the two delta = 1e-3
  tables above: on MoS2 at nk = 1024 the new spectrum differs from the one
  of the direct sum before the change of the Taylor test by 2.1e-9 of the
  largest entry, the same level as in the table.

  The difference between the kernels is rounding, not discretization. A
  reference in extended precision (numpy longdouble, the same branches)
  gives the error of each kernel. On MoS2 both are at the same level: at
  nk = 24 and 48, the direct sum is off by 3.2e-11 and 2.4e-10 of the
  largest entry, and the source kernel by 3.9e-11 and 2.7e-10. On the
  gapped honeycomb lattice at nk = 48 the two errors are 3.9e-14 and
  4.4e-13. The worst case we found is the three-sublattice triangular
  lattice of the tests at nk = 24: 1.8e-13 for the direct sum and 3.7e-11
  for the source kernel. It sits in the products of three bands, and it is
  in the coefficients and not in their evaluation (evaluating the same
  coefficients in extended precision changes nothing). Simplex by simplex
  the two kernels have errors of the same size; only in the sum over
  simplices do the errors of the direct sum cancel and those of the sources
  do not. The likely reason, not proven: the coefficients of the nodes are
  of order 1/gap^3 and their rounding differs from simplex to simplex,
  while the recursion of the direct sum makes the same rounding in the
  simplices that share a node. Compensated (double-double) arithmetic in the build
  would cure this; it was not tried, because the level is five orders of
  magnitude below the discretization error. The new test asserts agreement
  to 1e-9 of the largest entry at nk = 40 (two bands) and nk = 24 (three
  bands), where the measured differences are 3e-13 and 3.7e-11.

  Most clusters are pairs. A pair stored as its Taylor series needs about
  25 coefficients, and the number of pairs grows as nk^3 at fixed delta
  (the triangles times the chance that two nodes are closer than delta/4):
  at nk = 512 on MoS2 there are 6.0M pairs against 1.4M merged nodes. So a
  pair is evaluated in closed form from its two nodes, which took the memory
  at nk = 512 from 146M Taylor coefficients (2.3 GB) to 7M. At nk = 1024
  there are 41M pairs, 3.3M Taylor clusters with 93M coefficients (1.5 GB)
  and 5.5M nodes. What stays open: the pairs and clusters then dominate the
  cost per frequency, about 0.45 s per frequency at nk = 512 and 2 s at
  nk = 1024 on 4 cores, and their number grows as nk^3. Merging
  the pairs that are equal (symmetry-related k-points, spin copies), or a
  far-field expansion of all the sources, would reduce this.
- **Below the absorption edge.** The 1/z^2 prefactor amplifies the
  O(1/nk^2) residual of the sum rule that cancels it. Inside the gap the
  integrand is smooth and the grid is better; not measured in detail below
  w = 0.3 t.
- **QTCI for three bands.** The count did not saturate up to R = 9 on MoS2.
  Whether it saturates at R = 11 or 12, as on the honeycomb lattice, is
  open. Fewer fields would help: the weights could be grouped
  by their resonant factor before the interpolation, which the code does
  only for fields that are exactly equal.
- **Metals.** The occupation factors in the weights jump at the Fermi
  surface, and the velocity gauge carries the intraband response through
  the cancellation between the five families. Neither was checked for the
  triangle path.
- **An exact product.** The integral of 1/((z - A)(2z - B)) over a triangle
  with A and B linear is a dilogarithm divided by the Jacobian of (A, B). It
  would make the products exact, but it needs a separate limit wherever the
  two gradients are parallel. The ordering average already gives second
  order, so it was not built.
- **Not run.** The spinful MoS2 model with spin-orbit coupling (six bands),
  the length gauge (the triangle path refuses it), and 3D.
- **Shift current and sum frequencies.** With two different frequencies the
  products have resonances at w1, w2 and w1 + w2. In principle the same
  prism construction applies to each product; not built.

## Repairs of the seventh bug audit

- **The Taylor rule of the direct sum (fixed, 14 November 2026); the source kernel keeps
  its rule (measured, not changed).** Since the source kernel was built, both kernels
  sent a cluster of nodes to its Taylor series when its span was below delta/4, and to
  the recursion of the divided differences otherwise, whatever the frequency. Far from
  resonance, at a distance abs(x) from z much larger than the span s, the recursion loses
  eps (abs(x)/s)^r in a divided difference of order r: r = 2 for a single resonance on a
  triangle and r = 3 for a prism piece. One triangle at energies 0.5, 0.5 + 0.4 s, 0.5 + s
  and one at 0.5, 0.5 + 0.02 s, 0.5 + s, at w = 0 and delta = 1e-6, against a 200-point
  Gauss-Legendre quadrature: the single resonance lost up to 5.5e-4 of its value, at s
  from 0.26 to 2 delta, and the product 1.1e3 times its value at s = delta and still 2e-2
  of it at s = 30 delta. At delta = 1e-3 the product lost up to 1.2e-6 of its value. On
  resonance there was a second loss: the series skipped its first-order term, which
  vanishes only for the exact mean of the nodes, while the rounded mean is off by
  eps abs(m). That is eps abs(m)/delta relative where abs(y) is about delta, 2.4e-10 of a
  triangle at m = 0.5 and delta = 1e-6 against mpmath at 50 digits.

  The direct sum (`kernel="simplices"`) now decides per frequency, with the rule of the
  first version of this module: a cluster of nodes of span s at a distance y of its mean
  from the pole goes to its series when s < y/4, and the series keeps its first-order term
  in both kernels. Over the two shapes, spreads from 0.3 to 10 delta, far from resonance
  and on it, the worst error of the single resonance against mpmath is now 9e-15, against
  5.5e-4 before. It costs more per simplex, since a far cluster now takes a series of 15
  to 30 terms instead of three or four logarithms: one product of three bands, on one
  core, took 15 microseconds near resonance and 9.5 far from it, against 3.2 and 3.0 with
  the rule on delta. Two smaller ratios were tried and dropped. With s < y/10 the far
  product took 22 microseconds, and with s < 10^(-3/r) y at order r (4.2 and 7.7
  microseconds) a prism piece on resonance lost 4e-11 of its value.

  The source kernel, the default, cannot decide per frequency, and it keeps the rule on
  delta. At the delta of an optical spectrum its loss is small: on the three-band model of
  the tests at nk = 64 and delta = 1e-3 (66 frequencies on [0.5, 7] t), the old direct sum
  is within 5.0e-12 of the new one and the source kernel within 5.1e-11. It matters for
  nearly flat bands at a small delta. We built a cure and did not keep it. While building
  the sources, it bounded for every simplex and term the rounding error of the fixed
  branches over the frequency window, eps times the sum over the leaves of the Newton table
  of abs(coefficient) times a bound of the leaf on the window. When this was above a budget
  times a lower bound of the peak of the term on the simplex (pi/(s + pi delta) for one
  resonance, divided by 2(G + s) + 4 delta for a product, with G the largest distance
  between the two resonant energies), the simplex and term made no sources and were summed
  at every frequency with the rule of the direct sum. This is the rule of kubosimplex.py
  before its energy bins. A scan on the same model, nk = 32, 20 frequencies, two threads,
  against the direct sum (the time is the source sum with its build, the row without a
  budget the old kernel):

  | delta | budget | fraction summed per frequency | time (s) | error / max abs(sigma) |
  |---|---|---|---|---|
  | 1e-3 | 3e-11 | 10.2% | 1.58 | 2.1e-13 |
  | 1e-3 | 3e-10 | 4.8% | 0.85 | 3.4e-13 |
  | 1e-3 | 3e-9 | 0.56% | 0.21 | 4.9e-12 |
  | 1e-3 | 3e-8 | 0 | 0.12 | 2.6e-11 |
  | 1e-3 | none | 0 | 0.12 | 2.6e-11 |
  | 1e-5 | 3e-11 | 10.2% | 1.65 | 2.1e-13 |
  | 1e-5 | 3e-10 | 5.0% | 0.83 | 3.0e-13 |
  | 1e-5 | 3e-9 | 1.6% | 0.36 | 2.3e-12 |
  | 1e-5 | 3e-8 | 0.63% | 0.23 | 1.9e-11 |
  | 1e-5 | none | 0 | 0.10 | 7.5e-11 |

  And at nk = 64, delta = 1e-3, 66 frequencies:

  | budget | fraction | time (s) | error / max abs(sigma) |
  |---|---|---|---|
  | 3e-9 | 0.68% | 0.96 | 5.0e-11 |
  | 6e-9 | 0.31% | 0.59 | 2.0e-11 |
  | 1e-8 | 0.11% | 0.47 | 4.2e-11 |
  | 2e-8 | 0.01% | 0.32 | 3.1e-11 |
  | none | 0 | 0.31 | 5.1e-11 |

  On three nearly flat bands of spread 0.45, 1 and 3 delta on the 2x2 mesh, at delta =
  1e-6, the old source kernel loses 3.0e-9, 5.4e-9 and 8.6e-10 of the largest value over
  the window; a budget of 3e-9 leaves 1e-14, 6e-9 leaves 5e-11 at the spread of 3 delta,
  and 1e-8 leaves 1e-9 there. Relative to the value at each frequency, which is what a
  user reads off resonance, the source kernel there is off by 4e-4 at w = 0, by 1e-2 at
  w = 2 t and by more than the value at w = 0.3 t, where the terms nearly cancel, against
  1e-15 to 9e-11 for the direct sum. So no budget both cures the flat bands and keeps the cost:
  3e-9 cures them and costs 2.6 times the old time at nk = 64 and delta = 1e-3 (the
  smallest of three lock holds, 0.95 s against 0.36 s), for no gain in precision there,
  since what is left at 5e-11 is the rounding of the vertex form of simplices below that
  budget (only 3e-11 or less brings it to 2e-13, at 14 times the cost). The open options
  are: a budget of 3e-9 at that cost, the energy bins and the split along the Newton table
  of kubosimplex.py, which keep the far simplices cheap, or leaving it. For a small delta
  or flat bands, `kernel="simplices"` of `_sigma_from_fields` is exact, but the public
  call cannot select it. The direct sum,
  now the exact reference, is slower with the rule per frequency: 38 s against 9.3 s for
  that model at nk = 64 and 66 frequencies (the smallest of three holds).

  Two tests pin the repair. One triangle and one prism against the quadrature to 1e-11 of
  the value, far from resonance and on it, at delta = 1e-6; with the old rule it failed
  at 2.8e-10 at its first case. And the three nearly flat bands against the quadrature of
  every triangle, the direct sum to 1e-11 of the largest value over a window with
  frequencies far from the bands and on them; with the old rule it lost 6.1e-10 to 1.8e-9.
- **Only the vertices the weights read (built, 14 November 2026).** The fields of the
  triangle and qtci paths built and rotated all 19 Cartesian derivatives of H per batch
  of k-points (3 first, 6 second, 10 third), while `_weights_jit` reads only h^a, h^b,
  h^c, h^ab, h^ac, h^bc and h^abc of the requested triplets: 3 of 19 for `xxx`.
  `shg._eigenbasis_derivatives` now takes the set of vertices to build
  (`shg.triplet_vertices(trip)` here), and it leaves zero, without building them, the
  derivatives along an axis on which every bond vector vanishes, so a planar cell never
  builds its z vertices. The grid path of `shg_conductivity` asks for the vertices of its
  triplets and h^c for every c in the velocity gauge, and for orders 1 and 2 in the length
  gauge. The arrays that are built are the same, bit for bit, as in the full build (a test
  checks every vertex on a planar cell and on a bilayer, whose z vertices are not zero).
  Wall time of `_fields`, two threads, the smallest of three lock holds with old and new
  back to back in each, on a shared machine: gapped honeycomb, `xxx`, nk = 256, 3.33 s to
  1.42 s (2.3x); three-band triangular lattice, the six in-plane triplets, nk = 128,
  0.97 s to 0.76 s (1.3x), where more vertices are read and the weights take a larger
  part. The largest peak memory of the process over the holds was 428 MB before and 466
  MB after, so the skipped vertices do not lower the peak of this small model; the cause
  of the 38 MB was not looked into.
- **The description of the method in the docstrings (fixed).** The docstrings of
  `optical_conductivity` and `shg_conductivity` in `conductivity.py` said that the triangle
  integration interpolates the weights linearly inside each simplex. The code takes one
  weight per simplex, the average over its vertices, times the exact simplex average of
  the resonant factor; the docstrings now say so. The one of `optical_conductivity` also
  described the 3D source kernel of its first version, with the kept tetrahedra summed
  whole at every frequency and a gain of 4 to 14; it now describes the energy bins, the
  split of a kept tetrahedron into its close edges and faces, and the sets of point
  sources that give their sum through their moments far from a frequency, without a
  gain, since the direct kernel was not timed again. No number changes.
- **Review of these repairs (14 November 2026).** Each repair was broken on purpose in a
  copy of the tree, to see which test fails (the rule of `_small` alone in one copy; the two
  first-order terms together in another, where the direct sum reads only the first and the
  source kernel only the second). With the rule on delta
  alone back in `_small`, the two tests of the repair fail, at 2.8e-10 of a triangle far
  from resonance and at 6e-10 of the largest value of the three bands. With the first-order
  term of the series skipped in the direct sum they fail too, but by little: 5.5e-11 of a
  triangle and 1.5e-11 of the three bands, against a tolerance of 1e-11. With the term
  skipped in the source kernel no test failed, since no test had bands flat enough for the
  source kernel to be exact. A third test now pins the term in both kernels: three bands of
  spread 0.1 and 0.2 delta at delta = 1e-7, where every triangle is one cluster that both
  kernels do by its series at every frequency, against the quadrature of every triangle. Both
  kernels give it to 7e-14 and 2e-14 of the largest value; without the first-order term
  either one is off by 2.8e-10, and the committed code fails it for both.

  The loss of the source kernel was measured again on the three nearly flat bands of the
  tests, relative to the largest value over the window, with the direct sum next to it:

  | delta | spread | source kernel | direct sum |
  |---|---|---|---|
  | 1e-6 | 0.1 delta | 7.0e-15 | 6.8e-15 |
  | 1e-6 | 0.45 delta | 3.0e-9 | 9.8e-15 |
  | 1e-6 | 1 delta | 5.4e-9 | 3.1e-15 |
  | 1e-6 | 3 delta | 8.6e-10 | 3.0e-14 |
  | 1e-7 | 0.1 delta | 6.5e-14 | 6.5e-14 |
  | 1e-7 | 0.45 delta | 9.1e-8 | 2.1e-14 |
  | 1e-7 | 1 delta | 1.8e-7 | 5.7e-13 |
  | 1e-7 | 3 delta | 5.7e-9 | 6.1e-13 |

  So the loss of the default kernel is a fraction of the peak of the spectrum that grows 30
  times at a spread of delta or below, and 7 times at 3 delta, when delta goes down 10
  times, and it is exact below a spread of delta/4.
  The decision of the review is to keep the source kernel as it is. A budget that sends
  some simplices to a sum per frequency is a tuned threshold: 3e-9 costs 2.6 times at the
  delta of an optical spectrum for no gain there, and 6e-9 does not cure the spread of
  3 delta. The cure that keeps the cost is the one of kubosimplex.py, the energy bins of the
  far simplices and the split along the Newton table, where a cluster is a series far from
  it and a divided difference close to it. It is not ported, and it is the open item of this
  note. Until then the docstring of `shg_conductivity` says what the triangle integration
  does not resolve: values many orders of magnitude below the peak, for nearly flat bands at
  a very small delta.

  A test now checks the callers of the vertices and not only the builder. For the grid in
  both gauges, the triangle integration, the injection current, the Berry curvature dipole
  and the dc terms, the result with the full build of the 19 vertices in place of the
  subset is the same bit for bit. The second-harmonic routes run on a bilayer with a tilted
  interlayer hopping and a uniaxial strain, where the vertices with mixed indices (h^xz,
  h^yz, h^xyz) are finite, one component per call. With h^ac left out of
  `triplet_vertices` the triangle route of this test fails, and so does the comparison of
  the triangle integration with the grid for three bands. In the grid the partner
  component with b and c exchanged asks for the same vertex, so there the omission changes
  nothing.

  The review found the same loss in one more place and repaired it. For a cluster of
  equal nodes (a span below 1e-12 of the distance to the pole, in practice an exactly flat
  transition energy) both kernels kept only the leading term of the series. The rounded
  mean of equal nodes is not always the node: (0.7 + 0.7 + 0.7)/3 is 0.7 minus 1.1e-16. So
  an exactly flat band lost eps abs(m)/delta on resonance, 1.1e-9 of the largest value at
  delta = 1e-7 on the three flat bands of the tests (energies 0.5 and 0.7), in both kernels.
  The series now keeps its first-order term there too, which costs one more term per
  cluster, and the test above includes the exactly flat bands.
- **The two points the review left to a decision (decided 14 November 2026).** The
  first-order term of the series for equal nodes stays. It is not a new branch: the case
  of equal nodes now keeps the term that the series of close nodes already keeps, for the
  same reason, the rounded mean. It costs one more term per such cluster, and
  `test_bands_flatter_than_the_broadening` pins it in both kernels, where the exactly flat
  bands lose 1.1e-9 of the largest value without it.

  The default source kernel keeps its rule on delta, and the port of the energy bins is
  not started. The loss is a fraction of the peak of the spectrum, 5e-11 at the delta of
  an optical spectrum and up to 5e-9 at delta = 1e-6 and 2e-7 at delta = 1e-7, and only
  for bands whose spread on a triangle is between delta/4 and a few delta. It is seen in
  the values far from the resonances of such bands, and the docstring of
  `shg_conductivity` says so. The route to the root that the review named does not remove
  a tuned number. `kubosimplex.py` still decides with a budget which simplices leave the
  vertex form (`_tol = 3e-11` of the peak of the simplex, and `_budget = 1e-2` of it for
  the split along the Newton table). Its energy bins and its sets of point sources only
  make that budget cheap, where the budget of 3e-9 tried here cost 2.6 times. So the port
  is the cure that was dropped plus the far field that pays for it, and it is a rewrite of
  the default kernel: the moments of a prism piece around the centre of an energy bin, for
  the two families of coefficients (plain and times 1/z), the sets of point sources with
  one decision per set and frequency, the split of a kept simplex, and the compensated
  sum. It needs its own review, and timing tables up to nk = 1024 that do not fit a shared
  workstation, so it waits for a sign-off. A rule with no budget at all exists in
  principle: every cluster by its series far from a frequency, summed through the bins,
  and by its divided difference close to it, which are the branches of the direct sum. It
  gives up the vertex form, which merges the simplices that share a k-point, and its cost
  was not estimated by a run. A keyword on the public call that selects the direct sum was
  considered and not added: the direct sum is about 100 times slower (38 s against 0.31 s
  at nk = 64 and 66 frequencies on the three-band model), and it would put a choice of
  kernel on a public method for one corner. `_sigma_from_fields(kernel="simplices")` stays
  the exact reference. What would decide the port is a calculation that needs the response
  far from the resonances of nearly flat bands at a delta of 1e-6 or below.
