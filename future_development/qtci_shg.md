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
  and it makes the two kernels take the same branches. So we run the
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
