# QTCI and the triangle method for the optical conductivity

This note records where quantics tensor cross interpolation (QTCI) helps the
k-integral of an optical response and where it does not, with the
measurements that decided it. The outcome is in the code as
`integration="triangle"` and `integration="qtci"` of
`h.get_optical_conductivity()` (`conductivitytk/kubosimplex.py`, tests in
`tests/conductivity/test_optical_simplex.py`). What is left open is at the end.

All numbers are for the spinless honeycomb lattice with a sublattice
imbalance of 0.3 (gap 0.6 t), Re sigma_xx, T = 1e-3 t, and the error is
measured against the uniform mesh at nk = 8192 (67M diagonalizations),
which itself agrees with the converged triangle and qtci results to 1e-4 to
3e-4. Below that level the methods cannot be ranked with this reference.

## QTCI of sigma itself: a loss

The first idea was to interpolate the integrand of sigma(omega) on a
quantics grid of (kx, ky), one interpolation per frequency. The resonant
shell E_c - E_v = omega has width delta, so the integrand is sharp, and the
hope was that QTCI refines only near the shell.

- At delta = 1e-2 it loses: the uniform mesh at nk = 512 (262k points,
  error 1e-3) beats QTCI at R = 12 (378k evaluations, error 4.5e-3).
- At delta = 1e-3 it is comparable, not better: R = 12 at tolerance 1e-5
  gives 6e-4 with 365k diagonalizations, against 3e-3 for the uniform mesh
  at nk = 2048 (4.2M). The error is controlled by the tolerance (it comes
  out 20 to 70 times the requested value) and not by the rank cap (raising
  `maxbonddim` from 400 to 800 changes nothing, the rank saturates at 115
  for R = 10 and 256 for R = 12). Seeding the pivots on the resonant shell
  instead of at random points barely helps.
- Each frequency needs its own interpolation, while the uniform mesh gives
  all frequencies from one diagonalization, so for a spectrum this is a
  loss in every regime.
- Interpolating the joint function f(kx, ky, omega), with omega as a third
  quantics variable, is a clear loss: at Rk = 8, Rw = 6, delta = 1e-2 it
  diagonalized 73% of the k-points at rank 404, with an error of 9e-3.

## The triangle method: the real gain

The sharp part of the integrand is only the resolvent 1/(z - dE(k)); dE and
the weights are smooth. Interpolating dE linearly inside each triangle of
the mesh and integrating the resolvent exactly (Hermite-Genocchi, the 2D
analogue of the analytic tetrahedron method of Lambin and Vigneron, PRB 29,
3430 (1984)) gives every frequency from one mesh, at any delta:

| method | diagonalizations | max relative error, omega in [0.65, 3] t |
|---|---|---|
| uniform nk = 4096 | 16.8M | 4e-3 |
| triangle nk = 256 | 65k | 1e-3 |
| triangle nk = 1024 | 1.0M | 3e-4 |
| qtci R = 8 + triangle | 35k | 1e-3 |
| qtci R = 10 + triangle | 69k | 3e-4 |
| qtci R = 12 + triangle | 81k | 3e-4 |

The triangle method alone gives about a factor 1000 in diagonalizations
against the uniform mesh at equal error. QTCI of the smooth fields (dE and
the weights, one interpolation each, sharing one cache of diagonalizations)
adds a factor 2 at 1e-3 and about 15 at 3e-4, because the number of
diagonalizations saturates with R: ranks stay at about 43 for dE and 60 for
the weight. At delta = 1e-2 the triangle result at nk = 512 agrees with the
converged uniform mesh at nk = 2048 to 1e-4 at all six frequencies, which is
the independent check of the formula.

The triangle result converges at second order in 1/nk (the Hall
conductivity at omega = 0 has errors 4.8e-3, 1.2e-3, 5.4e-4 at nk = 30, 60,
90). For a smooth periodic integrand, a frequency inside the gap or a large
delta, the uniform mesh converges exponentially and is the better choice;
the docstring and the guide say so.

The weights are averaged over degenerate multiplets at each k before they
are interpolated. For exact degeneracies at every k (a spinful copy) this
changes nothing in the triangle sum, because the pairs of a multiplet share
the same dE and the sum is linear in the weights; it matters for an
interpolation in which a single w_nm would depend on the basis that eigh
picks. On the spinful Haldane model, switching the averaging off changed
the triangle result by 8e-13 and the qtci result by 7e-8, so on the models
tried the unaveraged fields were smooth as well (possibly because eigh
returns pure spin states for a spin-block Hamiltonian; not checked). Only
the direct basis-rotation test exercises the averaging.

## What is left open

- **Cost per frequency (built, 6 November 2026).** The triangle sum used to
  be evaluated triangle by triangle at every frequency. The branches of
  the formula (exact divided difference, or the Taylor series of a
  confluent edge or triangle) depend on the vertex energies and on delta,
  but not on omega. So the sum is a fixed list of sources: psi_2 at the
  vertex energies, and a few point terms at the middle of the confluent
  edges and triangles. This is the weighted joint DOS, a sum of linear
  B-splines, stored by its knots. We build the list once, with the vertex
  sources indexed by k-point, so the six triangles that share a vertex
  share one source. Sources with exactly the same energy are merged, which
  also merges the copies of a spinful model and the points related by an
  exact symmetry. Then each frequency costs one sum over the sources. The
  old kernel is kept as `_sigma_from_fields(..., kernel="simplices")` for
  the test that compares the two. Wall times on 4 cores of one workstation,
  both kernels on the same pair fields, the new time including the build
  of the sources (0.55 to 0.6 s at 2M triangle-pairs), delta = 1e-3 unless
  noted, frequencies on [0, 7] t:

  | model | mesh | frequencies | bands | old (s) | new (s) | max difference / max abs(sigma) |
  |---|---|---|---|---|---|---|
  | gapped honeycomb | 256^2 | 701 | 2 | 7.7 | 0.33 | 1.8e-12 |
  | gapped honeycomb | 512^2 | 701 | 2 | 30.8 | 1.32 | 8.2e-12 |
  | gapped honeycomb | 1024^2 | 701 | 2 | 122.7 | 5.17 | 2.1e-11 |
  | gapped honeycomb, delta = 1e-2 | 1024^2 | 701 | 2 | 122.3 | 5.38 | 1.4e-11 |
  | gapped honeycomb | 1024^2 | 50 | 2 | 8.9 | 0.97 | 2.6e-11 |
  | gapped honeycomb, spinful | 512^2 | 701 | 4 | 123.1 | 2.43 | 5.7e-12 |
  | Haldane, spinful | 512^2 | 701 | 4 | 123.8 | 2.82 | 2.6e-10 |
  | dimerized chain | 20000 | 701 | 2 | 0.70 | 0.21 | 1.8e-16 |

  The gain is about 24 at two bands and about 45 to 50 for the spinful
  copies, where the merge removes the duplicate energies. The 1024^2 mesh
  keeps 0.6M vertex sources out of 2M triangles, because the merge also
  catches the symmetry-related k-points. The difference with the old
  kernel is not a discretization error, it is rounding. Both kernels
  cancel terms of size 1/tolc^2 (tolc = 1e-2 delta) in the divided
  differences, and the old kernel's own comment bounds this at about
  eps delta/tolc^2. A reference in extended precision (numpy longdouble,
  same branches) gives the size of it: for the spinful Haldane model at
  delta = 1e-3 the old kernel is off by 3.2e-10, 2.3e-11 and 1.0e-10 of the
  largest entry at nk = 128, 256 and 512, and the new one by 5.0e-10,
  9.9e-11 and 2.1e-10. On the gapped honeycomb at nk = 1024 the two errors
  are 2.0e-11 and 2.8e-11. So the new path is up to 4 times noisier than the
  old one, and both are at the rounding level of the formula. Compensated
  (Kahan) summation in the evaluation changed nothing, so the error is in
  the terms and not in the accumulation. The new test asserts agreement to
  1e-10 of the largest entry at nk = 40, where the measured difference is
  about 1e-12.

  Binning the joint DOS on a fine energy grid was not built. It is
  approximate, with an error of order (bin/delta)^2, so agreement at 1e-10
  needs a bin of about 1e-5 delta. That is more points than there are
  sources, so it would not be faster than the exact route. What stays open:
  the cost is still linear in the number of frequencies times the number
  of sources (about 6.5 ms per frequency at 0.6M sources on 4 cores). A
  far-field expansion of the sources (the 1D fast multipole method for the
  Cauchy kernel) would remove that factor, at the price of a controlled
  truncation. This is not urgent: on the same 4 cores the bands,
  velocities and pair fields of the 1024^2 gapped honeycomb take 9.7 s,
  more than the 5.2 s of the triangle sum at 701 frequencies.

  The gain survives in larger unit cells. We take n x n supercells of the
  gapped honeycomb (sublattice imbalance 0.3, spinless), with and without a
  random onsite energy of width 0.2 t, at 201 frequencies, delta = 1e-3 t,
  on 4 cores. The random onsite energy removes the exact degeneracies, so
  no sources merge. Times are in seconds, "fields" is the diagonalization,
  the velocities and the pair fields.

  | cell | bands | nk | pairs | fields | old sum | new sum | max diff / max |
  |---|---|---|---|---|---|---|---|
  | 1x1 | 2 | 256 | 1 | 0.61 | 2.22 | 0.14 | 2e-12 |
  | 2x2 | 8 | 128 | 12 | 0.35 | 6.63 | 0.42 | 2e-9 |
  | 3x3 | 18 | 96 | 57 | 1.12 | 17.7 | 1.24 | 2e-10 |
  | 4x4 | 32 | 64 | 126 | 6.06 | 17.5 | 1.10 | 9e-10 |
  | 2x2, random | 8 | 128 | 16 | 0.35 | 8.96 | 1.22 | 5e-10 |
  | 3x3, random | 18 | 96 | 81 | 1.02 | 25.4 | 3.56 | 3e-10 |
  | 4x4, random | 32 | 64 | 256 | 5.39 | 35.7 | 4.88 | 4e-10 |

  So the new sum is 13 to 16 times faster in the clean cells and about 7
  times faster in the random ones, where it gains only from building the
  sources once. In the clean cells the pruning drops the pairs with zero
  velocity matrix elements (126 of 256 at 4x4). The number of stored pairs
  grows as (bands/2)^2 for an insulator, so in large cells the cost is the
  pairs and the fields, not the frequencies: at 32 bands the fields
  already cost as much as the sum.
- **Many bands.** The fields are ordered by band index, so they have kinks
  where bands cross. The triangle sum still converges (the crossings are a
  set of measure zero), but the QTCI rank grows. Not measured beyond two
  bands.
- **Metals.** The occupation factors jump at the Fermi surface within a
  width T. The vertex-mean weight is first order there, and in the qtci path
  the step costs rank. The Drude term is the same in all three
  integrations. Linear vertex weights (a Blochl-type correction) would make
  the interband part second order; not built.
- **Second harmonic generation and the shift current** (`conductivitytk/shg.py`)
  have the same structure with 1/(z - dE)^2 and two resonant shells. The
  same Hermite-Genocchi construction works with one more antiderivative.
  This is the most useful next step, because SHG spectra are usually
  over-broadened to converge the k-sum.
- **3D.** The Kubo module has no 3D branch (`current.derivative` has none).
  With one, tetrahedra need the third divided difference of
  psi_3, and the gain over a uniform nk^3 mesh should be larger still.
