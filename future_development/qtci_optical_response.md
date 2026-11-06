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

## The band window for large unit cells

In a cell of n orbitals the full calculation diagonalizes n bands at every
k-point, rotates three n x n velocity matrices, and stores about (n/2)^2
pairs. So the memory per k-point grows as n^2 and the time as n^3. At
n = 1084 the velocities alone take 56 MB per k-point, and a 24 x 24 mesh
needs 32 GB before a single pair is stored. Most of these bands do not
matter for the optics at low frequency, so the question is which bands we
can drop and what we lose.

The keyword is `ewindow` of `h.get_optical_conductivity()`, the same name
that the DOS routines use for an energy window around the Fermi level. A
number W keeps every band that comes inside (-W, W), a pair (emin, emax)
gives an asymmetric window. The same window can be given as band counts,
`nv` bands below and `nc` above the Fermi level, the names of the BSE and
the excitonic optics. It is off by default, and the default result does
not change. The code is in `conductivitytk/kubowindow.py`, the tests in
`tests/conductivity/test_optical_window.py`.

Only the bands of the window are diagonalized: scipy's `eigh` with
`subset_by_index` for a dense H(k), and a shift-invert solver around the
Fermi level above `limits.densedimension`. Below 200 orbitals the batched
numba `eigh` of all the bands is used and then cut, since it is batched
over k-points; the crossover with the subset solver was not measured. The
velocity is never built as an n x n matrix. The lattice-gauge
derivative acts on the window vectors through the sparse hoppings, at cost
nnz times nw, and the commutator i[H, r] becomes i(E_n - E_m)<n|r|m>, at
cost n nw^2. This is exact for eigenvectors, and the full-window test
checks it against the full path to 1e-10.

### The rule for the bands

The triangle and qtci integrations interpolate the fields by band index,
so a pair index must mean the same two bands at every vertex of a
triangle, and in every call of the qtci cache. A window by value does not
give this: in a metal, or in any cell with dispersive bands, the bands
inside (-W, W) change from one k-point to the next. So the window is a
fixed range of band indices in the sorted spectrum, the same at every
k-point, and the rule is this:

- with `ewindow`, every band that enters the window at some k-point of the
  mesh is kept at every k-point;
- with `nv`, `nc`, the range is [min N - nv, max N + nc), with N(k) the
  number of bands below the Fermi level, so in a metal every band that
  crosses the Fermi level is kept as well;
- the range does not cut a degenerate multiplet at its edges at a k-point
  where the window asks for the edge band, since there the state that is
  kept would be fixed by the eigensolver and not by the physics. Where the
  edge band is outside the window, a cut multiplet only touches pairs that
  the window drops anyway.

The range is estimated from all the eigenvalues on 16 k-points of the
mesh. Then the whole mesh is done with four probe bands below and four
above the range. If a probe band enters the window, or the edge cuts a
multiplet where it should not, the range grows. The probe bands are kept
when the growth fits inside them, and the mesh is done again only when it
does not. With
this rule the windowed calculation has the same labels and the same kinks
at band crossings as the full one. A window that holds all the bands is
then the full calculation, and the test checks this to rounding. The qtci
path fixes the range on its coarse 32 x 32 scan, and it warns if a later
k-point needs more bands. On the sparse path the solver does not give the
band index, so the bands are counted from the Fermi level instead. For an
insulator this is the same labelling. For a metal the labels change where
a band crosses the Fermi level, which spoils the interpolation in the
triangles cut by the Fermi surface. This has not been measured.

### What is exact and what is biased

Let us look at a pair that the window drops, with a nonzero occupation
factor. One band is below the Fermi level and one above, and one of the two
is outside (emin, emax). So its transition energy is larger than
min(-emin, emax). As a result Re sigma(omega), the absorption, is exact for
omega below the window, up to the Lorentzian tails of the dropped pairs,
of order delta/(dE - omega)^2. Im sigma is different. Every pair adds to it
a Kramers-Kronig term (omega - dE)/((omega - dE)^2 + delta^2), which decays
only as 1/dE. For a diagonal component the forward and reverse terms of a
dropped pair add up to about 2 omega/dE^2, so the bias of Im sigma is
linear in omega. The measurement shows exactly this: on the 10 x 10 cell
with W = 1.5 t, the bias of Im sigma_xx divided by omega is 1.47e-3,
1.48e-3, 1.50e-3 and 1.56e-3 at omega = 0.1, 0.3, 0.5 and 0.8 t.

### Measured against the full calculation

The cells are n x n supercells of the honeycomb lattice with a sublattice
imbalance of 0.3 t and a random onsite energy of width 0.4 t, spinless, so
no band is degenerate. The integration is the triangle one, at T = 0.01 t,
delta = 0.02 t, 151 frequencies on [0, 3] t, sigma_xx. The errors are the
largest deviation from the full calculation on the same mesh for omega
below W - 5 delta, divided by the largest Re sigma_xx. Times are wall times
of the whole call, peak memory is the maximum resident set of the process,
on 4 cores of one workstation that was shared with other jobs (load 10 to
19). The same runs an hour apart differed by up to 30% in time, so the
times are good to about that; the errors, the bands kept and the memory
do not change from run to run.

| cell | orbitals | nk | W (t) | bands kept | error Re | error Im | time (s) | memory (GB) |
|---|---|---|---|---|---|---|---|---|
| 4x4 | 32 | 96 | full | 32 | | | 18.0 | 2.27 |
| 4x4 | 32 | 96 | 1.0 | 8 | 1.5e-3 | 5.0e-2 | 5.3 | 0.50 |
| 4x4 | 32 | 96 | 1.5 | 18 | 9.8e-5 | 4.8e-3 | 8.1 | 0.97 |
| 4x4 | 32 | 96 | 2.0 | 24 | 1.3e-5 | 7.6e-4 | 11.3 | 1.44 |
| 4x4 | 32 | 96 | 2.5 | 30 | 9.3e-7 | 5.1e-5 | 14.0 | 2.03 |
| 7x7 | 98 | 48 | full | 98 | | | 35.4 | 4.76 |
| 7x7 | 98 | 48 | 1.0 | 24 | 1.7e-3 | 5.4e-2 | 11.7 | 0.63 |
| 7x7 | 98 | 48 | 1.5 | 56 | 8.4e-5 | 4.2e-3 | 18.6 | 1.82 |
| 7x7 | 98 | 48 | 2.0 | 71 | 1.8e-5 | 1.1e-3 | 24.8 | 2.70 |
| 7x7 | 98 | 48 | 2.5 | 86 | 2.8e-6 | 1.6e-4 | 40.5 | 3.75 |
| 10x10 | 200 | 32 | full | 200 | | | 78.5 | 8.45 |
| 10x10 | 200 | 32 | 1.0 | 47 | 2.0e-3 | 6.4e-2 | 15.8 | 0.83 |
| 10x10 | 200 | 32 | 1.5 | 104 | 1.5e-4 | 7.0e-3 | 26.4 | 2.60 |
| 10x10 | 200 | 32 | 2.0 | 142 | 2.1e-5 | 1.2e-3 | 37.7 | 4.46 |
| 10x10 | 200 | 32 | 2.5 | 175 | 2.5e-6 | 1.5e-4 | 50.2 | 6.56 |

What we read from it: the error of Re sigma falls by about a factor 10 for
every 0.5 t of window, and it does not depend on the cell size, since it
is set by the distance between omega and the dropped transitions. It is
almost flat in omega (on the 10 x 10 cell at W = 1 t the deviation is
-4.3e-4 at omega = 0.1 t and -6.4e-4 at 0.8 t, in units of e^2/hbar), as
the tails delta/(dE - omega)^2 of transitions far above omega should be,
so it should grow with delta; we did not vary delta to check this. The
error of Im sigma is 20 to 50 times larger at the same window and falls at
the same rate. The memory follows the bands kept: a window of 1 t takes a
tenth of the memory of the full 10 x 10 calculation. The time gains less,
because on the honeycomb lattice a window of 1.5 t already keeps half of
the bands, and the time of the triangle sum follows the number of pairs.
Split on the 10 x 10 cell (two runs each): the bands and velocities take
21 to 25 s for the full calculation, 11.5 to 11.8 s at W = 1 t and 16 to
17 s at W = 2.5 t, and the whole call 68 to 72, 14 to 18 and 52 to 56 s.
So the window buys memory in every case, and time in proportion to the
bands it drops.

A first version of the range search did the mesh again for every band it
added. On the 10 x 10 cell at W = 2.5 t this doubled the time of the
bands, and on the twisted bilayer below it ran away. There every band has
a valley partner within 1e-7 t at every k-point, and the rule "never cut a
degenerate multiplet" kept adding the next pair: a window of 0.8 t grew to
536 bands in 6900 s. Two changes fixed this. The edge is kept whole only
at the k-points where the window asks for the edge band; where the edge
band is outside the window, its pairs are dropped pairs anyway. And four
probe bands are computed beyond each edge, and they are kept when the
growth fits inside them. The same window now keeps 162 bands in 19 s.

### A cell the full path cannot do

We take the twisted bilayer graphene of `specialhamiltonian.tbg(n=9)`, a
twist of 3.48 degrees with 1084 orbitals per cell, spinless, interlayer
hopping 0.12 t, at charge neutrality, so it is a Dirac semimetal. The
integration is the triangle one at T = 0.01 t, delta = 0.01 t, 151
frequencies on [0, 0.5] t, sigma_xx. The interlayer hopping gives a
velocity along z through i[H, r_z], so the pair fields keep nine Cartesian
components here.

The full path stores the velocities of all 1084 bands, 56 MB per k-point,
and about 294,000 interband pairs, (1084/2)^2. At nk = 6 (36 k-points) it takes 205 s
and 16.3 GB. Both grow with the number of k-points, so nk = 24 would need
about 260 GB, and on this machine the full path stops at nk = 6. At nk = 6
the windowed path can be compared with it:

| W (t) | bands kept | error Re | error Im | time (s) | memory (GB) |
|---|---|---|---|---|---|
| full | 1084 | | | 205 | 16.33 |
| 0.3 | 24 | 5.5e-3 | 1.1e-1 | 12.8 | 0.40 |
| 0.5 | 60 | 3.3e-3 | 1.2e-1 | 13.2 | 0.48 |
| 0.8 | 162 | 1.4e-3 | 6.3e-2 | 18.6 | 0.94 |

The errors are taken for omega below W - 5 delta, relative to the largest
Re sigma_xx, the Drude peak at omega = 0 (0.67 e^2/hbar). They are larger
than on the honeycomb cells and fall more slowly with the window, because
a semimetal has no gap: the dropped transitions start right above W, not
above W plus a gap. The Re error is again flat in omega (4.1e-3, 2.3e-3 and
1.2e-3 at omega = 0 for the three windows), so it is the sum of the
Lorentzian tails and not an effect of the edge of the window.

Then the windowed path at W = 0.5 t, which keeps 61 bands, as a function
of the mesh. The difference is the largest deviation of Re sigma_xx from
the nk = 36 result for omega below 0.45 t, relative to its largest value:

| nk | k-points | time (s) | memory (GB) | difference to nk = 36 |
|---|---|---|---|---|
| 6 | 36 | 13.2 | 0.48 | 3.1e-1 |
| 12 | 144 | 44.1 | 0.66 | 6.6e-2 |
| 18 | 324 | 97.4 | 1.01 | 3.3e-2 |
| 24 | 576 | 167 | 1.49 | 1.5e-2 |
| 36 | 1296 | 379 | 2.86 | |

From nk = 12 to 24 the difference falls by 4.4, so the result converges at
second order in 1/nk, as on the small cells. At nk = 24 and 36, Re sigma_xx
is 0.2146 and 0.2155 at omega = 0.05 t, 0.4633 and 0.4643 at the peak at
0.2 t, and 0.2549 and 0.2510 at 0.4 t, in units of e^2/hbar. The time per
k-point is about 0.3 s for the whole call; we did not split it here. The
subset eigensolver still reduces the whole 1084 x 1084 matrix to
tridiagonal form, so the window does not remove that n^3; it removes the n^2 per k-point of the memory and the
(n/2)^2 of the pairs.

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
  already cost as much as the sum. The section on the band window for
  large unit cells is the answer to this.
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
- **The band window, what it does not do yet.** The bias of Im sigma is
  linear in omega and smooth, so in principle one number per component
  removes most of it. That number is the weight of the dropped pairs
  divided by their transition energy squared, and it needs the bands that
  were dropped, so it is not computed. The pair fields keep every
  Cartesian component with a nonzero velocity (nine in a bilayer, where
  the interlayer hopping gives a velocity along z), even when only one is
  asked for; keeping only the requested ones would cut the memory of the
  fields by that factor. The labels of the sparse path in a metal are not
  tested beyond the dense-sparse comparison on an insulator.
- **Real-space Kubo-Greenwood with KPM, the next step.** The window takes
  the k-space path to a few thousand orbitals per cell. For 10^4 to 10^6
  sites (a large disordered sample, a moire cell at small angle) the next
  step is a real-space Kubo-Greenwood path with the Chebyshev expansion of
  `kpmtk/`, in the spirit of KITE. It has no k-mesh and no band labels,
  and it will be cross-checked against this windowed path on the cells
  where both run.
- **3D.** The Kubo module has no 3D branch (`current.derivative` has none).
  With one, tetrahedra need the third divided difference of
  psi_3, and the gain over a uniform nk^3 mesh should be larger still.
