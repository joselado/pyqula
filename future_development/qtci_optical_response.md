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
the weights, one interpolation each, sharing one cache of diagonalizations;
the code now runs one interpolation of all of them, see the section on one
interpolation of all the fields)
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

## One interpolation of all the fields (7 November 2026)

The qtci path first ran one cross interpolation per field: dE of every
pair, and the real and imaginary part of every weight, all sharing one
cache of diagonalizations. Now it runs one interpolation of all of them at
once (`qtcitk/meshfields.qtci_stacked_fields`), with the field index as one
more tensor site, as the mean-field density matrix already does. The
question is which form needs fewer diagonalizations at the same error, and
whether it costs wall time.

What changes is this. One TCI per field puts its pivots where that field
needs them, and two fields share a diagonalization only when they happen to
pick the same k-point. The stacked TCI shares its pivots in k between all
the fields, so fields with the same kinks (the K points, the band edge) ask
for the same k-points. Its tolerance is relative to the largest value of
any field, not of each field, and its rank is roughly that of all the
fields together, so it grows with their number.

The models are the gapped honeycomb lattice (sublattice imbalance 0.3, two
bands, five fields), the three-band MoS2 model of `qtci_shg.md`
(`three_band_tmdc`, spinless, ten fields) and a 2x2 supercell of the
gapped honeycomb lattice with a random onsite energy of width 0.2 t (eight
bands, 80 fields). T = 1e-3, delta = 1e-3, tolerance 1e-6, 48 frequencies
on [0.65, 3] t, 28 on [0.8, 3.5] eV and 28 on [0.3, 3] t. The error is
max|sigma_qtci - sigma_triangle| / max|sigma_triangle| over the frequencies
and the components, with the triangle on the same 2^R mesh, so it is only
the interpolation error. As a reference for what "equal error" means, the
triangle itself moves by 2.9e-3 and 7.9e-4 from nk = 256 to 512 and from
512 to 1024 on the honeycomb lattice, by 2.1e-3 and 4.6e-4 on MoS2, and by
2.6e-2 and 8.0e-3 from 128 to 256 and from 256 to 512 on the supercell. So
every interpolation error below is at least 20 times smaller than the error
of the mesh, and both forms give the same spectrum at the level the mesh
allows. The rank is the largest rank of one field for the per-field form,
and the largest bond dimension for the stacked one. Wall times of the whole
call on two cores of a workstation that other jobs kept busy (load 5 to
10), so they are good to about 20%.

| model | R | form | diagonalizations (share) | rank | error | time (s) |
|---|---|---|---|---|---|---|
| gapped honeycomb | 8 | triangle | 65.5k (1) | | | 0.7 |
| gapped honeycomb | 8 | per field | 51.2k (0.78) | 65 | 1.3e-5 | 4.0 |
| gapped honeycomb | 8 | stacked | 27.9k (0.43) | 100 | 1.0e-5 | 2.0 |
| gapped honeycomb | 9 | triangle | 262k (1) | | | 2.7 |
| gapped honeycomb | 9 | per field | 109k (0.41) | 67 | 3.7e-5 | 8.1 |
| gapped honeycomb | 9 | stacked | 34.2k (0.13) | 101 | 1.3e-5 | 4.4 |
| gapped honeycomb | 10 | triangle | 1.05M (1) | | | 11.2 |
| gapped honeycomb | 10 | per field | 150k (0.14) | 66 | 1.9e-5 | 12.8 |
| gapped honeycomb | 10 | stacked | 43.7k (0.04) | 102 | 1.5e-5 | 8.2 |
| MoS2 | 8 | triangle | 65.5k (1) | | | 2.0 |
| MoS2 | 8 | per field | 65.4k (1.00) | 114 | 3.2e-5 | 25.0 |
| MoS2 | 8 | stacked | 62.1k (0.95) | 214 | 2.2e-5 | 21.6 |
| MoS2 | 9 | triangle | 262k (1) | | | 8.9 |
| MoS2 | 9 | per field | 229k (0.88) | 118 | 1.6e-5 | 46.7 |
| MoS2 | 9 | stacked | 145k (0.55) | 220 | 3.4e-5 | 24.4 |
| supercell 2x2 | 7 | triangle | 16.4k (1) | | | 1.1 |
| supercell 2x2 | 7 | per field | 16.4k (1) | 119 | 6.6e-7 | 50.2 |
| supercell 2x2 | 7 | stacked, cap 400 | 16.4k (1) | 400 | 5.2e-5 | 25.5 |
| supercell 2x2 | 7 | stacked, cap 800 | 16.4k (1) | 460 | 1.2e-5 | 44.3 |
| supercell 2x2 | 8 | triangle | 65.5k (1) | | | 5.0 |
| supercell 2x2 | 8 | per field | 65.5k (1) | 161 | 4.8e-6 | 192 |
| supercell 2x2 | 8 | stacked, cap 400 | 65.2k (0.99) | 400 | 2.2e-4 | 46.8 |
| supercell 2x2 | 8 | stacked, cap 800 | 65.4k (1.00) | 572 | 2.0e-5 | 102 |
| supercell 2x2 | 7 | stacked, cap 800, tolerance 1e-7 | 16.4k (1) | 525 | 1.6e-6 | 74.8 |
| supercell 2x2 | 8 | stacked, cap 800, tolerance 1e-7 | 65.5k (1.00) | 700 | 1.1e-6 | 120 |

What we read from it:

- With few fields the stacked form wins. On the gapped honeycomb lattice it
  needs 3.2 times fewer diagonalizations at R = 9 and 3.4 times fewer at
  R = 10 (43.7k against 150k), and its count grows slowly with R (28k, 34k,
  44k), while the per-field count still grows by 1.4 to 2.1 per step. On MoS2
  it needs 1.6 times fewer at R = 9. The errors are the same within a
  factor 3 either way, and the stacked call is 1.2 to 2 times faster, because
  it runs one interpolation instead of five or ten.
- On the disordered supercell neither form saves anything: both diagonalize
  the whole mesh at R = 7 and 8. The random onsite energy turns every band
  crossing into a narrow anticrossing, a kink of the fields ordered by band
  index, and the interpolation has to resolve every one of them. The plain
  triangle on the same mesh takes 5 s, against 100 to 200 s for qtci, so
  there the triangle is the right choice.
- The stacked rank grows with the number of fields. At the cap of 400 of
  one field, the stacked interpolation of the 80 supercell fields stopped
  at the cap, before its tolerance, with an error of 2.2e-4 at R = 8, 47
  times the per-field one. With a cap of 800 it reaches rank 460 and 572
  and an error of 1.2e-5 and 2.0e-5, which is the error it gives on the
  other models at this tolerance, in half the time of the per-field form
  at R = 8. The per-field error on this model is smaller, 6.6e-7 and
  4.8e-6, because each of the 80 fields is converged relative to its own
  maximum. To compare at equal error, the stacked form at a tolerance of
  1e-7 gives 1.6e-6 and 1.1e-6, as small as the per-field
  form, in 75 s against 50 s at R = 7 and 120 s against 192 s at R = 8. So
  at equal error the two forms are within a factor 1.6 in time either way
  on this model.

So the qtci optical conductivity uses the stacked form now, with a rank cap
of 800 (`meshfields.stacked_maxbonddim`, shared with the qtci LDOS), and
the per-field form stays in `meshfields.qtci_fields` for the qtci DOS and
the second-harmonic conductivity. There is no keyword to choose: on no model
did the stacked form lose by more than a factor 2 in diagonalizations or
in time once the cap was raised, and at equal tolerance its error stays at
the level the tolerance asks for. The test that compares qtci with the
triangle on the same mesh passes unchanged.

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

## 3D crystals (7 November 2026)

The question was whether the optical conductivity of a 3D crystal can use
the same machinery, and whether the tetrahedra gain over the uniform mesh
at small delta as the triangles do in 2D.

What was built. The Kubo velocities of `conductivitytk/kubo.py` now exist in
3D, in the atomic gauge, with the intracell bonds included as in 2D:
`_cell_factors` gives the three derivative orders, the Jacobian a_i/(2 pi)
and the cell volume, and `current.derivative` has a 3D branch, so the
single-k velocity of `operators.get_velocity` works too. The grid sum needed
nothing else. `integration="triangle"` in 3D uses the six tetrahedra per
cube of `dostk/simplexdos.py`, which share the (0,0,0)-(1,1,1) diagonal of
the reduced mesh, with the vertex average of the weights and the
Hermite-Genocchi simplex average of the resolvent, 3! psi_3[e_0..e_3]. The
direct kernel of `kubosimplex.py` already took any number of vertices,
since its divided differences are those of `simplexdos._dd`. The source
kernel was first written for segments and triangles only, so 3D used the
direct kernel; it now takes tetrahedra too and is the default in 3D, see
"The source kernel on tetrahedra" below. `integration="qtci"` in 3D
raises NotImplementedError, from `kubo.py`, so the qtci path itself was not
touched. In 3D sigma is a conductivity, in units of e^2/(hbar a).

What was checked (`tests/conductivity/test_optical_3d.py`). The band velocity
of the simple cubic band at a generic k is the finite difference of the
band, to 1e-8. The whole velocity matrix of an eight-site cubic cell with a
staggered onsite, intracell bonds included, is the sparse full-bond-vector
construction of `kpmkubo.py`, an independent code, to 1e-12, in the
single-k and in the batched path. A stack of decoupled Haldane layers with
spacing c gives c sigma_3D = sigma_2D on the grid, to 1e-12, and nothing
along z. The f-sum rule holds: on a 3D insulator at nk = 24 the frequency
integral of Re sigma is pi W, with W from the second k-derivative of H, to
1e-3 in the plane and 1e-4 along z (the difference is the mesh mean of the
band curvature, which falls off exponentially with nk). The tetrahedra keep
the spectral weight of the mesh: their frequency integral is the one of the
grid on the same mesh, to 1e-4.

Decoupled layers and the tetrahedra. For a field that does not depend on
k3, the three tetrahedra above a triangle of the in-plane mesh fill the
prism on it, and that triangle is cut along (0,0)-(1,1). This is the 2D
mesh of the same layer in the basis (a1,-a2), not in the basis (a1,a2), whose
triangles are cut along (1,0)-(0,1). So the transition energies are
interpolated exactly as in 2D in the basis (a1,-a2). The weights are not:
a tetrahedron counts one of its projected vertices twice in its vertex
average. As a result c sigma_3D and the 2D result in the basis (a1,-a2)
differ at second order, by 8e-3, 2.7e-3 and 7.3e-4 relative at nk = 8, 16
and 32 (Haldane, delta = 0.1, T = 0.05). The anomalous Hall conductivity,
whose weights are exact linear integrals, does agree to rounding (see
`triangle_anomalous_hall.md`). We kept the decomposition of `simplexdos.py`:
cutting the cubes along (1,0,0)-(0,1,1) instead makes the tetrahedra project
onto the triangles of the basis (a1,a2), but it lengthens the diagonal of
the face-centered cubic mesh, and the 3D Green's function test of
`tests/green/test_green_simplex.py` then misses its bound (2.3e-2 against
2e-2 at nk = 32).

The measurement. The model is the Haldane layer (t2 = 0.1, staggered onsite
0.2) stacked along z with spacing c = 2 and the interlayer hopping
diag(0.05,-0.05), a mass that depends on kz. It is an insulator with a gap
from -0.15 to 0.15 and C = 1 at every kz. We take 40 frequencies from 0.1 t
to 4 t, delta = 1e-2 t, T = 1e-3 t, the interband part only, and four cores.
The reference is the Richardson extrapolation of the tetrahedra at nk = 64
and 96 (the tetrahedra at nk = 96 are within 6e-3, 1.1e-2 and 7e-3 of it for
xx, xy and zz). Errors are relative to the largest value of each component,
the largest over the frequencies and, in brackets, the median.

| integration | nk | k-points | time (s) | xx | xy | zz |
|---|---|---|---|---|---|---|
| tetrahedra | 16 | 4096 | 0.4 | 0.10 (2.3e-2) | 0.16 (4.5e-2) | 0.21 (2.1e-2) |
| tetrahedra | 24 | 13824 | 1.2 | 5.0e-2 (1.2e-2) | 8.7e-2 (2.3e-2) | 0.12 (9.8e-3) |
| tetrahedra | 32 | 32768 | 2.7 | 3.7e-2 (7.7e-3) | 6.4e-2 (1.3e-2) | 6.9e-2 (4.9e-3) |
| tetrahedra | 48 | 110592 | 8.7 | 2.0e-2 (3.1e-3) | 3.8e-2 (5.7e-3) | 3.1e-2 (2.1e-3) |
| tetrahedra | 64 | 262144 | 18 | 1.3e-2 (1.8e-3) | 2.6e-2 (3.1e-3) | 1.6e-2 (1.4e-3) |
| tetrahedra | 96 | 884736 | 50 | 5.9e-3 (8.2e-4) | 1.1e-2 (1.4e-3) | 7.0e-3 (6.3e-4) |
| grid | 32 | 32768 | 0.35 | 0.98 (5.2e-2) | 0.92 (6.7e-2) | 1.8 (0.13) |
| grid | 64 | 262144 | 3.1 | 0.47 (1.4e-2) | 0.45 (1.9e-2) | 0.70 (4.3e-2) |
| grid | 96 | 884736 | 10 | 0.20 (5.9e-3) | 0.19 (9.8e-3) | 0.40 (2.7e-2) |
| grid | 128 | 2097152 | 24 | 0.13 (5.4e-3) | 0.12 (6.3e-3) | 0.23 (1.8e-2) |
| grid | 160 | 4096000 | 47 | 8.0e-2 (3.4e-3) | 7.7e-2 (4.7e-3) | 0.15 (1.3e-2) |

What this means. The largest error of the tetrahedra falls at second order,
and it sits at the absorption edge, where the transition energy has its
minimum and the linear interpolation is worst, as in 2D. At equal largest
error the tetrahedra need about 100 times fewer k-points: 8e-2 takes nk = 24
(1.4e4 points, 1.2 s) with the tetrahedra and nk = 160 (4.1e6 points, 47 s)
with the grid, and 1e-2 is out of reach of the grid in memory (it extrapolates
to nk of about 450, 9e7 points), while the tetrahedra reach it at nk = 64 to
96. Along z, where the velocity is small and the grid error is largest, the
gain is the largest. The median error, away from the edge, gains less: the
grid at nk = 160 is about the tetrahedra at nk = 24 (along z) to 48 (in the
plane), 300 to 40 times fewer points. The times of this table are those of
the direct kernel, about 1.7 microseconds per k-point and frequency for the
one band pair of this model on four cores, against 0.3 for the grid, so the
gain in time was about 40 at a largest error of 8e-2, shrinking as the
number of frequencies grows. The source kernel below brings it to about 130
at 40 frequencies and 80 at 400. With many bands the diagonalization
dominates again and the gain in k-points is a gain in time.

### The source kernel on tetrahedra (7 November 2026)

The question was whether the source kernel, which gave 10 to 50 in 2D,
also gains in 3D, and at what precision.

What was built. The branch of a simplex is now its Newton table, read from
the top, for any number of vertices. The divided difference over the sorted
nodes of spread s is the difference of the two divided differences with one
node less, divided by s, and a cluster of nodes with s < delta is not split
further but kept as one Taylor series point for psi_K over it. So one code
path (`_simplex_sources` and `_point_value` in `kubosimplex.py`) takes
segments, triangles and tetrahedra, and two, three or four equal nodes, or
two equal pairs, end in a point whose series stops at its first term. The
series over four nodes converges at a rate of at least 3/4 on resonance, so
it runs up to about 130 terms there and a few far from it.

This alone is not enough in 3D. The vertex form of a simplex is a sum of
large terms that cancel, and at a frequency far from the simplex it loses
eps (|x|/s)^3 of a tetrahedron, against (|x|/s)^2 of a triangle. On random
tetrahedra with energies in [0.2, 3] t and frequencies in [0, 4] t, the
plain source kernel lost up to 2e-9 of the peak of a tetrahedron at
delta = 1e-2 t, 4e-7 at delta = 1e-3 t, and more than the whole value at
delta = 1e-6 t for a spread of a few delta (mpmath at 80 digits confirmed
that the direct kernel is the exact one there). A rule that does not see
the frequency cannot avoid this, so the sources now see the frequency
window, and there are two new kinds of source.

- A simplex kept whole. For every simplex and pair we bound the rounding
  error of its vertex form over the window (`_vertex_error`: eps times the
  sum over the leaves of the Newton table of the coefficient times the
  largest |psi_K| on the window). Above 3e-11 of the peak of the simplex,
  the simplex is one point source evaluated per frequency with
  `simplexdos._dd`, the rule of the direct kernel.
- A bin of far simplices. A simplex whose nodes lie within half the
  distance from the center C of its energy bin to the window (512 bins
  over the range of +-dE) is expanded around C,
  K! psi_K[e] = sum_k K! k!/(k+K)! h_k(e - C) (z - C)^-(k+1),
  which converges at a rate 1/2 or better at every frequency of the
  window and has no cancellation. The bin keeps the sum of these moments
  to order 60, so it costs 60 terms per frequency whatever the number of
  simplices in it. This takes all the reverse terms at -dE of a positive
  window, and the transitions far above it.

The build runs in parallel over chunks of simplices; the vertex
coefficients are added in one serial pass, since the simplices that share
a vertex add to one entry.

The precision (`tests/conductivity/test_tetrahedron_sources.py`). We take
random tetrahedra, generic, and with two, three or four nodes or two pairs
exactly equal, or within 1e-14, 1e-9, 1e-5, 0.3 delta, delta and 3 delta of
each other, at delta = 1e-2, 1e-3 and 1e-6, on frequencies that include the
node energies, so that the peak is sampled; and the same tetrahedra at
negative energies with the window [0, 4] t, where most of them go to the
bins. The two kernels agree to 1e-11 of the peak of each tetrahedron
(of its largest value on the window, for the negative ones), except at 3
entries out of about 3.5e4 at delta = 1e-2, and at a few far ones, where
they differ by up to 2.8e-11. There mpmath at 150 digits shows that the
source kernel is exact to 2e-16 and that the difference is the direct
kernel's own: its exact branch takes spreads up to 20 times |c - z|, and
near resonance for a tetrahedron of spread about delta, or at the far end
of the window for a wide one, this loses up to 3e-11. So the floor is
1e-11 against mpmath at those entries, and 1e-11 against the direct kernel
at all the others. A bound of 1e-10 instead of 3e-11 left 1.5e-11 in the
source kernel itself. On the layered model below the full sums of the two
kernels agree to 2.6e-13, 7.6e-13 and 2.6e-12 of the largest entry at
nk = 24, 48 and 64.

What it costs. A simplex kept whole costs as much per frequency as in the
direct kernel (since the change in "The 2D cost of the tetrahedron
kernel" below, only at the frequencies near it), so the gain is set by
how many are left. Fraction of the
simplex pairs kept whole (a pair is a tetrahedron with the transition at
dE, or the reverse one at -dE), on the layered model with the window
[0.1, 4] t:

| nk | delta | without bins | with bins |
|---|---|---|---|
| 24 | 1e-2 | 23% | 4% |
| 24 | 1e-3 | 55% | 17% |
| 48 | 1e-2 | 35% | 7% |
| 48 | 1e-3 | 58% | 20% |
| 64 | 1e-2 | 44% | 9% |

The tetrahedra kept whole are forward terms inside the window, 8% to 13%
of them at delta = 1e-2 (17% at nk = 64) and 34% to 39% at
delta = 1e-3. The fraction grows
with nk, since the spreads shrink like 1/nk while the loss grows like
(|x|/s)^3, and it grows as delta falls, since the bound compares with the
peak, about 1/delta for a narrow tetrahedron.

The measurement. The layered Haldane model of the 3D table above,
delta = 1e-2 t, T = 1e-3 t, the interband part, frequencies on [0.1, 4] t,
four cores of one workstation with other jobs on its other cores. Both
kernels run on the same pair fields, interleaved in one session; "fields"
is the diagonalization, the velocities and the pair fields, common to
both, so the time of a calculation is fields plus kernel. Times in
seconds, the smaller of two runs (one run of the direct kernel at nk = 64
and 400 frequencies).

| nk | k-points | fields | frequencies | direct | sources | gain |
|---|---|---|---|---|---|---|
| 24 | 13824 | 0.17 | 40 | 1.06 | 0.18 | 6 |
| 24 | 13824 | 0.17 | 400 | 10.6 | 0.75 | 14 |
| 48 | 110592 | 1.3 | 40 | 7.6 | 1.46 | 5.2 |
| 48 | 110592 | 1.3 | 400 | 75.7 | 7.7 | 9.8 |
| 64 | 262144 | 3.0 | 40 | 15.7 | 3.85 | 4.1 |
| 64 | 262144 | 3.0 | 400 | 155.7 | 21.8 | 7.1 |

The grid at the same largest error. In the 3D table above, the tetrahedra
at nk = 24 have a largest error of 5.0e-2, 8.7e-2 and 0.12 for xx, xy and
zz, and the grid needs nk = 160 (4.1e6 points) for 8.0e-2, 7.7e-2 and 0.15.
In the same session the whole public call with the tetrahedra at nk = 24
took 0.36 s at 40 frequencies and 0.93 s at 400, and the grid at nk = 160
took 45.8 s and 71.6 s. The finer tetrahedron meshes have no grid
counterpart that fits in memory, as said above.

What this means. With the source kernel the tetrahedra pay much less per
frequency: about 0.12 to 0.19 microseconds per k-point and frequency on
four cores at nk = 24 to 64, plus a build of 0.1 to 1.9 s, against 1.5
to 1.9 for the direct kernel. So the gain grows with the number of frequencies,
14 at nk = 24 and 400 frequencies, and falls slowly with nk, 7 at nk = 64,
because more tetrahedra are kept whole on finer meshes. Without the bins
the same calls at 400 frequencies took 3.3, 30 and 59 s, a gain of 2.7 to
3.8 against the direct kernel of that session: the bins are what makes the source kernel pay in 3D. Against the grid
at the same largest error the tetrahedra are now about 130 times faster at
40 frequencies and 80 times at 400, instead of 40 with the direct kernel.
In 2D the tetrahedron change first made the source kernel slower, and
the next subsection gives that time back.

### The 2D cost of the tetrahedron kernel (7 November 2026)

The question was why the 2D source kernel became slower with the change
above, and whether the rule that keeps a simplex whole, written for the
cubic loss of the tetrahedra, is needed in 2D at all. The first numbers,
0.59 s against 0.57 s on the gapped honeycomb and 2.24 s against 1.83 s on
the spinful Haldane model, came from 14 threads of a loaded workstation.

The measurement. Three cores of one workstation, numba and BLAS threads at
3, with other jobs sharing the cores, so only the ratios inside one row
mean something. The three kernels are loaded in one process and called in
turn, ten times each on the same saved pair fields: the kernel of 671795e,
before the tetrahedra, the kernel of aacdd26, and the kernel now. The
models are those of "Precision of the two kernels": the spinful Haldane
model (t2 = 0.2, Fermi level at 0.3 t) and the gapped honeycomb
(imbalance 0.3), T = 0.01 t, delta = 1e-3 t, 141 frequencies on [0, 7] t.
Times of the kernel in seconds, the smallest of the ten calls and, in
brackets, the median.

| model | mesh | 671795e | aacdd26 | now |
|---|---|---|---|---|
| spinful Haldane | 512^2 | 2.86 (3.01) | 3.66 (3.82) | 2.77 (2.94) |
| spinful Haldane | 256^2 | 0.46 (0.59) | 0.64 (0.73) | 0.53 (0.56) |
| gapped honeycomb | 512^2 | 0.79 (1.00) | 0.90 (1.06) | 0.80 (0.90) |

Separate processes in alternation, four passes of two calls, gave the
same picture with more noise: 2.33, 3.07 and 2.53 s for the spinful
Haldane model at 512^2 in one hour, 2.82, 3.60 and 2.51 s in another.

Where the time went. The kernel of aacdd26 lost its time in the build of
the sources, not in the sum per frequency. On the spinful Haldane model at
512^2 its build took about 1.5 s against 0.8 s, of which 0.53 s were the
moments of the bins, 60 terms for every one of the 2.1M binned triangle
pairs (all the reverse terms at -dE), and 0.28 s the serial pass that adds
the vertex coefficients. Its sum per frequency was in fact cheaper, about
1.3 s against 1.5 s, since the bins take the reverse terms out of the
vertex sources (0.14M of them against 0.29M). The triangles kept whole cost
about 0.17 s of that sum: the rule keeps 336, 5592 and 68112 forward
triangle pairs whole at nk = 128, 256 and 512 (0.3%, 1.1% and 3.2% of
them), 43842 point sources after the merge at 512^2.

Is the rule needed in 2D? With the rule off (a bound of 1e300 instead of
3e-11), the spinful Haldane model is within 1.5e-12, 1.3e-12 and 8.6e-13
of the largest entry of the longdouble reference at nk = 128, 256 and 512,
against 7.0e-13, 6.6e-13 and 7.5e-13 with the rule, so at delta = 1e-3 it
buys a factor of two at most. At delta = 1e-5 it is needed: without it the
error is 2.4e-10 at nk = 128 and 2.3e-11 at 256, with it 2.8e-12 and
1.1e-12, and a bound of 1e-10 already gives 6.3e-12 and 8.9e-12. At
delta = 1e-2 the error is 9e-14 or less either way. Bounds between 3e-11
and 1e-10 at delta = 1e-3 do not move the error in one direction (2.3e-12
at nk = 256 for 6e-11), since a triangle kept whole also breaks the
cancellation of the vertex coefficients it shares with its neighbors. So
the rule stays as it is in 2D, and the triangles it keeps were made
cheaper instead. `tests/conductivity/test_source_kernel_2d_precision.py`
checks the kernel against the longdouble sum, now in the test itself, at
nk = 128 on both models: to 2e-12 of the largest entry at delta = 1e-3,
and to 1e-11 at delta = 1e-5, which fails without the rule.

What was changed (`kubosimplex.py`).

- A binned simplex adds only the orders it needs. Its term of order k is
  below r^k of the first, with r its own rate over the window, so it
  stops at r^k < 1e-18, which is what 60 terms give at the largest rate
  1/2. The reverse triangles stop after 8 terms on average.
- The passes over the simplices skip the pairs whose branch belongs to
  another pass, and the vertex pass runs in parallel over the band pairs,
  whose entries are their own. These two leave the output bitwise the
  same, and it still does not depend on the number of threads.
- A simplex kept whole is close to some frequency of the window but far
  from most of them. It now also adds its moments, to 60 terms, to a
  second set of moments of its energy bin, and the bin keeps the largest
  distance R of these nodes from its center C. At a frequency with
  (w - C)^2 + delta^2 >= (R/0.5)^2 they all converge at a rate 1/2 or
  better and the bin gives their sum; at the others each one goes through
  `simplexdos._dd` as before. One decision per bin and frequency serves
  both loops, so no simplex is counted twice or left out. On the spinful
  Haldane model at 512^2 the kept-whole triangles sit in 17 bins, and
  0.7% of the (simplex, frequency) pairs are left to `_dd`. This moves the
  result by 1.6e-14 of the largest entry, and the errors against the
  longdouble reference above are those of aacdd26 to two digits.

3D. On the layered Haldane model of the table above at nk = 24
(interband part, T = 1e-3 t, delta = 1e-2 t, frequencies on [0.1, 4] t,
same three cores), the source kernel takes 0.20 s at 40 frequencies and
0.73 s at 400, against 0.27 s and 1.27 s for aacdd26, the smallest of six
calls in separate processes, and the result moves by 8e-15 of the largest
entry. The tetrahedra kept whole are wider than the triangles, a median
distance of 0.23 t from the center of their bin, so a third of their
(simplex, frequency) pairs are still near and left to `_dd`.

What was still open then. In 3D these near pairs were the cost. One wide
tetrahedron makes R large for its whole bin, so a set of moments per class
of distance inside a bin would let the narrow ones go to the moments
earlier. The other route is to keep only the close edge of such a
tetrahedron per frequency, so that the rest cancels like (|x|/s)^2 and the
edges, shared by four to six tetrahedra, merge. The second route was
built, together with a far-field sum of all the point sources, see the
next subsection.

### Split tetrahedra and point sources far from resonance (7 November 2026)

The question was how to make the tetrahedra kept whole cheaper in 3D
without losing precision, and where the rest of the time per frequency
goes.

Why the tetrahedra were kept whole. On the layered model at nk = 24 the
tetrahedra kept whole, 7000 point sources after the merge, are wide
(median spread 0.34 t) but have a close pair of nodes: 99.5% of them have a gap below 3 delta. The loss of the vertex
form comes from that pair, not from the whole tetrahedron. Read down the
Newton table: the top split divides by the wide spread s and cancels little,
while the edge of spread s' of a few delta, far from resonance, cancels like
|x|/s'. So we keep in vertex form what is safe and evaluate per frequency
only the nodes of the table that cancel too much.

The rule (`_decompose` in `kubosimplex.py`). A tetrahedron whose vertex form
loses more than `_tol` = 3e-11 of its peak (the old rule, unchanged) gets
a rounding budget, a fraction `_budget` of `_tol` of its peak. Top down, a
node of the table whose vertex form stays within its budget stays in vertex
form; a node whose two children, evaluated per frequency, would each stay
within half of its budget is split, with half of the budget to each child;
any other node is evaluated per frequency with `simplexdos._dd`. The
budgets of the leaves add up to that of the tetrahedron, so the bound of
the old rule still holds. The decision is taken once, in the count pass,
and stored in the branch flag as a bit per node, so the other passes only
read it. Since the edges and faces are shared by neighboring tetrahedra,
they merge like the Taylor sources.

Which budget. The precision is the hard part, because the parts of a split
tetrahedron are much larger than the whole: on the 9886 tetrahedra that
the old rule keeps whole in one direction or both, against mpmath at 110 digits and relative to
the largest entry of the whole sum, the kept-whole sum is exact to
6.0e-15 (the direct kernel 7.2e-15), and the split one to 2.9e-13, 2.7e-13
and 8.0e-14 with budgets 1, 1/4 and 1/10. On the whole sum, against an
mpmath reference of all 165888 tetrahedron pairs at 40 frequencies
(13 minutes on three cores), the direct kernel is exact to 3.2e-14 and the
source kernel to 1.8e-13 before the change; with the split this is 3.5e-13,
4.6e-13, 3.4e-13, 2.2e-13, 1.6e-13 and 1.8e-13 for budgets 1, 1/4, 1/10,
1/30, 1/100 and 1/1000. So we take 1/100. Note that these numbers move by
some 30% between neighboring budgets, so differences below that are the
rounding of the largest entry, not a trend. With 1/100 the 7000 kept
point sources become 4742 edges and 4679 faces after the merge; with
budget 1 they would be 1991 edges.

The split alone pays little. The time per frequency of the kept part falls,
but the rest of the sum stays: at 400 frequencies the kernel went from
0.49 s to 0.44 s, and at 40 frequencies it was 3% slower, since the build
does more. A first version took the decision in all four passes and made
the 2D kernel 25% slower; storing it in the flag removed that.

Where the time per frequency goes. The evaluation of the split kernel at
400 frequencies took 0.32 s: 0.18 s for the 5015 Taylor sources, 0.11 s for
the nodes evaluated per frequency, 0.03 s for the energy bins and 0.025 s
for the 3741 vertex sources. So the Taylor sources, narrow clusters of
spread below delta, were the main cost, and they are evaluated at every
frequency although almost all frequencies are far from them.

The point sources far from resonance. Every point source, a Taylor series
or a node evaluated per frequency, now belongs to a set: its energy bin,
with the two kinds apart, since the second is wider. A set keeps the
moments of its sources around its center C and the largest distance R of
their nodes from C. The expansion of a source of m+1 nodes for psi_K has,
besides the powers of 1/(z - C) of the bins, the terms
psi_(K-i)(C - z) h_(i-m)(y)/i! for the orders i < K, a logarithm and its
products (`_add_node_moments`). At a frequency with
(w - C)^2 + delta^2 >= (R/0.5)^2 the set gives the sum of its sources,
with the orders that its rate R/|z - C| needs at that frequency, and at the
others each source is evaluated one by one. The moments are built from the
merged sources, in parallel over the sets and in a fixed order inside a
set, so the result does not depend on the number of threads. The layered
model has 316 sets.

The measurement. Three cores of the workstation, other jobs on the
machine (load 7 to 9), all versions loaded in one process and called in
turn on the same saved pair fields, eight calls each, the smallest time.
"Sets only" is the new code with the budget 0, so every kept tetrahedron
stays whole and only the sets act.

| case | before | sets only | sets and split |
|---|---|---|---|
| 3D, nk = 24, 40 frequencies | 0.149 | 0.127 | 0.127 |
| 3D, nk = 24, 400 frequencies | 0.607 | 0.433 | 0.300 |
| 2D spinful Haldane, 512^2, 141 frequencies | 2.098 | 1.312 | 1.321 |

The 3D case is the layered model of "The source kernel on tetrahedra",
interband, T = 1e-3 t, delta = 1e-2 t, on [0.1, 4] t; the 2D case is that
of "The 2D cost of the tetrahedron kernel". So the 3D kernel is 2 times
faster at 400 frequencies and 15% faster at 40, and the 2D kernel 1.6
times faster. The split is what gives the second factor of 1.4 in 3D at
400 frequencies, and it does nothing in 2D, where the kept triangles were
already cheap. At 40 frequencies the build, about 0.1 s, is now most of the
time.

What it costs in precision. On the 3D case the source kernel is within
2.1e-13 and 2.3e-13 of the direct kernel at 40 and 400 frequencies
(1.8e-13 and 2.6e-13 before), and within 2.0e-13 of the mpmath reference
(1.8e-13 before). In 2D, against the longdouble references of "Precision of
the two kernels", the spinful Haldane model is within 7.0e-13, 6.6e-13 and
7.5e-13 of the largest entry at nk = 128, 256 and 512 and delta = 1e-3,
2.75e-12 and 1.09e-12 at nk = 128 and 256 and delta = 1e-5, and 9.0e-14
and 3.5e-14 at delta = 1e-2, the numbers of before to two digits. On the
random tetrahedra of `test_tetrahedron_sources.py` the worst entries
against mpmath are those of before, 4.0e-12, 2.9e-12 and 1.8e-12 of the
peak at delta = 1e-2, 1e-3 and 1e-6, which sit in tetrahedra in vertex
form. A new test checks that three tetrahedra with close pairs are split
into edges and faces and agree with the direct kernel to 1e-12 of their
peak.

What was measured and left out. The energy bins of the far simplices
still run their Horner sum over all 60 orders, although their simplices
stop after 30 orders on average in 3D (8 in 2D). Skipping the orders that
vanish gives the same numbers bitwise, but it took 0.294 s against 0.300 s
at 400 frequencies and 1.336 s against 1.321 s in 2D, inside the noise, so
it was not kept.

What is left open. At 40 frequencies in 3D the build is now the cost, and
in it the count, the merge and the moments passes, each 0.01 to 0.03 s on
three cores. The energy bins of the far simplices and the sets could also
take the rate of each frequency for their own truncation, as the sets now
do. The budget 1/100 is set on one model; a model with many more close
pairs could want a check of the precision again. The next subsection
answers these three points.

### The budget on more models, and the error that grew with nk (7 November 2026)

The questions were whether the budget 1/100 is safe on other models, why
the error of the source kernel grew from 2e-13 to 7e-13 of the largest
entry between nk = 24 and 48 on the layered model while the direct kernel
stayed near 5e-14, and what the build costs at 40 frequencies.

The reference. An mpmath sum of every tetrahedron took 13 minutes at
nk = 24, too slow for six models and three broadenings. The reference is
now a hybrid. Where the nodes of a tetrahedron are within 0.3 |z - c| of
their mean c, we take the moment series around c in longdouble, 48
orders, which does not cancel. Elsewhere we take the vertex form in
longdouble, where its rounding bound, 8 eps times the sum of |terms|
times the weight, is below 1e-15 of the largest entry over the square
root of the number of tetrahedron pairs, and mpmath for the rest (0.6% to
7% of the entries). This takes 20 s at nk = 24 and 2 minutes at nk = 48 on
three cores, and it agrees with the full mpmath reference to 7e-16 of
the largest entry. A bound 100 times stricter moved it by 6e-17 on the
model with close pairs.

The models. All are 3D, with the interband part, T = 1e-3 t and 40
frequencies on [0.1, 4] t:

- layered: the Haldane layers of "3D crystals", interlayer mass 0.05 t, at
  nk = 24 and 48;
- layered, tm = 1e-3: the same with an interlayer mass of 1e-3 t, so that
  most tetrahedra have a close pair of nodes along kz;
- cubic metal: the spinful simple cubic band with Rashba coupling 0.3 t,
  an exchange field (0.1, 0.2, 0.3) t and the Fermi level 1 t inside the
  band, so the weights live near the Fermi surface;
- diamond: first-neighbor diamond lattice, with its nodal lines at E = 0;
- diamond, spinful: the same with the Fu-Kane-Mele spin-orbit coupling
  0.1 t, so every band is a Kramers pair.

Why the error grew with nk. We summed each class of tetrahedra alone, with
both kernels and with its own reference: forward at dE and reverse at -dE,
and in each the vertex form, the binned ones and the split ones. On the
layered model at nk = 24 and delta = 1e-2 the forward tetrahedra in vertex
form carry all the error, 1.2e-13 against 9e-15 for the direct kernel on
the same class, while the split ones are at 5e-15 and the binned ones at
1e-17. The reason is that the vertex sources cancel at every frequency: the
sum of |coefficient| |psi_3| over them is 5.0e3 times the largest entry at
nk = 24 and 8.3e3 at nk = 48. A plain running sum of such terms loses eps
times this times a factor that grows with the number of terms, which is
the growth with nk. The direct kernel does not see it, since far from z
it takes the Taylor series of each tetrahedron, which does not cancel.
We tried three fixes on the layered model at delta = 1e-2.

- Vertex coefficients in double-double, accumulated without rounding:
  2.2e-13 to 2.1e-13 at nk = 24, so the coefficients are fine and the loss
  is in the sum per frequency. Not kept.
- A compensated sum (Knuth's two-sum) of the vertex sources only: 6.6e-14
  at nk = 24 and 1.2e-13 at nk = 48 (with the double-double coefficients,
  which change nothing measurable).
- A compensated sum of every term of the sum per frequency: 2.8e-14 and
  5.2e-14, against 3.2e-14 and 6.5e-14 for the direct kernel.

The third one is the fix (`_kadd` in `kubosimplex.py`). Two sources are
added plainly before each compensated addition: on four models this gives
the same error as compensating every source, while blocks of 4 and 16 gave
up to 1.5 and 2.5 times more. What is left is the rounding of each term,
eps times the root of the sum of the squared terms, and it moves by about
50% with the order of the summation; the result on the cubic metal moved
from 3.8e-13 to 2.2e-13 just with the order of the merged sources.

The error table. Largest error over frequencies and components, relative
to the largest entry of the reference. "Before" is the kernel of b343d03;
the other columns are the kernel now, with the budget b and the vertex
bound `_tol` = 3e-11 unless noted.

| model | nk | delta | direct | before | b = 1 | b = 1/10 | b = 1/100 | b = 1/1000 | b = 0 | b = 1/100, `_tol` = 3e-12 |
|---|---|---|---|---|---|---|---|---|---|---|
| layered | 24 | 1e-2 | 3.2e-14 | 2.2e-13 | 4.3e-14 | 3.3e-14 | 6.2e-14 | 6.4e-14 | 6.4e-14 | 6.5e-15 |
| layered | 24 | 1e-4 | 2.3e-14 | 1.1e-13 | 2.3e-13 | 9.3e-14 | 7.1e-14 | 9.0e-14 | 9.3e-14 | 9.0e-15 |
| layered | 24 | 1e-6 | 4.3e-14 | 1.5e-13 | 3.4e-13 | 6.8e-14 | 6.8e-14 | 9.0e-14 | 9.0e-14 | 1.1e-14 |
| layered | 48 | 1e-2 | 6.5e-14 | 7.4e-13 | 5.6e-14 | 5.6e-14 | 7.9e-14 | 7.1e-14 | 7.1e-14 | 1.1e-14 |
| layered | 48 | 1e-4 | 7.2e-14 | 5.9e-14 | 1.2e-13 | 7.6e-14 | 1.1e-14 | 9.6e-15 | 1.2e-14 | 2.0e-15 |
| layered | 48 | 1e-6 | 5.0e-13 | 1.4e-13 | 2.4e-13 | 8.0e-14 | 6.8e-15 | 5.9e-15 | 5.3e-15 | 2.8e-15 |
| layered, tm = 1e-3 | 24 | 1e-2 | 2.1e-13 | 4.0e-13 | 2.0e-13 | 9.1e-14 | 7.6e-14 | 7.6e-14 | 7.6e-14 | 6.6e-15 |
| layered, tm = 1e-3 | 24 | 1e-4 | 1.1e-13 | 1.4e-13 | 1.2e-13 | 1.2e-13 | 1.1e-13 | 1.2e-13 | 1.1e-13 | 1.1e-13 |
| layered, tm = 1e-3 | 24 | 1e-6 | 5.4e-14 | 5.6e-14 | 1.2e-13 | 1.7e-13 | 5.6e-14 | 5.5e-14 | 5.4e-14 | 5.5e-14 |
| cubic metal | 24 | 1e-2 | 1.7e-14 | 5.9e-13 | 3.5e-13 | 2.5e-13 | 2.2e-13 | 2.2e-13 | 2.2e-13 | 8.6e-15 |
| cubic metal | 24 | 1e-4 | 1.7e-14 | 4.5e-13 | 2.0e-13 | 3.4e-13 | 3.2e-13 | 3.2e-13 | 3.2e-13 | 3.5e-15 |
| cubic metal | 24 | 1e-6 | 1.5e-14 | 3.1e-13 | 1.7e-13 | 1.6e-13 | 1.6e-13 | 1.6e-13 | 1.6e-13 | 3.7e-15 |
| diamond | 24 | 1e-2 | 5.3e-14 | 2.3e-13 | 2.1e-13 | 1.5e-13 | 2.2e-13 | 2.2e-13 | 2.1e-13 | 3.6e-14 |
| diamond | 24 | 1e-4 | 3.5e-14 | 1.3e-13 | 1.6e-13 | 1.1e-13 | 1.6e-13 | 1.6e-13 | 1.6e-13 | 3.0e-14 |
| diamond | 24 | 1e-6 | 5.2e-14 | 1.6e-13 | 1.1e-13 | 9.6e-14 | 1.3e-13 | 1.3e-13 | 1.3e-13 | 3.5e-14 |
| diamond, spinful | 24 | 1e-2 | 1.8e-13 | 3.9e-13 | 1.3e-13 | 1.2e-13 | 9.3e-14 | 9.2e-14 | 9.2e-14 | 1.4e-14 |
| diamond, spinful | 24 | 1e-4 | 1.6e-13 | 4.9e-13 | 2.4e-13 | 1.9e-13 | 1.7e-13 | 1.7e-13 | 1.7e-13 | 1.2e-14 |
| diamond, spinful | 24 | 1e-6 | 1.9e-13 | 5.2e-13 | 1.9e-13 | 1.4e-13 | 1.3e-13 | 1.3e-13 | 1.3e-13 | 1.3e-14 |

What the table says. With the compensated sum the budget 1/100 is as
precise as keeping the tetrahedra whole (b = 0) on every model, while 1
and 1/10 are worse at small delta (3.4e-13 and 1.7e-13 where b = 0 gives
9e-14 and 5e-14). So the budget stays at 1/100. Before the compensated sum
the budgets could not be told apart, since every column sat on the floor
of the plain sum. The kernel now is better than before, or equal, on every
row but one: the diamond at delta = 1e-4, 1.6e-13 against 1.3e-13, inside
the 50% that the order of the summation moves. On the layered model the error
no longer grows with nk, so a rule tied to the number of tetrahedra is not
needed. What still separates the source kernel from the direct one on the
metal and the diamond is not the budget but the vertex bound `_tol`: each
tetrahedron in vertex form may lose 3e-11 of its own peak, and the metal,
whose weights sit on a few tetrahedra near the Fermi surface, adds these
up to 1.6e-13 to 3.2e-13. With `_tol` = 3e-12 every row is below 4e-14 except
the one at tm = 1e-3 and delta = 1e-4, where both kernels are 1.1e-13
from the reference. That floor is shared with the direct kernel, so it
points to the per-frequency divided differences of `simplexdos._dd`; it
was not followed further. Note also that the direct kernel is not the better
of the two everywhere: it is at 5.0e-13 on the layered model at nk = 48
and delta = 1e-6, where the source kernel is at 7e-15.

The single logarithm. A point source has two nodes or more (a single node
is a vertex source), so `_add_node_moments` never writes the order 0 of
`Ls`, and its logarithm was computed for nothing. The terms of orders 1
to K-1 of a far set also share one logarithm, ln(Cs - z), so a far set
now costs one logarithm per frequency instead of K.

The bins of the far simplices. Each bin now keeps the largest distance of
its nodes from its center and takes the orders that its rate needs at each
frequency, as the sets do. At 400 frequencies in 3D this took the kernel
from 0.353 s to 0.327 s, in 2D from 1.457 s to 1.429 s, and at 40
frequencies it is inside the noise. It was kept.

The build at 40 frequencies. On the layered model at nk = 24 the build
is about three quarters of the call, on three cores, the smallest of six
calls: the count pass 0.016 s, the write pass 0.008 s, the moments of
the bins 0.026 s, the vertex pass 0.012 s, the merge of the 91332 point
sources into 14436 0.035 s (0.012 s of it the lexicographic sort of their
keys), the moments of the sets 0.008 s, the sum of the 16 copies of the
moments of the bins 0.006 s. Two simple changes were tried and dropped: a
sort by energy first with the ties sorted in numba was slower than the
lexicographic sort (0.017 s against 0.012 s), and 6 groups of moments
instead of 16 saved 0.008 s but would cap the parallel pass at 6 threads.
Two were kept: the merge of the vertex sources takes the default sort
instead of the stable one, since equal keys only need a deterministic
order (0.055 s against 0.077 s for the 2.1M keys of the 2D model), and
the test for vanishing vertex coefficients runs on a float view of the
array. The merge is shared with `shgsimplex.py`, whose results move at
the rounding level too.

The cost. Three cores of the workstation with other jobs on the machine,
the kernels loaded in one process and called in turn on the same saved
pair fields, ten calls each, the smallest time in seconds. The 2D case is
the spinful Haldane model at 512^2 and 141 frequencies of "The 2D cost of
the tetrahedron kernel", the 3D case the layered model at nk = 24 and
delta = 1e-2. The last two columns change only the bound `_tol` (and the
budget) in 3D and 2D alike.

| case | before | now | `_tol` = 1e-12, b = 1 | `_tol` = 3e-12, b = 1/100 |
|---|---|---|---|---|
| 2D, 512^2, 141 frequencies | 1.350 | 1.376 | 2.659 | 2.354 |
| 3D, nk = 24, 40 frequencies | 0.121 | 0.112 | 0.159 | 0.196 |
| 3D, nk = 24, 400 frequencies | 0.297 | 0.274 | 0.312 | 0.778 |

So the 2D kernel costs the same within the noise (the medians are 1.436 s
and 1.443 s), and the 3D kernel is 7% to 8% faster. In 2D the error against
the longdouble references of "Precision of the two kernels" falls from
7.5e-13 to 7.8e-14 at 512^2, from 6.6e-13 to 2.9e-13 at 256^2 and from
7.0e-13 to 3.5e-13 at 128^2 (delta = 1e-3). A tighter `_tol` doubles the
2D cost, so it is not for 2D. In 3D, `_tol` = 1e-12 with the budget 1
costs 42% at 40 frequencies and 14% at 400 against the kernel now (31% and
5% against the kernel before). It gives at most 5.4e-14 on every row of
the table, the level of the direct kernel on that row, and 1.1e-13 on the
shared floor, measured with the compensated sum on every source.

`tests/conductivity/test_tetrahedron_sources.py` has a new test: on the
layered model at nk = 24 the two kernels agree to 1.5e-13 of the largest
entry. The kernel now gives 6.3e-14 there, and the kernel before 2.2e-13.
Since the order of equal keys in the merge may differ between machines,
and the error moves by about 50% with that order, a failure of this test
on another machine near the bound means the bound, not the physics.

What is left open. Whether 3D should take the tighter `_tol` = 1e-12 with
the budget 1, 15 to 80 times more precise on the metal and the diamond for
14% to 42% more time, is a choice left open. The 1.1e-13 that both
kernels share on the model with close pairs at delta = 1e-4 points to
`simplexdos._dd`. The merge of the point sources is a third of the build
at 40 frequencies; a merge written in numba, which sorts and sums in one
pass, would be the next thing to try.

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
  catches the symmetry-related k-points. These times and differences are
  those of the kernels as first built; the precision fix below changed
  both kernels.

  Precision of the two kernels (fixed, 6 November 2026). The difference
  in the last column was first read as the rounding level of the formula.
  It was not. Both kernels switched from the exact divided difference to a
  second order Taylor series when the spread s of a triangle was below
  tolc = 1e-2 delta. The cancellation of a divided difference grows like
  eps (|x|/s)^2, with |x| = |c - z| and c the mean energy, so a spread
  just above tolc far from resonance loses a lot. For the triangle 0.5,
  0.5 + 0.4 s, 0.5 + s at omega = 0 and delta = 1e-6, against mpmath at 60
  digits, the direct kernel was off by 0.10, 0.027 and 4e-4 at s = 3e-8,
  1e-7 and 1e-6, and the source kernel by 1.0, 0.063 and 7e-4. The
  second order series had a second defect: it took the variance of the
  uncentered energies, which loses eps c^2/delta^2 on resonance (1e-4 of
  the triangle at delta = 1e-6, 1e-10 at delta = 1e-3). Segments in 1D
  were fine to about 2e-8.

  The direct kernel now calls `simplexdos._dd`, which decides per
  frequency: a cluster of nodes goes to its Taylor series when its spread
  is below 0.05 |c - z|. The source kernel decides once for all
  frequencies, so it cannot compare with |c - z|. It expands a triangle
  or an edge when s < delta, and runs the series until it converges. The
  term of order n is bounded by (2 s/(3 |c - z|))^n and |c - z| >= delta
  at any real omega, so the series converges everywhere, in up to about 80
  terms on resonance and a few far from it. The series is evaluated on
  w = (e - c)/(z - c), so no large power of 1/(z - c) appears. Worst
  relative error of one triangle over spreads from 1e-4 to 100 delta and
  |c - z| up to 1e4 delta (10 t at delta = 1e-3) or 1e6 delta (1 t at
  delta = 1e-6):

  | kernel | delta | before | after |
  |---|---|---|---|
  | direct | 1e-3 | 4e-4 | 9e-13 |
  | source | 1e-3 | 3e-4 | 4e-8 |
  | direct | 1e-6 | 2 | 2e-12 |
  | source | 1e-6 | 2 | 1e-4 |

  The 4e-8 and 1e-4 left in the source kernel sit at a spread just above
  delta and the largest |x|, where its divided difference still loses
  eps (|x|/s)^2. Below s = delta it is exact to rounding.

  The reference for the full sums is now a numpy longdouble sum (eps 1e-19)
  with the per-frequency rule and 24 Taylor terms, checked against mpmath
  to 1e-15 per triangle over the same map. The reference used before kept
  the branches of the kernels, so it saw the rounding but not the
  truncation of the second order series. Error as a fraction of the
  largest entry, delta = 1e-3, 141 frequencies on [0, 7] t:

  | model | nk | source before | source after | direct before | direct after |
  |---|---|---|---|---|---|
  | Haldane, spinful | 128 | 5.8e-10 | 5.0e-12 | 5.0e-10 | 4.2e-14 |
  | Haldane, spinful | 256 | 7.5e-11 | 3.6e-12 | 3.3e-11 | 1.1e-13 |
  | Haldane, spinful | 512 | 2.8e-10 | 1.7e-12 | 1.0e-10 | 1.5e-13 |
  | gapped honeycomb | 128 | 7.9e-13 | 7.5e-13 | 3.8e-13 | 2.2e-14 |
  | gapped honeycomb | 256 | 2.6e-12 | 2.6e-12 | 1.8e-12 | 3.6e-14 |
  | gapped honeycomb | 512 | 8.1e-12 | 8.2e-13 | 8.3e-12 | 6.6e-14 |

  So the 1e-10 floor of the spinful Haldane model was the switch rule, and
  it is gone. The direct kernel is now at 1e-13. The source kernel keeps
  about 1e-12, which is the cancellation in its vertex sources, the
  B-spline form of the sum, for triangles of ordinary spread. At nk = 40
  on the spinful Haldane model it is 1.7e-11, before and after, from edges
  just above delta in wide triangles; the test of the two kernels keeps
  its 1e-10. Re sigma_xx of the gapped honeycomb with the triangle at
  nk = 256 (the row of the first table) moved by 4e-13 of its largest
  value, so no physics result of this note changes.

  We scanned the threshold of the source kernel on the spinful Haldane
  model at nk = 512. With s < alpha delta, alpha = 0.01 (the old rule),
  0.1, 0.25, 0.5 and 1 give 2.8e-10, 2.9e-11, 1.6e-11, 5.2e-12 and
  1.7e-12, with 6.6k, 23k, 48k, 85k and 146k edge sources against 0.29M
  vertex sources. We took alpha = 1, since the series still converges at
  a rate 2/3 on resonance. A second rule, s < 0.05 times the distance from c
  to the frequency grid, also expands the triangles far from the whole
  spectrum, the reverse terms at -dE among them. At a factor 0.01 it made
  1.3M simplex sources out of 17k and cost 6 s per 141 frequencies
  instead of 1 s, for 8e-12 instead of 1.7e-12. It was not kept.

  Cost. The source kernel pays for the longer series: on the spinful
  Haldane model at 512^2 and 701 frequencies it took 7.2 s against 3.1 s
  before, on the gapped honeycomb 2.3 s against 1.9 s. The direct kernel
  is about as fast as before (115 s against 104 s, 19 s against 25 s),
  because most triangles are far from most frequencies and take a short
  series instead of three logarithms. These are 14 threads of one
  workstation at a load of 7 to 22 from other jobs, so only the ratios
  mean something. The open point of this paragraph, a simplex kept as a
  point source that decides per frequency between its series and its
  divided difference, was built on 7 November 2026 for whole simplices,
  with a bound of the loss over the frequency window that decides which
  simplices need it; see "The source kernel on tetrahedra" in the 3D
  section.

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
  set of measure zero), but the QTCI rank grows. Measured on a 2x2
  supercell with a random onsite energy (eight bands, 80 fields, see the
  section on one interpolation of all the fields): qtci diagonalizes the
  whole mesh at R = 7 and 8 in both forms, so it saves nothing there and
  the plain triangle is 20 to 40 times faster. Band tracking by overlap of
  eigenvectors would remove the kinks of the anticrossings; not built. With
  more fields than this (a 4x4 cell has about 1300) the stacked rank would
  pass the cap of 800, which the 80 fields of the 2x2 cell did not reach
  (572 at R = 8); not measured.
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
- **3D, what is still open (built otherwise, 7 November 2026, see "3D
  crystals").** The source kernel for tetrahedra: the cost of the direct
  kernel grows as the tetrahedra times the pairs times the frequencies, so
  a long spectrum on a fine mesh is where it would help. It needs psi_3 at
  the vertices and the confluent pieces of a tetrahedron (edges, triangles
  and the whole simplex) as point sources, with their own Taylor series.
  Also open: qtci in 3D, a choice of the tetrahedron diagonal adapted to the
  lattice (the shortest one in reciprocal space), and the SHG and the
  excitonic optics in 3D, whose velocities now exist.
