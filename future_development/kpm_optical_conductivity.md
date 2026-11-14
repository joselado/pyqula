# The Chebyshev (KPM) Kubo conductivity: what is built and what is left open

The linear conductivity of a system with 10^4 to 10^6 orbitals cannot go through the bands: a
supercell of that size has no affordable diagonalization, and a finite flake has no k-mesh.
Since 6 November 2026 `conductivitytk/kpmkubo.py` computes it from a Chebyshev expansion of
the Hamiltonian with random vectors. Two entry points use it:

- `h.get_optical_conductivity(integration="kpm", ...)` gives the absorptive part of
  sigma_ab(w), so Re sigma_xx(w) and the circular dichroism Im sigma^A_xy(w).
- `h.get_dc_conductivity(fermi=..., ...)` gives the DC tensor from the Kubo-Bastin formula,
  with the Fermi sea, so sigma_xy is the Hall conductivity of a Chern insulator.

This note records the formulas, what was measured, where the band path is still the better
choice, and what is not built.

## The formulas

We rescale the Hamiltonian into (-1,1), H~ = (H - b)/a, with the edges from Lanczos padded by
5%. Everything comes from one object, the double moment

    mu^ab_nm = g_n g_m Tr[v_a T_n(H~) v_b T_m(H~)] / V,

with g_n the kernel and V the number of k-points times the cell volume. The velocity is the one
of `conductivity.py`, v_a = i sum_R t_ij(R) d^a_ij exp(2 pi i k.R) with the full bond vector
d_ij = R + r_j - r_i, built entry by entry on the sparse hoppings. For a finite system this is
v = i[H,X]. Because the operator is the same, the two paths share every sign convention.

With the broadened delta function delta(E - H) = sum_n phi_n(E) T_n(H~), where
phi_n(E) = (2 - delta_n0) T_n(x)/(pi a sqrt(1 - x^2)) and x = (E - b)/a, the spectral density
of two currents is Tr[v_b delta(E - H) v_a delta(E' - H)]/V = sum_nm mu^ab_nm phi_n(E') phi_m(E).
In the eigenbasis this is sum_nm v^a_nm v^b_mn delta(E - E_n) delta(E' - E_m)/V, meaning that
it is the numerator of the Lehmann sum of `conductivity.py` with the two energies kept apart.

The absorptive part follows from the delta-function part of the Lehmann sum:

    sigma^H_ab(w) = pi int dE [f(E) - f(E + w)]/w  Tr[v_b delta(E-H) v_a delta(E+w-H)]/V.

The diagonal is Re sigma_aa, and the off-diagonal imaginary part is Im sigma^A_xy. At w = 0 the
ratio becomes -df/dE, so the Drude peak is the same formula. The energy integral runs on
Chebyshev-Gauss nodes, which absorb the 1/sqrt(1 - x^2) of the first delta, and a numba kernel
builds T_n(x + w/a) of the second by its recursion.

The DC tensor is the Kubo-Bastin formula. We re-derived the Chebyshev form of Garcia, Covaci and
Rappoport (PRL 114, 116602 (2015), arXiv:1410.8140, from the TeX source) with
G+(e) = -i sum_n (2 - delta_n0) exp(-i n arccos e) T_n(H~)/(a sqrt(1 - e^2)). The derivative of
G gives their Gamma_nm, and the prefactor is

    sigma_ab = 1/(pi a^2) int de f(a e + b) sum_nm (2-delta_n0)(2-delta_m0) mu_nm Gamma_nm(e)/(1-e^2)^2,
    Gamma_nm(e) = (e - i n s) exp(i n t) T_m(e) + (e + i m s) exp(-i m t) T_n(e),

with s = sqrt(1 - e^2) and t = arccos e, which is their Eq. (3) with 4/Delta E^2 = 1/a^2. There
is one trap in the sign. Garcia writes v_a delta v_b dG+ - v_a dG- v_b delta, and in the
Lehmann representation this gives the antisymmetric part with the opposite sign of the retarded
Kubo formula of `conductivity.py`. The order of Crepieux and Bruno, v_a dG+ v_b delta -
v_a delta v_b dG-, gives the retarded one. So the code uses Garcia's expression with a and b
exchanged. We checked it against the band path, which gives sigma_xy 2 pi = -C with the C of
`h.get_chern()`, for both signs of the Haldane flux. The moments do not depend on the Fermi
level or on T, so a scan over Fermi levels costs one more energy integral each.

## Resolution: what delta means here

`delta` keeps the meaning it has in the band modes: the half width of the absorption line of
one transition. A transition has two broadened deltas, at E_n and at E_m. Two Lorentzians of
half width eta convolve into one of half width 2 eta, so with the Lorentz kernel each delta gets
eta = delta/2 and the line is the Lorentzian of `integration="grid"`. With the Jackson kernel
two Gaussians give a Gaussian of half width sqrt(2) eta, so eta = delta/sqrt(2), and the line is
a Gaussian with no tails, narrower by sqrt(1 - x^2) towards the band edges. In both cases
the number of moments is 2 npol, the package convention of `kpmtk/kernels.py`, and npol
is about a/delta, so a delta of 0.05 t on a spectrum of half width 3.2 t needs npol of 100 to 200.

There is one more difference with the band path, and it is a property of the formula, not an
error. The KPM weight of a transition is [f(E) - f(E + w)]/w evaluated inside the broadened
deltas, while the band path uses (f_n - f_m)/(E_m - E_n) at the exact energies. Far from a
resonance, in the tail of a Lorentzian, the two differ: at w = 0 in a gapped system the band
path gives a small Lorentzian tail of the interband peak, and the KPM gives nothing. We measured
that the KPM agrees with a Lehmann sum that uses the same two broadened deltas to 2 to 3%, and
with the band path to 3 to 5%, on an 18-orbital cell at the same k-mesh.

## Cost and memory

Per random vector and k-point the code builds the left set L_n = T_n(H~) v_a |r> and the right
set R_m = v_b T_m(H~) |r>, and mu_nm = L_n^dagger R_m is one dense complex matrix product per
pair of directions. That is 2M sparse products and N M^2 operations at BLAS speed for M = 2 npol
moments. The memory is 2 nd N B complex numbers, with nd the number of Cartesian
directions and B the number of polynomials stored at once. B comes from `memory=` (in GB, 2 by
default). When B < M the left set is recomputed M/B times, so a small budget costs sparse
products but never memory. We use mu^ba = (mu^ab)^T, so in 2D three products are needed and not
four. The sparse product is a numba kernel parallel over the rows, which falls back to a serial
version below 5e4 nonzeros, because the thread launch costs more than the product there (this
took the Hall test from 43 s to 6.6 s). The random vectors run one after the other, because
running them in parallel multiplies the stored set.

The post-processing does not depend on N. Both integrals are linear in mu, so we evaluate them
per random vector, and the spread gives the error bar.

## What was measured

All the runs are on four cores of a workstation that other jobs shared, so a wall time can be
off by a factor of two in either direction. T = 0.02 t unless said otherwise. The error of a
spectrum is the largest deviation over 81 frequencies in [0, 4t], divided by the peak of the
reference.

### The mid-size cell against the bands

The model is the gapped honeycomb lattice (sublattice imbalance 0.5, gap 1t) in a 12x12
supercell, 288 orbitals, with uniform onsite disorder of width 0.6t, below the gap. For
`integration="grid"` this is affordable. The first question is how far the Gamma point of this
cell is from the converged k-mesh of the band path:

| delta | grid nk=4 against nk=6 | grid nk=1 against nk=6 |
| --- | --- | --- |
| 0.1 | 0.5% | 39% |
| 0.05 | 1.9% | 68% |
| 0.025 | 5.0% | 93% |

So 288 orbitals at the Gamma point are still far from the thermodynamic limit, and the cell is
a check of the method, not of the physics. The KPM with the Lorentz kernel and the exact trace,
against the band path on the same k-points:

| delta | npol | KPM nk=1 against grid nk=1 | time |
| --- | --- | --- | --- |
| 0.1 | 96 | 4.5% | 10 s |
| 0.05 | 192 | 2.4% | 14 s |
| 0.025 | 384 | 1.5% | 43 s |

At nk=2 and delta=0.05 it is 2.4% against the grid at nk=2. The residual goes down with npol,
since it is the difference between the Lorentz kernel and a true Lorentzian and the occupation
weight in the tails (see the resolution section). With the Jackson kernel the difference with the
Lorentzian grid is 30% at delta=0.05: this is the line shape (a Gaussian is about 1.5 times
taller at the same half width), not an error, so a KPM curve is laid over a grid curve only
with `kernel="lorentz"`.

The random vectors against the exact trace, at the Gamma point. The error bar is the standard
error over the vectors, and the last column is the largest deviation in units of the local
error bar:

| delta | vectors | error against exact | mean error bar | largest deviation | time |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 4 | 12% | 4.6% | 2.8 | 5 s |
| 0.1 | 16 | 10% | 2.4% | 1.6 | 20 s |
| 0.1 | 64 | 2.2% | 1.2% | 1.5 | 87 s |
| 0.05 | 4 | 18% | 4.8% | 3.5 | 11 s |
| 0.05 | 16 | 13% | 2.5% | 2.6 | 65 s |
| 0.05 | 64 | 3.3% | 1.3% | 2.1 | 201 s |
| 0.025 | 4 | 29% | 5.4% | 3.5 | 26 s |
| 0.025 | 16 | 16% | 2.9% | 3.1 | 117 s |
| 0.025 | 64 | 4.7% | 1.5% | 2.6 | 452 s |

The error bar falls as 1/sqrt(vectors), as it should. With 4 vectors it underestimates the error
(3.5 error bars at the worst frequency), which is the small-sample estimate of the spread, so
we quote error bars from 8 vectors or more. A sharper delta needs more vectors for the same
relative error, because each frequency then averages over fewer transitions. At 288 orbitals the
random vectors are expensive: the exact trace (288 basis vectors) is cheaper than 64 random ones.

The f-sum rule. The diamagnetic weight from one Chebyshev expansion of the Fermi function with 16
vectors is W = 0.2861, against 0.2870 from `kubo.sum_rule_weight`. The integral of the KPM
absorption gives 0.2938, 2.4% above. At a single k-point the sum rule holds only when the
k-integral is converged, since it comes from an integration by parts over the Brillouin zone, so
this 2.4% is mostly the Gamma point. For a finite flake, where v = i[H,X], the rule is exact, and
`test_f_sum_rule_of_a_finite_flake` finds the integral within 1% of pi W.

### The Hall conductivity against the Chern number

The Haldane model (t2 = 0.2, Fermi level shifted by 0.3 into the gap) has C = 1 from
`h.get_chern()`, so the target is 2 pi sigma_xy = -1. We use n x n supercells, 8 random vectors
and T = 0.01, with and without onsite disorder of width 1t:

| orbitals | disorder | nk | npol=100 | npol=200 |
| --- | --- | --- | --- | --- |
| 72 | 0 | 1 | -0.976 +- 0.040 | -0.974 +- 0.040 |
| 72 | 0 | 2 | -0.993 +- 0.028 | -0.991 +- 0.028 |
| 72 | 1 | 1 | -0.985 +- 0.048 | -0.983 +- 0.048 |
| 72 | 1 | 2 | -1.000 +- 0.029 | -0.998 +- 0.029 |
| 288 | 0 | 1 | -1.013 +- 0.025 | -1.011 +- 0.025 |
| 288 | 0 | 2 | -1.014 +- 0.007 | -1.011 +- 0.007 |
| 288 | 1 | 1 | -1.024 +- 0.025 | -1.022 +- 0.025 |
| 288 | 1 | 2 | -1.017 +- 0.007 | -1.015 +- 0.007 |
| 100352 | 1 | 1 | | -1.0005 +- 0.0003 (2 vectors) |

The primitive cell with the exact trace and nk=12 gives -1.0008 (this is the test). Every entry
is within about two error bars of -1, the plateau is the same at Fermi levels -0.3, 0 and 0.3 in
the gap, and the error bar at 10^5 orbitals is small with only 2 vectors, because the trace
self-averages in a big cell. The 10^5 run took 77 s with 3.1 GB, after 111 s to build the Haldane Hamiltonian.
The DC sigma_xx is below 1e-6 in the clean cells and of order -1e-4 to -3e-3 with disorder: a
negative DC conductivity is not physical, so this is the noise floor of the Bastin integral and
not a result.

### Large cells

The same disordered gapped honeycomb, now in a 224x224 supercell (100352 orbitals) and a 708x708
supercell (1002528 orbitals), at the Gamma point, with the Jackson kernel. No band calculation
can do these. The time per vector includes the post-processing of 81 frequencies (2 to 11 s,
independent of N), and the memory is the peak resident size:

| orbitals | delta | npol | vectors | time per vector | memory | mean error bar | one-vector spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100352 | 0.1 | 86 | 4 | 16 s | 2.1 GB | 0.40% | 0.80% |
| 100352 | 0.05 | 171 | 4 | 30 s | 3.8 GB | 0.48% | 0.96% |
| 100352 | 0.025 | 341 | 2 | 83 s | 7.0 GB | 0.68% | 0.96% |
| 1002528 | 0.1 | 86 | 2 | 56 s | 5.4 GB | 0.08% | 0.11% |
| 1002528 | 0.05 | 171 | 2 | 193 s | 5.4 GB | 0.13% | 0.18% |

Two steps come before the vectors: the Lanczos edges (15 s at 10^5 orbitals, 50 s at 10^6) and
building the Hamiltonian (6 s and 21 s). The runs used a budget of `memory=5`. At 10^5 orbitals
the stored set then fits whole up to npol of about 310, and at 10^6 the budget holds 62
polynomials, so the left set is recomputed 3 times (npol = 86) and 6 times (npol = 171). The
7.0 GB at npol = 341 is above the budget by a conjugated copy of the left block (1 GB), which
was removed afterwards, by the sparse matrices and by the interpreter. These runs were made with
a budget formula that also counted a third block that is never stored, so after the fix a given
`memory=` stores 5/4 as many polynomials.

The time per vector is the second vector; the first one at 10^6 took 145 s and 231 s, since it
also compiles the kernels and builds the sparse operators. The peak of the spectrum is 0.3098 at
10^6 against 0.3101 at 10^5 (delta = 0.1), and 0.3244 against 0.3246 (delta = 0.05), so the
10^5 cell is already converged in size at this resolution.

The one-vector spread, the scatter of a single random vector relative to the peak, is 1% at
10^5 orbitals and 0.1 to 0.2% at 10^6. At 288 orbitals it is about 9%. This is roughly the 1/sqrt(N) of a self-averaging
trace, within the crude spread that 2 to 4 vectors give, and it means a big cell needs one or two
vectors. As a physical check, we compare the 10^5 Gamma-point spectrum with the band path
on three other realizations of the 288-orbital cell at nk=4, with a Gaussian line of the same
half width (the Jackson line shape). The peak sits at w = 2.2t in both, with heights 0.310 and
0.304 at delta = 0.1. The largest deviation is 2.4% at delta = 0.1 and 6.8% at delta = 0.05, the
same size as the spread of the reference over its three realizations (4.0% and 6.6%), so the
small cells limit this comparison, not the KPM.

## What wins and what loses

The KPM wins when the cell is large: at 10^5 orbitals one vector gives the spectrum to 1% in 30 s
at delta = 0.05, and the Hall plateau to 3e-4, where the band path cannot diagonalize at all. It
also wins for a disordered system, where a large cell at the Gamma point is the honest model and a
small cell with a dense k-mesh is not: the 288-orbital cell differs from its own nk=6 limit by
40 to 90% at the Gamma point.

The bands win in three places, and we record them plainly.

- Small cells. At 288 orbitals the band path at nk=4 is converged to 2% in about 15 s, while the
  KPM needs the exact trace or 64 vectors for 2 to 5%. Below a few thousand orbitals,
  `integration="grid"` or `"triangle"` with a k-mesh is the better choice.
- Sharp features. The cost of the KPM grows as npol^2, and npol grows as 1/delta, so halving
  delta costs four times the products (30 s to 83 s per vector at 10^5 orbitals). A clean crystal
  with a sharp edge below the KPM resolution is the case for `integration="triangle"`, which
  integrates the resolvent exactly and resolves delta = 1e-3t.
- The line shape. A KPM curve is a Lorentzian only with the Lorentz kernel, and then only to 2
  to 5%, and a tail far from a resonance carries the occupation weight of the broadened deltas,
  not of the exact energies. The Jackson kernel gives a Gaussian line, narrower towards the band
  edges.

## What is open

- The GPU. The cost is a sparse product and a dense complex matrix product per block, the shape
  that fits a card. Following the gpu-backend rule, nothing was moved to the device. The
  natural path is the block recursion of `kpmtk/pairmomentsjax.py` with a zgemm on the device,
  behind `gpu.get_gpu()`. A consumer card is a float32 card, and whether single precision is
  enough for the double moments was not measured.
- The reactive part at finite frequency, Im sigma_xx(w) and Re sigma_xy(w) for w > 0, that is,
  the Kubo-Bastin Hall conductivity at finite frequency. The same moments give it, either from a
  Hilbert transform of the absorptive part or from the finite-frequency kernel
  (f(E) - f(E'))/(E' - E)/(w + i eta - (E' - E)) on a double energy grid. Neither is built.
- The second harmonic (SHG). It needs a triple expansion, Tr[v_a T_n v_b T_m v_c T_l], as in
  KITE. The cost goes as N npol^3, so it needs a different contraction.
- KITE (Joao et al., R. Soc. Open Sci. 7, 191809 (2020)) was the reference for the conventions,
  but it was not run here. The checks are against the band path of `conductivity.py`, against
  the Chern number, and against the f-sum rule.
- `intraband=False` and `interband=False` are not available with `integration="kpm"`, since
  the expansion never builds eigenstates. An unknown `integration=` name raises an error that
  lists every accepted name, "kpm" included.

## Repairs of the seventh bug audit

The circular dichroism, the off-diagonal part i Im sigma^A_xy of the
absorptive tensor that the module advertises, was never tested. The only
comparison of `absorptive()` with the band path looked at the diagonal,
on a real Hamiltonian, where sigma^A_xy vanishes anyway, and the Haldane
test goes through `bastin()`. So an exchange of the two Cartesian indices
in the assignment of `absorptive()` would flip the sign of the dichroism
and pass every test. The code itself is right: on the Haldane model
(t2 = 0.2, Fermi level shifted by 0.3, nk = 12, T = 0.05, delta = 0.1, the
Lorentz kernel and the exact trace) the KPM value of sigma^A_xy follows the
Hermitian part (sigma_xy + conj(sigma_yx))/2 of the band path to 2.1e-2,
against a peak of 0.68, while its transpose is 1.37 away.
`test_kpm_circular_dichroism_matches_the_band_calculation` in
`tests/conductivity/test_kpm_conductivity.py` now pins it. On a copy of the
code with the two indices exchanged it fails, while the diagonal test
passes.
