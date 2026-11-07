# The Fermi search of the simplex integration

Every triangle and tetrahedron mean field, the qtci mean field and the
triangle Fermi level of a normal state locate the Fermi level once per
iteration with `dmtk.simplexdm.fermi_from_energies`. The question is how
many evaluations of the charge N(mu) this needs, since at T>0 each
evaluation runs the quadrature of the Fermi function on every simplex
within 40 T of mu. Profiling the square-lattice V1 mean field at nk = 8 gave
0.226 of the 0.328 s of the loop in this search, with about 11 evaluations
per search.

## What changed

The old search was brentq on the whole band range with `xtol=1e-14`,
followed by two evaluations at mu0 +- 1e-8 to check for a gap, and two more
brentq calls when the charge was held on an interval. The new search is a
safeguarded Newton iteration, in the shape of rtsafe: the bracket is kept,
and bisection takes over when a Newton step leaves the bracket or does not
halve the previous step. The derivative dN/dmu is the density of states of
the same simplices. A new kernel, `_charge_dos_jit`, returns the charge and
the DOS in one pass. At T>0 it reads both at the same knots and
Gauss-Legendre nodes as `_weights_T` and `_dos_T`. It sums in blocks of
1024 simplices, so the sum does not depend on the number of threads. The
search starts from the Fermi level of the uniform mesh: each state is
counted as a step that is half occupied at its energy, and the result is
interpolated between two states, which costs one `np.partition` and no
charge evaluation.

Two things make one evaluation cheaper too. The occupied fraction of a
simplex is now written in closed form, the sum of the vertex weights,
instead of being taken from `_step_weights`. `_step_weights` allocates
small arrays in `_cut` and `_vert` at every call, and at T>0 it is called
at every quadrature node. The closed form agrees with the sum of the
weights to 8e-16 on random simplices with p = 2, 3 and 4, ties included.
The simplices are also built once per search and not once per evaluation.
`simplex_weights` and all of its helpers are unchanged, so the density
matrix, the anomalous Hall conductivity and the chi weights are untouched.
`simplex_charge` now goes through the new kernel.

A flat simplex, with all its vertex energies equal, is a Fermi step
f(e - mu). Its derivative f(1-f)/T is its DOS at T>0, while `_dos_T`
returns zero for it. The new kernel counts that derivative, so a flat band
at the Fermi level does not look like a gap.

## The gap

The task assumed that at T>0 the charge is strictly monotonic and the gap
check is not needed. In exact arithmetic this is true, but not for the
charge we compute. `_weights_T` cuts the Fermi function at 40 T, so in a
gap wider than 80 T the computed N(mu) is exactly flat on
[E_v + 40 T, E_c - 40 T]. Here E_v is the highest vertex energy below and
E_c the lowest one above. The DOS is exactly zero there, and a Newton
iteration alone would bisect to an arbitrary point of the plateau. So the
search keeps the plateau rule at every T. When the DOS is exactly zero and
the charge is held to 1e-12, it returns (E_v + E_c)/2, the middle of the
gap between the vertex energies. This takes no extra evaluation, since
E_v and E_c are read from the band energies. When the charge is not held
there, the bracket jumps over the plateau.

This is not exactly what brentq returned. The old code took the middle of
the interval where |N - target| <= 1e-12 nb, with the edges placed inside
the band tails. At T=0 the tails are sharp, and on the asymmetric gapped
honeycomb below the two answers agree to 7e-15. At T>0 the old edges sit
some tens of T inside the gap, at a distance set by the DOS of each
band edge. On the asymmetric honeycomb the old level is 1.2e-4 above the middle
at T=1e-3 and 2.4e-3 above it at T=1e-2. On the symmetric one the old
level is the middle up to roundoff, but that roundoff is 1.1e-8 at
T=1e-3, because N is flat to 1e-15 at the old edges.

At T=0 a bracketing search can also miss the middle completely. When the
flat charge comes out 1e-16 below the target, brentq sees no sign change
inside the gap and stops at the conduction edge. The check at mu0 +- 1e-8
then sees the band above, takes the level for a metal, and returns E_c.
This happens with both gapped honeycomb models at nk = 32 when the charge
is the sum of `simplex_weights`. The old search itself, which used
`simplex_charge`, returned the middle on these two models at nk = 32 and
64, to 9e-15. So the old answer was right here, but it depended on the
sign of a roundoff. The new search does not depend on it, since the plateau is
found from D = 0 and not from the sign of N - target.

None of this has an effect on any computed quantity. On a plateau every weight is constant in mu,
the Bloechl term included (D = 0), so the density matrix is identical, and
the band energy summed with H - mu plus mu N does not move either.

## Accuracy against the old search

Measured with the same script on the aacdd26 tree and on the new one. The
metals are spinless at filling 0.3. The gapped honeycomb has a staggered
onsite energy of +-0.5 (symmetric case), or +-0.6 together with a
second-neighbor hopping of 0.2 (asymmetric case), both at half filling.

| case | nk | T | old evaluations | new evaluations | abs(mu_new - mu_old) |
|---|---|---|---|---|---|
| square metal | 64 | 0, 1e-3, 1e-2 | 11 | 4 | 1e-15 at most |
| square metal | 256 | 0, 1e-3, 1e-2 | 10 to 11 | 3 | 9e-16 at most |
| honeycomb metal | 256 | 0, 1e-3, 1e-2 | 9 to 11 | 3 | 1.2e-14 at most |
| chain | 400 | 0, 1e-3, 1e-2 | 9 to 10 | 1 to 3 | 0 |
| cubic metal | 16^3 | 0, 1e-3, 1e-2 | 11 | 4 | 6.5e-15 at most |
| cubic metal | 32^3 | 0, 1e-2 | 11 | 4 | 7.4e-15 at most |
| symmetric insulator | 64 | 0 | 145 | 1 | 0 |
| symmetric insulator | 64 | 1e-3, 1e-2 | 52 to 56 | 1 | 1.1e-8, 2.3e-10 (old roundoff) |
| asymmetric insulator | 64 | 0 | 145 | 1 | 7e-15 |
| asymmetric insulator | 64 | 1e-3, 1e-2 | 52 to 54 | 1 | 1.2e-4, 2.4e-3 (see the gap) |

The weights of `simplex_weights` at the new level hold the filling to
3.3e-16 in every case. The test `tests/scf/test_simplex_fermi_search.py`
checks the metals (1D, 2D and 3D, at T = 0, 1e-3 and 1e-2) against brentq
on the sum of the weights to 1e-12. It checks both insulators against the
middle of the gap, and the old answer inside the gap. It also checks the
DOS against a finite difference of the charge, a flat band at T=0 and
T>0, and at most five evaluations per search, counted by wrapping
`_charge_and_dos`.

Newton converges only linearly where the DOS vanishes as the level is
approached: a band edge at a tiny filling, or a Dirac point that is not
on the mesh. A flat band at T=0 with the filling inside its jump has no
root, and the search bisects to the energy of the band in about 50
evaluations, as brentq did. The count of five holds away from these
cases.

## Timing

A desktop workstation, three performance cores, numba and BLAS at three threads,
the same script on the two trees, the best of three calls after a warm-up.
The time is that of `fermi_from_energies` on given band energies.

| case | T | aacdd26 (s) | new (s) | speedup |
|---|---|---|---|---|
| square, nk = 256 | 0 | 0.267 | 0.0104 | 26 |
| square, nk = 256 | 1e-3 | 0.446 | 0.0171 | 26 |
| square, nk = 256 | 1e-2 | 0.547 | 0.0319 | 17 |
| honeycomb, nk = 256 | 0 | 0.218 | 0.0083 | 26 |
| honeycomb, nk = 256 | 1e-3 | 0.501 | 0.0229 | 22 |
| honeycomb, nk = 256 | 1e-2 | 1.47 | 0.0657 | 22 |
| cubic, 32^3 | 0 | 0.552 | 0.0192 | 29 |
| cubic, 32^3 | 1e-2 | 20.8 | 0.203 | 102 |

The acceptance asked for 3 at 32^3 and T=1e-2 and for 2 at nk = 256 and
T=1e-3. Roughly a factor 3 comes from the evaluations, 11 against 3 or 4.
The rest is the cost of one evaluation, a factor 7 at T=0 in 2D and 37 at
T=1e-2 in 3D, from the closed-form fraction at the quadrature nodes.

The triangle mean field of the spinless square lattice (`Vinteraction`,
V1 = -1.5, filling 0.3, T = 1e-7, `mf` random, six iterations, so seven
Fermi searches) was timed in the same way:

| nk | aacdd26 per iteration (s) | new per iteration (s) | Fermi searches, old to new (s) | evaluations, old to new |
|---|---|---|---|---|
| 64 | 0.120 | 0.029 | 0.686 to 0.066 | 75 to 23 |
| 128 | 0.224 | 0.069 | 1.22 to 0.143 | 73 to 21 |

The Fermi search time includes the diagonalization of the mesh for the
band energies, which now dominates it. So the loop is now set by the
density matrix and the diagonalization, and not by the search.

## Open

`tetrahedron_density_matrix.md` still quotes 33 s for the Fermi search at
32^3 and finite temperature, and a factor 20 between T>0 and T=0. Both
numbers are from the old search. That file is outside this change, so it
was left as it is.
