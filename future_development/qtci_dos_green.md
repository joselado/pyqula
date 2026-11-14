# The triangle method and QTCI for the density of states and the bulk Green's function

This note records what the linear simplex integration (segments in 1D, triangles in 2D,
tetrahedra in 3D) and its QTCI variant give for the density of states and for the bulk
Green's function of a crystal, with the measurements that decided what went into the code.
It follows `qtci_optical_response.md`, which found the same structure for the optical
conductivity. What is built is in `dostk/simplexdos.py`, reached as
`h.get_dos(integration="triangle")` and `h.get_dos(integration="qtci")` (2D and 3D), with
tests in `tests/dos/test_dos_simplex.py`, `tests/dos/test_dos_qtci_3d.py` and
`tests/green/test_green_simplex.py`. The local density
of states of every site uses the same kernel, reached as `h.get_ldos(integration="triangle")`
and `h.get_multildos(integration="triangle")`. What is open is at the end.

## The construction

The DOS is rho(E) = sum_n int dk w_n(k) delta(E - E_n(k)), and at a finite broadening it is
-Im of the same sum with 1/(E + i delta - E_n(k)) divided by pi. The bands E_n(k) and the
weights w_n(k) are smooth; only the resolvent is sharp. So we interpolate E_n and w_n
linearly inside each simplex of a uniform mesh and integrate the resolvent over the simplex
exactly. This is the analytic tetrahedron method of Lambin and Vigneron (PRB 29, 3430
(1984)), the same Hermite-Genocchi formula as `conductivitytk/kubosimplex.py`: the simplex
average of 1/(z - E) is d! times the d-th divided difference of the d-th antiderivative of
1/(z - E) over the vertex energies. The result is exact for any delta, delta = 0 included,
where -Im/pi is the density of a linear function on the simplex (the tent in 2D).

The kernel was first written apart from the one of `kubosimplex.py`, for one reason. There
the divided difference switched to a Taylor series when the spread of the vertex energies
was below 1e-2 delta. The cancellation of a plain divided difference grows like
(|E - z|/spread)^d, so for a nearly flat simplex far from resonance it loses digits even
when the spread is much larger than delta. At the delta of an optical spectrum this
matters little; at delta = 1e-6, or 0, it matters a lot: for a triangle at energies 0.5,
0.5 + 0.4 s, 0.5 + s, with E = 0 and delta = 1e-6, that rule has a relative error of 0.10,
0.027 and 4e-4 at s = 3e-8, 1e-7 and 1e-6, against a 50-digit evaluation, and the kernel
here is exact to rounding. This kernel switches on the spread over |c - z|, with c the
mean vertex energy, and expands every cluster of close nodes in its Taylor series with
the complete homogeneous symmetric polynomials, up to 14 terms. One routine covers d = 1,
2 and 3 and any number of confluent nodes. Over a map of spread from 1e-4 to 100 delta
and |c - z| up to 1e6 delta, its worst relative error on one triangle is 1.6e-12, at the
edge of the switch. The direct kernel of `kubosimplex.py` now calls this routine. The
source kernel there cannot use it, because it fixes the branch of a triangle once for all
frequencies; see the item on the cost per frequency in `qtci_optical_response.md`.

The weights are the vertex mean over each simplex, averaged over degenerate multiplets at
each k, as in `kubosimplex.py`. For the DOS the weight is one, or <u_n|O|u_n> for an
operator with a matrix. For the bulk Green's function G_ij(z) = (1/N) sum_k (z - H_k)^-1 it
is the band projector u_n,i u_n,j^*, so the same kernel gives the full matrix.

## The DOS of graphene

Spinless first-neighbor graphene, t = 1, 321 energies from -3.2 to 3.2 in steps of 0.02,
so the grid hits the van Hove energies +-1 and the Dirac point. The reference is the exact
DOS (the elliptic-integral formula), convolved with the Lorentzian numerically for finite
delta. The error is the largest absolute error over the grid, in states per t per unit
cell, where the DOS is of order 0.4 in the band and about 1.5 at the van Hove peak at
delta = 1e-3. In parentheses, the largest error at more than 0.05 from +-1 and 0.
Diagonalizations of the 2 x 2 Bloch Hamiltonian; wall time on four cores, for all 321
energies.

delta = 1e-3:

| method | diagonalizations | max error | time (s) |
|---|---|---|---|
| uniform nk = 1024 | 1.0M | 0.34 (0.026) | 1.5 |
| uniform nk = 2048 | 4.2M | 0.073 (5e-3) | 5.9 |
| uniform nk = 4096 | 16.8M | 6.9e-3 (5e-4) | 23 |
| triangle nk = 128 | 16k | 0.017 | 1.1 |
| triangle nk = 256 | 65k | 4.4e-3 | 3.2 |
| triangle nk = 512 | 262k | 1.1e-3 | 11 |
| triangle nk = 1024 | 1.0M | 7.7e-4 (2.8e-4) | 39 |
| triangle nk = 2048 | 4.2M | 7.6e-4 (6.9e-5) | 143 |
| qtci R = 8 | 27k | 4.4e-3 | 4.1 |
| qtci R = 9 | 41k | 1.1e-3 | 12 |
| qtci R = 10 | 52k | 8.0e-4 (2.8e-4) | 39 |
| qtci R = 11 | 65k | 7.8e-4 (6.9e-5) | 140 |
| qtci R = 12 | 74k | 7.7e-4 (1.4e-5) | 532 |

delta = 0 (an energy grid shifted by 0.0037, so that no point sits on a singularity):

| method | diagonalizations | max error | time (s) |
|---|---|---|---|
| triangle nk = 256 | 65k | 0.016 (7.5e-3) | 0.18 |
| triangle nk = 512 | 262k | 2.1e-3 | 0.6 |
| triangle nk = 1024 | 1.0M | 5.1e-4 | 2.3 |
| triangle nk = 2048 | 4.2M | 2.6e-4 (7.9e-5) | 9.1 |
| qtci R = 9 | 41k | 2.1e-3 | 1.5 |
| qtci R = 10 | 51k | 5.2e-4 | 1.9 |

The qtci tolerance is 1e-6; at 1e-8 the error does not move and the number of
diagonalizations roughly doubles. At delta = 1e-2 the triangle mesh at nk = 1024 agrees with
the reference to 8e-5 (3e-5 away from the singular points), and the uniform mesh converges
to 7.6e-5 at the van Hove points and 2e-6 elsewhere from nk = 2048 on; so the reference
itself is good to about 1e-4 at +-1 and much better elsewhere. At delta = 1e-3 the
reference at exactly E = +-1 is worse, about 7.6e-4 (the numerical convolution of the log
singularity with a narrow Lorentzian): the triangle mesh at nk = 2048 and at nk = 2049 both
give 7.6e-4 there, while away from the singular points the error keeps falling by four per
doubling, 2.8e-4 to 6.9e-5. So the rows below 1e-3 are limited by the reference at +-1, and
the number in parentheses is the one that measures the method.

What we learn from this table:

- At delta = 1e-3 the triangle mesh reaches the error of the uniform mesh at nk = 4096 with
  40k to 65k diagonalizations instead of 16.8M, a factor of 250 to 400. The uniform mesh
  needs a k-spacing below delta/v, while the triangle mesh only needs to resolve the bands.
- At delta = 0 there is no uniform-mesh counterpart at all. The triangle DOS converges to
  the exact DOS of graphene, at close to second order away from the van Hove energies.
- QTCI of the two band energies saturates at 50k to 75k diagonalizations, with ranks of 45
  for both bands, even though the bands have a cone at K. It adds a factor of 6 at 1e-3,
  20 at 2.8e-4 and 65 at 6.9e-5 (away from the singular points) over the triangle mesh, and
  R = 12 reaches 1.4e-5 with 74k diagonalizations, where the triangle mesh would need about
  16.8M by its second-order convergence (not run). The cost moves to the simplex sum on the
  4096 x 4096 mesh (532 s for 321 energies).
- In wall time the 2 x 2 model is a mixed result. Diagonalizing a 2 x 2 matrix costs less
  than the simplex sum at 321 energies (about 0.15 to 0.4 microseconds per simplex, band and
  energy), so the triangle mesh at nk = 512 takes 11 s against 23 s for the uniform mesh at
  nk = 4096, and qtci saves diagonalizations but not time. At delta = 0 the kernel only
  visits the energies inside each triangle and the cost is a few seconds. For a
  Hamiltonian of tens of orbitals the diagonalization dominates, and in principle the gain
  in time follows the gain in diagonalizations; this was not measured.

## A gapped four-band model

Spinful honeycomb with Kane-Mele 0.05, Rashba 0.2, sublattice imbalance 0.3 and an
exchange field (0.1, 0.05, 0.15), four non-degenerate bands with a gap of 0.2 around zero.
341 energies from -3.4 to 3.4, shifted by 0.0037. There is no exact DOS here, so the
reference at delta = 1e-3 and at delta = 0 is the triangle mesh at nk = 2048; the errors of
the triangle rows are therefore a self-convergence, and those of the uniform and qtci rows
are against it. The independent check is at delta = 0.1, where the triangle mesh at nk = 32,
64 and 128 agrees with a uniform mesh at nk = 2048 to 8e-3, 2e-3 and 5e-4 (the DOS is of
order 1), which is second order. The DOS peaks at 1.5 at delta = 1e-3.

delta = 1e-3:

| method | diagonalizations | max error | time (s) |
|---|---|---|---|
| uniform nk = 1024 | 1.0M | 0.030 | 3.7 |
| uniform nk = 2048 | 4.2M | 6.3e-3 | 15 |
| uniform nk = 4096 | 16.8M | 5.0e-4 | 57 |
| triangle nk = 256 | 65k | 4.1e-3 | 6.7 |
| triangle nk = 512 | 262k | 9.7e-4 | 23 |
| triangle nk = 1024 | 1.0M | 1.9e-4 | 81 |
| qtci R = 8 | 46k | 4.1e-3 | 9.1 |
| qtci R = 9 | 79k | 9.6e-4 | 25 |
| qtci R = 10 | 112k | 1.6e-4 | 84 |

delta = 0:

| method | diagonalizations | max error | time (s) |
|---|---|---|---|
| triangle nk = 256 | 65k | 8.0e-3 | 0.26 |
| triangle nk = 512 | 262k | 1.6e-3 | 0.9 |
| triangle nk = 1024 | 1.0M | 4.5e-4 | 3.6 |
| qtci R = 9 | 78k | 1.6e-3 | 4.6 |
| qtci R = 10 | 99k | 5.9e-4 | 5.6 |

The picture is the same as for graphene. The triangle mesh matches the uniform mesh at
nk = 4096 with about 400k diagonalizations, a factor of about 40 (the uniform mesh does
better here than for graphene because there is no Dirac point). QTCI saturates at about
100k diagonalizations, ranks 50 to 58 for the four bands, and adds a factor of 3 at 1e-3 and
of 9 at 2e-4.

The sz-projected DOS of the same model, the weight being <u_n|sz|u_n>, which varies with k
because of the Rashba term. At delta = 0.1 the triangle mesh at nk = 32, 64 and 128 agrees
with the uniform mesh at nk = 2048 to 4.6e-3, 1.2e-3 and 2.9e-4 (the projected DOS is of
order 0.25 there), second order again. At delta = 1e-3, where it is of order 0.55:

| method | diagonalizations | max error | time (s) |
|---|---|---|---|
| uniform nk = 2048 | 4.2M | 2.7e-3 | 19 |
| uniform nk = 4096 | 16.8M | 2.1e-4 | 75 |
| triangle nk = 512 | 262k | 8.5e-4 | 23 |
| triangle nk = 1024 | 1.0M | 1.7e-4 | 83 |
| qtci R = 9 | 153k | 8.4e-4 | 33 |
| qtci R = 10 | 248k | 1.5e-4 | 139 |

The triangle mesh gains a factor of about 16 over the uniform mesh here. QTCI now has eight
fields (four energies, four weights, ranks 48 to 74), and its gain over the triangle mesh
drops to 1.7 at R = 9 and 4 at R = 10. At delta = 0 it is the same: qtci at R = 10 gives
4.0e-4 with 245k diagonalizations, the triangle mesh at nk = 1024 4.0e-4 with 1.0M. A
tolerance of 1e-8 instead of 1e-6 raises the count to 405k to 419k and changes nothing.

## The cubic lattice in 3D

The simple cubic lattice with t = 1, one band from -6 to 6, 321 energies from -6.4 to 6.4
shifted by 0.0037. The mesh is split into six tetrahedra per cube along the main diagonal.
The independent check is at delta = 0.1, where the tetrahedra at nk = 16, 32 and 64 agree
with a uniform mesh at nk = 256 to 2.5e-3, 6.1e-4 and 1.5e-4 (the DOS peaks at 0.14), so
second order. At delta = 1e-2 and 0 the reference is the tetrahedron mesh at nk = 160.

| method | diagonalizations | max error, delta = 1e-2 | time (s) | max error, delta = 0 |
|---|---|---|---|---|
| uniform nk = 128 | 2.1M | 0.021 | 1.1 | |
| uniform nk = 256 | 16.8M | 4.3e-3 | 8.3 | |
| uniform nk = 400 | 64M | 1.4e-3 | 31 | |
| tetrahedron nk = 20 | 8k | 3.6e-3 | 2.2 | 6.2e-3 |
| tetrahedron nk = 40 | 64k | 8.8e-4 | 20 | 1.9e-3 |
| tetrahedron nk = 80 | 512k | 2.2e-4 | 135 | 8.4e-4 |

In diagonalizations the gain in 3D is the largest, about a factor of 1000 at 1e-3, as the
note on the optical conductivity expected: the uniform mesh needs (v/delta)^3 points. In
wall time, for one band, it is not a gain: the tetrahedron sum at 321 energies costs about
as much as the 64M one-band diagonalizations (20 s at nk = 40 against 31 s). At delta = 0
the sum is cheap (1.5 s at nk = 80). The interest of the tetrahedra in 3D is therefore for
many-orbital Hamiltonians, and for delta = 0.

## The bulk Green's function

The question is whether `bloch_selfenergy` should get a mode="triangle" beside the
existing mode="adaptive", which decimates exactly along one direction (Sancho-Rubio) and
integrates the other k adaptively. The model is the spinless honeycomb with a sublattice
imbalance of 0.2 and a Haldane term of 0.05, two gapped bands, at the van Hove energy of
the M point, E = sqrt(1.04) = 1.0198, and at E = 0.5 in the band. The reference is the
adaptive mode at error=1e-8; its own self-convergence (1e-6 against 1e-8 gives 1e-8) and the
convergence of the triangle mesh to it from below are the evidence that it is right. The
error is the largest absolute error over the four elements of the 2 x 2 cell Green's
function, whose largest element is 1.8 to 2.5 at the van Hove energy and 0.76 at E = 0.5.
One energy per call; the triangle time includes the diagonalization of the mesh.

At the van Hove energy:

| delta | adaptive, error=1e-5 | adaptive, error=1e-6 | triangle nk = 801 | triangle nk = 3201 |
|---|---|---|---|---|
| 1e-2 | 0.26 s, 1.5e-7 | 0.47 s, 4e-9 | 1.3 s, 1.5e-4 | 19 s, 1e-5 |
| 1e-3 | 0.63 s, 2.3e-7 | 1.1 s, 4e-9 | 1.2 s, 8.8e-4 | 18 s, 6e-5 |
| 1e-4 | 1.8 s, 2.4e-7 | 2.6 s, 6e-9 | 1.1 s, 5.7e-4 | 19 s, 6.7e-5 |
| 1e-5 | 4.7 s, 2.4e-7 | 5.2 s, 7e-9 | 1.2 s, 4.3e-3 | 19 s, 4.8e-4 |
| 1e-6 | 13 s, 5.9e-6 | 10 s, 1e-8 | 1.3 s, 5.4e-3 | 19 s, 1.1e-3 |

At E = 0.5, in the band:

| delta | adaptive, error=1e-5 | adaptive, error=1e-6 | triangle nk = 801 | triangle nk = 3201 |
|---|---|---|---|---|
| 1e-2 | 0.28 s, 3e-7 | 0.51 s, 1e-8 | 1.3 s, 2.2e-5 | 19 s, 1.4e-6 |
| 1e-4 | 0.86 s, 2.9e-7 | 1.5 s, 1e-8 | 1.2 s, 1.1e-4 | 19 s, 7.2e-6 |
| 1e-6 | 2.4 s, 2.9e-7 | 3.6 s, 1e-8 | 1.2 s, 1.3e-4 | 19 s, 1.3e-5 |

The two methods agree, which validates both: at delta = 1e-2 the triangle mesh converges to
the adaptive result at second order (9.6e-3, 2.3e-3, 6.1e-4, 1.5e-4, 3.8e-5, 9.6e-6 from
nk = 101 to 3201 at the van Hove energy). The triangle mesh behaves as expected in delta: in
the band its error does not depend on delta below 1e-4, which is the point of the method.
But the adaptive mode is not the uniform mesh of the DOS tables. It is already exact along
one direction and adaptive along the other, so it does not pay (v/delta)^2 points: its time
grows only from 0.3 s to 3 to 13 s as delta goes from 1e-2 to 1e-6, and it reaches 1e-7
where the triangle mesh at nk = 3201 gives 1e-5 in the band and 1e-3 at the van Hove energy,
where the log singularity is smoothed on the scale of the mesh. Per call the adaptive mode
wins at equal accuracy everywhere, with one exception: at the van Hove energy and
delta = 1e-6, if an accuracy of 5e-3 is enough, the triangle mesh at nk = 801 gives it in
1.3 s, while the adaptive mode needs 10 s at least (it gets slower, not faster, with a looser
error, see below).

The one place where the triangle mesh can win is a sweep, since one diagonalization of the
mesh serves every energy. For 41 energies from 0.9 to 1.1, across the van Hove energy:

| delta | method | time (s) | max error |
|---|---|---|---|
| 1e-3 | triangle nk = 801 | 4.8 | 9.6e-4 |
| 1e-3 | triangle nk = 1601 | 18 | 2.7e-4 |
| 1e-3 | adaptive error=1e-5 | 22 | 5e-7 |
| 1e-4 | triangle nk = 801 | 4.8 | 1.6e-3 |
| 1e-4 | triangle nk = 1601 | 18 | 4.5e-4 |
| 1e-4 | adaptive error=1e-5 | 39 | 8e-7 |

So for a sweep at an accuracy of 1e-3 the triangle mesh is 4 to 8 times faster, and for
anything tighter the adaptive mode is faster and a thousand times more accurate. A mode that
only wins at a loose tolerance on 2D sweeps is not worth a new branch of `bloch_selfenergy`,
so it is not wired in for 1D and 2D. The function stays as
`simplexdos.green_triangle(h,energies,nk,delta)`, tested against the adaptive mode and
against the DOS (-Im Tr G/pi equals the triangle DOS to rounding on the same mesh).

In 3D `bloch_selfenergy` had no mode at all, so there the tetrahedra are wired in as
`bloch_selfenergy(h,mode="triangle")`, for 3D only and for the bulk Green's function only;
in 1D and 2D this mode raises and points to `mode="adaptive"`. On a two-site diamond
crystal with a sublattice imbalance of 0.3, at delta = 0.3 t, the uniform mesh converges
fast (1e-4 at nk = 24), while the tetrahedra converge at roughly second order: the error of
the full 2x2 matrix is 3e-3 to 1.8e-2 of its largest entry at nk = 24 and 1e-3 to 6e-3 at
nk = 48, largest in the gap. So at a large broadening a uniform sum is the better choice,
and the tetrahedra are for a small delta, where the uniform mesh needs nk of order
bandwidth/delta in each of the three directions.

A side observation on the adaptive mode: a loose error makes it slow at small delta, not
fast. At delta = 1e-6 and error=1e-3 it did not finish in 900 s, at error=1e-4 it took 37 s,
and at error=1e-6 10 s. At delta = 1e-5, error=1e-3 took 197 s against 4.7 s at 1e-5. The
quadrature seems to spend its budget on a poorly resolved integrand before it reaches
the tolerance; error=1e-5 or tighter is the better default at small delta.

## The local density of states

The LDOS on site i is the DOS with the projector on site i as the weight. So the weight of
band n on site i is the sum of |u_n,a|^2 over the orbitals a of that site, its spin and
Nambu components included, averaged over degenerate multiplets like the other weights. With
an operator it is the local matrix element Re<u_n|P_i O|u_n>, the convention of every mode
of `get_ldos` since 13 November 2026 (it was <u_n|O|u_n> times the density on the site
before, see the section on the repairs of the seventh bug audit). The simplex kernel already sums one channel per weight, so one
diagonalization of the mesh and one simplex sum give every site at every energy. The code is
`simplexdos.ldos_triangle_counted` and `simplexdos.ldos_qtci_counted`, reached as
`h.get_ldos(integration="triangle")` (modes "arpack" and "diagonalization"),
`h.get_multildos(integration="triangle")` (mode "diagonalization") and
`ldos.ldosmap(h,integration="triangle")`, with "qtci" in 2D. The accepted values come from
`ldos.get_integration_names()`. The output is one value per site, so the nrep replication
and the files `LDOS.OUT`, `MULTILDOS/` and `DOSMAP.OUT` are the ones of the diagonalization
mode. For `MULTILDOS/DOS.OUT` the DOS on the ten times finer grid comes from the same band
energies, with no new diagonalization. Tests are in `tests/ldos/test_ldos_simplex.py`.

The checks are exact relations or independent paths. The weights of a band add up to one
over the sites, so the sum of the LDOS over the sites is the triangle DOS of `h.get_dos` on
the same mesh, to 1e-10 relative, in 1D, 2D and 3D, at delta = 0 and 1e-2. At delta = 0.1
the triangle LDOS at nk = 60 to 120 equals the diagonalization mode at nk = 150 to 3e-3
relative, on a Lieb lattice with three inequivalent sites and, with sz as the operator, on
spinful graphene with Rashba coupling and an exchange field along a generic direction. A
spinful Hamiltonian without spin terms gives twice the spinless LDOS to rounding, so the
spin is summed correctly.

The benchmark is spinless graphene, t = 1, with a sublattice imbalance m = 0.3, so the two
sites differ. Here there is an exact reference. The total DOS is rho_0(sqrt(E^2 - m^2))
|E|/sqrt(E^2 - m^2), with rho_0 the DOS of pristine graphene, and the band at energy E has a
weight (E + m)/(2E) on the site at +m, meaning that the two LDOS are in the ratio
(E + m)/(E - m). At delta = 1e-3 the reference is this LDOS convolved numerically with the
Lorentzian. There are 161 energies from -3.2 to 3.2, shifted by 0.0037. The error is the
largest absolute error over both sites and all energies; in parentheses, the largest error
at more than 0.05 from the singular energies (the gap edges +-m, the van Hove energies
+-sqrt(1 + m^2) and the band edges). The LDOS is of order 0.1 to 0.4 in the band. The mesh is
odd and divisible by 3 (see the flat simplices below). Wall time on four cores, for all 161
energies and both sites; "grid" is `multi_ldos` in mode "diagonalization", which is what
the triangle replaces. The machine carried a load of about three from other jobs during the
runs, so the times are an upper bound rather than a clean measurement.

| method | delta | diagonalizations | max error | time (s) |
|---|---|---|---|---|
| grid nk = 256 | 1e-3 | 66k | 1.5 (0.12) | 1.9 |
| grid nk = 512 | 1e-3 | 262k | 0.57 (0.051) | 8.1 |
| grid nk = 1024 | 1e-3 | 1.0M | 0.16 (0.018) | 33 |
| triangle nk = 129 | 1e-3 | 17k | 0.015 (9.8e-4) | 1.8 |
| triangle nk = 255 | 1e-3 | 65k | 2.7e-3 (2.6e-4) | 4.2 |
| triangle nk = 513 | 1e-3 | 263k | 2.4e-4 (5.4e-5) | 18 |
| triangle nk = 1023 | 1e-3 | 1.0M | 3.0e-4 (9.5e-6) | 53 |
| qtci R = 8 | 1e-3 | 49k | 1.6e-3 (2.4e-4) | 11 |
| qtci R = 9 | 1e-3 | 99k | 6.1e-4 (4.8e-5) | 33 |
| qtci R = 10 | 1e-3 | 125k | 4.5e-4 (1.1e-5) | 78 |
| triangle nk = 129 | 0 | 17k | 0.082 (3.0e-3) | 0.05 |
| triangle nk = 255 | 0 | 65k | 0.011 (7.9e-4) | 0.14 |
| triangle nk = 513 | 0 | 263k | 0.018 (2.2e-4) | 0.57 |
| triangle nk = 1023 | 0 | 1.0M | 7.2e-3 (7.5e-5) | 1.8 |
| qtci R = 8 | 0 | 50k | 0.026 (7.0e-4) | 6.7 |
| qtci R = 9 | 0 | 96k | 0.012 (2.2e-4) | 8.4 |
| qtci R = 10 | 0 | 125k | 4.4e-3 (7.5e-5) | 10 |

What we learn from this table:

- The picture is the one of the DOS. At delta = 1e-3 the grid at nk = 1024 is still off by
  0.16, while the triangle mesh at nk = 255 gives 2.7e-3 with 16 times fewer
  diagonalizations. Away from the singular energies the triangle error falls by four to
  five per doubling, second order. The full max stops at about 3e-4 from nk = 513 on,
  which is the numerical convolution of the reference at the van Hove energies, as for the
  DOS.
- At delta = 0 the full max does not converge smoothly. It sits at the energy closest to a
  gap edge, where the exact LDOS jumps from zero to a finite value, and where it lands
  depends on the mesh. Away from these points the error is second order. A pointwise check
  at E = 0.6, close to the gap edge, shows the same: the ratio of the two LDOS misses
  (E + m)/(E - m) by 8e-3, 1.8e-2, 1.0e-2 and 3e-4 at nk = 51, 99, 201 and 399, while at
  E = 1.4 and -2.1 it is within 2e-4 from nk = 99 on.
- QTCI has six fields here (two band energies and four site weights) and saturates at about
  125k diagonalizations, a factor of 8 below the triangle mesh at the same accuracy at
  R = 10. In wall time it loses on this two-band model, as for the DOS: the cost is in the
  cross interpolations and in the simplex sum, not in the 2 x 2 diagonalizations.
- The number of sites enters the simplex sum as the length of the inner loop over
  channels; the simplex average of the resolvent is computed once for all of them. Per
  energy the LDOS of the two sites took 0.33 s at nk = 1023, against 0.12 s for the DOS
  at nk = 1024 in the first table, which was measured in another session, so the
  comparison is rough. For qtci the number of fields is the number of bands times one plus
  the number of sites, so qtci is for small cells.

The grid row stops at nk = 1024 because `multi_ldos` in mode "diagonalization" keeps every
eigenvector of the mesh as a Python list; at nk = 2048 that is 8M arrays. The triangle path
keeps only the band energies and the site weights.

## One interpolation of all the fields (7 November 2026)

The qtci tables above ran one cross interpolation per field (each band
energy, each weight), all sharing one cache of diagonalizations. The other
choice is one interpolation of all the fields at once
(`qtcitk/meshfields.qtci_stacked_fields`), with the field index as one more
tensor site, as the mean-field density matrix does. The question is which
one needs fewer diagonalizations at the same error, without losing wall
time. The stacked TCI shares its pivots in k between the fields, so bands
with the same kinks ask for the same k-points. Its tolerance is relative to
the largest value of any field, and its rank grows with the number of
fields.

The models are the graphene and the four-band model of the sections above
(the four-band one with the sz weight, eight fields), the LDOS of gapped
graphene of the section on the local density of states (six fields), and a
2x2 and a 4x4 supercell of graphene with a sublattice imbalance of 0.3 and
a random onsite energy of width 0.2 t (8 and 32 bands). delta = 1e-3,
tolerance 1e-6, 321 energies (341 for the four-band model, 161 for the
LDOS). The error is max|rho_qtci - rho_triangle| / max|rho_triangle| over
the energies (and the sites), with the triangle on the same 2^R mesh, so it
is only the interpolation error. As a reference, the triangle itself moves
by 1.2e-3 from nk = 256 to 512 on graphene (1.0e-3 from 512 to 1024, the
van Hove peak), 5.1e-3 on the four-band sz DOS, 1.2e-3 on the LDOS and
3.7e-2 on the 2x2 supercell, 8.5e-3 on the 4x4 one and 5.4e-2 on the LDOS
of the 2x2 one, all from 256 to 512. So every
interpolation error below is at least 30 times smaller than the error of
the mesh. The rank is the largest rank of one field for the per-field
form, the largest bond dimension for the stacked one. Wall times of the
whole call, 321 energies included, on two cores of a workstation that
other jobs kept busy (load 5 to 10).

| model | R | form | diagonalizations (share) | rank | error | time (s) |
|---|---|---|---|---|---|---|
| graphene | 8 | triangle | 65.5k (1) | | | 6.3 |
| graphene | 8 | per field | 30.5k (0.47) | 46 | 6.8e-6 | 8.2 |
| graphene | 8 | stacked | 22.3k (0.34) | 45 | 6.3e-6 | 6.8 |
| graphene | 9 | triangle | 262k (1) | | | 21.2 |
| graphene | 9 | per field | 41.9k (0.16) | 47 | 8.2e-6 | 22.4 |
| graphene | 9 | stacked | 32.7k (0.12) | 45 | 8.3e-6 | 21.5 |
| graphene | 10 | triangle | 1.05M (1) | | | 76.1 |
| graphene | 10 | per field | 52.7k (0.05) | 46 | 1.0e-5 | 76.3 |
| graphene | 10 | stacked | 36.0k (0.03) | 47 | 6.0e-6 | 75.2 |
| four bands, sz | 8 | triangle | 65.5k (1) | | | 13.2 |
| four bands, sz | 8 | per field | 60.1k (0.92) | 70 | 2.6e-5 | 23.4 |
| four bands, sz | 8 | stacked | 32.2k (0.49) | 120 | 1.6e-5 | 23.5 |
| four bands, sz | 9 | triangle | 262k (1) | | | 61.7 |
| four bands, sz | 9 | per field | 156k (0.60) | 73 | 2.1e-5 | 69.7 |
| four bands, sz | 9 | stacked | 42.5k (0.16) | 121 | 1.5e-5 | 63.0 |
| LDOS, gapped graphene | 8 | triangle | 65.5k (1) | | | 3.3 |
| LDOS, gapped graphene | 8 | per field | 46.9k (0.72) | 52 | 6.8e-6 | 6.5 |
| LDOS, gapped graphene | 8 | stacked | 19.9k (0.30) | 55 | 1.2e-5 | 3.9 |
| LDOS, gapped graphene | 9 | triangle | 262k (1) | | | 11.2 |
| LDOS, gapped graphene | 9 | per field | 93.8k (0.36) | 54 | 3.3e-5 | 17.9 |
| LDOS, gapped graphene | 9 | stacked | 27.3k (0.10) | 57 | 1.3e-5 | 16.4 |
| supercell 2x2 | 8 | triangle | 65.5k (1) | | | 21.5 |
| supercell 2x2 | 8 | per field | 65.5k (1.00) | 116 | 7.2e-6 | 35.3 |
| supercell 2x2 | 8 | stacked | 63.8k (0.97) | 203 | 1.3e-5 | 29.0 |
| supercell 2x2 | 9 | triangle | 262k (1) | | | 76.4 |
| supercell 2x2 | 9 | per field | 245k (0.93) | 132 | 5.3e-6 | 111 |
| supercell 2x2 | 9 | stacked | 203k (0.77) | 220 | 1.0e-5 | 112 |
| supercell 4x4 | 8 | triangle | 65.5k (1) | | | 93.5 |
| supercell 4x4 | 8 | per field | 65.5k (1) | 165 | 5.9e-6 | 163 |
| supercell 4x4 | 8 | stacked, cap 400 | 65.5k (1) | 402 | 2.7e-4 | 132 |
| supercell 4x4 | 8 | stacked, cap 800 | 65.5k (1) | 588 | 4.7e-6 | 288 |
| supercell 4x4, again | 8 | per field | 65.5k (1) | 165 | 5.4e-6 | 178 |
| supercell 4x4, again | 8 | stacked, cap 800 | 65.5k (1) | 589 | 4.8e-6 | 310 |
| supercell 4x4 | 8 | stacked, cap 800, tolerance 1e-7 | 65.5k (1) | 678 | 4.5e-7 | 175 |
| LDOS, supercell 2x2 | 8 | triangle | 65.5k (1) | | | 15.6 |
| LDOS, supercell 2x2 | 8 | per field | 65.5k (1) | 152 | 8.9e-6 | 151 |
| LDOS, supercell 2x2 | 8 | stacked, cap 400 | 65.3k (1.00) | 400 | 1.3e-4 | 52.3 |
| LDOS, supercell 2x2 | 8 | stacked, cap 800 | 65.5k (1.00) | 502 | 1.0e-5 | 76.1 |

What we read from it:

- With few fields the stacked form wins in diagonalizations, at the same
  error: 1.3 to 1.5 times fewer on graphene (two fields), 3.7 times fewer on
  the sz DOS at R = 9 (42.5k against 156k, eight fields) and 3.4 times fewer
  on the LDOS at R = 9 (27.3k against 93.8k, six fields). In wall time the
  two forms are within 10% on the DOS, because the simplex sum at 321
  energies is most of the time and the diagonalizations of two or four
  bands are cheap; the interpolation alone takes 6 s against 16 s on the sz DOS at
  R = 9 and 1.6 s against 4.8 s on the LDOS.
- On the disordered supercells qtci saves little or nothing in either
  form: 77 to 100% of the mesh is diagonalized, because the random onsite
  energy turns the band crossings into narrow anticrossings, kinks of the
  bands ordered by energy. The plain triangle on the same mesh is then as
  fast or faster.
- The stacked rank grows with the number of fields. On the 4x4 supercell
  (32 band energies) it stopped at the cap of 400 of one field, before its
  tolerance, with an error of 2.7e-4, 45 times the per-field one. With a
  cap of 800 it reaches rank 588 and the per-field error, 4.7e-6, but the
  interpolation then takes 190 s against 86 s, and the whole call 288 s
  against 163 s. A second run gave 200 s against 101 s for the
  interpolation and 310 s against 178 s for the call. The stacked form at a
  tolerance of 1e-7 took only 99 s for the interpolation in one run; we do
  not trust that number, because on another machine the stacked TCI ran the
  same six sweeps at 1e-6 and 1e-7, at rank 586 and 680, in 291 s and
  266 s, while one TCI per field took 178 s there.
- The LDOS of the 2x2 supercell has more fields (72, the 8 band energies
  and 64 site weights), and there the stacked form at a cap of 800 is the
  faster one: rank 502, the per-field error, 76 s against 151 s. So what
  decides is not the number of fields alone. In principle one TCI per field
  costs the number of fields times the cost of one field, while the cost of
  the stacked TCI grows faster than linearly with its rank, and the 32 band
  energies of the 4x4 cell are few fields of high rank (165 each).

So the answer differs by model for the DOS: the stacked form needs up to
3.7 times fewer diagonalizations with a few fields, and on the 32-band
supercell it is 1.7 to 1.8 times slower for the whole call at the same
error, and 2.0 to 2.2 times for the interpolation alone. The qtci DOS keeps
one interpolation per field (`meshfields.qtci_fields`) as the default. The
qtci LDOS, where the stacked form won or tied on every model (3.4 times
fewer diagonalizations on gapped graphene at R = 9, twice as fast on the
supercell), uses it now, with a rank cap of 800
(`meshfields.stacked_maxbonddim`, shared with the qtci optical
conductivity). There is no keyword to choose the form for the DOS: the
stacked form wins by more than a factor 2 on some models (in
diagonalizations), but on the 32-band supercell it loses by 1.7 to 1.8 in
the wall time of the call, not by more than 2, and qtci saves nothing on
that model in either form. The graphene tables above keep the numbers of
the per-field form, which is still the one of the DOS.

## The qtci DOS in 3D (7 November 2026)

The question is whether qtci pays in 3D, where the mesh grows as nk^3. The
qtci DOS now runs on a 2^R x 2^R x 2^R mesh, with three quantics
coordinates, and the tetrahedra of the section on the cubic lattice are
summed on the reconstructed band energies. The quantics mesh is in
`qtcitk/meshfields.py`, shared with the qtci density matrix of the mean
field. The test is `tests/dos/test_dos_qtci_3d.py`: on the same mesh the
qtci DOS is the tetrahedron DOS to the interpolation tolerance, in both bit
orderings, and a stack of decoupled layers gives the 2D qtci DOS of the
layer. For that last test note that the six tetrahedra of a cube share its
main diagonal, so their projections on the layer are the two triangles of
the square cut along (0,0)-(1,1), while the 2D mesh cuts it along
(1,0)-(0,1). So the stack is the layer written in the lattice basis
(a1,-a2). With unit weights the two agree exactly, because a band that does
not depend on kz is linear in each tetrahedron in the same way as in the
triangle under it, and the three tetrahedra above one triangle fill its
prism uniformly.

### The bit ordering

A mesh point has three coordinates of R bits each. We tried two orders of
the bits along the tensor train, both with the scales in order: interleaved,
3R sites of dimension 2 (x1 y1 z1 x2 y2 z2 ...), and fused, R sites of
dimension 8, one per scale. The fused train only has the bonds between two
scales, so its largest rank is lower, but on those bonds the two orders give
the same rank: 3, 6, 6, 6, 4 on the cubic band at R = 6 in both, and at most
154 (interleaved) against 159 (fused) on the gapped layered model below at
R = 6. What differs is the number of points: a site of dimension 8 makes
every update sample 16 times more entries than a site of dimension 2 at the
same rank. Tolerance 1e-6, one TCI per band:

| model | R | ordering | diagonalizations (share) | largest rank | time (s) |
|---|---|---|---|---|---|
| cubic | 5 | interleaved | 3151 (0.096) | 9 | 0.32 |
| cubic | 5 | fused | 5015 (0.153) | 6 | 0.32 |
| cubic | 6 | interleaved | 3492 (0.013) | 9 | 0.76 |
| cubic | 6 | fused | 13142 (0.050) | 6 | 0.88 |
| layered semimetal | 5 | interleaved | 32763 (1.00) | 120 | 1.03 |
| layered semimetal | 5 | fused | 32768 (1) | 64 | 0.63 |
| layered semimetal | 6 | interleaved | 231706 (0.88) | 193 | 4.0 |
| layered semimetal | 6 | fused | 262144 (1) | 188 | 3.9 |
| layered insulator | 5 | interleaved | 32344 (0.99) | 110 | 0.99 |
| layered insulator | 5 | fused | 32768 (1) | 64 | 0.64 |
| layered insulator | 6 | interleaved | 223319 (0.85) | 159 | 4.2 |
| layered insulator | 6 | fused | 262144 (1) | 159 | 3.4 |

So the rank where the two can be compared is the same, and the interleaved
order needs 1.6 to 3.8 times fewer diagonalizations on the cubic band and
12 to 15% fewer on the layered models at R = 6. The code keeps the
interleaved order in 3D (`meshfields.ordering3d`); the other one is
`ordering="fused"` of `simplexdos.dos_qtci_counted`. A rank of 64 at R = 5 in
the fused train is the largest possible on that bond (8 x 8), so at 32^3 the
layered bands are not compressed at all.

### Share of the mesh and wall time

Three models. The simple cubic band, t = 1, one band. A layered two-band
model: gapped graphene layers (sublattice imbalance 0.3) stacked along z
with the interlayer hopping diag(tm,-tm), so the mass is m(kz) = 0.3 + 2 tm
cos kz. With tm = 0.3 the mass changes sign and the bands touch on two
nodal lines, the layered semimetal; with tm = 0.1 the mass stays between 0.1
and 0.5, the layered insulator. The DOS is at delta = 0 on 201 energies,
tolerance 1e-6, the interleaved order. The error is
max|rho_qtci - rho_tet| / max|rho_tet| with the tetrahedra on the same
mesh, so it is the interpolation error alone. "Sum" is the time of the
tetrahedron sum alone, which is the same for the two methods, since qtci
sums the same 6 n^3 tetrahedra on the reconstructed mesh. Four cores of a
workstation that other jobs kept busy (load 6 to 9 on 16 cores), so the
times are indicative. The largest resident memory was 1.9 GB, at R = 7.

| model | R | mesh | method | diagonalizations (share) | rank | error | time (s) | sum (s) |
|---|---|---|---|---|---|---|---|---|
| cubic | 5 | 32^3 | tetrahedron | 32768 (1) | | | 0.16 | 0.15 |
| cubic | 5 | 32^3 | qtci | 2628 (0.080) | 8 | 1.7e-15 | 0.29 | |
| cubic | 6 | 64^3 | tetrahedron | 262144 (1) | | | 0.63 | 0.55 |
| cubic | 6 | 64^3 | qtci | 3685 (0.014) | 9 | 1.3e-15 | 0.78 | |
| cubic | 7 | 128^3 | tetrahedron | 2.10M (1) | | | 3.71 | 2.90 |
| cubic | 7 | 128^3 | qtci | 4078 (0.0019) | 9 | 1.5e-15 | 3.16 | |
| layered semimetal | 5 | 32^3 | tetrahedron | 32768 (1) | | | 0.21 | 0.18 |
| layered semimetal | 5 | 32^3 | qtci | 32757 (1.00) | 120 | 9.2e-07 | 0.98 | |
| layered semimetal | 6 | 64^3 | tetrahedron | 262144 (1) | | | 1.03 | 0.75 |
| layered semimetal | 6 | 64^3 | qtci | 235376 (0.90) | 191 | 9.8e-07 | 5.08 | |
| layered semimetal | 7 | 128^3 | tetrahedron | 2.10M (1) | | | 6.65 | 3.70 |
| layered semimetal | 7 | 128^3 | qtci | 885066 (0.42) | 211 | 1.1e-06 | 14.9 | |
| layered insulator | 5 | 32^3 | tetrahedron | 32768 (1) | | | 0.42 | 0.22 |
| layered insulator | 5 | 32^3 | qtci | 32418 (0.99) | 109 | 1.6e-06 | 1.02 | |
| layered insulator | 6 | 64^3 | tetrahedron | 262144 (1) | | | 1.14 | 0.84 |
| layered insulator | 6 | 64^3 | qtci | 218652 (0.83) | 161 | 1.9e-06 | 4.34 | |
| layered insulator | 7 | 128^3 | tetrahedron | 2.10M (1) | | | 7.10 | 4.06 |
| layered insulator | 7 | 128^3 | qtci | 724443 (0.35) | 178 | 4.7e-06 | 10.3 | |

What we read from it:

- In diagonalizations qtci pays more in 3D than in 2D, as expected from the
  nk^3 growth of the mesh. The cubic band is a sum of three cosines, which
  has rank 9 and is interpolated exactly (an error of 1e-15), and the
  number of points barely grows with R: 2.6k, 3.7k and 4.1k, so 0.19% of the
  128^3 mesh, 500 times fewer points than the tetrahedra. On the layered
  models the share falls from 1 at 32^3 to 0.83 to 0.90 at 64^3 and 0.35 to
  0.42 at 128^3, so a factor of 2.1 to 2.4 per step in R. The rank is 110 to
  210, four times the one of graphene in 2D (about 45), since the band now
  depends on kz through the mass as well; in principle the rank of the 2D
  layer multiplies with the one of m(kz). The semimetal, with its nodal lines,
  costs 20% more points than the insulator at 128^3.
- In wall time qtci does not pay on these models. On the cubic band it is
  15% faster at 128^3 (3.2 s against 3.7 s) and slower below, and on the
  layered models it is 1.5 to 5 times slower at every R. The reason is the
  same as in 2D, only stronger: the tetrahedron sum, the same for both, is
  most of the time (2.9 s of 3.7 s on the cubic band at 128^3), so qtci can
  only save the rest: there 0.8 s of diagonalizations become 0.26 s of
  interpolation, and a 1 x 1 or
  2 x 2 diagonalization costs less than the cross interpolation at rank
  180. qtci would pay in wall time for many bands, where one diagonalization
  costs nb^3, if the bands stay smooth, and at R = 8, where the share of the
  layered models should fall again by a factor of about 2. At R = 8 the tetrahedron
  sum alone holds 100M tetrahedra, 3.2 GB of vertex indices, so that mesh
  was not run.

So 3D is where qtci saves most k-points, but not wall time: the
tetrahedron sum on the full reconstructed mesh is what costs, for both
methods, and what is left to win is the diagonalization of a few bands.

## Repairs of the seventh bug audit (13 November 2026)

The audit is `bug_audit_7.md`. All timings below were taken on a shared machine with two
numba and BLAS threads, the old code (HEAD at `9bcf52d`) and the new one run back to back,
three times each, keeping the smallest wall time; only the ratio means something. The
tests named here are in `tests/ldos/test_ldos_operator_local.py`,
`tests/dos/test_simplex_sum_order.py`, `tests/dos/test_simplex_tetrahedra.py`,
`tests/ldos/test_ldos_simplex.py`, `tests/dos/test_dos_simplex.py` and
`tests/green/test_bloch_selfenergy_batch.py`.

**The LDOS with an operator.** The weight of state n on site i was
$\langle u_n|O|u_n\rangle\,|u_n(i)|^2$ in the diagonalization modes (grid, triangle, qtci,
`get_multildos`). This product is quadratic in the projector of one state, so inside a
degenerate multiplet it depends on the basis the diagonalization returns, and the multiplet
average of the triangle code does not remove that. On the Néel state of spinful graphene
(`add_antiferromagnetism(0.5)`, every band a pair) a seed exchange of $10^{-10}$ only picks the
basis inside each pair, and the $s_z$ map at E = 1.2, delta = 0.05, nk = 30 was 0.270 on site
A with the seed along z and $6.6\times10^{-5}$ with the seed along x (triangle), 0.232 against
$6.0\times10^{-5}$ in grid mode. The weight is now the local matrix element
$\mathrm{Re}[u_n^*(a)\,(O u_n)(a)]$ summed over the orbitals of the site, which is linear in the
projector, so its sum over a multiplet is the trace $\mathrm{Re\,Tr}(P_M P_i O)$, and it is
$-\mathrm{Im}\,\mathrm{diag}(GO+OG)/2\pi$, what `mode="green"` and the KPM map already returned. The
convention was chosen during the audit, which records the alternatives it rejected. Both seeds now
give the same staggered map to $10^{-7}$ in grid, triangle and `get_multildos`, and in qtci at
nk = 32. With Rashba coupling and a tilted exchange field `mode="arpack"` and `mode="green"`
agree to $10^{-6}$ for $s_x$, $s_z$ and a random Hermitian matrix, the $s_z$ map is the local spin
density $|u(i\uparrow)|^2-|u(i\downarrow)|^2$ to $10^{-10}$, and the map summed over the sites is
`h.get_dos(operator=...)` on the same mesh to rounding. On a lattice polarized along z the
states are eigenstates of $s_z$ and the map is the old one. The sparse branch of
`get_multildos`, which keeps the `num_bands` states closest to the energy, gives the local
spin density of those states to $10^{-8}$. Eight of the twelve tests of
`test_ldos_operator_local.py` fail on the old code; the sum rules and the polarized lattice
hold for both conventions.

**The qtci LDOS with an energy offset.** The one interpolation of all the fields measures
its error against the largest value of any field, so the band energies set the absolute
tolerance of the site weights. On graphene with a sublattice imbalance of 0.3, nk = 128,
delta = 1e-3 and the default tolerance 1e-6, the error against the triangle LDOS was 2e-6 to
8e-6 of the largest LDOS, 7e-5 to 1.5e-4 with every onsite energy shifted by 30 (four runs)
and 3e-4 to 6e-4 shifted by 100 (three runs); the global pivot search draws random numbers,
so the error changes from run to run. The energies now enter the interpolation measured
from their mean over the Brillouin zone, $\mathrm{Re\,Tr}(H_0)/n$, which is exact since the
hoppings to other cells average to zero, and the mean is added back after. The error is now
1e-6 to 8e-6 at the three shifts. The rescaling of every field to order one before the
stacking, the other half of the finding, was not done in this repair; it was decided and
built on 14 November 2026, one scale per family and not one per field, see the next
paragraphs. The same normalization in
`kubosimplex.sigma_qtci`, where $W$ goes as a length squared, was repaired on 14 November
2026, with the transition energies and the weights each divided by its own maximum
(`qtci_optical_response.md`).

**The unit of energy in the qtci LDOS (decided and built, 14 November 2026).** With the
Hamiltonian, the energies and delta all multiplied by a factor lam, the LDOS is the same
LDOS divided by lam, and the triangle sum on the full mesh has this to $10^{-14}$. The qtci
LDOS did not have it. The band energies go as lam and the site weights do not, so the unit
of energy decided which of the two got the tolerance. On the same graphene (nk = 128,
delta = 1e-3 in units of the hopping, tolerance 1e-6, the error against the triangle on the
same mesh, three seeds of the pivot search each):

| lam | error before | error now |
|---|---|---|
| 1 | 2.1e-6 to 5.0e-6 | 6.7e-6 to 1.3e-5 |
| 30 | 4.4e-6 to 5.6e-6 | 6.5e-6 to 1.3e-5 |
| 1/30 | 2.8e-5 to 6.9e-5 | 4.7e-6 to 7.6e-6 |
| 1000 | 1.9e-4 to 2.1e-4 | 3.1e-6 to 1.3e-5 |
| 0.001 | 1.6e-3 to 8.7e-3 | 1.1e-5 to 1.7e-5 |

A model written in meV is the row lam = 1000. Now the energies, measured from the zone
mean, and the weights are two families of `qtci_stacked_fields`: the keyword `families`
divides each family by its largest value on the coarse scan before the interpolation and
multiplies it back after. `kubosimplex.sigma_qtci` calls the same keyword for its
transition energies and weights, so there is one implementation of the scale. The finding
asked for every field at order one. This is not what is built: a scale per field brings a
weight that is zero up to rounding (a band with no weight on a site) to order one, and
the interpolation then follows the noise at full rank.

At lam = 1 the error is larger than before. Over eight seeds it is 3.9e-6 to 1.6e-5 with
the scale (median 9.4e-6) against 1.7e-6 to 5.0e-6 without it (median 3.0e-6), so the
factor 3 is not the scatter of the seeds. We looked at which family limits the map. With
the energies scaled to a largest value kappa and the weights to 1, by hand and with the
largest values of the full mesh, four seeds each (no scale at all is kappa = 3 here):

| kappa | error of the energies | error of the weights | error of the LDOS | rank |
|---|---|---|---|---|
| 3, no scale | 0.8e-6 to 1.6e-6 | 2.9e-6 to 3.9e-6 | 2.1e-6 to 5.0e-6 | 54 to 55 |
| 1 | 1.2e-6 to 2.5e-6 | 1.0e-6 to 1.3e-6 | 2.8e-6 to 5.5e-6 | 58 to 59 |
| 10 | 1.1e-6 to 1.7e-6 | 1.1e-5 to 1.6e-5 | 3.8e-6 to 1.6e-5 | 53 to 55 |
| 100 | 1.1e-6 to 1.3e-6 | 0.8e-4 to 1.8e-4 | 3.4e-6 to 8.9e-6 | 48 to 50 |
| 3000, the bandwidth over delta | 0.9e-6 to 1.5e-6 | 1.8e-3 to 3.1e-3 | 1.6e-4 to 2.1e-4 | 42 to 43 |

The field errors are relative to the largest value of the family. The energies are
resolved to the tolerance at every kappa, and the weights to kappa times the tolerance, as
they should. The largest error of the LDOS does not follow either of them until kappa is
the bandwidth over delta, where the weights take over. Below that it moves between 2e-6
and 1.6e-5 with no order in kappa. The row kappa = 1 is the family scale applied by hand,
and it gives 2.8e-6 to 5.5e-6, where the public route gives 6.7e-6 to 1.4e-5 with the same
four seeds. The two differ in the scale of the weights by 3%: the public route takes the
largest value of the coarse scan, 0.969, and not the one of the full mesh, 0.998. So the
factor 3 at lam = 1 is real for this model and this scale, but it is not a loss of
resolution of the energies or of the weights. The largest error of the map depends on
where the errors of the energies fall against the peaks of width delta, and a change of
3% in a scale moves it as much as the factor in question. The seeds only move the global
pivot search, so their scatter is smaller than this. All these errors are 100 times below
the 1.2e-3 that the triangle LDOS itself moves from nk = 256 to 512, and the scale costs
3 to 4 in the rank. `tests/ldos/test_ldos_qtci_energy_unit.py` pins the invariance at
lam = 1e4 and 1e-4. There the old code is off by 8e-3 and 7e-3 and the new one by 5e-6 to
1e-5 and by 8e-6 to 2e-5 over five seeds, against a bound of 2e-4.

**The zero of energy in the qtci DOS (built, 14 November 2026).** The qtci DOS runs one
interpolation per field, each relative to its own largest value, so it does not depend on
the unit of energy. It did depend on the zero, as the LDOS did before its repair. With
every onsite energy shifted by c and the energies of the DOS shifted alike (the same
graphene, nk = 128, delta = 1e-3, three seeds), the error against the triangle was 7e-6
at c = 0, 3.6e-4 to 3.7e-4 at c = 100 and 4e-2 to 7e-2 at c = 1e4, with ranks 43, 31 and
16. The DOS now measures the band energies from the zone mean too
(`meshfields.zone_mean_energy`, shared with the LDOS), and the error is 2e-6 to 8e-6 at
the three shifts, with rank 43.

Measured from the mean, a flat band at the mean is rounding noise, and one interpolation
per field normalizes it by its own largest value. This was already so without a shift for
a flat band at zero: the flat band of the Lieb lattice took the full rank and every point
of the mesh (rank 64 and 4096 of 4096 diagonalizations at nk = 64, rank 256 and 65536 of
65536 at nk = 256). With the centring alone it would be so at every shift. `qtci_fields`
now takes the keyword `families` too: a field below 1e-12 of the largest field of its
family on the whole coarse scan is set to zero, as a field that vanishes there. On the
Lieb lattice the flat band then has rank 0 at c = 0 and at c = 5, with 3965 of 4096
diagonalizations at nk = 64 and 16827 of 65536 at nk = 256, and the error against the
triangle is 5e-8 and 1.6e-7 (delta = 1e-2). `tests/dos/test_dos_qtci_energy_shift.py`
pins the shift at c = 100 and 1e4, where the old code is off by 5e-4 and 5e-2 against a
bound of 7e-5, and the flat band at both shifts, where the old code has rank 64 at c = 0.

**A centre per band, and weights without a scale (after a review, 14 November 2026).** The
two paragraphs above describe the first shape of this repair, and its numbers. A review
found two defects in it, and the shape that is in the code is this one. First, the zone
mean loosens a band that is far from it. With an onsite energy of 20 on one sublattice and
0 on the other, the lower band lies within 0.45 of zero and 10 below the mean, so its
tolerance was relative to 10.45: the qtci DOS was off by 2.0e-4 to 2.6e-4, against 3.8e-6
to 7.6e-6 in the committed code. Second, the scale of the weights family of the LDOS
brought a family of rounding noise to order one. With `operator="sz"` and no magnetism
every weight vanishes up to rounding, and the LDOS at nk = 256 took 65536 of 65536 points
at rank 512, against 18405 points at rank 45 in the committed code.

Now the band energies are the centred fields of both routes (keyword `centred` of
`qtci_fields` and of `qtci_stacked_fields`): each band is measured from the middle of its
own range on the coarse scan, and the middle is added back after. Its tolerance is then
relative to its width, whatever its position and whatever constant is added to the onsite
energies, and the zone mean is not used any more. A centred field below 1e-12 of the band
energies before the shift is a constant, a flat band, and is not interpolated. In the LDOS
the band energies are also the one family that is scaled, by the largest of the centred
values, and a family that is rounding noise as a whole (every band flat) is not scaled. The
site weights have no unit, so they are in no family and enter as they are, as in the
committed code. Measured on the new code:

| case | committed code | first shape | now |
|---|---|---|---|
| DOS, band far from the mean, error | 3.8e-6 to 7.6e-6 | 2.0e-4 to 2.6e-4 | 2.0e-6 to 3.6e-6 |
| DOS of graphene, shift 0, 100 and 1e4, error | 7e-6, 3.7e-4, 4e-2 to 7e-2 | 2e-6 to 8e-6 | 1.7e-6 to 1.4e-5 |
| DOS of the Lieb lattice, nk = 256, points at shift 0 and 5 | 65536 and 18388 | 16827 and 16827 | 19524 and 21304 |
| LDOS of graphene, lam = 1, error over eight seeds | 1.7e-6 to 5.0e-6 | 3.9e-6 to 1.6e-5 | 1.0e-6 to 3.7e-6 |
| LDOS of graphene, lam = 1e4 and 1e-4, error | 8e-3 and 7e-3 | 5e-6 to 2e-5 | 1.8e-6 to 3.1e-6 and 1.2e-6 to 6.5e-6 |
| LDOS with sz, no magnetism, nk = 256, points and rank | 18405, 45 | 65536, 512 | 22218, 47 |
| LDOS with sz, exchange (0.3, 0.4, 0), nk = 256, points and rank | 25802, 51 | 65536, 565 | 37096, 67 |

So the factor 3 of the first shape at lam = 1 is gone: the error is now at the level of the
committed code or below it, at every unit. The two last rows are a cost. The energies are
now resolved against the width of a band and not against the largest distance from the
mean, which is 2 to 4 times tighter on these models, and the LDOS with sz takes 20% and
44% more points than the committed code for the same `tolerance`. The error of these two
cases against the triangle was not measured. The interpolation without the two keywords
gives the same numbers as the committed code bit for bit, on a synthetic set of fields.
`tests/dos/test_dos_qtci_energy_shift.py` has the band far from the mean (2.6e-4 on the
first shape against a bound of 3e-5), and `tests/ldos/test_ldos_qtci_energy_unit.py` the
sz weights without magnetism (65536 points on the first shape against a bound of half the
mesh).

**The simplex sum at finite delta.** The kernel gathered the weights of the p vertices of
every simplex at every energy, for every channel. It now adds the simplex average of the
resolvent onto the vertices first,
$A_n(k,E)=\sum_{s\ni k} \langle 1/(E+i\delta-E_n)\rangle_s$, and then contracts with the
weights of every mesh point in one matrix product, $\sum_{k,n} A_n(k,E)\,W_n(k)/p$, the same
sum in another order. A parallel task owns one band, one energy and one slice of the
simplices, and adds into its own row of mesh points. The energies go in blocks that keep the
resolvent near 64 MB. When a block has fewer bands times energies than there are threads,
the simplices are cut in slices, one per thread, so one band at one energy also uses every
thread; the kernel then holds one complex number per mesh point and thread. On a 3x3
honeycomb supercell (18 orbitals) with an onsite modulation,
nk = 90 and 200 energies: the DOS (one channel) 6.8 s before and 6.6 s after, the LDOS of
the 18 sites 11.0 s and 6.9 s. The simplex averages themselves are not cheaper, so the gain
is the channel part, and with one channel there is none: the resident memory of the DOS went
from 0.39 GB to 0.47 GB, for the vertex resolvent of one block of energies. On the one-band
cubic lattice at nk = 128 (12.6M tetrahedra) and 6 energies the DOS took 31.8 s and 1.73 GB
before and 24.9 s and 0.99 GB after, with a spread of 25 to 37 s between the three runs of
the new code, so in 3D the one-channel sum is not slower either.

**The 3D Green's function.** `green_triangle` kept the band projector of every band at every
k, $n^3$ numbers per k-point, about 8 GB at nk = 100 and n = 8. It now keeps the eigenvectors,
$n^2$ per k-point, averages the vertex resolvent $A_n$ over each multiplet instead of the
projectors (the two give the same sum), and builds the projectors for one chunk of the mesh at
a time. On a 2x2x2 cubic supercell (8 orbitals), one energy: nk = 40 took 5.3 s and 1.44 GB
before and 2.6 s and 0.57 GB after, nk = 60 16.9 s and 3.9 GB before and 9.1 s and 0.80 GB
after, and nk = 100, the default of `bloch_selfenergy`, runs in 38 s and 2.0 GB after; the old
code needs 15.8 GB there (see the large cases below).

**Sweeps of the 3D selfenergy.** `bloch_selfenergy(mode="triangle")` diagonalizes the mesh at
every call. `bloch_selfenergy_batch(h,energies,mode="triangle",nk=...)` now calls
`green_triangle` once for all the energies. On spinful diamond with a sublattice imbalance of
0.3, nk = 40 and 10 energies, the batch took 16.6 s before (a loop of single calls) and 6.7 s
after. One call at one energy went from 2.0 s to 1.1 s and one call at 10 energies from 7.7 s
to 6.1 s, from the simplex sum above. In 3D the simplex sum costs about as much as the mesh,
not ten times less as in 2D. The energy sweeps of the embedding now go through the batch
too, see the end of this section.

**The batch of the diagonalization.** `_Fields` diagonalized 2^15 k-points per batch
whatever the number of orbitals, except for the LDOS. The batch now goes down as $1/n^2$, so
the Bloch matrices and the eigenvectors of one batch stay near 64 MB. The triangle DOS at
delta = 0 of a 5x5 honeycomb supercell (50 orbitals) at nk = 200 took 17.9 s and 1.95 GB
before and 15.7 s and 0.53 GB after; the operator-weighted DOS of a 4x4 supercell (32
orbitals) 14.4 s and 3.5 GB before and 14.1 s and 0.78 GB after. With an operator at 50
orbitals the old code did not finish in 230 s under a cap of 6 GB, and the new one took 23 s
and 0.81 GB.

**The tetrahedra.** The (6 nk^3, 4) array of vertex indices was built as a list of six stacks,
a concatenation and a copy, about three times the array at its peak. It is now filled in place,
with the same order. At nk = 128 the build took 2.0 s with a peak of 1.33 GB before and 0.55 s
with 0.55 GB after, for an array of 0.40 GB; at R = 8 the peak is 4.4 GB in place of
10.0 GB (see the large cases below). The array is still rebuilt at every call. In
`h.get_anomalous_hall_conductivity(nk=64)` of a stack of Haldane layers with 20 Fermi levels
(one build per level) the 20 builds took 4.6 s of 23.0 s before and 1.2 s of 22.2 s after. A
cache keyed by the mesh would save that 1.2 s, but it would keep 3.2 GB alive between calls
at R = 8, so it was not added; passing the tetrahedra from the caller to each Fermi level is a
change of `topologytk/anomaloushall.py`.

**The sign of the operator-weighted DOS.** The test of the Zeeman-split copy of graphene
accepted both signs of the $s_z$-weighted DOS. It now asserts $d_0(E-J)-d_0(E+J)$, and a
second case puts the exchange along y and weights with $s_y$, which is imaginary and
antisymmetric. With the operator transposed in a copy of the code the $s_y$ case fails, while
$s_z$, real and diagonal, cannot see the transpose.

**The review of these repairs (14 November 2026).** A second pass read every repair again,
broke each repaired line on purpose in a copy of the tree, and ran the tests on that copy.
The old code was also used as a reference: 36 arrays (the 1D, 2D and 3D Green's function, the
DOS with and without an operator at delta = 0 and 0.05 in 1D, 2D and 3D, the LDOS without
an operator, the 3D selfenergy, the tetrahedra) agree between the two trees to
$2\times10^{-14}$. Three things changed.

The first is the parallel loop of the vertex resolvent. The first version had one task per
band and energy. On a mesh above $2^{22}$ points a block holds one energy, so a one-band
model ran on one thread, and a sweep of energies lost the parallel loop over energies that
the old kernel had. On the one-band cubic lattice at nk = 162 (25.5M tetrahedra), 4
energies and delta = 0.05 it took 46 s against 33 s for the old kernel. This was measured
with two threads only; with more threads the old kernel uses one per energy, and the first
version stayed on one. With the slices of the simplices described above
it takes 29 s, with a peak of 1.6 GB against 3.1 GB for the old code. The kernel also adds
directly into its output now, so it no longer holds a second copy of each row. The other
ratios hold with the final kernel: the LDOS sum of 18 sites 11.3 s before and 6.7 s after,
the one-channel DOS 6.2 s and 6.2 s, the batched 3D selfenergy of spinful diamond at nk = 40
and 10 energies 17.4 s and 6.5 s, the Green's function at 8 orbitals and nk = 40 5.2 s
and 1.44 GB before and 2.4 s and 0.58 GB after. The timings of the batch of the
diagonalization and of the tetrahedra were not repeated. The sliced kernel was timed here
with two threads; with eight threads see the large cases below.

The second is a test that did nothing. The test of the multiplet average of the Green's
function used the spin pairs of the diamond lattice, which are degenerate at every k. There
the vertex resolvents of a pair are equal, so the average changes nothing, and the test
passed with the average removed. The average matters at a point degeneracy of the mesh,
where the neighbouring points are not degenerate. Graphene with K on the mesh does not work
either: the two K points cancel each other for the basis LAPACK returns. The new test uses
the chain written with two sites per cell at an even nk, whose two bands touch at the zone
boundary only. It fails without the average, and it also asserts that the two sites have
the same local Green's function. The old code gives the same matrix on that chain to
rounding.

The third is `delta_discard` of the grid LDOS (`ldostk/ldoswaves.py`), which drops the
states far from the energies of the maps. The window was
$[\min E-d\,\delta,\ \min E+d\,\delta]$, so with several energies the states at the upper
energies were dropped. It is now $[\min E-d\,\delta,\ \max E+d\,\delta]$. The test in
`tests/ldos/test_ldos_delta_discard.py` asserts that the discard changes each map by less
than the Lorentzian tail $1/(\pi d^2\delta)$; it fails on the old window.

The keyword `nk` of `bloch_selfenergy_batch` is now in its signature and not read from the
other keywords, and a test checks that the fallback loop receives it. The decisions of the
first pass stand: no cache of the tetrahedra (the first pass measured the
build at 5% of a sweep of 20 Fermi levels at nk = 64, and a cache keeps 3.2 GB alive at
R = 8), no cache of the eigenvectors of the 3D selfenergy, the vertex form for every kind,
one degeneracy tolerance for the whole mesh of the Green's function, and the energies of
the qtci LDOS centred without a rescaling of the weights (the last one was decided again
on 14 November 2026: the energies and the weights are scaled as two families, see above).

**The large cases, on a compute node (14 November 2026).** The cases that do not fit in the
6 GB of the shared machine were run on a compute node with 32 GB, with two numba threads and
one BLAS thread. The old tree is `9bcf52d`, and the new one is the same tree with only the
four files of these repairs (`simplexdos.py`, `selfenergy.py`, `ldos.py`, `ldoswaves.py`).
Each case ran twice, old and new back to back, and the checksums of the two trees agree
to all printed digits in every case. The smallest wall time of the two runs and the peak
resident memory:

| Case | Old | New |
|---|---|---|
| `simplices(3,256)`, the R = 8 mesh, an array of 3.2 GB | 25.0 s, 10.0 GB | 4.9 s, 4.4 GB |
| `green_triangle`, 2x2x2 cubic supercell (8 orbitals), nk = 100, one energy | 88.8 s, 15.8 GB | 29.6 s, 2.0 GB |
| triangle DOS at delta = 0, 7x7 honeycomb supercell (98 orbitals), nk = 200, 11 energies | 33.8 s, 6.1 GB | 31.7 s, 0.54 GB |
| the same with `operator="sublattice"` | 47.3 s, 23.8 GB | 44.9 s, 0.73 GB |

So at the default nk = 100 of the 3D selfenergy the Green's function now fits in 2 GB, and
it is three times faster, because the old code spent its time filling the projectors. The
batch of the diagonalization buys memory and not time: with an operator, 98 orbitals need
33 times less memory. The sliced vertex kernel was run on the same node with eight numba
threads, on the one-band cubic lattice at delta = 0.05: at nk = 128 and 12 energies it took
22.2 s before and 12.2 s after, and at nk = 162 and 4 energies 24.9 s before and 9.0 s
after, with a peak of 2.4 GB against 3.0 GB. So with many threads the new kernel is faster
than the old one, not only with two.

**The energy sweeps of the embedding (14 November 2026).** With `gf_mode="triangle"`, `Embedding.multildos`, `Embedding.multidos` and `embedding.dos_impurity` called the selfenergy of one energy at every energy, so they diagonalized the nk^3 mesh at every energy. They now take the bulk selfenergy of the whole sweep from `green.supercell_selfenergy_batch`, which calls `bloch_selfenergy_batch`, and then do the Dyson step energy by energy. Every other `gf_mode` still goes energy by energy and keeps only the map of each energy, because the Green's functions of a large supercell at every energy of a sweep take a lot of memory (about 180 MB for the 7x7 honeycomb supercell and 100 energies of `examples/embedding/honeycomb_supercell_multildos`), so outside the tetrahedra nothing moves. The batch saves the diagonalization and not the simplex sum, which is still done per energy. On spinful diamond with a sublattice imbalance of 0.3 (4 orbitals) at nk = 40, one energy takes 1.33 s and the mesh is about 0.3 s of it, so the gain is small: with 10 energies `multildos` went from 11.3 s to 9.5 s and `dos_impurity` from 11.2 s to 8.7 s. The diagonalization weighs more with more orbitals. On a 2x2x2 supercell of the cubic lattice (16 orbitals) with an onsite modulation 0.3 cos(1.3x+0.7y+0.4z), which makes the eight sites different, and an onsite defect of 2 on one site, at nk = 24 and 10 energies, `multildos` went from 20.1 s to 7.7 s and `dos_impurity` from 19.2 s to 7.2 s, with a peak of 0.55 GB before and 0.61 GB after. These are the smallest of three runs with two threads on a shared machine, old and new back to back, so the ratio is what matters; the old code here is the tree with the batch of `bloch_selfenergy_batch` already in it, so the ratio is the one of the sweeps alone. The maps and the DOS agree with the old code to 1e-15 relative. The tests are in `tests/embedding/test_embedding_energy_sweep.py` (nine tests: the sweep against the single-energy calls to 1e-10 for the tetrahedra and exactly for the other modes, with a name, a potential, a bond matrix and an Operator object, and a count of the calls of `green_triangle`, one per sweep where the old code makes one per energy) and in `tests/green/test_supercell_selfenergy_sweeps.py`. `Embedding.multildos` now also takes `energies=`, the name of the grid in `h.get_multildos` and in `Embedding.multidos`; before, it took the keyword and returned its default grid of 20 energies. `green.green_operator_batch` does the same for the trace of `get_dos(mode="Green", gmode="triangle")`. It agrees with the loop to 2e-15 on spinful diamond at nk = 6, and `dos.py` does not call it yet. A complex energy in `mode="triangle"` now raises a `ValueError`; it was cast to its real part with only a ComplexWarning, so `Embedding.get_gf` at a complex128 energy returned the bulk at the real energy and the Dyson step at the complex one.

**The LDOS of an embedding with an operator (14 November 2026).** `Embedding.get_ldos` and `Embedding.get_gf` weighted the Green's function on one side, $-\mathrm{Im}\,\mathrm{diag}(AG)/\pi$, while every mode of `h.get_ldos()` returns $-\mathrm{Im}\,\mathrm{diag}(GA+AG)/(2\pi)$. Summed over the orbitals of a site the two are the same when $A$ is local to each site, because the trace over the orbitals of the site of $A_iG_{ii}$ and of $G_{ii}A_i$ is the same, and they are different for a bond operator. On the pristine honeycomb lattice with Rashba 0.3 and an exchange field (0.1, 0.2, 0.3), at E = 0.4, delta = 0.1 and nk = 12, they agree to 1e-17 for sx, sy and sz. They differ by 2.9e-2 for a random Hermitian matrix and by 5.5e-3 for the intracell hopping, with the same sum over the sites. The map is now `ldos.green2ldos` of the unweighted Green's function, and `get_gf(operator=A)` returns $(GA+AG)/2$, so its diagonal still gives the map. With `gf_mode="full"` the pristine embedding is the same k-sum as `h.get_ldos(mode="green")`, and the two now agree to 1e-10 for sz, sx, the random matrix and the hopping. A named operator with `nsuper=2` was built on the primitive cell and stopped with a dimension mismatch. A name, a potential or a callable is now built on the embedded cell, the supercell of the defect and the outer `nsuper`, and a matrix has to be the one of that cell. This is checked in three ways: a bond matrix of the 2x2 supercell at `nsuper=2` and nk = 12 agrees with `h.supercell(2).get_ldos(mode="green", nk=6)` to 4e-16, where the one-sided weight is off by 0.31; a potential f(r) with a defect gives f(r_i) times the plain map; and `Embedding(h, m=..., nsuper=2)` gives the map of the embedding of `h.supercell(2)` itself. An operator without a matrix, such as the valley operator, is refused with a message. The tests are in `tests/ldos/test_embedding_gf_operator.py`, where ten of the fifteen tests fail on the old code. The defect of the older tests of that file was selected with x = 0, where the honeycomb lattice has no site, so it was no defect; it is now on the first site.

**The complex energies of the tetrahedra (14 November 2026).** The repair above made `mode="triangle"` of the Bloch selfenergy raise for a complex energy, with a message that sent the caller to another mode. In 3D there is no other mode: `"adaptive"`, `"renormalization"`, `"full"` and `"full_adaptive"` all stop at dimensionality 3. So the density matrix of a 3D embedding, a contour integral at complex energies, could not be computed at all, and it never could. The Green's function depends on $z = E + i\delta$ alone, and `green_triangle` at the energy $\mathrm{Re}\,z$ with the broadening $\mathrm{Im}\,z$ is exactly the resolvent of the interpolated bands at $z$. So `_triangle_green` now takes a complex energy as the real energy with its imaginary part added to delta, and below the real axis, the other half of the contour, it returns $G(z^*)^\dagger$, since the Bloch Hamiltonian is Hermitian. A negative delta at a real energy gives the advanced function in the same way, as in the other modes, where it raised. On the real axis, $\mathrm{Im}\,z = 0$, it raises a `ValueError` that names the energy and the delta that were passed, also when the imaginary part of the energy cancels delta up to a few ulps. The four modes that stop in 3D now say in their message to use `mode='triangle'`, which an Embedding takes as `eb.gf_mode='triangle'`; its default, `"renormalization"`, is one of them. The kernel takes one broadening per call, so the energies of a batch are grouped by $|\mathrm{Im}\,z|$: a sweep at real energies is one call and one diagonalization with the same numbers as before, and each point of a contour is its own call. On spinless diamond with a sublattice imbalance of 1, the Green's function at $z = 0.3 + 0.85i$ differs from the mean of $(z-H_k)^{-1}$ over a $16^3$ mesh by 1.4e-2, 4.2e-3 and 1.1e-3 at nk = 6, 12 and 24, the second order of the tetrahedra. For a pristine insulator every band of the interpolant is full or empty in every tetrahedron, so the contour has to give the mean of the occupied projectors over the same mesh up to the tails of delta: `Embedding.get_density_matrix` with `gf_mode="triangle"` at nk = 6 is off by 1.3e-3 at delta = 1e-2 and by 1.3e-4 at delta = 1e-3, linear in delta as a Lorentzian tail is. In the diamond metal with the Fermi level in the lower band it is within 1.1e-3 to 3.1e-3 of the triangle density matrix at T = 0 on the same mesh, at delta = 1e-2. The contour takes 76 points for the insulator and 60 for the metal at the default `eps`, each one a call of the tetrahedra at the price of one energy. The whole contour takes 0.4 to 0.7 s at nk = 6 and 4.5 s at nk = 20, and one point takes 0.66 s at nk = 40, so the contour takes about ten minutes at the default nk = 100 by the scaling with nk^3, which was not run. The docstring of the density matrix says to pass an nk that resolves the bands. The batch of the 1D decimation had the same cast: `bloch_selfenergy_batch` turned a complex array into real energies with only a ComplexWarning. No routine of the library calls it with complex energies; they now go through the loop of the single calls. `get_dos(mode="Green", gmode="triangle")` now calls `green.green_operator_batch`, one diagonalization for the sweep: 3.91 s against 2.90 s on four-orbital diamond at nk = 30 with 10 energies, and 5.47 s against 1.90 s on a 16-orbital cubic supercell with an onsite modulation at nk = 16, the smallest of three runs back to back at two threads, with the same DOS to 7e-15. The tests are in `tests/green/test_bloch_selfenergy_batch.py` (ten new test cases, which fail on the code before, with the mesh mean in both half planes), `tests/embedding/test_embedding_density_matrix.py` (the pristine insulator in 3D, which raised before) and `tests/dos/test_dos_green_triangle_sweep.py` (one call of the tetrahedra per sweep, the single-energy traces, and the tetrahedron DOS of `mode="ED"` to 1e-10).

## Limits of the linear interpolation that were found on the way

- **Flat simplices.** A simplex whose vertices all sit at one energy is a delta function in
  energy. On graphene this happens in two ways: the van Hove contour at E = +-t is made of
  straight lines that run along the mesh for an even nk, and a K point sits at the centroid
  of a triangle when 3 does not divide nk, so its three vertices are related by C3. At
  delta = 0 such simplices are dropped (no grid point can see them); at a small finite
  delta they are spikes of weight 1/(2 nk^2) at the van Hove energy, which the grid point at
  exactly E = 1 sees as an error of 1/(2 nk^2 pi delta) per triangle. With nk = 8 and
  delta = 1e-6 that was a DOS of 1.5e4 at E = -1. An odd nk divisible by 3 avoids both. The
  docstring says so.
- **Band crossings.** The bands are ordered by energy at each k. Where two bands cross, the
  ordered bands have a kink inside the triangles, and the linear interpolation sees a wrong
  band there. On spinful graphene in an exchange field J = 0.4 along z, where the spin-up
  and spin-down cones cross on the E = 0 contour, the DOS differs from the sum of two
  shifted spinless DOS by up to 0.51, 0.31, 0.19 and 0.18 at nk = 61, 121, 241 and 481, at
  the energy closest to E = 0, while the mean error over the grid falls as 5.0e-3, 2.1e-3,
  5.3e-4 and 3.6e-4. So the error is of order one in a window of width of order v/nk around
  the crossing energy, and only the window shrinks. Band tracking by eigenvector overlap,
  or splitting a conserved quantum number before the integration, would remove it; neither
  is built.
- **1D is first order pointwise.** A linear band on a segment has a constant density, so the
  DOS of the chain is off by the variation of 1/|v| across the segment, an error of order
  1/nk (3.5e-3 relative at nk = 2001). In 2D and 3D the errors of the triangles at different
  positions on the constant-energy contour cancel to close to second order.

## What is left open

- **Cost per energy at finite delta.** The simplex sum visits every simplex at every energy,
  about 0.15 to 0.4 microseconds per simplex, band and energy on four cores. This is what
  makes the triangle mesh lose in wall time against cheap diagonalizations (two-band models,
  one-band 3D). Far from resonance each simplex is a short Taylor series in 1/(z - c), so
  the far field of a group of simplices with close centers c could be summed once per group
  (a one-dimensional fast multipole in energy), leaving the exact sum for the simplices
  within a few spreads of E. Not built. At delta = 0 the problem does not exist, since each
  simplex only visits the energies inside its range. Since 13 November 2026 the weights
  enter through one matrix product after the simplex averages (see the repairs of the
  seventh bug audit), so what is left per energy is the simplex average itself.
- **One diagonalization for the density-matrix contour of an embedding.** The contour of
  `Embedding.get_density_matrix` works with the tetrahedra since 14 November 2026 (see the
  complex energies of the tetrahedra above), with one diagonalization of the mesh per
  contour point, 60 to 76 points. The points are not known in advance and each one has its
  own imaginary part, while the kernel of `green_triangle` takes one broadening for all its
  energies. A mesh shared by the whole contour needs a broadening per energy in that kernel,
  or the eigenpairs of the mesh kept between the calls of one contour. Not built: the
  diagonalization is the smaller part of a point on cells of a few orbitals (0.3 s of 1.33 s
  on four-orbital diamond at nk = 40), so it would pay on larger cells only.
- **One normalization of the fields of the qtci DOS and LDOS (done, 14 November 2026).**
  Both routes measure each band energy from the middle of its own range, the LDOS divides
  the band energies by the largest of them, the weights enter as they are, and a flat band
  is not interpolated; see "A centre per band, and weights without a scale" in the repairs
  of the seventh bug audit. Four things are left. The LDOS with an operator takes 20% to
  44% more points than before for the same `tolerance` on the two cases measured, since
  the energies are resolved against the width of a band; the error it buys there was not
  measured. An operator with a large norm loosens the energies of the LDOS, as it did
  before, since the weights are not divided by it. The weights of the DOS with an
  operator that vanish up to rounding (sz without magnetism) are still interpolated as
  noise, one field at a time, as before; no case was measured. And
  `qtcitk/densitymatrix_qtci.py` gives the
  raw band energies to one stacked interpolation, so a constant shift of the onsite
  energies should loosen the mean-field density matrix in the same way; this is read from
  the code and not run.
- **The exact DOS of the interpolant at delta = 0 as a function.** The delta = 0 result is a
  piecewise-linear function in 2D (piecewise quadratic in 3D) with breakpoints at the vertex
  energies. Returning it as such, or its integral (the number of states below E, which gives
  the Fermi level of a metal exactly for the interpolant), would remove the energy grid from
  the sum rule. Not built.
- **Band crossings.** See above: band tracking by overlap of eigenvectors, or a split by a
  conserved quantum number (sz without spin-orbit coupling) before the integration.
- **QTCI with an operator, and in 3D (3D done, 7 November 2026).** The qtci path takes an
  operator (the weights are interpolated as one more field per band), in 2D and in 3D. The
  3D quantics mesh is built, see the section on the qtci DOS in 3D; it saves k-points but
  not wall time on few-band models, because the tetrahedron sum on the full mesh dominates.
  The qtci LDOS and the qtci conductivities stay 2D only and raise
  NotImplementedError in 3D. A sum of the tetrahedra on the tensor train itself, without
  the full mesh, is what would make qtci pay in wall time in 3D; it is not built.
- **Shared helpers with kubosimplex.py (done, 6 November 2026).** The cached cross
  interpolation of the fields is one function, `qtcitk/meshfields.qtci_fields`, called by
  the qtci DOS, the qtci LDOS and `kubosimplex.sigma_qtci`. The DOS and LDOS give the same
  numbers bit by bit as before. The direct kernel of `kubosimplex.py` calls `_dd` from
  here. Since 7 November 2026 `conductivitytk/shgsimplex.py` calls `qtci_fields` too,
  and the qtci DOS, LDOS and optical conductivity call `qtci_stacked_fields` (see the
  section on one interpolation of all the fields).
