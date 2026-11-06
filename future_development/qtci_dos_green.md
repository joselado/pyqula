# The triangle method and QTCI for the density of states and the bulk Green's function

This note records what the linear simplex integration (segments in 1D, triangles in 2D,
tetrahedra in 3D) and its QTCI variant give for the density of states and for the bulk
Green's function of a crystal, with the measurements that decided what went into the code.
It follows `qtci_optical_response.md`, which found the same structure for the optical
conductivity. What is built is in `dostk/simplexdos.py`, reached as
`h.get_dos(integration="triangle")` and `h.get_dos(integration="qtci")`, with tests in
`tests/dos/test_dos_simplex.py` and `tests/green/test_green_simplex.py`. What is open is at
the end.

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

The kernel is not the one of `kubosimplex.py`, for one reason. There the divided
difference switches to a Taylor series when the spread of the vertex energies is below
1e-2 delta. The cancellation of a plain divided difference grows like (|E - z|/spread)^d,
so for a nearly flat simplex far from resonance it loses digits even when the spread is
much larger than delta. At the delta of an optical spectrum this does not matter; at
delta = 1e-6, or 0, it does: for a triangle at energies 0.5, 0.5 + 0.4 s, 0.5 + s, with
E = 0 and delta = 1e-6, the kubosimplex kernel has a relative error of 0.10, 0.027 and
4e-4 at s = 3e-8, 1e-7 and 1e-6, against a 50-digit evaluation, and the new one is exact
to rounding. The new kernel switches on the spread over |c - z|, with c
the mean vertex energy, and expands every cluster of close nodes in its Taylor series with
the complete homogeneous symmetric polynomials, up to 14 terms. One routine covers d = 1,
2 and 3 and any number of confluent nodes.

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
so it is not wired in. The function stays as `simplexdos.green_triangle(h,energies,nk,delta)`,
tested against the adaptive mode and against the DOS (-Im Tr G/pi equals the triangle DOS to
rounding on the same mesh). It works in 3D, where `bloch_selfenergy` has no mode at all;
that would be the reason to wire it in, and it is a decision for the maintainer.

A side observation on the adaptive mode: a loose error makes it slow at small delta, not
fast. At delta = 1e-6 and error=1e-3 it did not finish in 900 s, at error=1e-4 it took 37 s,
and at error=1e-6 10 s. At delta = 1e-5, error=1e-3 took 197 s against 4.7 s at 1e-5. The
quadrature seems to spend its budget on a poorly resolved integrand before it reaches
the tolerance; error=1e-5 or tighter is the better default at small delta.

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
  simplex only visits the energies inside its range.
- **The exact DOS of the interpolant at delta = 0 as a function.** The delta = 0 result is a
  piecewise-linear function in 2D (piecewise quadratic in 3D) with breakpoints at the vertex
  energies. Returning it as such, or its integral (the number of states below E, which gives
  the Fermi level of a metal exactly for the interpolant), would remove the energy grid from
  the sum rule. Not built.
- **Band crossings.** See above: band tracking by overlap of eigenvectors, or a split by a
  conserved quantum number (sz without spin-orbit coupling) before the integration.
- **QTCI with an operator, and in 3D.** The qtci path takes an operator in 2D (the weights
  are interpolated as one more field per band); a 3D quantics grid is not built.
- **Shared helpers with kubosimplex.py.** `simplexdos.qtci_fields` is a generic copy of the
  cached cross interpolation inside `kubosimplex.sigma_qtci`, and the kernel is a more robust
  version of the one there. Once nobody else is editing `kubosimplex.py`, it can call
  `simplexdos` for both, which would also make its small-delta limit safe.
