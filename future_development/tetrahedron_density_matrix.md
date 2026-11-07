# The triangle and tetrahedron density matrix of the mean field

This note records how the linear simplex integration (triangles in 2D,
tetrahedra in 3D) was built for the density matrix of the self-consistent
mean field, and the measurements that decided its shape. The outcome is in
the code as `integration="triangle"` of `scftk.densitydensity.get_dm` and of
the entry points that go through it (`Vinteraction`, `hubbard`, and
`h.get_mean_field_hamiltonian(integration="triangle")`, which routes there).
The engine is `dmtk/simplexdm.py`, the tests are
`tests/scf/test_densitydensity_triangle.py`. What is left open is at the end.

## The question

The mean field reads the density matrix at a few lattice vectors d,

dm[d]_ij = int dk sum_n f(E_n(k)) e^{2 pi i k.d} conj(u_n,i(k)) u_n,j(k).

In an insulator the integrand is smooth and the uniform mesh converges
exponentially. In a metal at low temperature the Fermi function is a step,
and the uniform mesh only resolves it when the energy spacing of the mesh,
of order v_F/nk, is below T. So nk has to grow like v_F/T. The question is
whether we can integrate the step exactly instead, so that the mesh only has
to resolve the bands.

## The construction

We interpolate E_n(k) and the field X_n(k) = e^{2 pi i k.d} conj(u_n,i)
u_n,j linearly in each simplex of the uniform mesh, and integrate the step
exactly. Because X is linear in the simplex, the result is a scalar weight
per vertex, w_j = int_T theta(mu - E_L) lambda_j, with lambda_j the
barycentric coordinates. So the density matrix is the exact-diagonalization
kernel with the Fermi occupation replaced by w_n(k), summed over the
simplices that share k. This is the linear tetrahedron method of Bloechl,
Jepsen and Andersen, PRB 49, 16223 (1994). The weights come from a
decomposition of the occupied part of each simplex into sub-simplices (a
prism into three tetrahedra in the middle case), whose denominators never
vanish, also for degenerate vertex energies. They were checked against a
uniform sampling of the simplex in 1D, 2D and 3D.

The weights are averaged over degenerate multiplets at each k, which is the
same as averaging the projector, so the result does not depend on the basis
that eigh picks inside a multiplet.

## The Bloch phase

The phase e^{2 pi i k.d} is not linear in k. There are two choices: keep it
in the interpolated field, or interpolate the phase-free projector and
integrate the phase exactly. In a fully occupied simplex the first choice
gives each vertex the weight 1/nk^dim once summed over its simplices, so a
filled band gives exactly the uniform-mesh mean, which is exponentially
accurate. The exact phase gives the linearly interpolated projector times
the exact phase, which has a second-order error everywhere. On the
simplices cut by the Fermi surface both are of second order.

We measured both on a honeycomb metal at T=0 (spinless, Haldane t2 = 0.1,
sublattice imbalance 0.2, onsite 0.6, filling 0.4), for the onsite block and
the first-neighbor vectors the mean field reads. The exact phase is
emulated by sampling the linear fields of each coarse triangle on an m times
finer mesh, where the phase is exact at every point, so m = 1 is the phase
in the field and large m is the exact phase. The reference is the triangle
result at nk = 1024 and 2048, extrapolated in 1/nk^2 (the extrapolation
moved it by 1e-7). The largest error over the entries of the bond blocks,
and of the onsite block, which has no phase:

| nk | m = 1 (phase in field) | m = 2 | m = 4 | m = 8 | onsite block |
|---|---|---|---|---|---|
| 24 | 9.9e-4 | 7.7e-4 | 8.9e-4 | 9.2e-4 | 2.8e-4 |
| 48 | 2.5e-4 | 1.9e-4 | 2.2e-4 | 2.3e-4 | 7.3e-5 |
| 96 | 5.2e-5 | 4.7e-5 | 5.4e-5 | 5.6e-5 | 1.7e-5 |
| 192 | 1.4e-5 | 1.2e-5 | 1.4e-5 | 1.4e-5 | 4.2e-6 |

The phase in the field and the exact phase (m = 8) agree within 7% at
every nk, and both are of second order. So the exact phase buys nothing,
and only the phase in the field is built. As a consequence, an insulator at
T=0 gives exactly the density matrix of exact diagonalization on the same
mesh: the test with a Haldane insulator agrees to 1e-12 for every vector.

## Finite temperature

The Fermi function is integrated over the linear simplex through its
density of states. The weight at temperature T is the T=0 weight convolved
with -f'(x),

w_j(mu, T) = int dx (-f'(x)) w_j(mu + x, T=0).

w_j(mu, 0) is a polynomial in mu between two vertex energies. So the
integral is split at the vertex energies, at mu and at a fixed set of
multiples of T up to 40 T. Each piece is done by Gauss-Legendre quadrature
in the variable f(x), in which -f'(x) dx is flat. Above the highest vertex
energy the weight is constant, and that piece is exact. Against an adaptive
quadrature the error is below 3e-9 in units of the simplex volume, for T
from 1e-4 to 0.3 of the energy spread of the simplex, with 8 nodes per
piece. A first version split only at 2, 6 and 14 T. It had errors up to
6e-6, which went down slowly with the number of nodes, and the extra
breakpoints fixed that. A closed form exists, with divided differences of
polylogarithms, but it cancels badly when the vertex energies are close
compared with T, so it was not built.

What matters is that the mesh no longer has to resolve the Fermi function.
In the tables below, the triangle errors at T = 1e-2, 1e-3 and 1e-4 are the
same to two digits at every nk. At T=0 the weights are the step itself.

## The Fermi level and the total energy

The charge of the simplex integration, N(mu) = sum_T C_T(mu), with C_T the
occupied volume of each simplex, is continuous and monotonous in mu, also
at T=0. The eigenvalue count of a uniform mesh is a staircase instead. So
the Fermi level is located on the same simplices as the density matrix
(`simplexdm.get_fermi4filling`), and the density matrix holds the requested
filling to 1e-15 in every run below. In a gap, where the charge is held on
a whole interval of mu, the middle of the interval is returned.

The Fermi level has to come from the same integration. With the
uniform-mesh Fermi level and the triangle density matrix, a square-lattice
ferromagnet at filling 0.4 (exchange 1.2) loses 4.7e-3 of the charge per
orbital at nk = 32 and 7e-4 at nk = 64, at any T from 1e-4 to 1e-2. The
exact-diagonalization loop itself misses the filling by about 1e-4 at
nk = 32 to 128 (the last column of the tables), because its charge is a
staircase.

The band energy is sum w_n(k) E_n(k) with the same weights at T=0, which is
the exact band energy of the linear bands, plus the Bloechl term below. As
for `"ed"` and `"qtci"`, it is the T=0 sum at the Fermi level of the loop,
while the double counting is taken from the density matrix at T. The
un-shift by the Fermi level adds mu times the requested charge, filling
times the number of electron states, as `"ed"` does. In a normal state this
is the trace of the density matrix to 1e-12. It used the trace itself
before, which is not the same thing in a superconductor (see The total
energy of a superconductor).

## The Bloechl correction

Bloechl, Jepsen and Andersen add to the weights

dw_j = sum over the simplices T that contain j of D_T(mu) c_dim sum_{i in T} (e_i - e_j),

with D_T the density of states of the simplex at mu and c_3 = 1/40. The
coefficient comes from the mean of the interpolation error E_L - E of a
quadratic band over a simplex, sum_{i<j} e_ij.A.e_ij/(2 (dim+1)(dim+2)),
with e_ij the edges and A the Hessian. The sum over the simplices that
share a vertex turns the edge terms into e_i - e_j, because the star of a
vertex is centrosymmetric on this mesh. So c_dim = 1/(2 (dim+1)(dim+2)),
which is 1/40 in 3D, 1/24 in 2D and 1/12 in 1D. We checked c_2
numerically. On a curved square-lattice band at filling 0.3, the band
energy error at nk = 64 goes from 5.6e-4 to 2.1e-7 with 1/24, and only to
2.2e-4 with 1/40.

Does it apply to a matrix-valued weight? Yes. The correction is a scalar
weight per band, vertex and simplex, and the field enters linearly, so it
applies to the projector with its Bloch phase entry by entry. What it can
correct follows from sum_j dw_j = 0 in every simplex. It moves no charge
between bands, so it does not change the Fermi level. It acts only through
the variation of the field along the band, as
-c_dim D_T sum_{ij} (X_i - X_j)(e_i - e_j)/2. On a collinear ferromagnet
the magnetization is a difference of band charges, so the correction
cannot change it, and the tables show exactly that. For a generic field it
does not remove the whole 1/nk^2 error either, because part of that error
is in the band charges: on the model band above, a smooth field X(k)
improved by a factor 3 to 30, erratically. For X = E and for the bond
densities of a V1 metal it removes almost all of it.

It is not variational: there is no total energy whose stationary point is
the corrected mean field. So we checked that the loop reaches the same
state. With and without the correction, the extrapolated limits agree to
2e-11 in the bond density of the V1 metal below and to 1e-8 or better in
the total energies. It is on by default (`simplexdm.bloechl_default`), and
`tests/scf/test_densitydensity_triangle.py` checks the common limit.

## The benchmark

The runs below are 2D Hubbard mean fields (`Vinteraction`, Broyden mixing,
`maxerror=1e-9`) on four cores of one workstation. Other jobs loaded the
machine to 10 to 19 on 16 cores, so the wall times are only indicative.
Every run starts from the same guess, `mf="ferro"` (or `"random"` with seed
1), except the rows marked warm, which start from the converged state of
the previous mesh of the same method. "Diagonalizations" counts both the
Fermi search and the density matrix of every iteration, nk^2 each, for the
triangle and for exact diagonalization alike. For qtci it is the number of
eigh calls.

The reference is the triangle result without the correction at the two
largest meshes, extrapolated in 1/nk^2, since it converges cleanly at
second order. For the total energy and the bond density the reference is
the corrected triangle at the largest mesh, which agrees with that
extrapolation to 1e-8 or better. At T = 1e-2 the reference agrees with
exact diagonalization at nk = 2048 to 4.5e-8 in the magnetization and
2.3e-9 in the energy, which is the independent check of the whole scheme.
At T = 1e-3 exact diagonalization at nk = 1024 agrees with it to 2e-5.

### A Hubbard ferromagnet on the square lattice

One orbital, U = 6, filling 0.4 (0.8 electrons per site), mf="ferro". The
state is a partly polarized metal, m = 0.474 per site, with both spin Fermi
surfaces present. Here the correction changes only the total energy.

T = 1e-2:

| method | nk | diagonalizations | iterations | wall (s) | error in m | error in mu | error in E | filling |
|---|---|---|---|---|---|---|---|---|
| ed | 32 | 0.15M | 70 | 3 | 1.8e-02 | 1.1e-02 | 1.6e-04 | 0.399902344 |
| ed | 64 | 0.62M | 74 | 9 | 4.8e-03 | 8.1e-04 | 2.3e-05 | 0.400024414 |
| ed | 128 | 2.5M | 75 | 24 | 7.7e-04 | 2.0e-03 | 4.1e-06 | 0.399993896 |
| ed | 256 | 9.7M | 72 | 129 | 1.4e-04 | 1.3e-04 | 1.3e-06 | 0.400001526 |
| ed | 512 | 3.1M | 4 (warm) | 79 | 9.9e-07 | 6.6e-06 | 4.0e-07 | 0.399999619 |
| ed | 1024 | 10M | 3 (warm) | 142 | 1.8e-07 | 1.2e-06 | 7.0e-08 | 0.400000095 |
| ed | 2048 | 42M | 3 (warm) | 668 | 4.5e-08 | 2.9e-07 | 2.3e-09 | 0.399999976 |
| qtci | 8 | 6.1k | 15 | 2 | 2.3e-01 | 2.6e-01 | 6.3e-03 | 0.400000000 |
| qtci | 16 | 4.0k | 4 | 1 | 4.5e-01 | 2.1e-01 | 2.7e-02 | 0.400000000 |
| qtci | 32 | 13k | 12 | 1 | 4.7e-01 | 3.5e-01 | 2.0e-02 | 0.400000000 |
| triangle, no correction | 16 | 0.034M | 64 | 3 | 2.0e-02 | 6.1e-03 | 1.6e-02 | 0.400000000 |
| triangle, no correction | 32 | 0.15M | 69 | 8 | 4.9e-03 | 1.5e-03 | 4.1e-03 | 0.400000000 |
| triangle, no correction | 64 | 0.59M | 70 | 17 | 1.2e-03 | 4.0e-04 | 1.0e-03 | 0.400000000 |
| triangle, no correction | 128 | 2.4M | 70 | 37 | 3.2e-04 | 1.2e-04 | 2.6e-04 | 0.400000000 |
| triangle, no correction | 256 | 9.4M | 70 | 98 | 7.8e-05 | 2.8e-05 | 6.4e-05 | 0.400000000 |
| triangle, no correction | 512 | 3.1M | 4 (warm) | 24 | 2.0e-05 | 7.0e-06 | 1.6e-05 | 0.400000000 |
| triangle, no correction | 1024 | 13M | 4 (warm) | 75 | 4.9e-06 | 1.7e-06 | 4.0e-06 | 0.400000000 |

T = 1e-3, where "triangle" is the default, with the correction:

| method | nk | diagonalizations | iterations | wall (s) | error in m | error in mu | error in E | filling |
|---|---|---|---|---|---|---|---|---|
| ed | 32 | 0.014M | 5 | 0 | 4.5e-01 | 3.6e-01 | 2.5e-02 | 0.399902344 |
| ed | 64 | 0.53M | 63 | 6 | 4.7e-01 | 3.6e-01 | 2.5e-02 | 0.400024414 |
| ed | 128 | 2.8M | 82 | 29 | 3.5e-03 | 4.2e-03 | 3.3e-06 | 0.399993896 |
| ed | 256 | 11M | 80 | 128 | 8.1e-04 | 4.9e-04 | 1.5e-06 | 0.400001526 |
| ed | 512 | 4.2M | 6 (warm) | 50 | 3.5e-04 | 3.7e-04 | 5.0e-07 | 0.399999619 |
| ed | 1024 | 15M | 5 (warm) | 169 | 2.0e-05 | 1.1e-05 | 8.0e-08 | 0.400000095 |
| qtci | 8 | 5.4k | 13 | 1 | 2.3e-01 | 2.7e-01 | 6.5e-03 | 0.400000000 |
| qtci | 16 | 2.3k | 0 | 0 | 4.5e-01 | 2.1e-01 | 2.7e-02 | 0.400000000 |
| qtci | 32 | 5.3k | 3 | 1 | 4.7e-01 | 3.4e-01 | 2.0e-02 | 0.400000000 |
| triangle, no correction | 64 | 0.59M | 70 | 14 | 1.2e-03 | 3.9e-04 | 1.0e-03 | 0.400000000 |
| triangle, no correction | 256 | 9.4M | 70 | 80 | 7.8e-05 | 2.7e-05 | 6.4e-05 | 0.400000000 |
| triangle, no correction | 512 | 3.1M | 4 (warm) | 15 | 2.0e-05 | 7.0e-06 | 1.6e-05 | 0.400000000 |
| triangle, no correction | 1024 | 13M | 4 (warm) | 47 | 4.9e-06 | 1.8e-06 | 4.0e-06 | 0.400000000 |
| triangle | 16 | 0.034M | 64 | 3 | 2.0e-02 | 6.1e-03 | 2.6e-05 | 0.400000000 |
| triangle | 32 | 0.15M | 69 | 7 | 4.8e-03 | 1.5e-03 | 1.1e-06 | 0.400000000 |
| triangle | 64 | 0.59M | 70 | 14 | 1.2e-03 | 3.9e-04 | 1.2e-06 | 0.400000000 |
| triangle | 128 | 2.4M | 70 | 32 | 3.2e-04 | 1.2e-04 | 1.2e-07 | 0.400000000 |
| triangle | 256 | 9.4M | 70 | 70 | 7.8e-05 | 2.7e-05 | reference | 0.400000000 |

T = 1e-4, the triangle without the correction only (not run with it):

| method | nk | diagonalizations | iterations | wall (s) | error in m | error in mu | error in E | filling |
|---|---|---|---|---|---|---|---|---|
| ed | 64 | 0.41M | 48 | 6 | 4.6e-01 | 3.5e-01 | 2.5e-02 | 0.400024414 |
| ed | 128 | 2.7M | 80 | 36 | 4.7e-01 | 3.5e-01 | 2.5e-02 | 0.399993896 |
| ed | 256 | 11M | 83 | 151 | 8.4e-04 | 5.3e-04 | 1.5e-06 | 0.400001526 |
| ed | 512 | 6.3M | 10 (warm) | 89 | 2.7e-04 | 2.7e-04 | 4.9e-07 | 0.399999619 |
| ed | 1024 | 17M | 6 (warm) | 258 | 3.9e-05 | 5.4e-05 | 6.9e-08 | 0.400000095 |
| triangle, no correction | 64 | 0.59M | 70 | 14 | 1.2e-03 | 3.9e-04 | 1.0e-03 | 0.400000000 |
| triangle, no correction | 256 | 9.4M | 70 | 71 | 7.8e-05 | 2.7e-05 | 6.4e-05 | 0.400000000 |
| triangle, no correction | 512 | 3.1M | 4 (warm) | 13 | 2.0e-05 | 7.0e-06 | 1.6e-05 | 0.400000000 |
| triangle, no correction | 1024 | 13M | 4 (warm) | 39 | 4.9e-06 | 1.8e-06 | 4.0e-06 | 0.400000000 |

What these say:

- At T = 1e-3, exact diagonalization at nk <= 64 converges to another
  state, nearly nonmagnetic, with a total energy 2.5e-2 too high, and at
  T = 1e-4 it does so up to nk = 128. The triangle finds the ferromagnet at
  nk = 16. This is the clearest gain.
- For the magnetization the triangle needs about four times fewer k-points
  than exact diagonalization at T <= 1e-3: nk = 512 gives 2.0e-5, as exact
  diagonalization does at nk = 1024.
- For the total energy the corrected triangle at nk = 32 (1.1e-6) matches
  exact diagonalization at nk = 256 (1.5e-6), 64 times fewer k-points.
  Without the correction the triangle loses the energy at every mesh. The
  reason for this is that the total energy of exact diagonalization is
  stationary in the density matrix, so to leading order its error is
  quadratic in the error of the density matrix (its band energy, a plain
  sum of a step on the mesh, has an error of its own too, which is small
  here). The band energy of the uncorrected linear bands carries the
  1/nk^2 interpolation error directly.
- At T = 1e-2 exact diagonalization wins from nk = 512 on, where its
  convergence becomes exponential: 1e-6 in m, against 2e-5 for the
  triangle at nk = 512. This is the crossover we expected, at nk of order
  v_F/T.
- The triangle converges at clean second order, so two meshes extrapolate
  well: nk = 512 and 1024 give m to 5e-8 of exact diagonalization at
  nk = 2048.
- The iteration counts are the same for both methods, about 70. The
  continuous charge of the triangle does not make the loop converge
  faster here, and not slower either.

### A ferromagnet on the honeycomb lattice

Two orbitals, sublattice imbalance 0.3, U = 3, filling 0.35, T = 1e-3,
m = 0.244 per cell. The projectors depend on k, so the correction acts on
the density matrix, but the magnetization is still a difference of band
charges.

| method | nk | diagonalizations | iterations | wall (s) | error in m | error in mu | error in E | filling |
|---|---|---|---|---|---|---|---|---|
| ed | 32 | 0.83M | 400 (not converged) | 13 | 4.1e-02 | 7.7e-03 | 4.5e-04 | 0.350097656 |
| ed | 64 | 0.88M | 105 | 14 | 3.9e-02 | 4.5e-03 | 1.5e-04 | 0.349975586 |
| ed | 128 | 5.5M | 166 | 82 | 8.3e-03 | 5.4e-04 | 4.8e-06 | 0.350006104 |
| ed | 256 | 4.2M | 30 | 55 | 1.8e-03 | 1.7e-04 | 2.5e-07 | 0.349998474 |
| ed | 512 | 6.8M | 11 (warm) | 108 | 5.2e-04 | 9.4e-05 | 1.4e-07 | 0.350000381 |
| ed | 1024 | 23M | 9 (warm) | 349 | 4.1e-05 | 2.5e-05 | 5.3e-08 | 0.349999905 |
| qtci | 8 | 7.1k | 18 | 4 | 4.2e-02 | 6.7e-02 | 2.7e-03 | 0.350000000 |
| triangle, no correction | 32 | 0.057M | 26 | 4 | 5.2e-03 | 1.3e-03 | 1.9e-03 | 0.350000000 |
| triangle, no correction | 128 | 0.98M | 28 | 17 | 3.7e-04 | 9.4e-05 | 1.2e-04 | 0.350000000 |
| triangle, no correction | 512 | 5.2M | 8 (warm) | 43 | 2.4e-05 | 6.0e-06 | 7.5e-06 | 0.350000000 |
| triangle | 16 | 0.016M | 29 | 2 | 2.8e-02 | 7.2e-03 | 6.3e-05 | 0.350000000 |
| triangle | 32 | 0.057M | 26 | 4 | 5.1e-03 | 1.3e-03 | 7.7e-06 | 0.350000000 |
| triangle | 64 | 0.24M | 27 | 9 | 1.5e-03 | 4.0e-04 | 1.1e-06 | 0.350000000 |
| triangle | 128 | 0.98M | 28 | 17 | 3.6e-04 | 9.4e-05 | 7.4e-09 | 0.350000000 |
| triangle | 256 | 3.8M | 27 | 41 | 9.2e-05 | 2.4e-05 | 2.5e-09 | 0.350000000 |
| triangle | 512 | 5.2M | 8 (warm) | 42 | 2.3e-05 | 6.0e-06 | reference | 0.350000000 |

The magnetization gains a factor 4 in k-points again (the triangle at
nk = 512 gives 2.3e-5, exact diagonalization at nk = 1024 gives 4.1e-5), and
the correction changes its error by at most 6%. The total energy with the
correction at nk = 128 (7e-9) beats exact diagonalization at nk = 1024
(5e-8), with 64 times fewer k-points. Exact diagonalization also needs 105
to 400 iterations below nk = 256, the triangle 26 to 29.

### Bond densities of a spinless V1 metal

Spinless honeycomb, V1 = 1.5, filling 0.35, T = 1e-3, mf="random". There is
no symmetry breaking. The Fock term renormalizes the first-neighbor bond,
and the error is the one of the intracell bond density dm[(0,0,0)][0,1],
which is the case where the projector varies along the band.

| method | nk | diagonalizations | iterations | wall (s) | error in bond | error in mu | error in E | filling |
|---|---|---|---|---|---|---|---|---|
| ed | 32 | 0.08M | 37 | 2 | 7.1e-04 | 3.3e-02 | 4.0e-04 | 0.350097656 |
| ed | 64 | 0.33M | 38 | 8 | 8.7e-06 | 1.2e-02 | 3.7e-07 | 0.349975586 |
| ed | 128 | 1.1M | 33 | 22 | 4.3e-06 | 2.0e-03 | 1.6e-05 | 0.350006104 |
| ed | 256 | 4.6M | 33 | 96 | 3.7e-07 | 1.1e-03 | 1.4e-06 | 0.349998474 |
| ed | 512 | 1.6M | 0 (warm) | 44 | 1.3e-07 | 5.2e-04 | 3.1e-09 | 0.350000381 |
| triangle, no correction | 32 | 0.063M | 29 | 3 | 1.8e-04 | 1.7e-03 | 1.4e-03 | 0.350000000 |
| triangle, no correction | 128 | 1M | 29 | 12 | 1.1e-05 | 9.3e-05 | 8.9e-05 | 0.350000000 |
| triangle, no correction | 512 | 1.6M | 0 (warm) | 9 | 6.9e-07 | 5.5e-06 | 5.6e-06 | 0.350000000 |
| triangle | 16 | 0.017M | 31 | 2 | 4.1e-05 | 8.8e-03 | 3.3e-04 | 0.350000000 |
| triangle | 32 | 0.061M | 28 | 3 | 5.1e-07 | 2.0e-03 | 4.1e-06 | 0.350000000 |
| triangle | 64 | 0.25M | 29 | 5 | 6.8e-08 | 4.1e-04 | 5.4e-07 | 0.350000000 |
| triangle | 128 | 1M | 29 | 11 | 2.1e-09 | 1.1e-04 | 1.6e-08 | 0.350000000 |
| triangle | 256 | 4.1M | 29 | 26 | 1.0e-10 | 2.7e-05 | 6.2e-10 | 0.350000000 |

Without the correction the triangle loses the bond density to exact
diagonalization: 1.8e-4 at nk = 32, while exact diagonalization gives
8.7e-6 at nk = 64. With it, the triangle wins by a large factor: nk = 32
gives 5e-7, which exact diagonalization reaches only at nk = 256. The Fermi
level is the weak point of exact diagonalization here, 5e-4 at nk = 512
against 6e-6 for the triangle on the same mesh.

### An insulator

Honeycomb, sublattice imbalance 0.5, U = 1, half filling. Every simplex is
full or empty, so the triangle is exact diagonalization on the same mesh:
the total energies agree to 1e-15 at nk = 12 and 24, and the Fermi level is
the middle of the gap in both. There is nothing to gain, as expected, and
at these small meshes the triangle costs more wall time (1.0 s against
0.2 s at nk = 12), from its Fermi search and the compilation of its kernels.

### qtci

`integration="qtci"` is now the triangle integration on a mesh whose fields
come from a cross interpolation. The maintainer decided this on 2026-11-06,
after the Gauss-Kronrod version lost on both metals. What follows is first
why the old one failed, then the new construction and its measurements.

#### Why the Gauss-Kronrod rule failed

The first qtci density matrix integrated every entry the mean field reads
with a tensorized Gauss-Kronrod rule. On the square-lattice ferromagnet it
converged at nk = 8 to 64 to a state with m between 0.005 and 0.25 instead
of 0.474. On the honeycomb ferromagnet it first stopped with `ValueError:
maxsamplevalue is zero!` (a pivot that underflows, fixed in
`gkintegrate.integrate_robust`), and then landed at m = 0.315 instead of
0.244 after 400 iterations.

Why does the loop find the wrong state? We froze the mean field as a rigid
exchange split U m_in/2 of the square lattice (U = 6, filling 0.4), took
the Fermi level of each method from its own search, and read back the
magnetization m_out. A self-consistent state is a fixed point,
m_out = m_in. The qtci density matrix is integrated on the tensorized
Gauss-Kronrod rule of `qtcitk/gkintegrate.py`, with 4 ceil(log2 nk) + 1
nodes per axis, so the table gives the size of that grid. "ed, same size"
is the uniform mesh with the same number of points. The reference is the
triangle at nk = 512. T = 1e-3:

| m_in | reference | grid | qtci nodes | ed, same size |
|---|---|---|---|---|
| 0.05 | 0.0553 | 13x13 (nk = 8) | 0.1595 | 0.0888 |
| 0.05 | 0.0553 | 21x21 (nk = 32) | 0.0224 | 0.0567 |
| 0.05 | 0.0553 | 25x25 (nk = 64) | 0.0560 | 0.0436 |
| 0.05 | 0.0553 | 29x29 (nk = 128) | 0.0809 | 0.0464 |
| 0.05 | 0.0553 | 33x33 (nk = 256) | 0.0615 | 0.0523 |
| 0.474 | 0.4740 | 13x13 (nk = 8) | 0.4571 | 0.5148 |
| 0.474 | 0.4740 | 21x21 (nk = 32) | 0.4405 | 0.4558 |
| 0.474 | 0.4740 | 25x25 (nk = 64) | 0.4828 | 0.4672 |
| 0.474 | 0.4740 | 29x29 (nk = 128) | 0.4835 | 0.4744 |
| 0.474 | 0.4740 | 33x33 (nk = 256) | 0.4867 | 0.4711 |

Three hypotheses are ruled out by this table. The cross interpolation was
exact there: the old `get_dm_qtci` equalled the plain sum over the node grid
(`full_dm_gk`, removed since) to every printed digit, at every m_in and nk, so neither the
pivot search nor the tolerance plays a role. The Fermi search on the nodes
holds the charge exactly, 0.80000 in every case. And the temperature is not
the point: at T = 1e-2 the same table moves by 1.3e-2 at most, the errors
stay of the same size, and the benchmark rows at T = 1e-2 are as wrong as
those at T = 1e-3.

What remains is the rule itself. The occupied projector has a step at the
Fermi surface, and a Gauss-Kronrod rule is spectral only for a smooth
integrand. On a step it does no better than a uniform mesh of the same
size, which is a few hundred to a thousand points. The number of nodes
grows only like log2(nk)^2, so raising nk does not help: the error in m
goes from 1e-3 to 1e-1 between nk = 8 and 256, with no trend. The part that matters most
for the loop is the response at small m, which measures U times the
density of states at the Fermi level. The true map has m_out/m_in = 1.1 at
m_in = 0.05, so the paramagnet is only weakly unstable. On the 21x21 grid
the ratio is 0.45, the paramagnet becomes stable, and the loop from
`mf="ferro"` goes there (m = 0.005). The uniform mesh fails in the same way
for the same reason (exact diagonalization at nk <= 64 and T = 1e-3 in the
tables above). So qtci of the projector cannot resolve a metallic Fermi
step at any nk it offers, and this is not something a tolerance or a pivot
fixes.

#### The construction

The density matrix is the contraction of two smooth fields with the triangle
weights: the band energies E_n(k), and the field
X_n(k) = e^{2 pi i k.d} conj(u_n,i(k)) u_n,j(k), averaged over the degenerate
multiplet of n at each k. These are the fields `simplexdm.get_dm` contracts.
So qtci reconstructs them on a 2^R x 2^R mesh from quantics cross
interpolations, as the DOS and the optical conductivity do
(`qtcitk/meshfields.py`), and then applies the triangle weights of
`simplexdm`, with the Bloechl term, to the reconstructed mesh. In the code
this is `qtcitk/densitymatrix_qtci.py`:

- nk means the mesh, rounded up to a power of two, 2^R.
- The Fermi level is found on the triangle charge of the reconstructed
  energies (`simplexdm.fermi_from_energies`), so the filling is exact on
  that mesh, up to the interpolation.
- The band energy is the sum of the T=0 weights with the Bloechl term
  times the reconstructed energies, as `simplexdm.band_energy` does on the
  exact ones (`band_energy_qtci`).
- Of the entries (d,i,j) and (-d,j,i), which are complex conjugates, only
  one is interpolated.
- The SCF loop finds the Fermi level of H and then asks for the density
  matrix of H minus mu. A constant shift moves every band energy by mu and
  leaves the eigenvectors alone. So the Fermi search keeps its eigenpairs
  and its reconstructed energies, and the density matrix of the shifted
  Hamiltonian reuses both. The match is checked on the Bloch matrices
  themselves, and any other Hamiltonian starts from scratch. One
  diagonalization per visited point serves both passes of an iteration,
  while the triangle diagonalizes its mesh twice. The eigenvectors are
  kept only below the memory limit of the triangle,
  `simplexdm.max_memory_gb`, and they stay in memory until the next Fermi
  search replaces them.

#### One TCI per field, or one for all

The DOS interpolates each field with its own TCI. For the density matrix this
fails, for a reason that has nothing to do with the physics. A TCI
normalizes its tolerance by the largest value of its field. The
spin-off-diagonal entry of a collinear ferromagnet is zero up to rounding,
about 1e-17, so its own TCI interpolates the rounding noise, at full rank.
So we interpolate all the X_n with one TCI, with the field index as an extra
tensor site in front of the 2R bits (`meshfields.qtci_stacked_fields`).
The tolerance is then relative to the largest field, and every field shares
the pivots in k. The energies get a second TCI of the same kind.

We measured the X fields of the frozen honeycomb ferromagnet (exchange 0.183,
the Hubbard entries, 24 fields of which 8 are not zero), the frozen gapped
honeycomb with a tilted exchange field and the V1 entries (64 fields), and
the square-lattice Rashba metal with a tilted exchange field and the Hubbard
entries (6 fields).
"Points" is the number of distinct k-points the interpolation diagonalized.
The largest error against the exact fields is 1e-6 to 2e-6 in every row.

| model | R | per field: points | per field: rank, time (s) | stacked: points | stacked: rank, time (s) |
|---|---|---|---|---|---|
| honeycomb ferromagnet | 7 | 16384 (100%) | 128, 49 | 12481 (76%) | 45, 1.4 |
| same, noise fields removed by hand | 7 | 16094 (98%) | 50, 8.7 | | |
| gapped honeycomb | 7 | 16384 (100%) | 48, 70 | 12467 (76%) | 83, 5.1 |
| honeycomb ferromagnet | 8 | | | 25720 (39%) | 48, 3.0 |
| honeycomb ferromagnet | 9 | | | 29922 (11%) | 48, 3.0 |
| gapped honeycomb | 8 | | | 21627 (33%) | 85, 8.1 |
| gapped honeycomb | 9 | | | 20782 (8%) | 85, 6.2 |
| Rashba metal | 8 | | | 35185 (54%) | 72, 5.4 |
| Rashba metal | 9 | | | 54114 (21%) | 73, 7.8 |

With the field index as the last site instead of the first, the stacked
TCI visited 23k to 110k points at R = 8 and 9, up to five times more on the
gapped model, so it goes first. The DOS and the conductivity still use one
TCI per field. Their fields are not zero up to rounding, but they may gain
from the shared pivots too. Not measured.

#### Band crossings

The fields are sorted by energy at each k, so where two bands cross they
have a kink, as in the simplex DOS. None of the three benchmark states has
a crossing. The square ferromagnet is a rigid spin split of one band. In
the honeycomb ferromagnet the split, 0.37, is smaller than the sublattice
gap, 0.6, so the smallest gap between consecutive bands is 0.235 on the
mesh. With an exchange of 0.5 on the same honeycomb, the up and down bands
cross on a line, and the smallest gap on the mesh is 1.4e-3. There the
stacked TCI of the X fields at R = 8 needs 37317 points and rank 81, against
22446 points and rank 49 at exchange 0.183. So a crossing costs about 1.7
times the points here.

#### The mean field

This is the benchmark above: T = 1e-3, Broyden mixing, the references of
the tables above, four cores of a workstation loaded by other jobs (a load
of 6 to 10 on 16 cores), so the wall times are only indicative. The qtci
tolerance is 1e-6. "Per iteration" is the number of diagonalizations of the
last iteration, Fermi search and density matrix together, and in
parentheses the fraction of the mesh. For the triangle it is the mesh
twice. The last column but one is the difference in m from the triangle
at the same nk.

Square-lattice ferromagnet, maxerror = 1e-9:

| method | nk | diagonalizations | per iteration | iterations | wall (s) | error in m | error in mu | error in E | qtci minus triangle in m | filling |
|---|---|---|---|---|---|---|---|---|---|---|
| qtci | 32 | 0.072M | 1024 (100%) | 69 | 27 | 4.8e-03 | 1.5e-03 | 1.1e-06 | 2e-16 | 0.400000000 |
| qtci | 64 | 0.14M | 2164 (52%) | 70 | 30 | 1.2e-03 | 3.9e-04 | 1.2e-06 | 0 | 0.400000000 |
| qtci | 128 | 0.18M | 2562 (15%) | 70 | 40 | 3.2e-04 | 1.2e-04 | 1.2e-07 | 9e-16 | 0.400000000 |
| qtci | 256 | 0.22M | 2911 (4%) | 70 | 64 | 7.8e-05 | 2.7e-05 | 2e-15 | 2e-16 | 0.400000000 |
| triangle | 32 | 0.15M | 2048 | 69 | 7 | 4.8e-03 | 1.5e-03 | 1.1e-06 | | 0.400000000 |
| triangle | 64 | 0.59M | 8192 | 70 | 14 | 1.2e-03 | 3.9e-04 | 1.2e-06 | | 0.400000000 |
| triangle | 128 | 2.4M | 32768 | 70 | 32 | 3.2e-04 | 1.2e-04 | 1.2e-07 | | 0.400000000 |
| triangle | 256 | 9.4M | 131072 | 70 | 70 | 7.8e-05 | 2.7e-05 | reference | | 0.400000000 |

Honeycomb ferromagnet, maxerror = 1e-7 for qtci unless said, 1e-9 for the
triangle:

| method | nk | diagonalizations | per iteration | iterations | wall (s) | error in m | error in mu | error in E | qtci minus triangle in m | filling |
|---|---|---|---|---|---|---|---|---|---|---|
| qtci | 32 | 0.027M | 1024 (100%) | 25 | 11 | 5.1e-03 | 1.3e-03 | 7.7e-06 | 1e-08 | 0.350000000 |
| qtci | 64 | 0.12M | 4096 (100%) | 29 | 27 | 1.5e-03 | 4.0e-04 | 1.1e-06 | 9e-08 | 0.349999999 |
| qtci, maxerror 1e-9 | 64 | 1.2M | 4092 (99%) | 400 (not converged) | 327 | 1.5e-03 | 4.0e-04 | 1.1e-06 | 1e-08 | 0.350000002 |
| qtci | 128 | 0.45M | 14935 (91%) | 30 | 56 | 3.6e-04 | 9.4e-05 | 1.5e-08 | 2e-07 | 0.349999997 |
| qtci, tolerance 1e-8, maxerror 1e-9 | 128 | 0.50M | 16186 (99%) | 30 | 81 | 3.6e-04 | 9.4e-05 | 7.9e-09 | 2e-10 | 0.350000000 |
| qtci | 256 | 1.0M | 35515 (54%) | 29 | 93 | 9.2e-05 | 2.4e-05 | 7.6e-08 | 3e-08 | 0.349999993 |
| triangle | 32 | 0.057M | 2048 | 26 | 4 | 5.1e-03 | 1.3e-03 | 7.7e-06 | | 0.350000000 |
| triangle | 64 | 0.24M | 8192 | 27 | 9 | 1.5e-03 | 4.0e-04 | 1.1e-06 | | 0.350000000 |
| triangle | 128 | 0.98M | 32768 | 28 | 17 | 3.6e-04 | 9.4e-05 | 7.4e-09 | | 0.350000000 |
| triangle | 256 | 3.8M | 131072 | 27 | 41 | 9.2e-05 | 2.4e-05 | 2.5e-09 | | 0.350000000 |

What these say:

- qtci now reaches the triangle state on both metals at every nk, with the
  same errors against the references. On the square ferromagnet it is the
  triangle to rounding, because its fields are trivial: the projectors of
  the two spin bands are constant, and the largest rank is 9.
- On the square ferromagnet the saving grows with the mesh. At nk = 256 an
  iteration diagonalizes 4% of the mesh, 45 times fewer points than the
  triangle, and the whole run takes about the same wall time.
- On the honeycomb ferromagnet qtci needs nearly the whole mesh up to
  nk = 128, 91 to 100% per iteration, and half of it at nk = 256. It saves
  a factor 2 to 4 in diagonalizations, mostly because the triangle
  diagonalizes the mesh twice, and it is 2 to 3 times slower in wall time,
  since a 4 x 4 diagonalization costs less than the cross interpolation.
  That is the honest outcome for this model: the gain in points starts only
  at R = 8 and grows from there, as the frozen measurements above show
  (11% of the mesh at R = 9).
- The pivots of the cross interpolation change from one iteration to the
  next, and each entry carries an interpolation error, 8e-9 at tolerance
  1e-6 and 3e-10 at 1e-8, against the triangle on the same mesh at nk = 64.
  So the residual of the loop has a floor. At maxerror = 1e-9 the
  honeycomb run at nk = 64 did not converge in 400 iterations, while at
  maxerror = 1e-7 it stops after 29, as the triangle does at 1e-9. A
  tolerance of 1e-8 lowers the floor below 1e-9 and costs 10% more points
  at nk = 128. The filling is exact up to the same interpolation error,
  7e-9 at worst.

#### A gapped state

Honeycomb, sublattice imbalance 0.5, U = 1, half filling, T = 1e-7. The
reference is exact diagonalization at nk = 128. In a gap every triangle is
full or empty, so the triangle is exact diagonalization on the same mesh,
which converges exponentially, and qtci reproduces it up to the
interpolation error. The old Gauss-Kronrod path was run from the previous
commit.

| method | nk | diagonalizations | iterations | wall (s) | error in E | error in the charge order |
|---|---|---|---|---|---|---|
| Gauss-Kronrod qtci | 8 | 3380 | 7 | 6 | 1.0e-04 | 1.2e-04 |
| Gauss-Kronrod qtci | 16 | 5780 | 7 | 2 | 2.0e-05 | 4.9e-05 |
| Gauss-Kronrod qtci | 32 | 8820 | 7 | 3 | 6.7e-06 | 3.2e-05 |
| Gauss-Kronrod qtci | 64 | 12500 | 7 | 4 | 4.6e-06 | 5.4e-05 |
| qtci | 8 | 512 | 7 | 2 | 1.3e-04 | 3.2e-04 |
| qtci | 16 | 2048 | 7 | 3 | 7.0e-06 | 7.7e-05 |
| qtci | 32 | 8192 | 7 | 4 | 8.1e-11 | 2.4e-09 |
| qtci | 64 | 61406 | 14 | 14 | 2.8e-08 | 8.6e-09 |
| ed | 32 | 18432 | 7 | 0.4 | 8.1e-11 | 2.4e-09 |

The charge order is the difference of the charges of the two sublattices.
For the same number of diagonalizations the new qtci is better than the
Gauss-Kronrod rule: 7.0e-6 in E with 2048 against 2.0e-5 with 5780, and
8e-11 with 8192. The Gauss-Kronrod rule stops improving at about 5e-6 in E
and 5e-5 in the charge order. The frozen density matrix of the same model
with a tilted exchange field and the V1 entries says the same. Against
exact diagonalization at nk = 240, the Gauss-Kronrod rule gives 5.1e-5 with
289 points and 5.2e-6 with 441, and the new qtci gives 3.4e-6 with 256 and
7.9e-10 with 1024. So no gapped case is better with the old path, and it
was removed. At nk = 64 qtci needs most of the mesh again and sits on its
interpolation floor of about 1e-8. In a gap a uniform mesh of nk = 32 has
already converged, so there is no reason to go there.

#### Between iterations

Every iteration starts the cross interpolation from scratch, with the coarse
scan of 32 x 32 points. The first and the last iteration diagonalize about
the same number of points (1959 and 2164 on the square ferromagnet at
nk = 64, 32498 and 35515 on the honeycomb at nk = 256), so nothing is
learned from one iteration to the next. We tried to seed the TCI of a
nearby Hamiltonian (the honeycomb ferromagnet at exchange 0.184, seeded
from 0.183, R = 8) with the global pivots of the converged one. That set
has 10749 pivots, and the seeded run diagonalized the whole mesh, 65536
points in 64 s, against 23119 points in 5 s from scratch. So the naive
reuse is much worse, and it was not built. A reuse that keeps only the
pivots of each bond, and not their products, was not tried.

#### What is in the code now

- `get_dm_qtci`, `get_fermi4filling_qtci`, `full_dm_qtci` and
  `band_energy_qtci` in `qtcitk/densitymatrix_qtci.py` are the triangle
  construction above, and `scftk/densitydensity.py` calls them in its qtci
  branches. Since 7 November 2026 they work in 2D and in 3D, see qtci in
  3D below; a chain is refused.
- The Gauss-Kronrod path is gone: `gk_node_grid`, `full_dm_gk`,
  `band_energy_gk`, and the warning `warn_if_metal`, since metals work now.
  `gkintegrate.py` stays, because `topology.chern_qtci` integrates with it.
- Bogoliubov-de Gennes Hamiltonians are accepted. The weights are those of
  the quasiparticle bands at zero energy, the Fermi level is found on the
  normal state, and the band energy is (sum_{E<0} E_BdG + Tr h_e)/2. On a
  frozen s-wave state this band energy equals exact diagonalization on the
  same mesh to 1e-9. The total energy of the mean field is settled in The
  total energy of a superconductor.
- `tests/scf/test_densitydensity_qtci.py` checks that qtci equals the
  triangle on the same mesh for a frozen low-symmetry metal (complex
  entries, nonzero d), that the eigenpairs of the Fermi search are reused
  only for a shift, that the filling is held to 1e-6, that the mean field
  reaches the triangle state of the square ferromagnet at nk = 8 (m, mu and
  E to 1e-6, in about 10 s), and that an insulator gives the mean field of
  exact diagonalization.

#### qtci in 3D (7 November 2026)

The qtci mean field now runs in 3D: the band energies and the fields X_n
are interpolated on a 2^R x 2^R x 2^R mesh with three quantics coordinates,
in the interleaved bit order of `qtcitk/meshfields.py` (the comparison of
the two orders is in `qtci_dos_green.md`, the section on the qtci DOS in
3D), and the tetrahedron weights of simplexdm are applied to the
reconstructed mesh. `tests/scf/test_densitydensity_qtci_3d.py` checks that
the density matrix of a cubic metal with Rashba coupling and a tilted
exchange field (complex entries, bonds along x, z and (1,1,1)) is the
tetrahedron one of the same 8^3 mesh to 1e-6, and that the mean field of
the cubic Hubbard ferromagnet (U = 10, filling 0.3, T = 1e-3, a metal with
m = 0.590 and about 0.005 minority electrons) reaches the tetrahedron state
of the same mesh: m, mu and E agree to 1e-14 there, and the test asks for
1e-6. A partly polarized state, U = 7 (m = 0.279), agrees as well, to
1e-14, but took 140 s per integration on a loaded machine, so it is not in
the tests.

The question for the benchmark is whether the nk^3 growth of the mesh makes
qtci pay. One iteration of the mean field, the Fermi search and the
density matrix together, on a frozen Hamiltonian, at T = 0, tolerance
1e-6, with the onsite spin blocks of every site, the entries a Hubbard U
reads. Three models: the simple cubic band in an exchange field of 1.5 (two
bands, a ferromagnetic metal), and two spinful layered models, gapped
graphene layers (sublattice imbalance 0.3) stacked with the interlayer
hopping tm times the sublattice sign, so a mass m(kz) = 0.3 + 2 tm cos kz,
in an exchange field of 0.5 (four bands), with tm = 0.3 (the mass changes
sign, a semimetal before the exchange) and tm = 0.1 (the mass stays
positive). At filling 0.3 four bands hold 1.2 electrons, so the Fermi
level is inside the second band and both layered models are metals. The triangle diagonalizes the
mesh twice, once for the Fermi level and once for the density matrix. The
error is the largest difference of an entry from the tetrahedra on the same
mesh. The rank is the largest bond dimension of the TCI of the energies and
of the one of the X fields. Four cores of a workstation that other jobs
kept busy (load 6 to 9 on 16 cores).

| model | R | mesh | method | diagonalizations (share) | rank E, X | error | time (s) | resident memory (GB) |
|---|---|---|---|---|---|---|---|---|
| cubic | 5 | 32^3 | tetrahedron | 65536 (2) | | | 0.62 | 0.4 |
| cubic | 5 | 32^3 | qtci | 3210 (0.098) | 11, 1 | 2e-16 | 0.82 | 0.4 |
| cubic | 6 | 64^3 | tetrahedron | 524288 (2) | | | 3.86 | 0.6 |
| cubic | 6 | 64^3 | qtci | 5326 (0.020) | 12, 1 | 2e-16 | 3.57 | 0.6 |
| cubic | 7 | 128^3 | tetrahedron | 4.19M (2) | | | 30.1 | 2.0 |
| cubic | 7 | 128^3 | qtci | 6288 (0.0030) | 11, 1 | 1e-15 | 27.1 | 2.0 |
| layered, tm = 0.3 | 5 | 32^3 | tetrahedron | 65536 (2) | | | 0.85 | |
| layered, tm = 0.3 | 5 | 32^3 | qtci | 32768 (1) | 138, 161 | 4e-10 | 3.74 | |
| layered, tm = 0.3 | 6 | 64^3 | tetrahedron | 524288 (2) | | | 5.00 | |
| layered, tm = 0.3 | 6 | 64^3 | qtci | 260488 (0.99) | 244, 336 | 9e-09 | 26.3 | |
| layered, tm = 0.3 | 7 | 128^3 | tetrahedron | 4.19M (2) | | | 38.0 | 2.8 |
| layered, tm = 0.3 | 7 | 128^3 | qtci | 1.32M (0.63) | 358, 403 | 4.7e-06 | 86.6 | 4.4 |
| layered, tm = 0.1 | 5 | 32^3 | tetrahedron | 65536 (2) | | | 0.77 | |
| layered, tm = 0.1 | 5 | 32^3 | qtci | 32768 (1) | 127, 153 | 2e-09 | 2.90 | |
| layered, tm = 0.1 | 6 | 64^3 | tetrahedron | 524288 (2) | | | 5.20 | |
| layered, tm = 0.1 | 6 | 64^3 | qtci | 255631 (0.98) | 215, 313 | 6e-09 | 18.9 | |
| layered, tm = 0.1 | 7 | 128^3 | tetrahedron | 4.19M (2) | | | 40.2 | 2.8 |
| layered, tm = 0.1 | 7 | 128^3 | qtci | 1.22M (0.58) | 346, 404 | 2.3e-05 | 87.6 | 4.3 |
| layered, tm = 0.1 | 7 | 128^3 | qtci, cap 800 | 1.32M (0.63) | 345, 512 | 5.0e-09 | 112 | 4.4 |

What we read from it:

- On the cubic ferromagnet qtci diagonalizes 0.3% of the 128^3 mesh, 670
  times fewer points than the tetrahedra, and the fields are exact: the
  energies are cosines of rank 11, and the X fields of a collinear magnet
  without spin mixing are constant (rank 1). Even so it is only 10% faster
  (27 s against 30 s), because the tetrahedron weights and the Fermi search
  on 12.6M tetrahedra are most of the time, and they are the same for both.
- On the layered models qtci needs the whole mesh at 32^3 and 64^3, and 58
  to 63% at 128^3, so 3.2 to 3.4 times fewer diagonalizations than the
  tetrahedra, which diagonalize the mesh twice. It is 2.2 times slower in
  wall time at 128^3 and 3.6 to 5.3 times slower below, and it takes 4.4 GB
  against 2.8 GB. The ranks are high, 350 for the energies and 400 for the
  X fields at 128^3. In the exchange field of 0.5 the conduction band of one
  spin crosses the valence band of the other wherever the gap is below 1,
  and each crossing is a kink of the bands ordered by energy, which costs
  rank; the dependence on kz adds to it.
- At 128^3 the TCI of the X fields stops at the rank cap of 400 of
  `qtci_stacked_fields`, and the error rises from 1e-8 to 4.7e-6 and
  2.3e-5, above the tolerance. With the cap of 800 of the qtci LDOS
  (`meshfields.stacked_maxbonddim`) the layered model with tm = 0.1 stops at
  rank 512 with an error of 5.0e-9, at the same share of the mesh (0.63)
  and in 112 s instead of 88 s. So the qtci density matrix takes that cap
  now, in 2D as in 3D; it only changes a TCI that reached 400.

So in 3D qtci saves k-points, by a large factor on smooth bands and by about
3 on the layered models at 128^3, but it does not save wall time on any of
these models, since the tetrahedron weights on the full mesh cost as much
as the diagonalization of two or four bands. At T = 1e-3 the gap is wider:
the Fermi search at finite temperature took 33 s at 32^3 on the cubic band
for the tetrahedra and 37 s for qtci, against 0.6 s and 0.8 s at T = 0, and
131 s at 64^3; the weights at T = 1e-3 are 20 times slower than at T = 0 on
another machine (1.45 s against 0.08 s at 32^3), and the Fermi search,
which evaluates the charge at many trial levels, 20 times as well (10.2 s
against 0.5 s). That cost was the same for both methods. It is gone since
the Fermi search is a safeguarded Newton step with the simplex DOS as its
derivative (`simplex_fermi_search.md`): 4 charge evaluations instead of 11,
each one cheaper, so at 32^3 and T = 1e-2 the search takes 0.20 s against
20.8 s before, and 0.019 s against 0.55 s at T = 0.

## The total energy of a superconductor

The question is which constant the mean field adds back to the band energy
of a Bogoliubov-de Gennes state. The band energy is summed for the
Hamiltonian shifted by its Fermi level, so together with the double
counting it is the grand potential Omega = <H - mu N>. With `mu=None` the
driver adds mu N back. `"ed"` took N = N0, the requested charge, filling
times the number of electron states. qtci took the trace of the density
matrix, the charge of the paired state. In a normal state these are the
same number. In a superconductor they are not, because the Fermi level
comes from the normal state (as in `spectrum.get_fermi4filling`, which
removes the Nambu part before counting), and the paired state at that level
holds another charge.

For concreteness we take the spinful square lattice with first-neighbor
hopping, e(k) = 2(cos 2 pi k1 + cos 2 pi k2) in pyqula's sign, U = -2,
filling 0.3 and `mf="swave"`. As a reference we wrote the BCS mean field by
hand, with no pyqula: the trial state is the ground state of the BdG
Hamiltonian with a Hartree shift s and a gap D, and
Omega = mean_k 2 (e_k - mu) n_k + U (n^2 + F^2), with n the density per spin
and F = <c_dn c_up>, the Wick contraction of U n_up n_dn. At mu = -1.68 on
a 1600 x 1600 mesh it gives Omega = -0.6126340, D = 0.2131 and
N = 0.60014, and the same numbers on 400 x 400 and at T = 1e-3, since the
state is gapped. Three checks decide.

- At fixed mu the gradient of Omega in (s, D) is 6e-12, while the gradient
  of <H> = Omega + mu N is (0.470, -0.076). So <H> is not stationary at
  fixed mu, because the charge moves with the pairing.
- Hellmann-Feynman at fixed mu: dOmega/dU = 0.10139026 against
  <n_up n_dn> = 0.10139026, while d<H>/dU = 0.345.
- pyqula at `mu=-1.68` adds no constant at all, and it gives the grand
  potential of the hand-written mean field: -0.61263418 with `"ed"` and
  -0.61263417 with qtci at nk = 64, and -0.61261336 with both at nk = 32.
  So the band energy and the double counting of both paths were right, and
  only the constant was in question.

As a result Omega + mu N0, with N0 fixed, is the quantity that is
stationary at fixed mu and obeys Hellmann-Feynman, and <H> is neither. This
is what `"ed"` returned. It is also what a user of
`h.get_mean_field_hamiltonian()` gets, since for a spinful Hamiltonian
with `integration="ed"` that goes through `VJinteraction`, which adds the
same constant. With the Fermi level of the normal state at filling 0.3 the
hand-written mean field gives mu = -1.66514 and N = 0.60615 on an
800 x 800 mesh, Omega + mu N0 = -1.6206807 and <H> = -1.6309261. So the two
conventions are 1.03e-2 apart on a converged mesh; the 1.3e-2 quoted
before was on a coarse one.

So every integration adds mu N0 now, in one place in
`scftk/densitydensity.py`, after the three band energies. In a normal state
this changes nothing for the triangle, whose density matrix holds N0 to
1e-12, and it changes qtci by mu times its filling error, roughly 1e-8. The
triangle also accepts a BdG Hamiltonian now, with the construction of
qtci: the weights of the quasiparticle bands at zero energy, the Fermi
level of the normal state, and the band energy
(sum_{E<0} E_BdG + Tr h_e)/2. In a gapped superconductor every triangle is
full or empty, so at fixed mu the triangle is exact diagonalization on the
same mesh, to 1e-8 in the test.

The total energy at `mu=None`, U = -2, filling 0.3, against the reference
Omega + mu N0 = -1.6206807:

| nk | `"ed"` | `"triangle"` | qtci before | qtci now |
|---:|---:|---:|---:|---:|
| 16 | -1.6209185 | -1.6210916 | -1.6399936 | -1.6210916 |
| 32 | -1.6206118 | -1.6206856 | -1.6335518 | -1.6206856 |
| 64 | -1.6206583 | -1.6206841 | -1.6312758 | -1.6206841 |

Before, qtci converged to <H> = -1.6309 instead. What is left between
`"ed"` and the triangle is the Fermi level, not the energy. `"ed"` counts
the eigenvalues of the normal state on the mesh, a staircase (mu = -1.6836
at nk = 16 against -1.6564 for the triangle), while the triangle charge is
continuous in mu. The energy only moves with mu as mu (N - N0), 6e-3 here,
so a Fermi level that is 3e-2 off costs about 2e-4 at nk = 16. At nk = 64
the triangle is within 4e-6 of the reference and `"ed"` within 3e-5. The
charge of the paired state of `"ed"`, 0.6012 at nk = 16 and 0.6044 at
nk = 64, moves with that staircase too, and it is not a second bug.

`tests/scf/test_densitydensity_bdg_energy.py` checked this at nk = 16: the
three integrations agree to 1e-8 (triangle) and 1e-6 (qtci) at fixed mu,
and to 1e-3 and 1e-6 at a filling; the reported energy at `mu=None` is
Omega at the converged level plus mu N0, with Omega written out by hand
from the projector on the mesh, for each integration; when the pairing
is scaled by 1 + x at fixed mu, dOmega/dx = 3e-10 and d<H>/dx = -8e-3; and
dOmega/dU = <n_up n_dn> to 1e-6. The checks at a filling changed with the
next section.

## The Fermi level of a superconductor

The question is how to find the Fermi level of a BdG mean field at a
filling, so that the paired state holds the requested charge, and not the
normal state. Then N = N0, and the reported energy Omega + mu N0 is <H> at
that charge. When the loop cannot hold the charge (a state without a gap
on the mesh, see below), the reported energy stays Omega + mu N0, which
differs from <H> by mu (N - N0), with N - N0 in `scf.charge_error`.

Two designs were possible. The first is a root find of mu on the charge of
the paired state at each iteration, which costs a few more density
matrices per iteration. The second takes one step in mu per iteration and
converges it together with the mean field. We built the second, because
it costs no extra density matrix. The chemical potential is written as
mu = mu_n + x. Here mu_n is the Fermi level of the normal state of the
current Hamiltonian, found with the count of the integration in use, as
before, and x is an offset. Each density matrix gives the charge of the
paired state, N = Tr rho_e(0,0,0), the trace of its electron block, which
all three integrations compute (qtci now always computes the diagonal of
the onsite block for a BdG Hamiltonian, also on sites with no
interaction). The next offset is x - g (N - N0)/(2 kappa), with
kappa = dN/dmu of the normal state, from two Fermi searches at the
filling plus and minus 0.05 on the first Hamiltonian, and g a gain, 1 at
the start. The
plain loop holds the size of this step to `maxerror`, together with the
change of the mean field. Broyden mixing and the scipy solvers take x as
the last entry of their vector, so their fixed point has the step at rest,
and they take a step without the gain (see "The step under the
quasi-Newton solvers").
This is `scftk.superscf.BdGFermiLevel`. It is used by `densitydensity`
(`Vinteraction`, `hubbard`) for `"ed"`, `"triangle"` and `"qtci"`, and by
`VJinteraction`, the engine of `h.get_mean_field_hamiltonian()` for a
spinful Hamiltonian, for `"ed"`. `spectrum.get_fermi4filling` and the two
Fermi searches of the simplices are unchanged, since other code calls them
outside the mean field.

Writing mu around mu_n has two reasons. The Hartree shift moves mu_n, so
the offset only carries the effect of the pairing, of order Delta^2/W.
And when the pairing vanishes, the paired charge at mu_n is the normal
one, so x = 0 is exactly the old Fermi level. Three safeguards come from
the states without a gap on the mesh. The reason for all three is that the
charge of such a state is a staircase in mu, for `"ed"` and for the
triangles alike (see "The triangle of a BdG Hamiltonian is the step"), so the offset has to sit on a
level, where the smearing makes the charge continuous, or no offset holds
N0 at all, and a fixed step oscillates around a jump. First, the offset is
clipped to a window |x| <= 2 Delta^3/(Delta^2 + Delta_s^2), with Delta the
largest anomalous entry of the Hamiltonian and Delta_s = 1e-3 W,
W = (electron states)/kappa. The window is 2 Delta for a sizable pairing,
and it closes as Delta^3 when the pairing dies out, so that the old Fermi
level comes back. Second, the gain halves each time N - N0 changes sign,
down to 1/64, and grows back by 1.5 while it does not. Third, the step
gives up and sets x = 0 for the rest of the loop, which is then the old
loop, in two cases: after eight steps in a row held by the window, and
after eight changes of sign at the smallest gain with |N - N0| above 1e-6
(below that it is the noise of qtci). A gapped state never gets there.
When the step gives up, the loop converges with the charge error of the
old Fermi level, and it prints one line when that error is above
100 maxerror. The error is also in `scf.charge_error`.

The three safeguards came out of failures, all in the tests of
`tests/scf`. A window of 2 Delta that sets x back to zero when it leaves
it gave a cycle of period three for qtci at nk = 8. A window that does not
close made the loop slow when the pairing dies, since x stays at the edge
of a shrinking window and moves mu across the levels of the mesh (more
than 400 iterations against 41 for the Zeeman case below). The halving
gain with no floor and no give-up froze x at a wrong value on the
ferromagnetic chain of `tests/scf/test_unconventional_superconductivity.py`,
a triplet with nodes on the mesh points at nk = 6, which then did not
converge in 1000 iterations. A give-up after eight changes of sign at any
gain fixed that chain, but it made the result depend on the random
guess: on the ferromagnetic bichain with V1 = -2 and J1 = -1 of
`tests/scf/test_spinspin_nambu.py` and `tests/scf/test_spinspin_supercell.py`
about half of the random guesses gave up during the first 20 iterations
and ended at the old Fermi level, with a charge error of 0.197 per cell and
E = -3.10296, and the other half held the charge, E = -3.12367. Counting
only the changes of sign at the smallest gain, six random guesses out of
six hold the charge, the chain converges in 78 iterations (50 before) to
the same triplet, and the triangular ferromagnet of
`tests/scf/test_nonunitarity_triangular.py` at nk = 4, which the window
gives up, keeps its recorded d-vector.

For concreteness we take again U = -2 on the spinful square lattice, with
`solver="broyden_mixing"` and `maxerror=1e-10` (1e-8 for qtci). The
charge per cell and the energy, before and now:

| nk | filling | N before (`"ed"`, triangle) | N now | E before (`"ed"`, triangle) | E now (all three) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.3 | 0.60124, 0.61141 | 0.6 | -1.6209185, -1.6210916 | -1.6209165 |
| 32 | 0.3 | 0.60010, 0.60773 | 0.6 | -1.6206118, -1.6206856 | -1.6206118 |
| 64 | 0.3 | 0.60442, 0.60636 | 0.6 | -1.6206583, -1.6206841 | -1.6206342 |
| 16 | 0.2 | 0.41174, 0.41443 | 0.4 | -1.2282869, -1.2283682 | -1.2281283 |

Before, qtci gave the numbers of the triangle. Now N - N0 is below 1e-9
for `"ed"` and the triangle, and about 1e-8 for qtci, which is its
interpolation floor. The three integrations give the same Fermi level and
the same energy, to 1e-10 for `"ed"` and the triangle, since the paired
state is gapped: every triangle is full or empty, and the charge of
`"ed"` is continuous in mu, so the staircase of the normal state no
longer enters. The triangle on a 256 x 256 mesh gives E = -1.6206340 at
filling 0.3, and nk = 128 gives the same to 5e-12, so this is the
converged <H> at the requested charge, with mu = -1.68034 and
Delta = 0.2130. It lies 4.7e-5 above the Omega + mu N0 = -1.6206807 of the
hand-written mean field above, which was at the normal-state Fermi level
and so at the charge 0.606. At nk = 64 the three are within 2e-7 of it.
Before, the triangle was 4e-6 and `"ed"` 3e-5 away from the old
reference.

### The triangle of a BdG Hamiltonian is the step

A reviewer found that the triangle density matrix of a BdG Hamiltonian is
exactly the step density matrix of exact diagonalization on the same mesh.
The spectrum is E and -E, and the bands are sorted at each k, so the lower
half of the bands is below zero at every k: no simplex is ever partly
occupied, and a vertex on a level is occupied by half, the average of its
multiplet. But the Fermi level came from the continuous triangle count of
the normal state, a different number. So the density matrix held another
charge even with no pairing: 0.3477, 0.3604 and 0.3528 per state at
nk = 16, 32 and 64 for 0.35, and qtci 0.3613 at nk = 32. In a mean field
this lost the pairing (U = -1, filling 0.3, nk = 24: `"ed"` has
|F| = 0.0298, the triangle 0) and made a spurious ferromagnet (U = 2,
`mf="ferro"`, nk = 64: mz = 0.00586 with Nambu, 0 without). The Fermi
search of a BdG Hamiltonian in `simplexdm.get_fermi4filling` and
`get_fermi4filling_qtci` is now the count of exact diagonalization at
T = 0 on the same mesh, the count of the step, and the mean field then
corrects it on the paired charge. With it the triangle and qtci give the
pairing of `"ed"` on the same mesh, and a Nambu state without pairing the
magnetization of the normal state with `"ed"` at T = 0. The normal
triangle (mz = 0 there) is out of reach: the charge of the sorted BdG
bands is a staircase. The reproducers, before and after:

- The charge per state of the triangle at the Fermi level of its search,
  with no pairing, for 0.35: before 0.3477, 0.3604 and 0.3528 at nk = 16,
  32 and 64; now 0.3633, 0.3486 and 0.3499, the same as exact
  diagonalization to 1e-5. This is the staircase of the step: at nk = 16
  a shell of degenerate levels sits on the Fermi level and holds half.
- qtci at nk = 32: 0.3613 before, 0.3486 now, the number of the triangle.
- A Nambu mean field with U = 1 from `mf="potential"`, nk = 32, filling
  0.35, no pairing: the triangle held 0.3604 with E = -1.403163; now
  0.3486 with E = -1.402939, against 0.3501 and -1.402938 for `"ed"`.
- s-wave, U = -1, filling 0.3, nk = 24: before `"ed"` gave |F| = 0.0298
  with charge 0.3039 per state, and the triangle |F| = 0 with 0.2865; now
  both give |F| = 0.0283, charge 0.3000 and E = -1.5263868.
- U = 2 from `mf="ferro"`, filling 0.35, nk = 64: the Nambu triangle had
  mz = 0.00586 and E = -1.280984, now mz = 0.00293 and E = -1.280962, the
  energy of `"ed"` (mz = 0.00220 at T = 1e-7, 0.00282 at T = 0). The normal
  triangle has mz = 0 and E = -1.281071; the staircase of the step makes
  this small ferromagnet, in `"ed"` as in the Nambu triangle.

The tests are in `tests/scf/test_densitydensity_bdg_energy.py`.

Other states, all at nk = 16:

- U = -6 at filling 0.1, a strongly paired state: the old Fermi level put
  0.334 electrons per cell instead of 0.2, an error of 0.134. Now the
  error is 3e-11. The energy goes from -0.85288 to -0.83832.
- A d-wave state from V1 = -2 with U = 1 at filling 0.4, started from a
  `"dx2y2"` pairing: the old charge was 0.871 instead of 0.8, now it is
  0.8 to 1e-10. It needed 57 Broyden iterations for `"ed"` and 43 for the
  triangle, against 13 and 13 before, from a guess that was not recorded.
  From the guess `add_pairing(mode="dx2y2", delta=0.1)` it needed 48 and
  45, and now 20 and 20 (see "The step under the quasi-Newton solvers").
- U = -1 with the triangle at filling 0.3. The old Fermi level held 0.633,
  and on this mesh the pairing died. At the charge 0.600 it survives, with
  Delta = 0.029. The old code at the new mu, fixed, finds the same state
  and the same energy to 1e-10, so this is the charge and not a second
  difference.
- No pairing, U = 2 with Nambu and `"ed"`: the Fermi level and the energy
  are the old ones, to 3e-12. With the triangle they are now the ones of
  `"ed"` (see "The triangle of a BdG Hamiltonian is the step").
- s-wave with a Zeeman field 0.3, larger than the gap, where the pairing
  dies: with Broyden, `"ed"` goes from 41 to 89 iterations, to the old
  state and the old charge. The triangle goes from 19 to 109 iterations,
  and now ends at the charge of `"ed"` (0.6016) instead of 0.5859. With the step
  of the solvers it is 57 for both (see "The step under the quasi-Newton
  solvers").
- Attractive V1 = -2.5 on the spinful chain at filling 0.3: before, the
  extended s-wave singlet from `mf="swave"` held 0.514 electrons per cell
  and the triplet from `mf="pwave"` 0.713, instead of 0.6. At 0.6 both
  guesses go to the same triplet, E = -1.7771730 for both, in about 250
  plain iterations against 50 before, because the state from the singlet
  guess is not stable there and moves slowly to the triplet. So the
  singlet test of `tests/scf/test_bond_pairing_prefactor.py` runs at
  filling 0.2 now, where the singlet is the converged state.
- qtci at nk = 8 (U = -2, filling 0.3), where the gap is barely resolved:
  the old loop lost the pairing (Delta = 7e-4). Now it holds the charge,
  with Delta = 0.233 against 0.213 on a converged mesh.

The cost per iteration is the one of before, one density matrix and one
normal-state Fermi search, plus two Fermi searches at the start. What
changes is the number of iterations, because the loop has one more
unknown. For the s-wave state at filling 0.3 Broyden needs 27, 28 and 31
iterations at nk = 16, 32 and 64 with `"ed"`, against 21, 22 and 22, and
about 24 for qtci against 22; at filling 0.2, 11 to 19 against 9 to 11.
With the step of the solvers it is now 23, 23 and 24 (see the next
section).
The wall time moves with the iterations. It was measured on four reserved
cores of a workstation that had other jobs, with the old and the new code
run one after the other: at nk = 64 and filling 0.3 it is 3.1 s against
2.1 s for `"ed"`, 5.0 s against 3.8 s for the triangle, and 24 s against
22 s for qtci. `VJinteraction` with plain mixing needs about the same
number of iterations as before: 212, 204 and 206 at nk = 16, 32 and 64,
against 215, 208 and 213, with 18.5 s against 23.7 s at nk = 64.

`tests/scf/test_densitydensity_bdg_energy.py` checks now, at nk = 16: the
paired state holds the requested filling (0.3 and 0.2) to 1e-8 (1e-6 for
qtci) for each integration, and `h.get_filling()` reads it back to 1e-6;
the energy is Omega + mu N with N the charge of the projector, written out
by hand, so <H>; the three integrations agree at a filling to 1e-8 in the
energy and in the Fermi level (1e-6 for qtci); `VJinteraction` holds the
filling too; and a Nambu state without pairing keeps the normal-state
Fermi level to 1e-10, with the energy of a run at that fixed mu plus
mu N0. The checks at fixed mu are unchanged. Two more check the triangle
of a BdG Hamiltonian: with U = -1 at nk = 32 the triangle and qtci give
the pairing of `"ed"` (to 1e-6 and 1e-5), and at nk = 64 a Nambu state
with U = 2 from a ferromagnetic guess has the magnetization of the normal
state with `"ed"` at T = 0 (to 3e-4 for the triangle, whose Bloechl term
moves it by about 1e-4). qtci is left out of the gapless checks: on a
degenerate level its energy at a filling differs from the one at the
fixed level by a few k-states (3.7e-4), and its Nambu ferromagnet at
nk = 64 takes minutes.

Not touched. The KPM engines keep the normal-state Fermi level for a
Nambu Hamiltonian (`LaggedFermi.shift` searches it each iteration on the
electron-only Hamiltonian), and `VJinteraction` refuses a dense Nambu
Hamiltonian with `integration="kpm"`. `densitydensity_jax` has no BdG
path at all.

### The step under the quasi-Newton solvers

Why did one more unknown cost so many Broyden iterations, 48 instead of
13 for the d-wave state? The step of x was taken inside the function the
solver evaluates, and that step has a memory: its gain halves on each
change of sign of N - N0, and the give-up counts the steps held by the
window and the changes of sign at the smallest gain. So every trial
vector of Broyden mixing, and of the scipy solvers, moved the gain, and
the entry of x in the new vector, x - g (N - N0)/(2 kappa), was a
different number at each call for the same vector. A multisecant update
fits a linear model of that map, and a map that changes with the history
cannot be fitted. We checked this on the code before the change, logging
the state of `BdGFermiLevel` at each evaluation of one Broyden run at
nk = 16, `maxerror=1e-10`, spinful square lattice, `"ed"`:

- s-wave, U = -2, filling 0.3: the gain takes 7 values, from 1 down to
  0.28, in 29 evaluations. No give-up.
- d-wave, V1 = -2 and U = 1 at filling 0.4, from the guess
  `h.add_pairing(mode="dx2y2", delta=0.1)` taken as a mean field (the
  difference of the two Hamiltonians, since `mf="dx2y2"` is not a guess
  mode of `meanfield.guess`): the gain takes 17 values, down to 0.18, in
  50 evaluations. No give-up.
- s-wave, U = -2, with a Zeeman field 0.3: the gain goes down to 1/64,
  there are 6 changes of sign at the smallest gain, and the window holds
  the step 8 times in a row, so the step gives up.

So now `BdGFermiLevel.step(x, dm, adaptive=False)` is the step of the
solvers, and `generic_densitydensity` calls it for every solver except
`"plain"`. It depends only on x and the density matrix of the Hamiltonian
at x: the Newton step dx = -(N - N0)/kappa, with no gain and no damping,
clipped to the window of the pairing amplitude of h, which depends only on
h. The give-up has no stateless form, so it is replaced by one rule: when
|dx| is longer than the whole window, 2 bound, no offset in the window
holds the charge on the linear estimate, wherever x is, and the step goes
to x = 0, the Fermi level of the normal state. In a state whose pairing
dies, the charge error is a staircase of about one k-shell, 0.03 electrons
per cell at nk = 16, while the window closes as Delta^3, so this rule
takes over after a few tens of iterations and x stays at zero, which is
where the give-up of the plain loop ends too. The plain loop keeps its
gain, its damping and its give-up, since they came out of the failures of
the plain loop recorded above. `VJinteraction` has no call site to change:
its numpy engine refuses `solver=` and always runs plain mixing, and its
jax engine refuses a Nambu Hamiltonian.

Four other stateless steps were tried, all at nk = 16. The clipped step
with the damping of the plain loop, 0.5, and no rule for the window: the
d-wave state needs 24, but the Zeeman case gets worse, 114 for `"ed"`
against 89, because x bangs on the two edges of the shrinking window and
spoils the secant model of the pairing. The same with a damping of 1:
d-wave 20, Zeeman 113 and 100. A step that goes to zero when the clipped
step leaves the window: with damping 0.5 Zeeman 46 and 41 but d-wave 26
from a small guess (delta = 0.01), and with damping 1 Zeeman 118 and 179.
The rule on the whole window with damping 1 is the only one that is below
the old count for the three states with `"ed"` and with the triangle. The
give-up kept as a one-time switch inside the solver step gives about the
same counts (d-wave 24 to 15 over the guesses, Zeeman 59 and 55, qtci
Zeeman 81), but it is a memory again, so it was not taken.

Broyden iterations at nk = 16, `maxerror=1e-10`, before and after, with
the total energy and the paired charge N of both:

| state | integration | before | now | E | N |
|---|---|---:|---:|---|---|
| s-wave | `"ed"`, triangle, qtci | 27, 27, 27 | 23, 23, 23 | -1.62091646 | 0.6 |
| d-wave | `"ed"`, triangle, qtci | 48, 45, 59 | 20, 20, 20 | -3.73712961 | 0.8 |
| Zeeman | `"ed"` | 89 | 57 | -1.62640723 | 0.6015625 |
| Zeeman | triangle | 110 | 57 | -1.62640643 | 0.6015625 |
| Zeeman | qtci | 300 | 321 | -1.62640643 | 0.6015625 |

Before and after give the same energy to 1.5e-11 and the same N to 3e-10
in every row, so the fixed point did not move. For the gapped states the
three integrations also agree with each other to 1e-10. The Zeeman case
does not: its `"ed"` and triangle energies differ by 8.1e-7 and its Fermi
levels by 6.5e-4, before as now, since the state has no gap and the Fermi
level sits on a different place of the staircase. Two runs from the same
guess give the same count and the same energy to the last digit, on a
fixed number of BLAS threads; the test asks for the same count and the
energy to 1e-12, since a threaded BLAS may order its sums differently.
Over the d-wave guesses with delta = 0.01, 0.05, 0.2, 0.3 and 0.5 `"ed"`
needed 44, 50, 43, 63 and 23 iterations before and needs 23, 21, 19, 18
and 15 now. The s-wave state at nk = 32 and 64 needs 23 and 24, against
28 and 31. qtci with the s-wave state at nk = 8, where the gap is barely
resolved, converges in 16 iterations against 22, to the same energy. The
Zeeman case with `"ed"` at nk = 32 got slower, 53 against 42. The Zeeman
case with qtci gave 321 against 300, but its count is not reproducible
from run to run, so these two numbers do not rank the two steps. At
nk = 64 the Zeeman case does not converge in 1000 iterations, before as
now. Why the Zeeman case is slow, and what was done about it, is in "The
gapless Zeeman state" below. The wall time follows the iterations: on
three reserved cores of a loaded
workstation the d-wave state takes 0.44 s against 1.7 s with `"ed"`, and
the s-wave state at nk = 64 3.5 s against 5.3 s.

`tests/scf/test_bdg_fermi_level_solvers.py` checks that the solver step
gives the same next offset for the same offset and density matrix,
whatever was evaluated in between, while the adaptive step does not; that
the d-wave state needs fewer Broyden iterations with the solver step than
with the adaptive step forced on the solver, at the same energy to 1e-8
and the same charge, with the same count twice in a row; and that in the
Zeeman case both give the same energy, charge error and Fermi level to
1e-8.

Two smaller changes went with it. `BdGFermiLevel.warn` printed "its
quasiparticle spectrum has no gap on this mesh" after a loop stopped at
`maxite`, for instance `Vinteraction(V1=-1.5, filling=0.3, nk=8,
maxite=10)` on the Nambu square lattice, where the charge error of 7e-3 is
only the error of an unconverged loop. Now it prints only for a converged
loop. And `pairing_amplitude(h)`, called by `shift` on every iteration,
reordered and densified every hopping through `get_eh_sector`; it reads the
anomalous entries with the Nambu indices directly now, the same number to
1e-14. In the plain s-wave loop at nk = 16 one call went from 1.46 ms to
0.10 ms, from 12% of an iteration to below 1%. The rest of `shift` is the
Fermi search of the normal state, about 40% of an iteration of this
one-site cell, which was not touched.

#### The staircase of the offset

Is a better treatment of the staircase worth building? In a state with no
gap on the mesh the requested charge N0 falls inside a jump of the paired
charge, so no offset holds it, and the step sends x across the jump at
every evaluation (see "The gapless Zeeman state" below). We measured three
treatments on the Zeeman state against the gapped states of this section:
U = -2, filling 0.3, B = 0.3, `mf="swave"`, `maxerror=1e-10`, T = 1e-7,
with `"ed"` and the triangle, Broyden mixing and the plain loop. As a
reference we take the offset held at zero, which is the Fermi level of the
normal state: it has no staircase of x at all, so the gap between it and
the present step is what the oscillation of x costs. The root find is the other design
of "The Fermi level of a superconductor": at each Hamiltonian, `brentq` on
the paired charge inside the window, and the offset of the loop is the
root. The shell filling is a stand-in for option (b), the partly filled
level: when the charge at mu - eps and at mu + eps lie on the two sides of
N0, and differ by more than 100 kappa 2 eps, the density matrix is
w rho(mu - eps) + (1 - w) rho(mu + eps), with w chosen so that it holds N0
exactly. Here eps = 1e-5. This is the occupation a finite T gives to the
shell at mu, to order eps, but it costs two more density matrices per
iteration, while the same filling inside the density matrix would cost
none. Iterations, and the extra density matrices of the root find:

| nk | integration | solver | now | x = 0 | root find | its extra dm | shell filled |
|---:|---|---|---:|---:|---:|---:|---:|
| 16 | `"ed"` | Broyden | 41 | 36 | 36 | 527 | 38 |
| 16 | triangle | Broyden | 41 | 32 | 28 | 1193 | 38 |
| 32 | `"ed"` | Broyden | 36 | 28 | 27 | 438 | 34 |
| 32 | triangle | Broyden | 36 | 22 | 23 | 1000 | 34 |
| 16 | `"ed"` | plain | 350 | 363 | 322 | 3020 | 340 |
| 16 | triangle | plain | 355 | 368 | 320 | 6713 | 340 |
| 32 | `"ed"` | plain | 273 | 278 | 270 | 2453 | 272 |
| 32 | triangle | plain | 274 | 279 | 270 | 5628 | 271 |

The wall times, on three reserved cores of a loaded workstation, with one
run before the timed ones so that the numba compilation is not in them:
with Broyden at nk = 16, 0.89 s now, 0.91 s at x = 0 and 6.4 s with the
root find for `"ed"`, and 0.57, 0.51 and 7.5 s for the triangle; at
nk = 32, 2.1, 1.8 and 17 s for `"ed"`, and 1.0, 0.49 and 10 s for the
triangle. The plain loop at nk = 16 takes 7.1, 8.3 and 45 s for `"ed"`.

So the root find does not pay. It needs 9 to 15 density matrices per
iteration with `"ed"` and 20 to 40 with the triangle, whose paired charge
is a true step with half a level on it, so that `brentq` only stops at its
tolerance in x. It is 7 to 13 times slower with Broyden and 6 to 10 times
with the plain loop. It also changes the gapped states, since the offset is then the
exact root at each iteration: the s-wave state at nk = 16 needs 21
iterations instead of 23, the d-wave state 18 instead of 20, and the weak
U = -1 at nk = 16 11 instead of 13, at 11 extra density matrices per
iteration, 3.5 s against 0.70 s for the s-wave state with `"ed"`. The
reason why it cannot do better is that every run ends in the normal state:
the pairing dies, the window closes as Delta^3, and the root find, like
the step, leaves mu at the Fermi level of the normal state. The charge
error at the end, 1.6e-3 per cell at nk = 16 and -3.9e-4 at nk = 32, is the
staircase of the Fermi search of the normal state, and no offset touches
it.

What the oscillation of x costs is the gap between the first two
columns: 5 and 9 Broyden iterations at nk = 16 with `"ed"` and the
triangle, 8 and 14 at nk = 32, which on this cell is at most 0.5 s. The
root find recovers it, and in the plain loop and for the triangle at
nk = 16 it goes a few iterations below x = 0 too, but each of its
iterations costs 9 to 40 density matrices, so it loses in wall time
everywhere. In the plain loop the present step is already faster than
x = 0, by 5 to 13 iterations. So we leave the step as it is.

The shell filling answers a different question. It holds N0 exactly, to
1e-16, in the gapless state too, and `"ed"` and the triangle then end in
the same state, E = -1.626407234195 and mu = -1.6044293647 at nk = 16,
where today they differ by 8.1e-7 in the energy and 6.5e-4 in the Fermi
level (this energy is the band energy of the loop, of the Hamiltonian and
not of the interpolated density matrix, so it is a check of the state and
not the energy a built version would report). It never fires in the gapped states: the s-wave state at nk = 16,
the d-wave state and the weak U = -1 at nk = 16 and 24, with both
integrations and both solvers, keep their counts and their energies to
1e-12. The jump test is needed for that. Without it the interpolation
also fires on the smooth charge of a gapped state: at eps = 1e-5 it moves
the d-wave energy by 2.2e-11, and the weak U = -1 at nk = 16 no longer
converges in 1000 Broyden iterations; at eps = 1e-3 it moves the s-wave
energy by 6.6e-9. Its iterations are only a few below the
present ones, 3 and 2 with Broyden and up to 15 in the plain loop. With
the offset held at zero and the shells filled, Broyden needs 23 and 25
iterations at nk = 16 and 32, so most of the remaining cost of the
gapless state is the motion of x while the pairing still lives, which
the filled shell does not remove. That combination cannot hold the charge
of a gapped state, so it is a diagnosis and not a design. The
maintainer chose to build the shell filling inside the density matrix,
see "The shell filled in part" below.

#### The gapless Zeeman state

Why does the Zeeman state need 57 Broyden iterations at nk = 16 and 53 at
nk = 32, while the gapped states need 13 to 26, and why does it not
converge at all at nk = 64? The state is the s-wave one with U = -2 at
filling 0.3 on the spinful square lattice, with a Zeeman field B = 0.3
that kills the pairing, `mf="swave"`, `maxerror=1e-10` and T = 1e-7. We
logged x, the window and the charge error N - N0 at each evaluation of
one Broyden run, with the residual split into its mean-field part and its
offset part. As references we ran the same loop with the offset held at
x = 0, which is the Fermi level of the normal state as before the BdG
Fermi level, the same Hamiltonian without Nambu from a ferromagnetic
guess, and the plain loop. Broyden iterations with `"ed"`, before the
restart described below, and the iterations of the plain loop:

| nk | T | with the offset | x = 0 | no Nambu | plain loop |
|---:|---:|---:|---:|---:|---:|
| 16 | 1e-7 | 57 | 41 | 7 | 350 |
| 32 | 1e-7 | 53 | 30 | 7 | 273 |
| 64 | 1e-7 | > 300 | > 300 | > 300 | > 1000 |
| 16 | 1e-3 | 52 | 35 | 12 | |
| 32 | 1e-3 | 47 | 23 | 7 | |
| 64 | 1e-3 | 23 | 22 | 13 | |

All runs at a mesh end in the same state, the normal one with the charge
error of the staircase (1.6e-3 per cell at nk = 16, -3.9e-4 at nk = 32).
Three different things make them slow.

The offset has no fixed point while the pairing lives. The charge of the
paired state jumps by a shell of k-states when mu crosses a level, and N0
falls inside the jump. At nk = 16 the charge error goes between -0.014,
+0.017 and +0.033 per cell while x moves by 1e-2, and at nk = 32 between
-4.3e-3 and +3.5e-3. The Newton step, 0.05 and 0.016 long, takes x from
one side of the jump to the other at every evaluation. The window is
4 Delta wide, and Delta goes down slowly from the guess, so the rule on
the whole window fires only at evaluation 12 to 22, when Delta is 0.01 to
0.03. This is the difference between the first two columns, 16 iterations
at nk = 16 and 23 at nk = 32. A window that closes faster cannot fix it.
The gapped states need |x| up to 0.33 Delta (U = -1 at nk = 24, x =
-0.0095 with Delta = 0.028) and |x| W/Delta^2 up to 80, while x in the
oscillation of the Zeeman case has |x|/Delta from 0.05 to above 1 and
|x| W/Delta^2 from 3 to a few hundred, so the two ranges overlap. One
offset and one density matrix do not tell a gapped shell from a flipping
one.

With x = 0 the map is still almost discontinuous in the pairing. A level
of the normal state that sits on mu_n, filled in part by the smearing, is
pushed off it by the pairing: the BdG levels of the pair (k up, -k down)
are at -B +- sqrt(xi^2 + Delta^2), so the one at zero moves by
Delta^2/(2B). This shift crosses T at Delta of about 2.4e-4. In the run at
x = 0 and nk = 16 the charge error goes from +1.6e-3 at Delta below 4e-5
to -1.4e-2 above 3.5e-4, and the Hartree part of the mean field jumps
with it: the residual goes from 1.2e-3 to 3.4e-2 in one step. The
Broyden step that extrapolates the decay of the pairing lands in this
range, and the secant pairs on the two sides of the jump give a model
that is wrong on both. Without Nambu there is no pairing and no such
jump, and 7 iterations suffice.

At nk = 64 the normal state itself has no fixed point that a mixing
reaches at T = 1e-7. The run without Nambu ends in a cycle of period 10,
with the residual repeating 1.076e-3, 5.149e-3, 3.092e-3, ... and the
step-size scale moving, not stuck at its floor. The plain loop does not
converge in 1000 iterations either, with Nambu or without, and a memory
of 3 instead of 8 does not help. At T = 1e-3 the same states converge in
13, 23 and 22 iterations, to the same charge error. So the failure at
nk = 64 is the map of the normal state at T = 0, and the BdG Fermi level
plays no role in it. At nk = 16 and 32 a smearing of 1e-3 does not
change the picture: the charge error is still the one of the staircase,
and the counts stay high.

Why does the normal state fail at nk = 64 and not at 16 or 32? The
way we see it is by writing the normal state as a map of one number.
The field is along z and U is onsite, so the mean field is diagonal in
spin, and with the charge held by the Fermi search the only thing that
moves is the splitting s of the onsite up and down energies. The Zeeman
term gives s0 = 0.6 and the Hartree term adds -U m, with m = n_up -
n_down per site, so the map is s -> s' = 0.6 + 2 m(s), where m(s) is the
magnetization of the density matrix at the splitting s. We evaluated
m(s) with the Fermi search and the density matrix of the loop, on the
same mesh, and checked it against the logged loop. With `"ed"` at
T = 1e-7, m(s) is a staircase. Between s = 0.465 and 0.473 it is
-0.0625 (1101 up and 1357 down electrons in 4096 cells), and between
0.474 and 0.48 it is -0.06640625 (1093 up and 1365 down). The step in
between is one star of 8 k-points, related by the C4v symmetry of the
square lattice, that moves from up to down: m jumps by 16/4096 = 3.9e-3
and s' by 7.8e-3. The map sends the lower tread to s' = 0.475 and the
upper one to s' = 0.4672, so each tread maps into the other one, and no
fixed point sits on a tread. The fixed point is on the step: bisection
finds it at s = 0.47353025, where 8 up levels sit 1.46e-7 below mu and
8 down levels 1.46e-7 above it, 1.5 T, and the 16 states hold 8
electrons with fractional occupations, m = -0.063235. So the fixed point
exists, but only because the Fermi-Dirac weight at T = 1e-7 smooths the
step over about 1.3e-6 in s, and the slope of the map there is ds'/ds =
-5.9e3. A linear mixing is stable only for a mix below 2/5.9e3, about
3e-4. In Broyden mixing the pairs that straddle the step are dropped by
the restart, since the residual grows by a factor 24 across it, and the
pairs on one tread fit a flat map. To reach
`maxerror=1e-10` the loop would need s to 2e-14.

The logged loops show exactly this. Broyden mixing without Nambu
cycles between the two treads, with s between 0.4706 and 0.4744 and m
alternating -0.0625 and -0.0664, the same 8 k-points flipping every
time. The pair of levels nearest mu is at 2e-5 to 1.5e-3 from it,
far above T. The residual of the cycle, 1e-3 to 5e-3, is the size of
the step of the map, 7 orders of magnitude above `maxerror`, so a looser
tolerance does not help either: 1e-3 would not pass. The plain loop
ends in a cycle of period 4 between the same two treads. At nk = 16 and
32 the fixed point is on a tread. There a degenerate star of one spin
sits exactly on mu and is filled in part (the down star at nk = 16, and
the up star at nk = 32, with 1 electron in its 8 states), the nearest star of
the other spin is 5.7e-3 and 2.5e-2 away, and moving s moves the whole
star with mu, so m does not change and the map is flat. Both converge in
7 Broyden iterations. Which case one gets is a coincidence of the mesh,
not a defect of the Fermi search.

The slope at the fixed point goes as 1/T: ds'/ds = -5.9e3, -600, -61
and -6.7 at T = 1e-7, 1e-6, 1e-5 and 1e-4, with the two stars of the
step at 1.3 T to 1.5 T from mu, and -0.88 at T = 1e-3, where the two
stars sit at 0.6 T from mu and the step is smeared out. Broyden mixing of the normal state
with `"ed"` at nk = 64 does not converge in 300 iterations at T = 1e-6
and 1e-5, and converges in 29 at T = 1e-4 and in 13 at 1e-3. So the
smearing has to bring the slope down to about 10, which here means
T = 1e-4 or more, still a hundred times below the distance to the next
stars on this mesh, 1.3e-2 and 2.6e-2. With `integration="triangle"` the charge is a
continuous function of mu and the map is smooth: m(s) changes linearly,
from -0.06475 at s = 0.465 to -0.06684 at s = 0.48, and the slope of the
map at its fixed point, s = 0.469308, is -0.29. The normal state with the
triangle at T = 1e-7 converges in 7 Broyden iterations at nk = 16, 32 and
64, and in 62 plain iterations at nk = 64, to E = -1.62721044 at nk = 64
against -1.62721013 at nk = 32. With Nambu the triangle does not help:
from `mf="swave"` at nk = 64 and T = 1e-7, `"ed"` and the triangle both
stop at 1000 Broyden iterations without converging, since the triangle of
a BdG Hamiltonian is the step of exact diagonalization (see "The
triangle of a BdG Hamiltonian is the step").

So there is nothing to fix in the Fermi search, the stopping rule or
Broyden mixing: the map that `"ed"` defines at T = 1e-7 on this mesh has
its fixed point on a step 1e-6 wide, and no mixing reaches it. A
condition that recognizes such a step and switches the integration
would be a gate on one mesh, and none was added. What a user should do
with a metal at T to 0: use `integration="triangle"` for a normal state,
or take T large enough that the step of one star is smooth, 1e-4 or
more at nk = 64 here. For a superconductor whose pairing dies, the
triangle does not help, and a T of this size does: with Nambu at nk = 64
and T = 1e-4, `"ed"` converges in 41 Broyden iterations from
`mf="swave"`, to the energy of the normal state at the same T to 2e-12.

Is it the history of Broyden? A shorter memory makes it worse: m = 3
gives 80 and 60 at x = 0, and 308 and 327 with the offset; m = 16 gives
63 and 52 with the offset. Scaling the offset entry of the vector by 0.1
or 0.01, or leaving it out of the threshold of the warm-up, moves the
counts by at most 7. What helps is to
drop the secant pairs that straddle a jump. `broyden_mixing_solve` now
takes `restart=4.`: when the residual of one multisecant step is more than
4 times the one of the step before, it forgets its memory and seeds it
again with one Pratt step from the current point, as at the end of the
warm-up. None of the gapped states below restarts. Broyden iterations at
`maxerror=1e-10`, before and after the restart:

| state | integration | nk | before | now |
|---|---|---:|---:|---:|
| s-wave, U = -2, filling 0.3 | `"ed"`, triangle, qtci | 16 | 23, 23, 23 | 23, 23, 23 |
| s-wave, U = -2, filling 0.3 | `"ed"` | 32, 64 | 23, 24 | 23, 24 |
| d-wave, V1 = -2, U = 1 | `"ed"`, triangle, qtci | 16 | 20, 20, 20 | 20, 20, 20 |
| s-wave, U = -1, filling 0.3 | `"ed"` | 16, 24 | 13, 16 | 13, 16 |
| s-wave, U = -6, filling 0.1 | `"ed"` | 16 | 26 | 26 |
| s-wave, U = -2, filling 0.2 | `"ed"` | 16 | 13 | 13 |
| Zeeman | `"ed"`, triangle | 16 | 57, 57 | 41, 41 |
| Zeeman | `"ed"` | 32 | 53 | 36 |
| Zeeman | qtci | 16 | 69 to 346 | 50 to 85 |
| Zeeman | `"ed"` | 64 | > 1000 | > 1000 |

The gapped rows have the same energy to the last digit, so the restart
never happened there. The same holds for the smooth problems of
`scftk/broydenmixing.py`: the three seeds of the Lieb flake (57, 67 and
132), the honeycomb antiferromagnet with U = 3 (30, and 132 with
`mix=0.1`), the honeycomb CDW with V1 = 2 (31) and the 3x3 honeycomb
antiferromagnet (51). The Zeeman states with `"ed"` and the triangle end
in the same state as before, with the energy to 2e-11 and the charge
error to 2e-12. With `"ed"` and the triangle a factor 2 gives the same
counts as 4, and a factor 10 gives 42, 41 and 43. With the offset held at
zero the restart gives 36 and 28, so the cost of the offset in a gapless
state is now 5 and 8 iterations instead of 16 and 23. The plain loop
does not use Broyden mixing and is unchanged: 350 iterations at nk = 16,
with the same energy to the last digit.

The qtci count of the Zeeman state is not reproducible from one run to
the next, with the same guess, the same seed and three BLAS threads: four
runs of the code before gave 69, 74, 101 and 346 iterations, and three
runs now 50, 65 and 85. The 300 and 321 of the table above, with the
adaptive and with the solver step, are in the same range, so they do not
say that one step is faster than the other. The reason is that the
residual sits at 1e-9 to 1e-8, the
interpolation noise of qtci at a Fermi surface, while `maxerror=1e-10`
asks for less, so the loop stops when the noise happens to dip below the
tolerance. One run before even stopped at an energy 3.6e-6 below the
others. The gapped qtci states converge in 23 and 20 iterations every
time. A qtci run of a metal needs a `maxerror` above its noise floor.

What the restart does not touch is the case at nk = 64, which is the
normal state at T = 0 with its fixed point on a step (see above), and the pairing that dies slowly: the residual of
the pairing goes down by a factor 0.88 to 0.93 per step of the warm-up,
so about 14 of the 36 to 41 iterations are spent before the multisecant
phase starts. That is a pairing near its critical field, which decays
slowly in any linear mixing.

`tests/scf/test_bdg_fermi_level_solvers.py` checks that the restart gives
the Zeeman state at nk = 16 in fewer iterations than without it, with the
same energy and charge error to 1e-8, and that the d-wave state has the
same count and the same energy to 1e-12 with and without it.

#### The shell filled in part

How do we hold the charge of a gapless state, where N0 falls inside a
jump of the staircase? The maintainer picked to fill the shell at zero
energy in part, inside the density matrix. The shell is the set of BdG
levels with |E| < tol, with tol = max(40 T, 1e-6/(kappa Nk)), meaning
every level whose Fermi-Dirac weight is not a step (to 4e-18), with a
floor far below the level spacing 1/(kappa Nk) when T = 0. The jump is
read on the levels within 5 T of zero, the ones the Fermi-Dirac weight
fills in part, by more than 1/150. This is where the Fermi search of the
normal state puts a degenerate star that it fills in part: within
T ln 15 < 3 T when one electron or more of 16 states is in it.

The occupations of the shell are the ones of a chemical potential
shifted inside it by mu', to first order. At each k the levels of the
shell are grouped in multiplets degenerate to the rounding of the
energies, 1e-12, each multiplet takes the mean of its energies (and zero
when it is at zero within that rounding), and the shell Hamiltonian
sum_m E_m P_m - mu' Q is diagonalized, with P_m the projector of the
multiplet and Q = V^dag tau_z V the charge in the shell (tau_z = P_e -
P_h, V the eigenvectors of the shell). Its eigenvectors get the
Fermi-Dirac occupations of their eigenvalues at T (at T = 0 at a smearing
of tol/40). mu' is set so that the trace of the electron block is exactly
N0, a root find on these small matrices that costs no diagonalization of
the mesh. In a multiplet at zero energy, as the star of the Zeeman state,
this is the design that was signed off: the vectors of charge 1 get one
occupation f and their partners, of charge -1, get 1 - f, uniformly over
the degenerate subspace. Note that there the electron level of the star
and the hole level of a partner are degenerate at the same k, and the
diagonalization mixes them at will, so a uniform f on the whole multiplet
would break the particle-hole symmetry; the charge Q is what separates
them. The partner of a vector at k sits at -k with the opposite energy
and charge, so it gets 1 - f. Since the shell Hamiltonian is built from
projectors, the filled density matrix does not depend on the eigenbasis
the diagonalization picks inside a multiplet. Without the grouping it
did at T = 0: there the smearing of the shell is about 1e-12, the 1e-16
rounding of the degenerate energies entered the Fermi-Dirac weight as an
argument of order 1e-4, and a random unitary inside the star moved the
filled density matrix by 3.6e-7 with `"ed"` and 5.5e-7 with the
triangle. Now it moves it by less than 1e-12, also at T = 0.

Why not a single f for the whole shell, as in the design? Because a shell
whose levels are split by more than T then loses the order of their
energies. The chain with V1 = 1 at half filling from a ferromagnetic
guess, nk = 20, of `tests/scf/test_bdg_total_energy.py`, has a degenerate
shell at mu that has to split for its normal state to form. With one f
the Nambu state ended at E = -0.4620676 against -0.4670676 for the
normal state. With the shifted chemical potential the shell keeps the
order of its levels.

The shell is filled only when three conditions hold, and otherwise the
density matrix is bit for bit the one of before.

- The shell is armed: the pairing amplitude is below the smearing of the
  shell, T (Ts = tol/40 at T = 0), or the window of the offset x (see
  "The Fermi level of a superconductor") is narrower than the shell at
  T = 0, 1e-6/(kappa Nk). This is the case when the pairing has died out
  or was never there. The gate went through three versions. The window
  against the shell, 40 T, armed the s-wave state with U = -1 at
  T = 1e-2 (Delta = 0.025) in every iteration, since the window closes as
  Delta^3 and is 0.045 there. The window against T armed the state with
  U = -0.5 at nk = 32 and T = 1e-3 (Delta = 3.1e-3), whose energy then
  moved by 6e-7. The pairing alone, against T or 1e-8 times the
  bandwidth at T = 0, let the Zeeman state go round the gate: the paired
  charge jumps while the pairing is still about 1e-4, and Broyden mixing
  without its restart did not converge in 300 iterations, the pairing
  going between 7e-8 and 7e-6 and the shell armed in one iteration out
  of three. With the window at the T = 0 shell it converges in 51
  iterations, against 38 with the restart. So the answer to the question
  of whether the gate should compare with Delta itself is yes at T > 0,
  and the window enters only at T = 0, where it arms a pairing below
  about 8e-5 at nk = 16, far below any live one of the table below.
  While the pairing lives, the offset is
  what holds the charge. The ferromagnetic triplet chain of
  `tests/scf/test_unconventional_superconductivity.py` (nk = 6,
  Delta = 1.73, plain loop, a random guess) shows why. There the level
  at the node k = 1/2 converges to mu from below, by a factor 0.6 per
  iteration. Once in the shell it was filled to hold N0, the mean field
  pushed it 0.18 away from mu, and the loop went round a cycle of 22
  iterations, which did not converge in 1000 for four seeds out of four.
  Before, and with the condition, it converges in 73 to 80 iterations
  with a charge error of 0.1 per cell, since the plain loop gives the
  offset up there. What separates it from the Zeeman state is that the
  star of the Zeeman state stays at mu while it is filled, because the
  Fermi search of the normal state follows it, while nothing holds the
  level of the triplet chain at mu.
- The density matrix misses N0 by more than 1e-12 per cell. A shell whose
  charge is held already is left alone. This matters for the triangle:
  the chain with V1 = 1 above, with Nambu and the triangle, holds its
  charge with its own weights and ends at E = -0.4756336249, the energy
  of the normal chain with the triangle. When the fill replaced those
  weights at two evaluations, the loop went to the state of `"ed"`,
  -0.4670675939, 8.6e-3 higher.
- N0 lies strictly between the charges of the shell emptied and filled in
  charge, and the jump, the charge of the levels within 5 T, is more than
  10 times kappa 2 (5 T), the charge a smooth density of states puts in
  that window. A gapped state has no level there, and a smearing that
  resolves the level spacing makes the charge continuous, so neither is
  touched.

When the triangle density matrix passes the three conditions, it is
rebuilt on the occupations of exact diagonalization, the Fermi-Dirac
weight at T on the same mesh, and not on its own weights, and the shell
is filled on that one if it misses N0 in turn. The rebuilt matrix is
kept even when it holds N0 already and the fill does not fire on it. A
first version kept the triangle weights in that case, so the triangle
kept its own charge miss exactly where `"ed"` holds the charge: at
filling 0.30078125, a whole number of states of the mesh at nk = 16, it
ended with N - N0 = -4.3e-10 and an energy 2.1e-7 below `"ed"`, and the
energy jumped by 2.1e-7 from there to filling 0.3007812. The reason for
rebuilding at all is that at T > 0 the
triangle weights of a BdG Hamiltonian are not the step when a level sits
on zero: the simplices with an edge on that level have a density of
states at mu, and their Bloechl term moves 1/24 of a level to their far
vertices, at E = 0.29 and 0.77 in the Zeeman state at nk = 16. Without
the Bloechl term the weights are the step to 2e-7. This is what put the
triangle in another state than `"ed"` at nk = 16, 8.1e-7 away in the
energy and 6.5e-4 in the Fermi level, and not the place of the Fermi
level on the staircase as written in "The step under the quasi-Newton
solvers". A triangle density matrix whose shell is not filled keeps its
own weights. A first version took the weights at T = 0 on the whole mesh
whenever a level was in the shell, filled or not, and that moved gapped
states with the pairing alive: the s-wave state with U = -0.5 at nk = 32
and T = 1e-3 went from Delta = 3.12124e-3 and E = -1.4821049867843 to
3.12187e-3 and -1.4821049866146, and stopped depending on T.

Which band energy do we report? Once the shell is armed, the band energy
of the mean field is Tr H rho of the density matrix the loop holds, at
every T, filled or not. With `"ed"` this is (1/2)(sum_k sum_n f_n E_n /
Nk + Tr h_e), from the band energies the density matrix is built from,
plus the change the fill makes in the shell, (1/2) sum (n <w|H|w> - f E).
With the triangle it is the same sum on the occupations of its density
matrix. A shell that is not armed keeps the step band energy of before.
A first version reported the step band energy when the shell was not
filled and the band energy of the filled density matrix when it was. At
T > 0 the two differ by the Fermi-Dirac tail, so the energy jumped where
the fill starts: at T = 1e-3 with `"ed"`, from filling 0.30078125, held
without the fill, to 0.3007812, 1e-7 electrons per cell below, the energy
moved by 3.3e-7 more than mu dN. Now dE - mu dN is below 5e-11 for both
fillings, both integrations and T = 1e-7, 1e-4 and 1e-3 (the table
below), meaning that the energy is continuous across the fill and its
slope is the chemical potential, as it must be.

The price is that an armed state at T > 0 whose fill never fires now
reports Tr H rho at T and not the step band energy, which is a count at
T = 0. Every such row of the measurements has a dead pairing, or one
below T. With `"ed"` the energy moves up by the thermal energy, roughly
as T^2; with the triangle it moves down, which we did not look into:

| state | integration | Delta at the end | dE |
|---|---|---:|---:|
| U = -0.5, nk = 32, T = 2e-3 | `"ed"` | 2e-12 | 8.6e-6 |
| U = -1, nk = 64, T = 1e-2 | `"ed"` | 4e-12 | 5.0e-5 |
| U = -1, nk = 96, T = 1e-2 | `"ed"` | 2e-12 | 4.1e-5 |
| U = -1, nk = 96, T = 1e-2 | triangle | 2.6e-3 | -1.8e-5 |
| U = -1, nk = 96, T = 2e-2 | `"ed"` | 1e-13 | 1.9e-4 |

The Fermi level and the iterations of these rows are the ones of
before; only the reported energy changes. Note that the
triangle of the first row ends in a state with the pairing alive,
Delta = 3.1e-3, which is not armed and does not change, so there `"ed"`
and the triangle converge to different states, before as now. A Nambu
state with a dead pairing at T > 0 also differs now from the normal
state at the same T, which reports the step band energy. With a gate
that switched the convention when the fill fires instead, these rows
would not move, but the energy would jump at the start of the fill as in
the first version. The levels of a shell of 5 T, as in an earlier
version, left out the ones between 5 T and 40 T: at T = 1e-3 the band
energy of the filled density matrix differed from the reported one by
1.95e-8. The fill also moves the chemical potential by
mu', which the mean field adds to the Fermi level of the Hamiltonian it
returns, shifted by it, so that this Hamiltonian holds N0 by itself at
its T (to 1e-8 in the test). `"ed"` and the triangle find the Fermi level
of the normal state at different places of the staircase, the triangle
by a count at T = 0, and mu' makes up the difference, so the two report
the same Fermi level.

This is `scftk.superscf.ShellFilling`. `BdGFermiLevel` owns one, and
`generic_densitydensity` and `VJinteraction` pass it to `get_dm` as
`shell=`; the SCF object of `Vinteraction` keeps it as `scf.shell`.
`shell.start` says whether the shell is armed. While it is not, the
density matrix is the one without shell, on the device route too. Once
it is armed, `"ed"` on the CPU hands the eigenpairs of each batch to the
shell, which keeps only the k-points with a level in it. On the device
the density matrix stays on the device route, the band energies come
from the eigenvalues on the device, and only the k-points with a level
in the shell are diagonalized again on the host for their eigenvectors.
A first version moved the whole density matrix to the CPU once the shell
was armed: the 3x3 supercell of the Zeeman state with B = 0.3 at nk = 8
went from 281 calls of the device density matrix to 740 diagonalizations
of the Bloch Hamiltonians on the device, 17.5 s against 10.3 s. Now it
makes the 281 calls and none of the others, in 10.6 to 11.0 s against
8.0 s for the code before in the same session, with the same energy; the
difference is the eigenvalues on the device and the host eigenpairs at
the k-points of the shell in the armed iterations. The shell handles its
k-points in groups of the same number of levels, with one batched
diagonalization of the shell Hamiltonian per group and per value of mu'
in the root find, which is what makes a shell of 40 T affordable. A
density matrix called without shell is the one of before. qtci does not
use it.

The fill needs a degenerate level sitting on mu, and it does nothing
when the Fermi level of the normal state falls between two levels. The
reviewer's example: Rashba coupling 0.5 with an in-plane field of 0.6,
U = -3, nk = 16, T = 1e-7, `"ed"`. The pairing dies, but mu sits between
two levels at +-6.24e-4, the shell stays empty, and the charge error of
1.5625e-3 per cell stays, before as now (109 iterations,
E = -1.9128268877). There the step of the charge is a jump of the Fermi
search of the normal state, which rounds the target to a whole number of
states on purpose, and not of the paired state.

We measured it on three reserved cores of a loaded workstation, against
the code before the fill, with `maxerror=1e-10` and T = 1e-7 unless said.
Every case whose shell is not filled has the same iterations, energy,
charge and Fermi level to the last digit (the difference of the energies
is exactly zero), with both integrations: the s-wave states with U = -2 at
nk = 16, 32 and 64, U = -1 at nk = 16 and 24, U = -0.7 at nk = 24 at
T = 1e-7 and 1e-3, U = -6 at filling 0.1, U = -0.45 to -0.6 at nk = 16 to
32 and T = 5e-4 to 2e-3, the d-wave state with a Zeeman field 0, 0.15 and
0.3, the Rashba state above, the triplet chain over four seeds, the
spinless p-wave chain, and the chain with V1 = 1 with both integrations,
with Broyden mixing and the plain loop. For the version in the code we
measured again the 70 cases of the second review, in the same way. The
ones that are not armed are the same to the last digit, including the
s-wave states with U = -0.5 at nk = 32 and T = 1e-3 and 2e-3 with the
triangle, and U = -0.5 at T = 1e-3 with `"ed"`. The chain with V1 = 1 with
the triangle and the Rashba state with `"ed"` are armed and never filled,
and their energy moves by 4.4e-15 and 4.4e-16, the rounding of the other
sum; the dead-pairing rows at T > 0 move as in the table of the energy
above. The Zeeman state, iterations before and after, the charge error
N - N0 per cell before (after it is 1e-16 in every row), and the energy
and Fermi level after, the same for `"ed"` and the triangle:

| nk | T | integration | Broyden | plain | N - N0 before | E after | mu after |
|---:|---:|---|---:|---:|---:|---|---|
| 16 | 1e-7 | `"ed"` | 41, 38 | 350, 355 | 1.6e-3 | -1.6264072342 | -1.604429385 |
| 16 | 1e-7 | triangle | 41, 38 | 355, 361 | 1.6e-3 | -1.6264072342 | -1.604429385 |
| 32 | 1e-7 | `"ed"` | 36, 34 | | -3.9e-4 | -1.6273027010 | -1.638774186 |
| 32 | 1e-7 | triangle | 36, 34 | | -3.9e-4 | -1.6273027010 | -1.638774186 |
| 16 | 1e-4 | `"ed"` | 48, 46 | | 1.6e-3 | -1.6264072342 | -1.604449432 |
| 16 | 1e-4 | triangle | 113, 134 | | 1.6e-3 | -1.6264072342 | -1.604449432 |
| 16 | 1e-3 | `"ed"` | 61, 60 | | 1.6e-3 | -1.6264072182 | -1.604633875 |
| 16 | 1e-3 | triangle | 41, 41 | | 1.6e-3 | -1.6264072182 | -1.604633875 |
| 16 | 0 | `"ed"` | 781, 38 | | 2.0e-3 | -1.6264072342 | -1.604429365 |
| 16 | 0 | triangle | 37, 38 | | 1.6e-3 | -1.6264072342 | -1.604429365 |

Before, `"ed"` and the triangle differed by 8.1e-7 in the energy and
6.5e-4 in the Fermi level at every T. The energy of `"ed"` at T = 1e-7
moved by 3e-12 only, although the charge moved by 1.6e-3 per cell, since
the charge goes into a star at zero energy. At T = 1e-3 it moved by
3.7e-8, since now the energy is the one of the filled density matrix,
the Fermi-Dirac tail included. Most of the cost of the gapless state is
still the motion of x while the pairing lives (see "The staircase of the
offset"); with `"ed"` at T = 0 the loop no longer runs into the staircase
of the smearing of 1e-15, and goes from 781 iterations to 38. The
triangle at T = 1e-4 is the one row that got slower, 134 iterations
against 113 before and 43 in the first version, with 31 armed
evaluations of which 27 filled; we did not find why.

The filling sweep of the second review puts N0 on a whole number of
states of the mesh, 0.30078125 at nk = 16, where the normal Fermi search
holds the charge without the fill, and 1e-7 electrons per cell below it,
0.3007812, where it does not. dE - mu dN between the two, with mu the
mean of the two Fermi levels, and the charge error at 0.30078125, before
the fill, in the first version and now:

| T | integration | before | first version | now | N - N0 before | N - N0 first | N - N0 now |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1e-7 | `"ed"` | 4.0e-13 | 1.8e-13 | -2.8e-15 | 7.4e-13 | 3.2e-12 | 0 |
| 1e-7 | triangle | -7.1e-14 | 2.1e-7 | -4.3e-15 | -4.3e-10 | -4.3e-10 | 0 |
| 1e-4 | `"ed"` | -3.9e-14 | 4.7e-12 | -3.1e-14 | 4.2e-15 | 3.8e-15 | 6.0e-15 |
| 1e-4 | triangle | -1.6e-16 | 2.1e-7 | -3.1e-15 | -4.3e-7 | -4.3e-7 | 4.9e-15 |
| 1e-3 | `"ed"` | 3.0e-13 | 3.3e-7 | -4.6e-11 | -9.7e-15 | -8.9e-15 | -8.4e-15 |
| 1e-3 | triangle | 3.4e-15 | -4.8e-11 | -4.6e-11 | -4.3e-6 | 0 | 0 |

Before the fill the slope is right because nothing changes between the
two fillings except mu dN, while the charge at 0.3007812 is missed by
1e-7 per cell. The first version holds the charge but jumps by 2.1e-7
with the triangle (its own miss kept at 0.30078125) and by 3.3e-7 with
`"ed"` at T = 1e-3 (the two conventions of the energy). Now both
integrations give the same energy at both fillings, -1.6289141551 and
-1.6289139946 at T = 1e-7 and 1e-4, and -1.6289139088 and -1.6289137484 at
T = 1e-3, and the slope is mu.
`VJinteraction` with plain mixing, the engine of
`h.get_mean_field_hamiltonian()` for this Hamiltonian, holds the charge
too. At nk = 64 the Zeeman state still does not converge in 1000 Broyden
iterations, as before, since the normal state there cycles between two
treads whose stars are not at mu (see "The gapless Zeeman state").

A Nambu state without pairing at a fractional charge on the mesh now
holds that charge, which the normal state cannot. For U = 2 from
`mf="ferro"` at filling 0.35, nk = 64 and T = 0, the requested charge is
2867.2 electrons on the mesh. The normal state holds 2869.536: its Fermi
search rounds the target to a whole number of states, on purpose, and
the smearing of 1e-15 reads the rounding of the degenerate energies of a
star of one spin that sits on mu. The Nambu state fills that star in
part and holds 2867.2. Since the star is of one spin, the occupation of
the other spin is the one of the normal state, and the magnetization
moves by exactly the charge difference: mz = 0.0022461 against 0.0028164
for the normal state, which the Nambu `"ed"` also had before. The energy
is the same to 1e-10, -1.2809620317, since the star is at zero energy.
`test_nambu_normal_state_has_the_magnetization_of_the_normal_state` in
`tests/scf/test_densitydensity_bdg_energy.py` checks this relation: one
spin with the occupation of the normal state to 1e-8, the other moved by
the charge difference to 1e-8, the charge held to 1e-10 and the energy to
1e-8.

`tests/scf/test_bdg_shell_filling.py` checks, at T = 1e-7 and 1e-3, that
the Zeeman state holds N0 to 1e-10 with `"ed"` and with the triangle,
with the same energy to 1e-10 and the same Fermi level to 1e-8, and that
the returned Hamiltonian holds N0 by itself; that the band energy is the
one of the density matrix, summed over the hoppings, to 1e-12, filled
and not filled; that the energy moves by mu dN to 1e-10 between filling
154/512, a whole number of states at nk = 16, and 1e-7 electrons per
cell below it, with both integrations at T = 1e-7 and 1e-3; that a state
whose shell is not filled (the s-wave state with `"ed"` and
the triangle, and the U = -0.5 state at T = 1e-3 with the triangle) has
the same iterations, energy and density matrix, bit for bit, as with the
shell switched off; that the chain with V1 = 1 with Nambu ends in the
state of the normal chain, with `"ed"` and with the triangle; and that
the filled density matrix is Hermitian, particle-hole symmetric (the
hole occupations are 1 minus the electron ones of the other spin) and
the same to 1e-12, at T = 1e-7 and at T = 0, when the diagonalization
mixes the degenerate levels at zero energy with a random unitary.

### qtci from a random guess

Why did qtci with V1 on a BdG Hamiltonian stop with "Non-Hermitian BdG
mean field" from a random guess, and not from the s-wave or p-wave one?
The case is `Vinteraction(V1=-1.5, filling=0.3, nk=8,
integration="qtci")` on the spinful Nambu square lattice, plain loop,
`mf=None` after `np.random.seed(0)`, 1 or 2. The qtci density matrix is
Hermitian by construction, since of (d,i,j) and (-d,j,i) only one entry is
interpolated and the other is its conjugate. The exception is the onsite
diagonal entry, which is its own partner. Its field |u_i(k)|^2 is real, but
it is interpolated in the stacked complex TCI together with the other
fields, so it came out with an imaginary part of the size of the
interpolation error, and the loop wrote x and then conj(x) on the same
place. At tolerance=1e-6 this imaginary part was 5e-8, 2e-8 and 5e-9 for
the seeds 0, 1 and 2 at the iteration that failed, and the rest of the
electron block was Hermitian to zero exactly. The Hartree term sums it
over the neighbors and the spins, so the onsite mean field got an
imaginary diagonal about 12 times larger, 6.4e-8 per entry for seed 2, on
the electron and on the hole block. The check of `get_mf_bdg` is a norm of
1e-7 on the whole mean field, and this crossed it, 1.3e-7, after 228 to
311 iterations. The s-wave and p-wave states have fields of a low rank,
16 and 17, whose imaginary diagonal is at rounding, so the loop from
those guesses never saw it; the state from a random guess has a rank of
51. A normal-state qtci mean field has no such check, so there the
imaginary part went into the Hamiltonian silently.

Now `dm_entries_qtci` keeps only the real part of the onsite diagonal
entries, which is the Hermitian part of a matrix whose other entries are
Hermitian already. The mean field is then Hermitian to zero exactly at
every iteration, and the loop converges from the three seeds: in 503, 537
and 481 iterations to E = -2.33065877, -2.33065918 and -2.33065857,
against -2.33065878, -2.33065852 and -2.33065854 for `"ed"` from the same
guesses in 446, 506 and 472 iterations. So qtci is on the state of `"ed"`
to 1e-8, 7e-7 and 3e-8, below the interpolation tolerance. The spread of
`"ed"` over the seeds, 2.6e-7, is where the loop stops at maxerror. Note
that `np.random.seed` does not fix a qtci run: the pivots of the cross
interpolation are drawn with the `random` module of Python, so the same
seed gives a slightly different qtci energy from one run to the next, and
the test seeds both. Broyden mixing
from seed 0 fails in the same way before the change (226 evaluations) and
converges after it (230 evaluations, E = -2.33065804), with no gain in
time over the plain loop. `tests/scf/test_qtci_bdg_random_guess.py`
checks that the onsite diagonal of the qtci density matrix is exactly real
on the paired state `"ed"` reaches from seed 0, with a BdG mean field
Hermitian to 1e-12. The whole loop from a random guess, about 500
iterations, is too long for the suite. The check fails before the
change: on that frozen state the
imaginary part was 7e-10 to 1e-9 and the non-Hermitian part of the mean field
1.7e-8, from stacked fields of rank 51. A frozen state with real or
low-rank fields does not show it: the s-wave and p-wave states (rank 16
and 17) and a random BdG Hamiltonian on the 8x8 mesh give an imaginary
part at rounding, 1e-16.

## What wins and what loses

The wins. The filling is exact and the Fermi level continuous. A metal at
T <= 1e-3 lands in the right state on a coarse mesh, where exact
diagonalization does not. The magnetization needs about four times fewer
k-points. With the Bloechl correction, the total energy and the bond
densities need about 64 times fewer k-points. The convergence is clean
second order, so two meshes extrapolate well. And the result does not
depend on how low T is, down to T=0.

The losses. At T = 1e-2 and nk >= 512, exact diagonalization converges
exponentially and wins. The magnetization of a collinear ferromagnet keeps
an error of about 5/nk^2 per site, which the Bloechl correction cannot
touch. An insulator gains nothing. The cost per iteration is the one of
exact diagonalization plus the Fermi search, which at T > 0 and nk = 256
takes 0.5 to 0.8 s on another machine; in the tables the triangle was not
slower per iteration than exact diagonalization, but the machine was
loaded.

## What is left open

- The band charges. To go past second order for the magnetization, a
  correction has to move charge between bands: the mean of E_L - E over
  each simplex, estimated from second differences on the mesh, times the
  density of states of the band at mu, with mu refitted afterwards. The
  optimized tetrahedron method of Kawamura, Gohda and Tsuneyuki, PRB 89,
  094515 (2014), does this kind of thing in 3D. Not tried.
- RPA chi(q) and the anomalous Hall conductivity of metals with the same
  treatment of the occupations. chi(q) needs f(E_n(k)) - f(E_m(k+q)) over
  the simplices (the tetrahedron Lindhard function), and the Hall
  conductivity of a metal needs its Fermi-sea term with the step on the
  simplices. Neither is built.
- Non-Hermitian Hamiltonians are refused. Bogoliubov-de Gennes ones are
  accepted, and their Fermi level is found on the charge of the paired
  state (see "The Fermi level of a superconductor").
- For a BdG Hamiltonian the triangle is the step of exact
  diagonalization on the same mesh (see "The triangle of a BdG Hamiltonian is the step"), so in a state
  without a gap it keeps the staircase of `"ed"`. A triangle integration
  of a gapless BdG state with the continuous charge of the normal state
  would need the electron and hole bands unsorted. Not tried.
- The partly filled shell of a gapless BdG state is built (see "The
  shell filled in part"): once the pairing is below T, or the window of
  the offset below the shell at T = 0, and when the density matrix misses
  N0 inside a jump of its charge, "ed" and the triangle fill the levels
  within 40 T of zero at a chemical potential shifted inside the shell,
  so that the charge is exactly N0. The Zeeman state holds N0 to 1e-16
  from T = 0 to 1e-3, and "ed" and the triangle give the same energy and
  Fermi level, where they differed by 8.1e-7 and 6.5e-4. Once armed, the
  reported band energy is Tr H rho, so it is continuous across the fill
  with slope mu. Every state that is not armed is the same to the last
  digit. A state that is armed and never filled at T > 0, all of them
  with a dead pairing or one below T, now reports Tr H rho at T instead
  of the step band energy, which moves it by the thermal energy (8.6e-6
  for U = -0.5 at nk = 32 and T = 2e-3, 1.9e-4 for U = -1 at nk = 96 and
  T = 2e-2), and so a dead Nambu state at T > 0 no longer has the energy
  of the normal state at the same T. Whether that is the convention we
  want for those states is open. What it leaves open too: a gapless state whose pairing
  lives (a Bogoliubov Fermi surface, a node on the mesh) is not touched,
  since there a level is not held at mu while it is filled, and filling
  it made the triplet chain go round a cycle; a state whose Fermi level
  falls between two levels, as Rashba 0.5 with an in-plane field 0.6 and
  U = -3 at nk = 16 (levels at +-6.24e-4, a charge error of 1.5625e-3),
  keeps its charge error, since there is no degenerate level at mu to
  fill; and qtci does not fill the shell.
- qtci with V1 on a BdG Hamiltonian stopped with "Non-Hermitian BdG mean
  field" from a random guess. This is closed (see "qtci from a random
  guess"): the onsite diagonal of the qtci density matrix had an
  imaginary part of the size of the interpolation error, and now only its
  real part is kept. A run of the maintainer with 54 plain iterations and
  E = -2.3214034 against -2.3214029 for `"ed"` was close to the s-wave
  state, but its guess was not recorded, and its numbers were not
  reproduced here.
- A state with a Bogoliubov Fermi surface, or one whose pairing dies in the
  loop, still converges more slowly with the BdG Fermi level than with
  the Fermi level of the normal state, since no offset holds the charge
  on the staircase of the mesh (see "The gapless Zeeman state"). With the
  restart of Broyden mixing the Zeeman case needs 41 iterations at
  nk = 16 and 36 at nk = 32. With the offset held at zero it needs 36 and
  28. Before the restart it needed 57 and 53, against 41 and 30 at zero
  offset. With the shell filled in part it needs 38 and 34, and it holds
  the charge; without the restart it needs 51 at nk = 16. The root find per iteration, the other design, was measured
  and does not pay (see "The staircase of the offset").
  At nk = 64 the Zeeman case does not converge with either Fermi level,
  because the normal state at T = 1e-7 does not converge on that mesh,
  without Nambu too; at T = 1e-3 it converges in 23 iterations. This is
  understood and has no fix in the code (see "The gapless Zeeman
  state"): the fixed point of the normal state with `"ed"` sits on the
  step of one star of 8 k-points that crosses mu, smoothed by T over
  1.3e-6 in the splitting, where the map has a slope of -5.9e3. The
  normal state with the triangle converges in 7 iterations at nk = 64,
  and with `"ed"` from T = 1e-4 on. With Nambu the triangle keeps the
  step, so a superconductor whose pairing dies needs T of that size, or
  the unsorted BdG triangle of the item above. The qtci
  count of this state is not reproducible at `maxerror=1e-10`, which is
  below the noise of its residual.
- The user guide moved with the BdG Fermi level. The figures of the s-wave
  and triplet examples of the mean-field chapter were made again, and the
  numbers in the text are the new ones: Delta_0 from 0.30 to 0.29, the
  band minimum at K from -1.85 to -1.90, the hole copy at M from 0.9 to
  0.92, the triplet gap from 0.42 to 0.39 at nk = 20 and from 0.40 to 0.38
  at nk = 40. The qtci gap of the topological-invariants chapter changed
  too: at nk = 8 the old loop lost the pairing, instead of the 16% quoted,
  and now it is 10% too large. The run of the exchange chapter with a
  random guess and nk = 8 (a gap of 0.02) was not checked.
- `VJinteraction`, the default engine of `h.get_mean_field_hamiltonian`
  for a spinful Hamiltonian, does not take `"triangle"`. The entry point
  routes it to `Vinteraction`, which has no exchange terms J.
- The Fermi search and the density matrix diagonalize the mesh twice per
  iteration, as `"ed"` does. Sharing the eigenvectors would save a factor 2.
- A 3D mean-field benchmark of the tetrahedra against a dense mesh was not
  run; the tetrahedron density matrix is only checked against a dense mesh
  on a cubic metal, in the tests. The 3D qtci is measured against the
  tetrahedra on the same mesh (qtci in 3D above). The slow
  finite-temperature Fermi search of the tetrahedra (33 s at 32^3) is
  solved by the Newton search of `simplex_fermi_search.md` (0.20 s at
  32^3 and T = 1e-2); what is left of a 3D mean field at T > 0 is the
  weights themselves.
- Bands that cross are interpolated with a kink, as in the simplex DOS.
- qtci for a metal is built, as the triangle on a quantics-interpolated
  mesh (see qtci above). What is open there: on a two-orbital model it
  needs nearly the whole mesh up to 2^7 x 2^7 and loses in wall time; in
  3D it saves k-points (0.3% of 128^3 on the cubic band, about 60% on
  layered four-band models) but not wall time, because the tetrahedron
  weights on the full mesh dominate, and the X fields of the layered
  models reach rank 512 at 128^3; every iteration starts its cross interpolation from scratch,
  and a reuse of the pivots bond by bond was not tried; and the stacked
  TCI of all the fields may help the qtci DOS and conductivity too, which
  still use one TCI per field.
- At T well above the mesh spacing v_F/nk, a hybrid that uses the plain
  Fermi occupation on the simplices far from mu would keep the exponential
  convergence of exact diagonalization there. Not tried.
