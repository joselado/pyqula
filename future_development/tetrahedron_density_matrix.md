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
un-shift by the Fermi level uses the trace of the density matrix, as
`"qtci"` does since commit a8596b6.

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

`integration="qtci"` loses on these metals. On the square-lattice
ferromagnet it converges at nk = 8 to 64 to a state with m between 0.005
and 0.25 instead of 0.474, with a few thousand diagonalizations. On the
honeycomb ferromagnet it stopped at nk = 16 with `ValueError:
maxsamplevalue is zero!`. These are two different problems, and only the
second one is a bug that can be fixed in place.

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

Three hypotheses are ruled out by this table. The cross interpolation is
exact here: `get_dm_qtci` equals the plain sum over the node grid
(`full_dm_gk`) to every printed digit, at every m_in and nk, so neither the
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

The crash is a separate bug. One of the entries the loop reads, a
spin-off-diagonal element of the collinear state, is zero up to rounding:
at most 2.3e-16 on the whole grid. At one node it held the subnormal value
-1.04e-322. `integrate_robust` accepted that node as a nonzero pivot, but
the cross interpolation tests the value times the Gauss-Kronrod weight and
the normalization, here 0.0074 times 289, and the product underflows to
exactly zero. Now `integrate_robust` tests that same weighted value, so
such a node is skipped like a zero, and a field that is zero in this sense
at every node integrates to zero, the exact value of the rule.

What is in the code now:

- `gkintegrate.integrate_robust` seeds the cross interpolation only on a
  node whose weighted value is nonzero. This removes the crash, and it is
  shared with `topology.chern_qtci`.
- `get_dm_qtci` warns when a band crosses the Fermi level on the nodes,
  meaning that the integrand has a Fermi surface, and the warning names
  `integration="triangle"`. In a gap every band is entirely above or below
  the Fermi level, so there is no threshold and nothing is said. It warns
  and does not raise, because the qtci tests run metals on purpose (the
  Fermi search on the nodes and the total energy of the charge the density
  matrix holds).
- The docstrings of `get_dm`, `Vinteraction` and `get_dm_qtci` say that
  qtci is for gapped states.

The qtci rows rerun with the fix, T = 1e-3, errors against the same
references as the tables above:

| model | nk | diagonalizations | iterations | wall (s) | error in m | error in mu | error in E | filling |
|---|---|---|---|---|---|---|---|---|
| square | 8 | 5.4k | 13 | 1 | 2.3e-01 | 2.7e-01 | 6.5e-03 | 0.400000000 |
| square | 16 | 2.3k | 0 | 1 | 4.5e-01 | 2.1e-01 | 2.7e-02 | 0.400000000 |
| square | 32 | 5.3k | 3 | 1 | 4.7e-01 | 3.4e-01 | 2.0e-02 | 0.400000000 |
| square | 64 | 11k | 6 | 2 | 4.2e-01 | 3.5e-01 | 2.5e-02 | 0.400000000 |
| honeycomb | 8 | 7.1k | 18 | 7 | 4.2e-02 | 6.7e-02 | 2.7e-03 | 0.350000000 |
| honeycomb | 16 | 0.23M | 400 | 119 | 7.1e-02 | 6.6e-02 | 1.2e-03 | 0.350000000 |
| honeycomb | 32 | 14k | 13 | 8 | 5.0e-02 | 5.9e-02 | 1.9e-03 | 0.350000000 |

The square rows are the same as before to every digit, as expected, since
the fix does not touch the node sum. The honeycomb at nk = 16 now runs. It
needs 400 iterations, the cap, and its residual falls below 1e-9 only at
that last step, so the loop reports it converged. It lands at m = 0.315
instead of 0.244. These
runs were on one workstation with a load of 5 to 7 on 14 cores, so the
wall times are only indicative.

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
- Bogoliubov-de Gennes Hamiltonians are refused, and so are non-Hermitian
  ones. For BdG the weights at mu = 0 of the quasiparticle bands would do,
  with the Fermi search on the normal state, as `"ed"` does.
- `VJinteraction`, the default engine of `h.get_mean_field_hamiltonian`
  for a spinful Hamiltonian, does not take `"triangle"`. The entry point
  routes it to `Vinteraction`, which has no exchange terms J.
- The Fermi search and the density matrix diagonalize the mesh twice per
  iteration, as `"ed"` does. Sharing the eigenvectors would save a factor 2.
- A 3D mean-field benchmark was not run. The tetrahedron density matrix is
  only checked against a dense mesh on a cubic metal, in the tests.
- Bands that cross are interpolated with a kink, as in the simplex DOS.
- qtci for a metal. What would work is the route the DOS and the optical
  conductivity already take (`qtci_dos_green.md`, `qtci_optical_response.md`):
  a quantics cross interpolation of the smooth fields, the band energies
  and the projector entries of each band, on a 2^R x 2^R mesh, and then
  the triangle weights of `dmtk/simplexdm.py` on the reconstructed fields.
  This replaces the Gauss-Kronrod grid, its Fermi search and its band
  energy, and it changes what nk means for qtci. The two spin bands of a
  ferromagnet cross, so fields sorted by energy jump there and cost rank,
  which is the open problem those notes record too. Not built.
- At T well above the mesh spacing v_F/nk, a hybrid that uses the plain
  Fermi occupation on the simplices far from mu would keep the exponential
  convergence of exact diagonalization there. Not tried.
