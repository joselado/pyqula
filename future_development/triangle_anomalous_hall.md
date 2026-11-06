# The triangle integration of the anomalous Hall conductivity of a metal

This note records how the Fermi-sea Hall conductivity of a 2D metal was built
with the linear triangle integration, and what was measured. The outcome is
`h.get_anomalous_hall_conductivity()`, implemented in
`topologytk/anomaloushall.py` (exposed as `topology.anomalous_hall_conductivity`),
with `integration="triangle"` (the default) and `integration="grid"` in a
registry. The tests are `tests/topology/test_anomalous_hall_triangle.py`.
What is left open is at the end.

## The question

The intrinsic anomalous Hall conductivity is the Fermi-sea integral of the
Berry curvature,

sigma_xy = (e^2/h) (1/2 pi) sum_n int d^2k f(E_n(k) - mu) Omega_n(k).

In an insulator every band is either full or empty, the integrand is smooth
and a uniform mesh converges exponentially, to the Chern number. In a metal
the Fermi step cuts the mesh, and the uniform sum only converges once the
energy spacing of the mesh, of order v_F/nk, is below T, with an error that
oscillates with nk. The question is whether we can integrate the step
exactly on the triangles of the mesh, as the triangle density matrix of the
mean field does (`tetrahedron_density_matrix.md`), so that the mesh only
has to resolve the bands and the curvature.

Before this there was no routine for a metal. `topology.hall_conductivity`
is an alias of `topology.chern`, which has no Fermi level and no
temperature, and its Wilson-loop plaquettes are set to zero wherever the
number of occupied states changes inside the plaquette. `h.get_dc_conductivity()`
is the Chebyshev Kubo-Bastin formula for large sparse cells, broadened by
its resolution.

## The construction

The Berry curvature of each band is the Kubo sum over the other bands,

Omega_n = -2 Im sum_m <n|v_x|m><m|v_y|n> / (E_n - E_m)^2,

with the velocities of `conductivitytk/kubo.py`, v = dH/dk + i[H,r], which
carry the full bond vector, so the atomic gauge. We take E_n(k) and
Omega_n(k) on the uniform mesh, interpolate both linearly in each triangle
(two per square), and integrate theta(mu - E) Omega exactly. Since Omega is
linear in the triangle, this is one scalar weight per vertex, the weights
`simplexdm.simplex_weights` already gives for the density matrix, with the
finite-T weights as the T=0 ones convolved with -f'(x) and with the Bloechl
term. So the Hall conductivity is sum_{k,n} w_n(k) Omega_n(k) times
2 pi/A_cell, and `integration="grid"` is the same sum with w = f/nk^2.

A fully occupied triangle gives every vertex the uniform weight 1/nk^2, so
in an insulator the two integrations are the same number, to 1e-12 in the
test.

Degenerate multiplets are handled as one object. The weights are averaged
over the multiplet at each k, and the pairs inside a multiplet are left out
of Omega, with the same grouping. The two terms of a pair enter Omega_n and
Omega_m with opposite signs, so with equal weights they cancel exactly and
the multiplet gives its non-Abelian trace, which is gauge invariant. A
spinful copy of the Haldane model gives exactly twice the spinless result.

## Why the Kubo curvature and the atomic gauge

We chose the Kubo sum over the Wilson-loop plaquettes. The plaquettes need
a fixed number of occupied states at the four corners, which fails on every
plaquette the Fermi surface crosses, and they live in the lattice gauge.

The gauge matters for a metal, unlike for the Chern number. The lattice and
the atomic curvature of a band differ by the curl of <u_n|r|u_n>, a total
derivative, which integrates to zero over a full band but not over a Fermi
sea. We measured it on the Haldane model below with a uniaxial strain
(`strain.uniaxial_strain`, s = 0.3), at mu = 0.5 and T = 0.05, nk = 120: the
atomic gauge gives 0.69733 and the lattice gauge (velocities without the
intracell term) gives 0.68983. Without the strain the two agree to 1e-13,
since C3 symmetry makes the boundary term vanish, so a test on a
C3-symmetric model cannot see the gauge. The atomic result is the one that
equals the zero-frequency Kubo conductivity of `conductivity.py`, which is
checked against graphene and C3 isotropy there; the test asserts this
agreement to 1e-6 (it is 2e-10 with delta = 1e-7).

## Measurements

Two metals, at T = 1e-3 in units of the hopping, four Fermi levels each:

- the spinless Haldane model, t2 = 0.1 with a staggered onsite 0.2 (C = 1
  at half filling, gap from -0.25 to 0.25), at mu = 0.5, 0.8, 1.5 and 2.2;
- the spinful square lattice with Rashba 0.5 and out-of-plane exchange 0.6,
  at mu = -3, -2, -0.5 and 1.

The reference is the triangle at nk = 2048. It agrees with the uniform mesh
at nk = 2048 to 9e-7 (Haldane) and 3e-6 (Rashba), and with the triangle
without the Bloechl term to 1.4e-6, so it is good to about 1e-6. The error
below is the largest over the four Fermi levels. The timings were taken on a
workstation on four cores (numba threads), with other jobs on the remaining
cores. The time is per Fermi level, the mesh (diagonalization, velocities,
curvature) plus the weights; the mesh is shared by any number of Fermi
levels.

Haldane, largest error over the four Fermi levels:

| nk | uniform | triangle | triangle, no Bloechl | mesh (s) | triangle weights (s) |
| --- | --- | --- | --- | --- | --- |
| 32 | 5.4e-3 | 3.5e-3 | 9.1e-3 | 0.010 | 0.003 |
| 64 | 6.4e-3 | 8.8e-4 | 2.2e-3 | 0.037 | 0.007 |
| 96 | 9.9e-4 | 4.1e-4 | 9.8e-4 | 0.083 | 0.010 |
| 128 | 1.2e-3 | 2.3e-4 | 5.7e-4 | 0.145 | 0.016 |
| 192 | 7.4e-4 | 1.0e-4 | 2.4e-4 | 0.351 | 0.034 |
| 256 | 4.0e-4 | 5.7e-5 | 1.4e-4 | 0.562 | 0.041 |
| 384 | 1.2e-4 | 2.5e-5 | 6.1e-5 | 1.23 | 0.092 |
| 768 | 2.0e-5 | 5.5e-6 | 1.5e-5 | 4.98 | 0.34 |
| 1536 | 2.4e-7 | 7.0e-7 | 3.0e-6 | 20.1 | 1.31 |

Rashba plus exchange:

| nk | uniform | triangle | triangle, no Bloechl | mesh (s) | triangle weights (s) |
| --- | --- | --- | --- | --- | --- |
| 32 | 6.7e-3 | 3.7e-3 | 5.4e-3 | 0.010 | 0.007 |
| 64 | 8.9e-3 | 7.6e-4 | 1.3e-3 | 0.035 | 0.011 |
| 96 | 2.2e-3 | 3.8e-4 | 5.9e-4 | 0.079 | 0.015 |
| 128 | 3.1e-3 | 2.0e-4 | 3.2e-4 | 0.136 | 0.025 |
| 192 | 1.5e-3 | 8.7e-5 | 1.4e-4 | 0.312 | 0.048 |
| 256 | 8.6e-4 | 4.9e-5 | 7.9e-5 | 0.584 | 0.059 |
| 384 | 4.3e-4 | 2.2e-5 | 3.5e-5 | 1.43 | 0.126 |
| 768 | 9.6e-5 | 4.8e-6 | 8.2e-6 | 5.78 | 0.43 |
| 1536 | 1.3e-5 | 6.2e-7 | 1.5e-6 | 28.2 | 1.97 |

What a 1e-3 accuracy costs, defined as the smallest nk from which every
larger nk tested stays within 1e-3 at all four Fermi levels (the uniform
error is not monotonic in nk, so a single lucky nk does not count):

| model | uniform | triangle | wall time uniform | wall time triangle |
| --- | --- | --- | --- | --- |
| Haldane | nk = 192 | nk = 64 | 0.35 s | 0.044 s |
| Rashba | nk = 256 | nk = 64 | 0.59 s | 0.047 s |

So the triangle needs 3 to 4 times fewer k-points per direction, 9 to 16
times fewer in total, and 8 to 12 times less wall time. At 1e-4 the gap is
the same or larger: the uniform mesh needs nk = 768 for both models (5.0 s
and 5.8 s), the triangle nk = 256 and 192 (0.60 s and 0.36 s).

What converged and what did not:

- The triangle converges as 1/nk^2 at every Fermi level, smoothly, a factor
  close to 4 per doubling from nk = 64 on. The Bloechl term divides the error
  by 1.6 to 2.5 at the same nk, so it is on by default.
- The uniform mesh at T = 1e-3 oscillates with nk: on the Haldane model the
  error is 9.9e-4 at nk = 96 and 1.2e-3 at nk = 128, and on the Rashba model
  6.7e-3 at nk = 32 and 8.9e-3 at nk = 64. It only becomes better than the
  triangle at nk = 1536 on the Haldane model, where the mesh spacing
  starts to resolve T, and there both are below 1e-6.
- The weights cost 7 to 30% of the mesh for these two- and four-band
  models, and less with more bands, since the mesh grows as the cube of the
  number of bands and the weights linearly.
- At T = 0.02 the uniform mesh is converged to 1e-9 at nk = 150 on the
  Haldane model, and there the triangle at nk = 48 and 96 is off by 1.5e-3
  and 3.8e-4. So the triangle wins at low T and loses once T is above the
  mesh spacing, as for the density matrix. The test uses this regime, since
  a converged uniform reference is cheap there.

## Choices

- A new function, not an option of `topology.chern`: `chern` and its alias
  `hall_conductivity` have no Fermi level and no temperature, and their
  Wilson plaquettes are not defined on the Fermi surface. The name says what
  the caller wants, and none similar existed.
- `integration="triangle"` is the default. In an insulator it is the same
  number as the grid, and in a metal it is never worse at T below the mesh
  spacing.
- The result is in units of e^2/h with the sign of `h.get_chern()`, so it
  is the Chern number in a gap. `conductivity.py` and `kpmkubo` give
  sigma_xy in e^2/hbar with sigma_xy 2 pi = -C, so the same quantity there
  is minus this divided by 2 pi. The curvature is Cartesian while
  `get_chern` works in the reduced coordinates, so for a left-handed
  lattice basis we expect the two to differ by a sign. This is reasoned,
  not tested.

## What is left open

- Band crossings. The bands are ordered by energy at each k, so two bands
  that cross, or anticross within the mesh spacing, are interpolated with a
  kink, and Omega_n is peaked with opposite signs in the two bands. When the
  Fermi level cuts such a point the triangle makes an error of order one in
  a window of width of order v/nk around it. A 2 x 2 supercell of the
  Haldane model at mu = 0.5 and T = 0.05 shows it mildly: its folded bands
  cross, and the triangle at nk = 60 is off by 3e-4, about what the
  primitive cell gives at nk = 96, so the folding costs a factor of order
  one in nk here. The uniform mesh of the supercell at nk = 60 gives the
  primitive nk = 120 result to 1e-13, since it is the same set of states.
- 3D, the tetrahedron version. The weights exist (`simplexdm` in 3D), but
  the velocities of `conductivitytk/kubo.py` are only built in 1D and 2D.
- The qtci mesh of the density matrix was not tried here.
- Only the intrinsic, Fermi-sea term is computed. The Fermi-surface terms
  of disorder (side jump, skew scattering) are not in a band calculation.
- Nambu and non-Hermitian Hamiltonians are refused, as in the triangle
  density matrix.
- The user guide section (in the topological invariants page), the
  reference entry and the example `examples/2d/anomalous_hall/main.py` are
  written. The README FUNCTIONALITIES bullet and its notebook are not,
  since a bullet there comes with its own executed notebook.
