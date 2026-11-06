# The triangle and tetrahedron susceptibility of metals

This note records how the bare susceptibility chi_0(q, omega) of a crystal
was built with the linear simplex integration (segments in 1D, triangles in
2D, tetrahedra in 3D), and the measurements that decided its shape. The
outcome is in the code as `integration="triangle"` of
`chitk.chiAB.chiAB_q`, so every response that sums on the uniform mesh
through it takes the keyword: `h.get_chi()`, `h.get_spinchi_full()`,
`h.get_spinchi_ladder()`, `h.get_densitychi_RPA()`,
`h.get_rpa_kernel_poles()`, and the site-basis route of
`h.get_magnon_bands()`. The pair-basis route of the spin response
(`chitk/pairchi.py`) refuses it. The engine is `chitk/simplexchi.py`, the
tests are `tests/chi/test_chi_triangle.py`. What is left open is at the
end.

## The question

The Lindhard form in the convention of `chitk/chiAB.py` is

chi_ij(q, w) = int dk sum_nm M^ij_nm(k) (f(E_n(k)) - f(E_m(k+q))) / (E_n(k) - E_m(k+q) - w + i delta),

with M^ij_nm(k) = <n,k|A_i|m,k+q><m,k+q|B_j|n,k> and the Brillouin zone
normalized to one. Note the sign: the static response of a metal is
negative here, so the Stoner criterion of a Hubbard model reads
1 + U chi_0(Q) = 0 per spin, and U_c = -1/chi_0(Q).

The energies and the matrix elements are smooth in k. The sharp parts are
the two Fermi steps and, for the dynamic response, the resonance. On the
uniform mesh each pair is sampled at a point, so the mesh has to resolve
the steps, which takes nk of order v/T, and the resonance, which takes nk
of order v/delta. At T = 1e-3 t this is nk of a few thousand per
direction. The question is whether we can integrate the steps and the
resonance exactly, so that the mesh only resolves the bands.

## The construction at T = 0

Let us take one simplex of the mesh and one pair (n, m). We interpolate
E_n(k), E_m(k+q) and M(k) linearly inside the simplex. Then
f_n - f_m = theta_n (1 - theta_m) - theta_m (1 - theta_n), so the integral
is over the region A, where n is occupied and m is empty, minus the region
B, where m is occupied and n is empty. Each region is the intersection of
the simplex with two half-spaces, so it is a convex polytope.

We tile each region with sub-simplices in two stages. First we cut out the
part where E_n < mu, with the corner, prism and truncated-corner
decompositions of `dmtk/simplexdm.py`. Then each piece is cut again by
E_m > mu. This gives at most 2 x 2 triangles or 3 x 3 tetrahedra per
region, all of positive volume, and the tiling never leaves its region.
As a result the transition energy D = E_n - E_m is negative everywhere in
A and positive everywhere in B. So the static response has no resonance
inside any piece, and its delta = 0 limit is a real and finite integral.
This is the reason for the two-stage tiling: a signed decomposition (the
simplex minus a corner) would put the singular line D = 0 inside the
pieces.

Since M is linear in the simplex, each piece gives a weight per vertex j,
w_j = int_piece lambda_j / (D - w + i delta), with lambda_j the
barycentric coordinate. On a piece of dimension d, with node values
x_s = D_s - w - i delta, the Hermite-Genocchi formula with one repeated
node gives the average of mu_s g(D) as d! G[x_0, ..., x_d, x_s], with
G^(d+1) = g and mu_s the barycentric coordinates of the piece. For
g = 1/(z - D) this is d! psi_(d+1) in the notation of
`dostk/simplexdos.py`, and the response is minus its complex conjugate.
The divided differences are evaluated with the Newton table of the
simplex DOS, with the clusters of close nodes replaced by their Taylor
series. The psi of the nodes is evaluated once per piece and shared by
its d + 1 tables. This is the analytic tetrahedron method of the Lindhard
function (MacDonald, Vosko and Coleridge, J. Phys. C 12, 2991 (1979);
Rath and Freeman, PRB 11, 2109 (1975)) with the complex energies of
Lambin and Vigneron, PRB 29, 3430 (1984). We checked the vertex weights
of a single simplex against a Monte Carlo sampling of 4 million points in
1D, 2D and 3D, and the tiling against the occupied volume and the step
weights of `simplexdm`, to 1e-12.

A pair whose transition energy vanishes on the whole simplex is special:
n = m at q = 0, the two spin copies of a spin-degenerate band, or q a
reciprocal lattice vector. There the regions A and B are empty, but on a
mesh the pair stands for the intraband continuum at q -> 0, where
(f_n - f_m)/(E_n - E_m) tends to f'(E_n). So, as in `chiAB.lindhard_pair`,
the pair gives f'(E_n) i delta/(i delta - w), integrated with the Fermi
surface weights int_T lambda_j delta(mu - E_n), which are -Im/pi of the
same resolvent averages at zero broadening. The static response at q = 0
is then minus the DOS of the triangle integration, exactly.

The weights are summed over the simplices that share each vertex and
averaged over the degenerate multiplets of n at k and of m at k+q, so the
result does not depend on the basis that eigh picks inside a multiplet.
Then they are contracted with the matrix elements of each batch of
k-points. The cost is the one of the grid for the eigenvectors, plus the
weights, whose cost grows with the number of frequencies.

## Finite temperature

f(E) = int dx (-f'(x)) theta(x - E), and the same x enters both steps. So
the response at T is the T = 0 response at chemical potential x,
convolved with -f'(x), for any q and w. As in `simplexdm`, the integral
over x is split at the vertex energies of both bands, at 0, at the
multiples of T of `simplexdm._tbreaks` up to 40 T, and at the energies
where a cut through an edge of the simplex meets the resonance D = w.
Each piece is done by Gauss-Legendre quadrature in the variable f(x). A
pair whose two bands stay on one side of the window at every vertex has
an x-independent T = 0 response and is evaluated once.

The plain Gauss-Legendre nodes of `simplexdm` were not enough here. On the
square lattice at q = (1/2, 1/2), E(k+Q) = -E(k) up to the onsite term, so
D = 0 on the half-filling Fermi line, and for an even nk that line runs
along mesh edges. A triangle with an edge on D = 0 has a T = 0 response
with a logarithmic singularity at a breakpoint, ln|x - mu|, which plain
nodes resolve only slowly. We think this is the cause, since grading the
nodes fixes it, but we did not isolate those triangles. At T = 5e-2 the
triangle results with 8
plain nodes did not converge with nk at all: the errors were 2.6e-4,
3.7e-4, 2.7e-4 and 1.4e-4 at nk = 64 to 512. With nodes graded by a
smoothstep, phi = f1 + (f0 - f1) g(u), whose g' vanishes at both ends,
they become clean second order. Against 48 nodes with the first grading,
at nk = 64 and 128:

| rule | error at nk = 64 | error at nk = 128 |
|---|---|---|
| 8 plain nodes | 6e-4 | 5e-4 |
| 8 nodes, g' = 30 u^2 (1-u)^2 | 6e-6 | 5e-6 |
| 12 nodes, g' = 30 u^2 (1-u)^2 | 6e-7 | 4e-7 |
| 16 nodes, g' = 30 u^2 (1-u)^2 | 1e-7 | 8e-8 |
| 8 nodes, g' = 140 u^3 (1-u)^3 | 1.2e-6 | 1.2e-6 |
| 12 nodes, g' = 140 u^3 (1-u)^3 | 7e-8 | 5e-8 |

The code uses 12 nodes with g' = 140 u^3 (1-u)^3, a relative error of
about 1e-7, at 1.5 times the cost of 8 nodes.

When delta >= T the features in x are at least T wide, so the frequencies
share the nodes and the tiling, and the resonance breakpoints are dropped.
On the square lattice at nk = 128, T = delta = 1e-3, 40 frequencies, this
shared path agrees with the frequency-by-frequency path to 3e-9 of the
largest value. We also tried a shorter list of T multiples (1, 2, 4, 8,
16, 25 T). It is twice as fast and loses 1e-6 relative accuracy, about
the discretization error at nk = 512, so we kept the full list.

## The benchmark

All runs are spinless, so the numbers are per spin, on four cores of one
workstation that other jobs shared (load 2 to 6), so the wall times are
only indicative. The triangle is the public call with `delta=0`, the
exact static limit. "Grid" is the same Lindhard sum on the same uniform
mesh, with the degenerate pairs of `lindhard_pair` and delta = 1e-7, in a
version vectorized over k for this benchmark. The library grid of
`chiAB_q` gives the same numbers where we ran it (nk = 64 to 256 on the
square lattice, 48 and 96 on the honeycomb), up to the cutoff described in
the open list, but it costs about 0.3 ms per k-point at one frequency
here, so it was not run on the large meshes. The errors are absolute, in units of 1/t.

The references. At T = 5e-2, 1e-2 and 1e-3 the grid at the largest mesh
(nk = 8192 on the square lattice, 6144 on the honeycomb) is converged
exponentially, since the Fermi step needs nk of order v/T. The triangle
extrapolated in 1/nk^2 from its two largest meshes agrees with it to
6e-9 on the square lattice at all three temperatures, to 2e-11 on the
honeycomb at q = M, and to 3e-7 at q = 0 at T = 1e-3. This is the
independent check of the whole scheme. At T = 1e-4 the grid does not
converge on any mesh we can afford, so the reference is the extrapolated
triangle itself.

### The square lattice near half filling, q = (pi, pi)

One orbital, first-neighbor hopping t, onsite energy 0.1 t, so the Fermi
level sits 0.1 t below the van Hove energy. E(k+Q) = -E(k) up to the
onsite term, so the response at Q is the nesting peak, cut off by the
detuning 0.2 t between the two Fermi lines. chi_0(Q) = -0.61081 per spin
at T = 1e-3 and -0.61076 at T = 1e-4 and at T = 0, in units of 1/t.

| nk | triangle T=5e-2 | grid T=5e-2 | triangle T=1e-2 | grid T=1e-2 | triangle T=1e-3 | grid T=1e-3 | triangle T=1e-4 | grid T=1e-4 | triangle wall (s), T=1e-3 |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 1.3e-03 |  | 1.6e-03 |  | 1.7e-03 |  | 1.7e-03 |  | 0.2 |
| 64 | 3.4e-04 | 7.0e-04 | 3.2e-04 | 6.1e-03 | 4.5e-04 | 1.3e-02 | 4.6e-04 | 1.8e-02 | 0.3 |
| 128 | 8.5e-05 | 3.0e-06 | 7.3e-05 | 3.7e-03 | 1.0e-04 | 2.1e-02 | 1.0e-04 | 2.1e-02 | 0.6 |
| 256 | 2.1e-05 | 3.1e-09 | 1.8e-05 | 4.0e-04 | 1.1e-05 | 7.6e-03 | 8.0e-06 | 7.5e-03 | 1.7 |
| 512 | 5.3e-06 | 3.0e-09 | 4.5e-06 | 6.7e-06 | 3.9e-06 | 2.3e-03 | 2.8e-06 | 2.8e-03 | 5.2 |
| 1024 | 1.3e-06 | 3.0e-09 | 1.1e-06 | 2.0e-09 | 1.1e-06 | 4.8e-04 | 8.8e-07 | 5.2e-04 | 18 |
| 2048 | 3.4e-07 | 2.9e-09 | 2.8e-07 | 6.2e-12 | 2.8e-07 | 3.0e-05 | 2.2e-07 | 4.0e-05 | 66 |
| 4096 |  | 2.3e-09 |  | 4.9e-12 |  | 1.7e-06 |  | 5.9e-05 |  |
| 8192 |  | reference |  | reference |  | reference |  | 2.0e-05 |  |

The triangle wall times at T = 5e-2 are much larger, 1.2, 17, 200 and 770
s at nk = 64, 256, 1024 and 2048, because the window of 40 T then covers
most of the band and every simplex goes through the quadrature in x. At
T = 0 they are 0.2 s at nk = 256 and 19 s at nk = 2048.

### The square lattice at half filling

The same lattice without the onsite term, so the nesting is perfect and
chi_0(Q) grows like ln^2(1/T): -1.4179, -2.4274 and -3.7056 at T = 1e-2,
1e-3 and 1e-4. At T = 0 it diverges. The triangle at T = 0 then returns
a large finite number that grows with nk, -49 and -96 at nk = 32 and 64,
-15 and -17 at nk = 33 and 65, with a spurious imaginary part of 3e-3,
probably because the logarithm at a node with D = 0 is cut off at 1e-300
in the divided differences (we did not check this). So T = 0
at a perfect nesting vector is not a meaningful call, and nothing warns
about it.

| nk | triangle T=1e-2 | grid T=1e-2 | triangle T=1e-3 | grid T=1e-3 | triangle T=1e-4 | grid T=1e-4 | triangle wall (s), T=1e-3 |
|---|---|---|---|---|---|---|---|
| 32 | 4.7e-03 |  | 5.2e-02 |  | 3.3e-01 |  | 0.2 |
| 64 | 8.3e-04 | 2.4e-01 | 6.2e-04 | 6.2e+00 | 1.4e-01 | 7.4e+01 | 0.3 |
| 128 | 2.1e-04 | 5.1e-02 | 1.0e-03 | 2.7e+00 | 2.9e-02 | 3.6e+01 | 0.6 |
| 256 | 5.2e-05 | 4.5e-03 | 1.1e-04 | 1.1e+00 | 1.9e-03 | 1.7e+01 | 1.7 |
| 512 | 1.3e-05 | 5.7e-05 | 2.5e-05 | 3.6e-01 | 1.8e-04 | 8.1e+00 | 5.5 |
| 1024 | 3.4e-06 | 4.9e-08 | 6.2e-06 | 8.9e-02 | 1.9e-05 | 3.6e+00 | 20 |
| 2048 | 9.0e-07 | 3.4e-08 | 1.7e-06 | 1.1e-02 | 4.9e-06 | 1.5e+00 | 75 |
| 4096 |  | 2.7e-08 |  | 3.2e-04 |  | 5.2e-01 |  |
| 8192 |  | reference |  | reference |  | 1.5e-01 |  |

At T = 1e-4 the triangle converges faster than second order up to
nk = 512 and at second order from there; we did not look into why. It
is wrong by 5e-4 relative at nk = 256, where the grid is wrong by a
factor 5. The grid is still wrong by 4 percent at nk = 8192.

### Odd and even meshes at (pi, pi)

The sequences above are not monotonous at every step: near half filling
at T = 1e-3 the error at nk = 256 (1.1e-5) is smaller than the second
order trend from 128 (1.0e-4) predicts, and at half filling the error at
nk = 64 is smaller than at 128. For an even nk the line E(k) = E(k+Q),
which is the half-filling Fermi line k1 + k2 = 1/2, passes through mesh
points and runs along the diagonals of the triangles. For an odd nk it
does not. The same runs on odd meshes, errors against the same
references:

| nk | near half filling, T=1e-3 | near half filling, T=1e-4 | half filling, T=1e-3 | half filling, T=1e-4 |
|---|---|---|---|---|
| 33 | 1.3e-03 | 1.3e-03 | 7.0e-02 | 3.8e-01 |
| 65 | 2.9e-04 | 2.9e-04 | 1.1e-02 | 1.7e-01 |
| 129 | 6.7e-05 | 6.7e-05 | 5.0e-04 | 4.7e-02 |
| 257 | 1.7e-05 | 1.6e-05 | 4.3e-05 | 5.4e-03 |
| 513 | 4.4e-06 | 4.1e-06 | 2.1e-05 | 3.3e-04 |
| 1025 | 1.1e-06 | 1.0e-06 | 6.1e-06 | 1.1e-05 |
| 2049 | 2.8e-07 | 2.0e-07 | 1.6e-06 | 3.1e-06 |

Near half filling the odd meshes are second order from nk = 65 on, with
a factor 4 per doubling, so they are the safe choice at a commensurate
nesting vector; the docstring of `chiAB_q` says so. The even meshes are
not worse in size, only irregular. The odd and the even sequences
extrapolate to the same limit: to 3e-9 at T = 1e-3, and to 8e-8 at
T = 1e-4, where the triangle is its own reference, so this is the check
of that reference. At half filling and T = 1e-4 they agree to 8e-6, or
2e-6 relative. The grid on odd meshes, nk = 4097 to 16385 at T = 1e-3,
gives the same reference to 2e-9.

### A honeycomb metal

Two orbitals, sublattice imbalance 0.2 t, onsite energy -0.6 t, so the
Fermi level is 0.6 t above the Dirac energy, inside the upper band. The
response is the 2 x 2 matrix chi_ij of the two sublattices, and the error
is the one of its largest eigenvalue lambda = max eig(-chi_0), the
quantity of the Stoner criterion. At q = 0 the matrix carries the
interband terms between the two bands as well as the intraband term,
lambda = 0.34498 at T = 1e-3. At q = M = (1/2, 0), lambda = 0.18527.

q = 0:

| nk | triangle T=5e-2 | grid T=5e-2 | triangle T=1e-3 | grid T=1e-3 | triangle T=1e-4 | grid T=1e-4 | triangle wall (s), T=1e-3 |
|---|---|---|---|---|---|---|---|
| 24 | 1.8e-03 |  | 7.3e-04 |  | 7.3e-04 |  | 0.1 |
| 48 | 4.5e-04 | 7.2e-05 | 3.9e-04 | 2.2e-02 | 3.9e-04 | 2.2e-02 | 0.1 |
| 96 | 1.1e-04 | 8.4e-10 | 1.8e-04 | 5.5e-03 | 1.8e-04 | 2.2e-02 | 0.4 |
| 192 | 2.8e-05 | 2.8e-15 | 2.5e-05 | 5.4e-03 | 5.1e-05 | 2.1e-02 | 1.0 |
| 384 | 7.0e-06 | 3.9e-16 | 4.5e-06 | 7.8e-04 | 2.0e-05 | 2.1e-02 | 3.4 |
| 768 | 1.7e-06 | 8.3e-16 | 6.9e-07 | 1.1e-03 | 5.7e-06 | 1.2e-02 | 12 |
| 1536 | 4.3e-07 | 3.3e-16 | 4.2e-07 | 9.4e-05 | 1.4e-06 | 4.7e-03 | 47 |
| 3072 |  | 2.2e-16 |  | 5.8e-07 |  | 9.0e-04 |  |
| 6144 |  | reference |  | reference |  | 2.4e-04 |  |

q = M:

| nk | triangle T=5e-2 | grid T=5e-2 | triangle T=1e-3 | grid T=1e-3 | triangle T=1e-4 | grid T=1e-4 | triangle wall (s), T=1e-3 |
|---|---|---|---|---|---|---|---|
| 24 | 3.3e-04 |  | 2.0e-04 |  | 2.0e-04 |  | 0.1 |
| 48 | 8.1e-05 | 4.1e-06 | 5.8e-05 | 3.7e-04 | 5.8e-05 | 3.7e-04 | 0.2 |
| 96 | 2.0e-05 | 7.5e-10 | 1.6e-05 | 5.2e-04 | 1.6e-05 | 4.1e-04 | 0.5 |
| 192 | 5.0e-06 | 7.7e-12 | 4.6e-06 | 1.1e-04 | 4.6e-06 | 2.1e-04 | 1.3 |
| 384 | 1.2e-06 | 7.7e-12 | 1.1e-06 | 1.9e-05 | 1.1e-06 | 1.4e-05 | 4.7 |
| 768 | 3.1e-07 | 7.6e-12 | 2.7e-07 | 3.4e-06 | 2.7e-07 | 5.0e-07 | 17 |
| 1536 | 7.8e-08 | 7.2e-12 | 6.8e-08 | 2.6e-07 | 6.6e-08 | 8.5e-06 | 65 |
| 3072 |  | 5.8e-12 |  | 1.1e-09 |  | 2.0e-06 |  |
| 6144 |  | reference |  | reference |  | 4.2e-07 |  |

At q = M the grid error at T <= 1e-3 is much smaller than at q = 0,
4e-4 already at nk = 48; we did not look into why. The triangle still
gains a factor 2 to 4 in nk there. Its wall times at T = 5e-2 are again
large, 30 s at nk = 192 and 1800 s at nk = 1536.

### The critical U

The number a user wants from all this is U_c = 1/lambda, the Hubbard U at
which 1 + U chi_0(Q) first reaches zero, in units of t. For the square
lattice near half filling at Q and T = 1e-3 (converged U_c = 1.63717) and
for the honeycomb metal at q = 0 and T = 1e-3 (converged U_c = 2.89871):

| nk | square, triangle | square, grid | honeycomb, triangle | honeycomb, grid |
|---|---|---|---|---|
| 24 or 32 | 1.63258 |  | 2.89258 |  |
| 48 or 64 | 1.63596 | 1.60396 | 2.89545 | 3.09841 |
| 96 or 128 | 1.63690 | 1.69544 | 2.89722 | 2.94595 |
| 192 or 256 | 1.63715 | 1.65770 | 2.89892 | 2.94471 |
| 384 or 512 | 1.63716 | 1.64325 | 2.89874 | 2.89215 |
| 768 or 1024 | 1.63717 | 1.63847 | 2.89870 | 2.90765 |
| 1536 or 2048 | 1.63717 | 1.63726 | 2.89870 | 2.89792 |
| 3072 or 4096 |  | 1.63717 |  | 2.89870 |

The first value of nk is the honeycomb one, the second the square one.
The triangle gives U_c to 1e-3 relative at nk = 64 on the square
lattice and at nk = 96 on the honeycomb, and to 1e-5 at nk = 256 and
384. The grid needs nk = 1024 and 1536 for 1e-3, and 4096 and 3072 for
1e-5. Below that it scatters by up to 4 percent on the square lattice and
7 percent on the honeycomb, and not monotonously, so nothing tells the
user that it has not converged. In k-points the gain at 1e-5 is a factor
256 on the square lattice and 64 on the honeycomb. Through
`chi_AB_RPA` with the vertex -U/2 of sz on the spinful model, the RPA
response divided by chi_0 is 1/(1 - U/U_c) to 1e-8, which the tests check.

### A cubic metal

One orbital on the simple cubic lattice, onsite energy 0.1 t, q = (pi, pi,
pi), tetrahedra with six per cube. lambda = 0.52653 at T = 5e-2 and
0.49782 at T = 1e-3. At T = 1e-3 the reference is the tetrahedron result
extrapolated from nk = 32 and 64, since the grid needs nk of order v/T,
several thousand, in three dimensions.

| nk | tetrahedron T=5e-2 | grid T=5e-2 | tetrahedron T=1e-3 | grid T=1e-3 | tetrahedron wall (s), T=1e-3 |
|---|---|---|---|---|---|
| 8 | 5.7e-02 |  | 2.0e-02 |  | 1.5 |
| 16 | 5.0e-03 | 3.7e-03 | 4.6e-03 | 2.1e-02 | 5.5 |
| 24 | 2.2e-03 |  | 2.0e-03 |  | 13 |
| 32 | 1.2e-03 | 3.2e-03 | 1.1e-03 | 1.9e-02 | 23 |
| 48 |  |  | 5.0e-04 |  | 57 |
| 64 |  | 1.9e-04 | 2.8e-04 | 2.5e-03 | 102 |
| 128 |  | 7.7e-07 |  | 2.0e-03 |  |
| 256 |  | reference |  | 5.4e-04 |  |

At T = 5e-2 the tetrahedra converge at second order to the converged
grid (the error falls by 4.2 from nk = 16 to 32). At T = 1e-3 the
tetrahedra at nk = 64, 262k k-points and 100 s, beat the grid at
nk = 256, 17M k-points. The tetrahedron cost per k-point is high: up to 9
sub-tetrahedra per region and a quadrature in x on every tetrahedron near
the Fermi surface.

### The dynamic response

The same square lattice near half filling, at a generic q = (0.3, 0.1),
60 frequencies from 0.02 t to 6 t, T = 1e-3. The error is the largest
deviation over the frequencies, of the real and the imaginary part
together, divided by the largest abs(chi_0), which is 0.39 at
delta = 1e-2 and 0.43 at delta = 1e-3. The reference is the triangle at
nk = 2048. At delta = 1e-2 the grid at nk = 4096 agrees with it to
5.6e-5, which is the size of the error left in the reference (the
triangle errors fall by 4 per doubling). At delta = 1e-3 no grid we ran
is converged. The grid wall times are those of the vectorized version;
the library grid took 4.5 s at nk = 256 and agreed with it to 3e-7.

| nk | triangle, delta=1e-2 | triangle wall (s) | grid, delta=1e-2 | triangle, delta=1e-3 | triangle wall (s) | triangle wall (s), T=0 | grid, delta=1e-3 |
|---|---|---|---|---|---|---|---|
| 64 | 5.2e-02 | 9 |  | 7.1e-02 | 9 | 0.4 |  |
| 128 | 1.4e-02 | 17 |  | 2.0e-02 | 17 | 0.6 |  |
| 256 | 3.5e-03 | 38 | 3.3e-02 | 5.3e-03 | 37 | 2 | 2.3e-01 |
| 512 | 8.4e-04 | 102 | 3.4e-03 | 1.3e-03 | 102 | 8 | 1.1e-01 |
| 1024 | 1.7e-04 | 335 | 9.3e-04 | 2.6e-04 | 334 | 53 | 3.5e-02 |
| 2048 | reference | 1561 | 6.9e-05 | reference | 1573 | 536 | 1.4e-02 |
| 4096 |  |  | 5.6e-05 |  |  |  | 4.1e-03 |

At delta = 1e-3 the triangle at nk = 512 (1.3e-3) beats the grid at
nk = 4096 (4.1e-3), with 64 times fewer k-points, and its error does not
depend on delta: the columns of delta = 1e-2 and 1e-3 differ by a factor
1.5 at every nk. The grid error grows like 1/delta. At delta = 1e-2 the
triangle wins by a factor 4 to 5 in the error at equal nk, but the grid
reaches 1e-4 at nk = 2048 in 11 s, so there the triangle loses in wall
time. The T=0 results at delta = 1e-3 have the same errors to two digits
and differ from the T = 1e-3 ones by 1.8e-4 of the largest value at every
nk from 128 on, which is the physical effect of the temperature, not a
discretization error.

The dynamic response at finite T is expensive: 6 to 18 times the T = 0
cost, because every simplex near the Fermi surface goes through the
quadrature in x, and every node of it through the tiling and the
divided differences of all the frequencies. The T = 0 cost itself grows
faster than the number of simplices (8 s, 53 s and 536 s at nk = 512,
1024 and 2048); we did not find out where.

## The tests

`tests/chi/test_chi_triangle.py` checks invariants, each in a few seconds
once numba has compiled the kernels:

- the tiling of the part e < 0 of a segment, a triangle and a
  tetrahedron has the occupied volume and the step weights of
  `simplexdm`, for random vertex values;
- at T = 0.1 t the triangle reaches the converged library grid, static
  and at one frequency, and its successive differences fall at second
  order;
- the static charge response at q = 0 of a spinful honeycomb metal,
  summed over the sites, is minus the triangle DOS at the Fermi level to
  1e-10, and a small q gives the same limit;
- the sz response of a spinful paramagnet is twice the charge response of
  the spinless band to 1e-10, which goes through the degenerate pairs of
  the two spin copies;
- chi(q) = chi(-q) = chi(qy, qx) exactly on the square lattice, since the
  triangulation keeps inversion and that mirror, and the fourfold
  rotation, which the triangulation breaks, holds to 2e-3;
- U_c at (pi, pi) converges between nk = 48, 96 and 192 at T = 1e-3, and
  `chi_AB_RPA` with the sz vertex diverges at it;
- the dynamic response at delta = 0 is close to delta = 1e-6, and the
  spectral weight of the charge response is positive at positive
  frequency;
- the q = 0 sum rule and the small-q limit on a cubic metal, through the
  tetrahedra;
- the accepted values of `integration` are those of the registry, and the
  combinations that are not built raise.

With these, `tests/chi` passes, 194 tests.

## What wins and what loses

The wins. The static response at low temperature, which is what the RPA
instability of a metal needs. At T <= 1e-3 the triangle gives chi_0 and
U_c to 1e-5 with 4 to 256 times fewer k-points than the grid on the
square and honeycomb lattices, and the grid converges erratically before
it converges at all. The result does not depend on how low T is, T = 0
included, and the static delta = 0 limit is exact and real, so there is
no broadening to choose. The convergence is clean second order on a
generic mesh, so two meshes extrapolate: the extrapolated triangle
agrees with the converged grid to 6e-9. At perfect nesting the triangle
follows the ln^2(1/T) growth of chi_0(Q) down to T = 1e-4, where the
grid is wrong by 4 percent at nk = 8192. The dynamic response at small
delta converges at second order with an error that does not depend on
delta, so at delta = 1e-3 it needs 64 times fewer k-points than the grid.

The losses. Where T is above the energy spacing of the mesh the grid
converges exponentially: at T = 5e-2 and nk = 256 both take about 17 s,
but the grid is exact to 3e-9 and the triangle is off by 2e-5. The
triangle cost also grows with T, because the window of 40 T covers more
of the band, so on large meshes at high T it is the slower of the two. The dynamic response
at delta = 1e-2 is the same story. The dynamic response at finite T
costs 6 to 18 times the T = 0 one. In 3D the tetrahedra cost about 0.4 ms
per k-point at T = 1e-3, so nk = 64 takes 100 s. At a commensurate
nesting vector an even nk converges irregularly, and an odd nk has to be
used.

## What is left open

- The library grid of `chiAB_q` skips every pair with
  abs(f_a - f_b) < delta/100, and for a degenerate pair that test is
  applied to i delta f', so it drops the degenerate pairs with
  abs(f') < 0.01 at any delta. On the honeycomb metal at T = 5e-2 this
  makes the library grid differ from the same sum without the cutoff by
  3e-5 to 5e-5, and at the default delta = 0.1 the cutoff drops every
  pair with abs(f_a - f_b) < 1e-3. It is not changed here, since it
  changes the grid results; the benchmark used the sum without it.
- The cost of the dynamic response. The tiling and the divided
  differences are redone at every frequency and every quadrature node.
  The source kernel of `conductivitytk/kubosimplex.py`, which stores the
  weighted joint DOS by its knots and evaluates every frequency as one
  sum, would apply to the T = 0 pieces, since their branches depend on
  delta but not on w. Not built. The superlinear growth of the T = 0
  cost with nk was not looked into.
- A hybrid for high T: the plain Fermi occupation on the simplices far
  from the Fermi level, where the triangle buys nothing, would keep the
  exponential convergence of the grid there, as in the density-matrix
  note. Not tried.
- The local response, q = None, averages chiAB_q over the q-mesh, so it
  diagonalizes the whole mesh again at every q. It works with the
  triangle, but sharing the eigenvectors of k + q across q would save a
  factor nk^dim in diagonalizations. Not built.
- The pair-basis spin response (`chitk/pairchi.py`), used for an
  interaction between different sites, refuses `integration="triangle"`.
  Its pair fields are the same kind of smooth matrix elements, so the
  same weights would apply to its chi_0. Not built.
- Bands that cross are interpolated with a kink, as in the simplex DOS,
  which costs an error of order one in a window of width v/nk around
  the crossing. The models here have no crossing near the Fermi level.
- At T = 0 and a perfect nesting vector the response is infinite, and the
  triangle returns a finite number that grows with nk instead of
  refusing. A check for pieces with D = 0 on an edge at w = 0 and T = 0
  would catch it. Not built.
- The single-precision and GPU paths of the grid are not offered for the
  triangle.
