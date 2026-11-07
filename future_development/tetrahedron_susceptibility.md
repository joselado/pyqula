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
square lattice, 48 and 96 on the honeycomb), up to the pair cutoff it
had then (see the pair cutoff of the grid), but it costs about 0.3 ms per
k-point at one frequency here, so it was not run on the large meshes. The errors are absolute, in units of 1/t.

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

The wall times in this table are those of the first kernel. The next
section says why they grew faster than the mesh, and what they are now.

### The cost of the dynamic response

We will now see why the cost of the dynamic response grew faster than
the number of simplices, and how the frequencies were made cheap.

The superlinear cost at T = 0 came from the accumulation, not from the
integration. The weight kernel works on chunks of 32768 simplices, and
the weights of each chunk were added to the vertex weights W[k,n,m,w]
with a bincount over the whole of W. W is kept below `max_memory_gb` by
blocks of frequencies, so one block holds about 3e7 entries at any nk,
and the number of bincount passes is the number of chunks times the
number of blocks, which grows like nk^4 in 2D: 16 passes at nk = 512,
192 at nk = 1024 and 2304 at nk = 2048. At nk = 1024 one chunk took
0.50 s in the kernel and 0.46 s in the bincount, on four busy cores. A
serial scatter of each chunk into W costs the size of the chunk, and
this alone makes the cost linear in the number of simplices. The first
kernel stays in the code with this fix, as the reference kernel
`"simplices"` of `simplexchi._kernels`.

What does not depend on the frequency? At T = 0 the tiling of each
simplex into pieces is fixed, and a piece is given by the transition
energies D_s on its nodes and, per node, the coefficient
c_s = sgn vol sum_j C_sj G_j, where G_j are the matrix elements of the
vertex j, already contracted over the operators. The multiplet average
is moved from the weights to G, which gives the same contraction
exactly. So the response at w is a sum over the pieces of
sum_s c_s r_s(w), and the pieces are the sources of the default kernel
`"sources"`.

The merging by exact keys of `conductivitytk/kubosimplex.py` does not
carry over. There the branch rule does not depend on the frequency: a
simplex of spread s above delta is evaluated by partial fractions, which
loses eps (|x|/s)^d at a distance |x| from the resonance. For the
optical conductivity d = 2. Here the repeated node of the vertex weights
makes d = 3 for triangles and d = 4 for tetrahedra, so at nk = 2048,
with s about 0.01 and |x| up to 6, the loss is about 1e-8 per source,
and at delta = 0, which the static response uses, the rule has no scale
at all. Instead the pieces go into bins of transition energy, as in a
fast multipole sum on a line. A piece of spread s goes to the finest
level whose bin width b is at least s, so its nodes lie within b of the
center c of its bin. Far from the bin, |z - c| > 3 b, the average of
mu_s/(z - D) over the piece is a series in u = 1/(z - c) whose
coefficients are the complete homogeneous polynomials of the node
energies, the series of the Taylor branch of `dostk/simplexdos.py`, and
its terms fall at least like 3^-n, so 40 terms converge to 1e-19. The
moments of each bin are summed once over its pieces. Each frequency then
costs the series of the bins far from it, a few thousand, plus the
pieces of the bins near it, which are evaluated exactly, with the same
divided differences and the same branches as the reference kernel. The
near bins are a few per level, and they hold about 1 to 5 percent of the
pieces, fewer on finer meshes.

At finite T with delta >= T the frequencies share the nodes of the
quadrature in x, so each node only adds pieces, with the weight of the
node in their coefficients, and the same kernel applies. With delta < T
the nodes include the resonance of each frequency, and at first the call
went to the reference kernel; the section on fewer pieces at finite
temperature below says how the source kernel now takes this case too.

The numba loops split their range statically, so the bins, the near sums
and the simplices are dealt to the threads in tasks of about equal work;
before this the moments of the dense bins all went to one thread. The
pass that counted the pieces before writing them, which tiled every
simplex twice, is replaced by an upper bound from the knots of the
quadrature.

The same benchmark as above, delta = 1e-3, 60 frequencies, on four
cores of the shared workstation (load 5 to 8), with the first kernel and
with the source kernel in the same session, the first kernel first:

| nk | first kernel, T=0 (s) | source kernel, T=0 (s) | first kernel, T=1e-3 (s) | source kernel, T=1e-3 (s) | agreement |
|---|---|---|---|---|---|
| 256 | 1.7 | 0.16 | 39 | 2.7 | 1e-14 |
| 512 | 6.7 | 0.49 | 106 | 6.4 | 3e-14 |
| 1024 | 44 | 1.7 | 391 | 17 | 4e-14 |
| 2048 | 587 | 6.3 | 2249 | 61 | 7e-14 |

The agreement is the largest difference between the two kernels over T =
0 and T = 1e-3, divided by the largest abs(chi_0). A second run of nk =
1024 at T = 0 right after, with both kernels, gave 60 s and 3.3 s, so
the load moved the wall times by up to a factor 2 and the ratios are
the numbers to read. The gain is a factor 11 to 94 at T = 0 and 15 to
37 at T = 1e-3, and it grows with nk, because the bins get finer with
the pieces. The T = 0 cost now grows like the number of simplices, a
factor 3.1 to 3.8 per doubling of nk. At delta = 1e-2 the source kernel
takes 0.3, 1.4, 5.0 and 11 s at T = 0 and 6.3, 17, 36 and 99 s at T =
1e-3, more than at delta = 1e-3. We first thought this was the Taylor
branch of the near pieces; the section on fewer pieces at finite
temperature below shows that it was the load of the machine. The
errors against nk = 2048 are those of the first table to two digits,
at both temperatures and both broadenings.

Finite T is still about 10 times the T = 0 cost (17 times at nk = 256,
10 times at nk = 2048): the quadrature in x gives 14 to 19 pieces per
simplex at nk = 256 and 512, against 0.45 at T = 0, and each piece is
tiled and enters the moments once. Before, the ratio looked smaller at
large nk only because the T = 0 cost carried the bincount. On a profile
at nk = 512 the tiling takes about 40 percent of the finite-T time, the
moments 20 percent and the near pieces 35 percent. The section on
fewer pieces at finite temperature below divides the pieces of this case
by 2.8 and the time by 2.5 to 3.

Now the triangle against the grid at equal accuracy, at T = 1e-3. The
grid is the same Lindhard sum on the uniform mesh, with the band
energies from the same diagonalization and a numba sum over k on the
same four cores; its errors are those of the first table.

| nk | grid, delta=1e-3 | grid wall (s) | grid, delta=1e-2 | grid wall (s) |
|---|---|---|---|---|
| 512 | 1.1e-01 | 0.4 | 3.4e-03 | 0.4 |
| 1024 | 3.5e-02 | 1.1 | 9.3e-04 | 1.4 |
| 2048 | 1.4e-02 | 4.7 | 6.9e-05 | 4.5 |
| 4096 | 4.1e-03 | 19 | 5.6e-05 | 18 |

At delta = 1e-3 the triangle at nk = 256 has an error of 5.3e-3 in 2.7 s,
where the grid at nk = 4096 has 4.1e-3 in 19 s, so the triangle now also
wins in wall time, by a factor 7, and at nk = 512 (1.3e-3 in 6.4 s) it
is better than any grid we ran. At T = 0 the same triangle takes 0.16 s.
At delta = 1e-2 the grid still wins: 9.3e-4 in 1.4 s at nk = 1024,
against 8.4e-4 in 17 s for the triangle at nk = 512, and 6.9e-5 in 4.5
s at nk = 2048, which the triangle does not reach below nk = 2048.

### Fewer pieces at finite temperature

We will now see where the pieces of the finite-T response come from, and
three changes that remove most of them without changing the quadrature.

Where are the pieces? On a simplex the T=0 response R0(x) at chemical
potential x only depends on x through the tiling of A and B. The knots of
the integral over x include every vertex energy of both bands, so inside
one piece of x between two knots every vertex keeps its side of x. When
both bands keep one side on the whole simplex, the whole simplex is in A
(every E_n < x < every E_m) or in B, or A and B are empty, and R0 does
not depend on x there. These are the pieces of x between the energy
ranges of the two bands on the simplex, and at T = 1e-2 they are most of
the window of 40 T. The quadrature still put 12 graded nodes on each of
them, and each node tiled the simplex again into the same piece. We now
evaluate such a piece of x once, with the weight f(a) - f(b), which is
the sum of the 12 node weights to 1e-15 (the rule is exact for the
degree-6 g'), and the pieces of A and of B of one pair are summed into
one evaluation each. Both kernels do this, and a test checks the single
evaluation against the node sum to 1e-13.

The pieces of x where the tiling moves are left. On the square lattice
at q = (0.3, 0.1), D = E_n - E_m is of order one on most simplices, so
usually only one band crosses the window there and the other band stays
on one side. The region is then the part E < x or E > x of the moving
band. In 2D the part with two vertices needs two triangles and the part
with one vertex needs one; in 3D the part with three vertices needs
three tetrahedra. The source kernel writes such a region as the whole
simplex minus the other part when the other part has fewer pieces. The
whole simplex goes into the single evaluation of A or B of the pair, so
per node only a corner is left. A corner also has a smaller spread of
D, so its pieces fall in finer bins and fewer of them are summed
exactly near each frequency. The reference kernel keeps the plain
tiling, so the agreement of the two kernels tests this one.

The third change is the case delta < T. The reference kernel adds to
the knots the energies where a cut through an edge meets the resonance
D = w of each frequency, so its nodes depend on the frequency. These
knots only fall in pieces of x where the tiling moves, and only for the
frequencies inside the range of D on the simplex. The source kernel keeps the shared nodes for all frequencies and adds, for
each such simplex and frequency, the quadrature on the refined pieces of
x minus the one on the shared pieces of x, evaluated directly. This is
the reference result to rounding: on the square lattice at nk = 256,
delta = 1e-3 and T = 1e-2 the two agree to 5e-15. The static response
at finite T (delta = 0, one frequency) goes the same way and costs what
the reference kernel costs, within 10 percent at nk = 384 on a desktop.

The pieces on the square lattice at nk = 512, 60 frequencies, counted
in the kernel (the exact near evaluations are summed over the
frequencies):

| case | pieces | per simplex | near evaluations |
|---|---|---|---|
| T = 0 | 2.4e5 | 0.45 | 2.8e5 |
| T = 1e-3, before | 7.2e6 | 13.6 | 8.3e6 |
| T = 1e-3, one evaluation per fixed piece of x | 3.7e6 | 7.0 | 3.5e6 |
| T = 1e-3, and the complement tiling | 2.6e6 | 4.9 | 1.6e6 |
| T = 1e-2, before | 4.0e7 | 77 | 5.1e7 |
| T = 1e-2, one evaluation per fixed piece of x | 1.0e7 | 19 | 9.5e6 |
| T = 1e-2, and the complement tiling | 6.9e6 | 13 | 4.1e6 |

The wall times on four cores of the shared workstation (load 5 to 8),
square lattice, q = (0.3, 0.1), 60 frequencies from 0.02 t to 6 t. Each
cell was run with the code before these changes ("before"), with the
first and the third change but plain tiling ("fixed x"), and with all
three ("all"), next to each other in one session. The table gives the
last of three passes; "before" and "fixed x" were also run in two
earlier passes. The first pass was up to 25 percent faster for both
codes at delta = 1e-3, and the old code at delta = T = 1e-2, nk = 1024
took 205 s in the first two passes and 157 s in the last; the other
cells agree between the passes to 5 percent. Before, delta < T went to
the reference kernel; we ran it at nk = 256 only, where it took 529 s
(first pass).

| nk | delta | T | before (s) | fixed x (s) | all (s) | T = 0 (s) | all / T = 0 |
|---|---|---|---|---|---|---|---|
| 256 | 1e-3 | 1e-3 | 4.7 | 3.0 | 1.9 | 0.31 | 6.0 |
| 256 | 1e-3 | 1e-2 | 529 (reference kernel) | 7.6 | 5.5 | 0.31 | 18 |
| 256 | 1e-2 | 1e-3 | 5.0 | 3.2 | 2.0 | 0.32 | 6.2 |
| 256 | 1e-2 | 1e-2 | 21.9 | 6.2 | 3.9 | 0.32 | 12 |
| 512 | 1e-3 | 1e-3 | 11.3 | 5.6 | 4.1 | 0.88 | 4.6 |
| 512 | 1e-3 | 1e-2 | not run | 18.3 | 13.5 | 0.88 | 15 |
| 512 | 1e-2 | 1e-3 | 11.5 | 5.6 | 3.9 | 0.96 | 4.1 |
| 512 | 1e-2 | 1e-2 | 62.4 | 15.1 | 10.6 | 0.96 | 11 |
| 1024 | 1e-3 | 1e-3 | 30.8 | 12.8 | 10.2 | 2.82 | 3.6 |
| 1024 | 1e-3 | 1e-2 | not run | 63.1 | 38.7 | 2.82 | 14 |
| 1024 | 1e-2 | 1e-3 | 32.4 | 12.9 | 10.6 | 3.03 | 3.5 |
| 1024 | 1e-2 | 1e-2 | 157 | 40.8 | 32.8 | 3.03 | 11 |

The T = 0 column is the same in the three codes to 5 percent, since T =
0 does not go through any of the changes. All of them give the same
numbers: the largest difference between the new and the old code, over
every cell, is 2.9e-13 of the largest abs(chi_0) (at delta = T = 1e-2),
and 4e-14 or less at T = 1e-3; at delta < T the new source kernel agrees
with the old reference kernel to 5e-15. So at T = 1e-3 the gain is a
factor 2.5 to 3, and finite T now costs 3.5 to 6 times T = 0, against 10
to 16 before. At delta = T = 1e-2 the gain is a factor 5 to 6, and
finite T costs 11 to 12 times T = 0, against 65 to 75 before. With
delta < T the gain is a factor 97 at nk = 256, and the cost is 14 to 18
times T = 0. The ratio to T = 0 falls with nk.

What about the larger cost at delta = 1e-2 than at 1e-3, which the
section on the cost of the dynamic response saw? In this session the two
broadenings cost the same at T = 0 (0.31 and 0.32 s at nk = 256, 2.8
and 3.0 s at nk = 1024) and at T = 1e-3 (11.3 and 11.5 s at nk = 512
with the old code). A profile at nk = 512 gives the same near
evaluations at both, 3.53e6 and 3.48e6, and the same time in them. So
the difference seen before was the load of the machine in a run that was
not interleaved, and not the Taylor branch.

We also tried to move work from the near evaluations to the moments, by
summing a bin by its moments from 2 b on instead of 3 b, with 60 terms
instead of 40. The near evaluations fall by a third and the moments
cost 55 percent more, and the total did not move, so we kept 3 b.

What is left at finite T is the 12 graded nodes on every piece of x where
the tiling moves. Fewer nodes, or fewer multiples of T among the knots,
would change the quadrature itself, and the section on finite
temperature above says that the shorter list of multiples loses 1e-6
relative; we did not do it. On a profile at nk = 512 and T = 1e-3 the
tiling now takes about half of the kernel, the moments and the near
pieces a quarter each.

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
  combinations that are not built raise;
- the source kernel gives the reference kernel, which evaluates every
  piece at every frequency, to 1e-10 of the largest value: at delta =
  1e-3 and at delta = 0, at T = 0, at finite T with delta = T and with
  delta < T (delta = 1e-3 at T = 1e-2 and 5e-3, the static response at
  delta = 0), on the square lattice, on a spinful honeycomb metal with
  the degenerate pairs of the spin copies, on a chain and on a cubic
  metal, and the names of the kernels come from their registry;
- a piece of x where the tiling does not move, evaluated once times
  f(a) - f(b), is the sum over its 12 graded nodes to 1e-13, for
  triangles and tetrahedra, in A and in B, at delta = 1e-3 and 0. Both
  kernels take this shortcut, so the agreement of the two kernels does
  not test it.

With these, `tests/chi` passes, 196 tests.

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
delta, so at delta = 1e-3 it needs 64 times fewer k-points than the grid,
and with the source kernel it is also 7 times faster in wall time at
equal accuracy.

The losses. Where T is above the energy spacing of the mesh the grid
converges exponentially: at T = 5e-2 and nk = 256 both take about 17 s,
but the grid is exact to 3e-9 and the triangle is off by 2e-5. The
triangle cost also grows with T, because the window of 40 T covers more
of the band, so on large meshes at high T it is the slower of the two. The dynamic response
at delta = 1e-2 is the same story: the grid reaches 1e-3 ten times
faster. The dynamic response at finite T costs about 10 times the T = 0
one, through the quadrature in x. In 3D the tetrahedra cost about 0.4 ms
per k-point at T = 1e-3, so nk = 64 takes 100 s. At a commensurate
nesting vector an even nk converges irregularly, and an odd nk has to be
used.

## The pair cutoff of the grid

The library grid of `chiAB_q` used to skip every pair with
abs(fac) < delta/100, with fac the Lindhard factor of `lindhard_pair`.
For a degenerate pair fac = i delta f', so the test dropped every
degenerate pair with abs(f') < 0.01 at any delta, and at the default
delta = 0.1 it also dropped every pair with abs(f_a - f_b) < 1e-3. On a
honeycomb and a square metal, spinless and spinful, at T = 5e-2 and 0.1
and nk = 24, this made the grid differ from the same sum without any skip
by 7e-7 to 2e-4 at a generic q and by 7e-5 to 2e-3 at q = 0, where every
pair is degenerate. The kernels now skip a pair only when
abs(fac) < 1e-14 delta (`chiAB.PAIR_TOL`). This is a bound on the pair
itself, since its term is at most abs(fac)/delta at every frequency, so
the grid now equals the sum without skip to 1e-15, and
`tests/chi/test_chi_grid_lindhard_sum.py` checks it to 1e-12. At low T
the new rule keeps few extra pairs: some 35 to 40 T away from the Fermi
level a filled occupation rounds to exactly 1 and an empty one, exp(-e/T),
falls below 1e-14 delta.
The three numba kernels and the jax pair plan use the same rule; the jax
one ran on jax's CPU backend only, not on a device. On a spinful
honeycomb metal at nk = 64 and 128, 50 energies, `mode="matrix"` and
`"trace"`, the wall time of `chiAB_q` moved by -8 to +8 percent in
interleaved runs on a shared machine, while two repeats of the same run
differ by up to 25 percent, so the change is below what this measurement
resolves. The kernels alone, at one k-point, take 0.9 to 1.3 times the
old time for this 4-band cell, and 1.5 times for a 64-band supercell at
T = delta = 0.1, where the most new pairs enter (1.05 to 1.4 at T = 0.01).
That cost is the pairs the old cutoff dropped although they contribute,
so a large cell at high T pays it in full.

### The degenerate pairs of the pair basis

The pair-basis kernel of `chitk/pairchi.py`, the route of every
interaction that couples different sites, skipped every pair with
f_n - f_m = 0 exactly and had no degenerate pair term. So the question is
whether the two routes give the same bare response where both are
defined. The pair operator c^dag_a c_b of one site has the response of the
projector |a><b| in `chiAB_q`, so we compare `pair_chi0` with `chiAB_q`
given those projectors, on the same mesh, the same q, delta and an
explicit T on both sides (T = None means T = delta in `chiAB_q` and a
step in `pair_chi0`). The models are spinful metals: a square lattice
with onsite energy 0.4 t and a honeycomb with 0.3 t, each without and
with an exchange field 0.3 t along z, at nk = 24 and T = 0.05.

At a generic q, (0.137, 0.211), the two routes agree to 7e-16 at every
frequency and both delta. At q = 0 every pair is degenerate, n = m and the
spin-degenerate partners, and the pair basis lost the whole Fermi-surface
term: at omega = 0 the difference is 0.205 on the square lattice, which is
its whole bare chi_0 there since it has one band per spin, 0.242 with the
exchange field, and 0.030 and 0.060 on the honeycomb, the same at
delta = 1e-2 and 1e-1. At finite omega it decays as delta/omega, 4.0e-2 at
omega = 5 delta and 2.0e-3 at omega = 100 delta, which is exactly the
missing term f' i delta/(i delta - omega). At the nested q = (1/4, 1/4) of
a nk = 8 mesh at T = 0.1, where E(k) = E(k+q) on mesh points, the routes
differed by 7e-3 to 0.25. So the pair basis was inconsistent with the site
basis, and a metal on the pair route had no Pauli term in its static
uniform response. The public route sets T = delta, so the missing term
was finite on every public call in a metal.

`_accumulate` now applies the rule of `chiAB.lindhard_pair`: a pair with
abs(e1_n - e2_m) < `chiAB.DEGENERATE_TOL` gets i delta f'(e1_n) in place of
f_n - f_m and a zero energy difference, and a pair is skipped only when
that factor is exactly zero, so the pair-basis rule for the other pairs
did not change. For a step occupation f' is zero. The jax kernel
(`pairchijax.band_pair_plan`) builds its pair list from the same factors,
`pairchi.pair_factors`; it ran on jax's CPU backend only, where it agrees
with the numba kernel to 1.3e-15 on the four metals at q = 0 and at the
generic q. After the change the two routes agree to 1.6e-15 in every
case above, and `tests/chi/test_pair_chi0_degenerate_pairs.py` checks it
to 1e-12, at q = 0, a generic q and the nested q, static and dynamic,
plus the public `h.get_spinchi_full(RPA=False)` against `pair_chi_rpa`
with a zero interaction.

The Goldstone mode does not move, and it should not. The field of the
uniform spin rotation, the generator in the pair basis, has the matrix
element (E_m - E_n) G_nm between two states, which is zero on a
degenerate pair, so the new term never acts on it. We measured
||M v||/||v||, with M = 1 + K chi_0 at q = 0 and omega = 0 and v = K
times the rotation of the pair expectation values, on ferromagnetic
metals ordered by U and V1: a doped chain (two-site cell, U = 2, V1 = 0.5,
filling 0.3, nk = 40, also tilted by 0.37 rad), a square lattice (U = 4,
V1 = 0.5, filling 0.3, nk = 16) with the mean field at T = 1e-7 and at
T = 2e-2, and a metallic 120-degree spiral on the 3x3 triangular cell
(U = 8, V1 = 0.5, filling 0.4, T = 0.05, nk = 3). The residual is the
same before and after to every printed digit: 1.681e-1 and 1.705e-2 at
delta = 1e-2 and 1e-3 on the chain, 8.551e-2 and 8.582e-3 on the square
lattice at T = 2e-2, and 1.7262701793e-4 on the spiral at delta = 1e-3,
proportional to delta in each case, while the new term moves chi_0 of
the spiral by 1.3e-2 out of 9.3e-2. What does move is the rest of the
spectrum of M: on the chain at T = 2e-2, which is paramagnetic, its
smallest eigenvalue goes from 0.683 to 0.262, since the Stoner
denominator now has the Pauli term in it.

## What is left open

- The cost of the dynamic response at finite T, now 3.5 to 6 times the
  T = 0 one at T = 1e-3 and 11 to 18 times at T = 1e-2. What is left is
  the 12 graded nodes on every piece of x where the tiling moves; fewer
  nodes or fewer multiples of T would change the quadrature, and were
  not tried. The tetrahedra in 3D have not been timed with the source
  kernel.
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
