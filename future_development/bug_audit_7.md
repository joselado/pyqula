# Bug audit 7: a second reading of the commits of 4 to 13 November 2026

This record holds what a second reading of every commit from `ead1bd4` to
`9bcf52d` (178 commits, 4 to 13 November 2026) found: potential bugs and
optimization opportunities. The commits were split into thirteen groups by
topic, one reader per group went through the diffs, and a second agent per
group tried to refute each finding of medium or high severity from the code at
`9bcf52d`. Nothing was run, since the machine allowed one Python process at a
time and it was taken by other work. So every entry below rests on reading the
code, with the file and line at `9bcf52d`, and none of them is a measurement.
Where a number appears (a memory size, a count of diagonalizations) it is
counted from the code.

The repairs came on 13 and 14 November 2026, in seven groups by the files they
touch, each one repaired by one agent and checked by a second one. What each
group did is said at the end of its entries, and the measurements are in the
section "Repairs of the seventh bug audit" of the roadmap each entry names.

## Status

**Repaired, except where an entry says otherwise (14 November 2026).** Eight of
the nine potential bugs are repaired, and the ninth in part: the direct SHG
simplex sum decides its Taylor branch per frequency again, the default source
kernel does not. The seventeen optimizations are applied, two of them in part
(the 3D triangle self-energy and the finite-temperature weights of the triangle
density matrix). The repairs of the code bugs have tests that fail on the code
at `9bcf52d`; the example and the notes were corrected by hand.
Each optimization was timed with the old and the new code back to back, the
large cases on a compute node, and gives the same numbers to rounding. The repairs
found thirteen more problems, listed in "What the repairs found", and the last
section lists what is left open. The repairs were committed on 14 November
2026 (`8518a8e`), and after them the whole suite passes on a compute node, on
the old and the new code, with the tests marked slow. The choices that the
repairs left open were decided the same day, see "The decisions of 14 November
2026": eight of them changed the code, and one of those repaired a bug of the
Kondo-lattice loop with a complex hopping between the moments.

## Potential bugs

The two of high severity first, then the medium ones.

- **The LDOS with an operator depends on the basis inside a degenerate
  multiplet** (`dostk/simplexdos.py:398-409`, from `2c48c94`). The triangle
  and qtci weight of a band is $|u_{n,i}|^2\langle u_n|O|u_n\rangle$, formed per
  state, and the multiplet average comes after it. A sum of a product of two
  expectation values over a multiplet is not a trace over the multiplet, so it
  depends on the basis `eigh` returns. On a $PT$-symmetric Néel pair the $s_z$
  basis gives $2a-1$ and a rotated basis gives 0, while the invariant
  $\mathrm{Re\,Tr}(P_M P_i O)$ gives $2a-1$ in both. The module header
  (lines 39-42) says the multiplet average removes the basis dependence, which
  holds for the DOS, the operator, the Green's function and the plain site
  weights, but not for a site weight times an operator. The same product is in
  `ldos.py` (`mode="arpack"`, lines 424-433) and in `ldostk/ldoswaves.py`, so
  the grid mode has it too and it predates the triangle code. The repair needs
  a convention from the maintainer.
  Repaired on 13 November 2026: the LDOS with an operator is the local matrix
  element $\mathrm{Re}\langle u_n|P_iO|u_n\rangle$ in every mode of
  `h.get_ldos()` and `h.get_multildos()`, the convention of `mode="green"` and
  of the KPM map. Rejected: the restricted form $\langle u_n|P_iOP_i|u_n\rangle$
  (it loses the sum rule and depends on the unit cell for the valley operator),
  the product of the two multiplet averages (0 for the Néel pair), diagonalizing
  $O$ inside each multiplet (discontinuous), and a keyword that keeps the old
  product. The embedding LDOS follows the same convention since 14 November, see
  "What the repairs found". Record in `qtci_dos_green.md`.
- **The exchange of the 36-site spinon example is not periodic in its cell**
  (`examples/spinon/spin_liquid_large_cell/main.py:41`, from `cc23757`). The
  modulation is a cosine of the Cartesian $x$ with period $L=6$, but a
  translation by the second supercell vector shifts $x$ by $-L/2$ and flips
  the cosine. So on every bond with an odd second index $J[d] \neq J[-d]^T$.
  With `hold="pattern"` each direction enters the mean field on its own, so
  the converged spinon Hamiltonian is not Hermitian, and the RPA vertex $J(q)$
  is not Hermitian either. The numbers that `spinon_response.md` records from
  this example come from that model.
  Repaired: the example modulates the fractional coordinates of the bond along
  both cell vectors, from an origin that is not a symmetry point of the
  lattice. A modulation centered on the cell keeps three symmetries and gives
  only 12 different sites; this one gives 36 different sites and 36 different
  moments. Every stiffness stays positive and the transverse peak stays at
  $q=2b$. The new numbers are in `spinon_response.md`.
- **No check that a spinon exchange is reciprocal** (`spinon.py:135`, from
  `cd5a60d`). Nothing in the bond path compares $J[d]$ with $J[-d]^T$, so the
  example above passed silently. The Hartree-Fock field averages the two
  directions, but `hold="pattern"` and the recorded RPA vertex use one
  direction each. A check of $\max|V[d]-V[-d]^T|$ where the exchange is built
  would have raised.
  Repaired: `SpinonHamiltonian.get_mean_field_hamiltonian` raises a
  `ValueError` for both decouplings and every hold, with a relative tolerance of
  $10^{-8}$, and the message names the worst bond. The old example raises.
  Open: the generic `Vr` and `Jr` of `VJinteraction` and `Vinteraction`
  (`densitydensity.add_pair_interaction`, shared with the BSE and the density
  response) still averaged a function that is not reciprocal, with no error.
  Decided on 14 November 2026: they raise too, see "The decisions of 14
  November 2026".
- **The total energy of `hold="pattern"` on inequivalent bonds is not the one
  of the docstring** (`spinon.py:372`, from `cd5a60d`). The code returns
  $-2A+B$, with $-2A$ the band energy at the held hoppings and
  $B=\sum_b gJ_b|\chi_b|^2/2$ the double-counting term, while the docstring
  promises $-B$. Since $A\le B$, the two agree only when $\chi_b$ is
  proportional to $J_b P_b$ bond by bond, so on equivalent bonds.
  The same holds for `hold="phases"` when $\chi_{ij}$ is not aligned with the
  ansatz phase. The energies recorded so far are on equivalent bonds, so no
  recorded number moves, but any energy comparison on a modulated cell does.
  Repaired: the energy is the bond functional
  $E_{\rm band}+\sum_b2|t_b|^2/(gJ_b)$ for the free loop and both holds, which
  is stationary in the held amplitude. At a free fixed point it is the old
  number. The band energy is the one the engine computes before its double
  counting (`scf.band_energy` of `scftk/spinspin.py`), so no diagonalization is
  added. Five tests fail on the code before.
- **The cotunneling operator with `J=` given is a factor 2 off**
  (`kondotk/cotunneling.py:73`, from `80183d9`). It returns
  $t_f\,\overline{V}/J$ with $J$ the coupling of the loop, which is half of
  the mean-field image of $(\sigma\cdot S)c$ and half of what
  `compositespectrum.py` uses for the same operator. So the $ff$ part comes
  out 4 times too small and the $cf$ part 2 times. The default `J=None` is
  right, no example or guide snippet passes `J`, and
  `test_cotunneling.py:177` restates the formula, so it cannot catch it.
  Repaired: the amplitude is $2t_f\overline{V}/J$. A new test compares it with
  $-\sum_s\langle f_s^\dagger c_s\rangle$ of the converged state, to
  $9\times10^{-9}$, and fails on the old code by 0.38; another puts a phase on
  each f fermion, which pins the conjugate. The guide and the reference page
  say $2t_fV_j^*/J$. Record in `kondo_tunneling.md`.
- **`chern_mod_n` does not follow the order of $a_1$ and $a_2$**
  (`topologytk/symmetryeigenvalues.py:197`, from `b4973cb`). The rotation is a
  fixed counterclockwise one in Cartesian space and the invariant momenta are
  the same physical set whatever the labelling, so the result does not change
  when $a_1$ and $a_2$ are swapped, while `h.get_chern()` flips. On a
  left-handed pair of lattice vectors it returns $-C \bmod n$ for $n=3,4,6$,
  although the docstring, the guide and `topological_invariants.md` say it has
  the sign of `h.get_chern()`. Every test lattice is right-handed. A repair
  multiplies by the sign of $(a_1\times a_2)_z$ before the modulo.
  Repaired that way. A built-in lattice was hit: `honeycomb_lattice(4)`, (7),
  (9) and (13) are left-handed, and the Haldane model on `honeycomb_lattice(4)`
  gave 1 for $n=3$ against `get_chern()=-1`. Record in
  `topological_invariants.md`.
- **The SHG simplex kernels choose the Taylor branch from the spread against
  `delta` alone** (`conductivitytk/shgsimplex.py:266` and `:508`, from
  `6611de1`). This is the rule that `baaad2d` removed from `kubosimplex`,
  because a nearly flat simplex far from resonance loses precision in the
  plain divided difference. The SHG kernels have no per-simplex rounding
  budget like `kubosimplex._vertex_error`, and `6611de1` replaced a
  per-frequency test of `b9495a2`, so the direct kernel regressed too.
  Repaired in part. The direct sum (`kernel="simplices"`, the reference of the
  tests) decides per frequency again and keeps the first-order term of the
  series: its worst single-resonance error against mpmath goes from
  $5.5\times10^{-4}$ to $9\times10^{-15}$, and two new tests fail on the old
  code. It is about 4 times slower, but only the tests use it. The review found
  one more case of the same loss: an exactly flat transition energy kept only
  the leading term of its series in both kernels, and the rounded mean of equal
  nodes is not the node, so an exactly flat band lost $1.1\times10^{-9}$ of the
  peak at `delta=1e-7`. Both kernels keep the first-order term there now, and a
  new test fails on the old code. The default source kernel keeps its rule on
  `delta`, and with it the loss on nearly flat bands at small `delta`: up to
  $5\times10^{-9}$ of the peak at `delta=1e-6` and $2\times10^{-7}$ at
  `delta=1e-7`, which the docstring of `shg_conductivity` now states. A cure
  that sums the cancelling simplices per frequency was built, but at the
  budget that cures the flat bands it costs 2.6 times at `delta=1e-3` for no
  gain there, so it was not shipped. The route to the root of it is the port of
  the energy bins and the Newton-table split of `kubosimplex.py`; the
  alternatives are in `qtci_shg.md`.
- **The k-batch of the nonlinear dc mesh loop has a floor of 64 points**
  (`conductivitytk/nonlineardc.py:327`, from `3b3f280`). `max(64, ...)` wins
  over the 2e8-byte budget above about 40 bands. Counted from the code, one
  batch needs 1.1 to 1.25 GB at 100 bands and passes 6 GB at about 220 bands,
  so a slab of a few hundred bands runs out of memory with no guard.
  Repaired: the floor is one k-point. At 128 bands the peak goes from 2.25 GB
  to 0.50 GB, in 5.0 s before and 3.9 s after. The same fixed batch was in
  `nonlineardcfit.convergence_radius` (4096 k-points, over 5 GB at 200 bands),
  which now uses the same budget. Record in `nonlinear_response.md`.
- **The nonlinear Drude term of a 3D Hamiltonian sweeps only $x$ and $y$**
  (`conductivitytk/nonlineardrude.py:517-526`, from `5b55594`). Since the 3D
  lift the routine accepts a 3D Hamiltonian, but the fields and the currents
  still run over $x$ and $y$ only. For a spin splitting odd in $k_z$ every
  component comes out exactly zero, with no error.
  Repaired: in 3D the fields and the currents run over $x$, $y$ and $z$. On a
  cubic magnet with a $d_{xz}$ splitting the old sweep gave at most
  $2.8\times10^{-17}$, where $\sigma^{x;z}=4.4\times10^{-3}$; the test uses the
  $y\leftrightarrow z$ symmetry of the cubic lattice. Record in
  `nonlinear_spin_transport.md`.

## Optimization opportunities

The two of high severity first. Each one keeps the result unchanged, by the
reading of the second agent; none is timed. The timings of the repairs are on a
shared machine with two threads, the old and the new code back to back, unless
an entry says otherwise, so the ratios are what matters.

- **The trace mode of the triangle susceptibility carries every site
  component to the end** (`chitk/simplexchi.py:1477`, `1395-1408`, `1539`,
  from `2d23f85`). `mode="trace"`, the mode of `h.get_chi()`, sends one
  component per site through the whole kernel and averages them at the end.
  Every step after the matrix elements is linear in them, so averaging them
  first is exact and removes a factor of the number of sites from all the
  component work.
  Done. On the spinful 4x4 honeycomb cell (64 bands, `nk=6`) one $q$ goes from
  28.7 s to 9.9 s and the peak from 717 MB to 560 MB, agreeing to
  $4\times10^{-16}$. Three more pairs in the review gave 22.6 s to 9.5 s, so
  the gain is 2.4 to 2.9 times. Record in `tetrahedron_susceptibility.md`, with the timing
  of the next entry on a compute node.
- **`localspinchi.spin_chi` keeps one array per k-point**
  (`chitk/localspinchi.py:207-213`, from `c0922af`). A list of $N_k$ arrays of
  shape $(n_\omega,3m,3m)$ is built and `np.mean` stacks a second copy, so the
  memory grows with $N_k$ (and with $N_q$ for `q=None`) where a running sum
  needs one array. At 196 sites on the spinful route one k-point is about
  450 MB, so 9 k-points pass 6 GB. `kondotk/momentchi.py:118-122` has the same
  pattern.
  Done, in both. On 49 sites (9 k-points, 81 energies) the peak goes from 848
  to 475 MB. On a compute node with one thread, 98 sites in a field that mixes
  the spins, 49 k-points and 61 energies: 8.1 GB before and 1.3 GB after, at
  the same 1335 s, with the same arrays to the last bit. Record in
  `spinon_response.md`.
- **The local moment susceptibility diagonalizes the mesh once per q**
  (`kondotk/momentchi.py:119`, from `e5e8a5f`). With `q=None`,
  `moment_spinchi` calls `bare_spin_chi` for every q of the mesh, which gives
  $2n_k^{2d}$ diagonalizations for $n_k^d$ distinct matrices: about
  $2.6\times10^7$ against 3600 at the default `nk=60` in 2D. A cache with an
  index shift is exact, but the pair loop of `_accumulate` still runs
  $n_k^{2d} n_\omega$ times, so only a route through spectral matrices changes
  the scaling, and that one changes the result at the level of the binning.
  Done with the cache: every point of the mesh is diagonalized once. On the
  square Kondo lattice at `nk=8` it gives 0.70 s against 0.22 s with one energy
  and 7.4 s against 7.1 s with 200, where the pair loop dominates. The binned
  route is not built, by decision of 14 November 2026. Record in `spinon_response.md`.
- **The 3D triangle self-energy diagonalizes the mesh again at every
  energy** (`greentk/selfenergy.py:14-34`, from `95ccd07`).
  `bloch_selfenergy(mode="triangle")` calls `green_triangle` with one energy,
  which diagonalizes the $n_k^3$ mesh each time, although `green_triangle`
  serves any list of energies from one diagonalization. Every energy sweep
  pays this: `bloch_selfenergy_batch`, `Embedding.get_gf_exact`, `multildos`,
  the density-matrix contour and `get_dos(mode="Green", gmode="triangle")`.
  Done in part. `bloch_selfenergy_batch` serves a sweep from one
  diagonalization, and with `gf_mode="triangle"` `Embedding.multildos`,
  `Embedding.multidos` and `embedding.dos_impurity` go through it: 1.2 to 1.3
  times faster on four-orbital diamond at `nk=40` and 2.6 to 2.7 times on a
  16-orbital cubic supercell with a modulation and a defect at `nk=24`
  (`multildos` from 20.1 s to 7.7 s), the same maps to $10^{-15}$. Open: the
  two-line edit of `get_dos(mode="Green", gmode="triangle")`, which waits
  because `dos.py` holds other uncommitted work (it was checked on a copy, to
  $7.6\times10^{-16}$), and the density-matrix contour of a triangle embedding,
  which evaluates complex energies on an adaptive contour and cannot use the
  triangle mode at all. Record in `qtci_dos_green.md`.
- **The 3D triangle Green's function keeps every band projector of the mesh**
  (`dostk/simplexdos.py:405-412`, from `95ccd07`). At the default `nk=100` of
  `bloch_selfenergy` this is $10^6 n^3$ complex numbers, and the list of
  batches and their concatenation are alive at once, so the peak is about
  2 GB at $n=4$ and 16 GB at $n=8$. Keeping the eigenvectors, or summing over
  slabs of the mesh, gives the same matrix.
  Done with the eigenvectors. On a compute node, 8 orbitals at `nk=100`: 88.8 s
  and 15.8 GB before, 29.6 s and 2.0 GB after. Record in `qtci_dos_green.md`.
- **The finite-`delta` simplex sum recomputes the vertex average at every
  energy** (`dostk/simplexdos.py:216-228`, from `82804d4`). The average does
  not depend on the energy. The same loop is a `prange` over energies, so a
  call with one energy, as `bloch_selfenergy` makes, runs on one thread;
  a `prange` over simplices, as in `_simplex_dos0`, fixes both.
  Done: the simplex average of the resolvent goes onto the vertices first, and
  the simplices are cut in slices when there are fewer bands times energies
  than threads. With 8 threads, `nk=162` and 4 energies: 24.9 s before and
  9.0 s after. Record in `qtci_dos_green.md`.
- **`_Fields` diagonalizes in batches of $2^{15}$ k-points whatever the number
  of orbitals** (`dostk/simplexdos.py:359`, from `82804d4`). Only the site
  kind scales the batch with $n$. At $n=100$ one batch of the DOS is about
  5 GB, and twice that for the operator and Green's function kinds.
  Done: the batch shrinks as $1/n^2$ for every kind. On a compute node, 98
  orbitals at `nk=200`: 6.1 GB before and 0.54 GB after, 23.8 GB and 0.73 GB
  with an operator. Record in `qtci_dos_green.md`.
- **The grid optical conductivity holds the velocity matrices of the whole
  mesh** (`conductivitytk/kubo.py:333-334`, `369-436`, from `89ef3af`). Only
  the diagonalization runs in batches; the velocities and the $(N_k,n,n)$
  temporaries are built for the whole mesh before the k-sum, about $90n^2$
  bytes per k-point, although the sum is linear and can be accumulated batch
  by batch.
  Done: the k-sum goes one slab at a time. At 3D `nk=128`, with the k-mesh
  built as an array in both trees, the peak goes from 1219 to 595 MB for the
  conductivity and from 2136 to 876 MB for the Drude weight, at the same wall
  time, agreeing to $2.4\times10^{-13}$. With the old list builder of the 3D
  mesh, which was the peak from about `nk=128` on, the review measured 727 to
  566 MB at `nk=96`; that builder is now an array too, see "What the repairs
  found". Record in `qtci_optical_response.md`.
- **The 3D tetrahedra are rebuilt as an index array at every call**
  (`dostk/simplexdos.py:286-299`, from `05df314`). The $(6n_k^3,4)$ int64
  array costs 192 bytes per k-point and about three times that at its peak,
  and it is rebuilt by `simplex_weights`, `simplex_charge_dos`,
  `fermi_from_energies`, once per Fermi level of the anomalous Hall
  conductivity and twice per qtci or tetrahedron mean-field iteration. The
  notes give this as the reason a finer mesh was not run.
  Done: the tetrahedra are built in place, once per call, and the anomalous
  Hall conductivity builds them once for all Fermi levels (about 10% per level,
  the same result to the last bit). A mean-field iteration shares them between
  the Fermi search and the density matrix, see the entry on the two
  diagonalizations below. Record in `tetrahedron_density_matrix.md`.
- **The matrix elements of the triangle susceptibility are dense products**
  (`chitk/simplexchi.py:1525-1528`, from `2d23f85`). Every operator that
  `chiAB_q` passes is a site projector times a local operator, with only the
  rows of one site, so restricting the product to those rows is exact and a
  factor of the number of sites cheaper. The products are also rebuilt in
  every component block of `_chi_sources`.
  Done, with each component block building only its own operators. On the
  200-band 10x10 cell one $q$ goes from 26.1 s to 14.5 s after the change of
  the trace mode. Both changes together, on a compute node with 4 threads: one
  $q$ from 128.6 s to 11.9 s, the local response from 1119 s to 93 s, the peak
  from 3.2 GB to 1.7 GB, agreeing to $7\times10^{-15}$. Record in
  `tetrahedron_susceptibility.md`.
- **The finite-temperature weight of the triangle density matrix uses about
  200 quadrature nodes per simplex and band** (`dmtk/simplexdm.py:293-313`,
  from `4b7ac92`). Each node allocates arrays, and in 3D it calls
  `np.linalg.det`. The default $T=10^{-7}$ of the mean field always takes this
  branch. A closed form, with an exact moment formula when no vertex is in the
  window, removes most of it.
  Done in part: the weights are in closed form, with no allocation and no
  determinant, and the DOS comes from the same pass. On a cubic band at $32^3$
  this is 16 to 19 times faster at $T>0$, the same to $3\times10^{-20}$, and
  the review measured 2.25 s against 0.122 s at $T=10^{-7}$, 18 times. The
  moment formula was not built, and the 208 nodes of the charge and the weights
  are now the main cost of a 3D iteration at $T=10^{-7}$. Record in
  `tetrahedron_density_matrix.md`.
- **Every triangle mean-field iteration diagonalizes the mesh twice**
  (`dmtk/simplexdm.py:689-749`, from `4b7ac92`), once for the Fermi level and
  once for the density matrix, and three times when the eigenvectors do not
  fit in memory, since the first pass throws them away. On the CPU the
  eigenvalue pass calls the full `eigh` anyway.
  Done: the Fermi search keeps the eigenpairs and the simplices when they fit
  in `max_memory_gb`, and the density matrix of the same iteration reads them,
  so an iteration diagonalizes the mesh once. On a spinful 3x3 honeycomb cell
  at `nk=48` one iteration goes from 1.12 s to 0.85 s. Record in
  `simplex_fermi_search.md`.
- **The density matrix is contracted once per hopping direction**
  (`dmtk/simplexdm.py:743-745`, from `4b7ac92`), although the weighted
  projector does not depend on the direction. This one is only plausible: the
  flop count drops, but the repair trades one large product per direction for
  many small ones, so it needs a timing.
  Done and timed: with 7 to 9 directions one product for all of them is 1.3 to
  1.4 times faster; with one direction the old product stays, since it is 1.15
  times faster there. That branch selects on the number of directions, not on
  a size, and it is for the maintainer to look at. Both were measured at two
  threads only: the batched product of the new route runs on one thread, where
  the old one spread over the BLAS threads, so with many threads the gain may
  shrink. A script for that measurement exists, but it needs several BLAS
  threads, which the cluster runs of this audit kept at one, so it was not run.
  Record in `tetrahedron_density_matrix.md`.
- **Every nonlinear dc call builds every vertex up to third order**
  (`conductivitytk/nonlineardc.py:290-356`, from `3b3f280`). The third
  derivative is never read, and the Berry curvature dipole reads only the
  first derivative and one of the three blocks. Building what the term reads
  cuts the batched products by 1.4 for `term="all"` and by about 4 for the
  dipole.
  Done: each term builds only what it reads. On an 18-band Haldane cell at
  `nk=40` the dipole goes from 1.24 s to 0.53 s, the Drude term from 1.29 s to
  0.57 s and `term="all"` from 1.20 s to 0.91 s; the review measured 3.4, 2.7
  and 1.5 times, with the same output byte for byte. Record in
  `nonlinear_response.md`.
- **The SHG triangle and qtci fields build all 19 derivative operators per
  k-batch** (`conductivitytk/shgsimplex.py:887`, from `b9495a2`), while the
  kernel reads only those of the requested components: 3 of 19 for
  `components="xxx"`, and at most 9 for a planar 2D model, whose $z$
  derivatives vanish.
  Done: the fields build only the vertices the components read, and a
  derivative along an axis with no bond component stays an exact zero. On
  gapped honeycomb, `xxx` at `nk=256`: 3.33 s before and 1.42 s after. Record in
  `qtci_shg.md`.
- **A bond spinon state in a collinear field takes the general spinful
  route** (`chitk/localspinchi.py:134-155`, from `cd5a60d`). With $S_z$
  conserved, most matrix elements are zero, so the contraction does $9m^2$
  work per pair where about $m^2$ is needed, and the diagonalization of the
  two spin blocks costs 4 times less.
  Done: a Hamiltonian that does not mix the spins diagonalizes the two spin
  blocks apart. At 196 sites in a field along $z$: 27.1 s and 1.71 GB before,
  4.0 s and 0.78 GB after, agreeing to $2\times10^{-15}$. Since the decisions
  of 14 November 2026 a field in any direction takes this route, in its own spin
  frame. Record in `spinon_response.md`.
- **A k-dependent operator is rebuilt once per occupied state**
  (`vev.py:86`, `spectrum.py:209`, from `01cbbe6`). For valley, valley_x,
  valley_y and velocity, `kresolved_orbital_vev` and `ev2d` rebuild the Bloch
  matrix of the operator for every state, and with `delta=` they also run one
  `eigh` per state. `bandstructure.get_bands_nd` has the same pattern.
  Done: the four operators keep the matrix of the last k-point, as one object
  with its k-point, and hand out a copy. On a 162-site honeycomb cell
  `h.get_vev("valley_x",nk=10)` goes from 4.38 s to 1.08 s, `"valley"` from
  6.05 s to 1.14 s, `"velocity"` at `nk=4` from 5.46 s to 0.50 s, and with
  `delta=0.1` tau_z goes from 17.6 s to 0.40 s, the same results byte for
  byte. Record in `inplane_valley.md`.

## Smaller findings, confirmed and rated low

What was done with each one is at the end of its line.

- `symmetryeigenvalues.py:295`: $\kappa_1$ of the Z4 indicator depends on the
  inversion center when a weak index is nonzero (it changes by $2\nu_i$ mod 4
  for a shift by $a_i/2$), while the guide, the notebook and one test quote
  its values for the weak and strong phases as properties of the phase.
  Documented in the docstring, the guide, the reference page and the notebook,
  and the test checks the rule and the sign by two independent routes.
- `symmetryeigenvalues.py:436`: `corner_charge` with $n=6$ checks the filling
  at $\Gamma$ and M but not at K, so a metal near K returns a number. Repaired,
  it raises.
- `tests/topology/test_symmetry_eigenvalues.py:377`: the Kekulé corner charge
  test uses the default $n$, so it cannot tell the $C_6$ index from a
  fallback to $C_2$. Repaired, it passes `n=6`.
- `conductivity.py:174`, `:382`: the docstrings say the triangle weights are
  interpolated linearly inside a simplex; the code uses the vertex average.
  Repaired.
- `conductivity.py:183`: the docstring still describes the 3D source kernel
  of `7b27825`, before the split tetrahedra. Repaired.
- `conductivitytk/kubosimplex.py:1152`: the velocity elements of the whole
  mesh stay alive through the source build. Repaired: 237 to 178 MB on a
  32-band cell, the same result to the last bit.
- `tests/dos/test_dos_simplex.py:125`: the $s_z$-weighted DOS test accepts
  both signs. Repaired.
- `tests/conductivity/test_kpm_conductivity.py:56`: the off-diagonal KPM
  absorptive tensor (the circular dichroism) is never tested. Tested now; the
  code was right, and a copy with the indices transposed fails the test.
- `kpm_optical_conductivity.md:265` and `README.md:392`: two statements that
  later commits made stale (the error for an unknown integration, the gain
  factors of the triangle DOS). Both corrected.
- `kubosimplex.py:1224`: the commit message of `b3569bc` says the stacked
  TCI reaches the per-field error; its own table gives 4 to 18 times larger.
  Left: only the commit message is wrong, and no tracked text repeats it.
- `simplexdos.py:595`: the stacked TCI normalizes its error by the largest
  field, so the site weights inherit a tolerance set by the energies
  (plausible, not checked by a run). Confirmed by a run and repaired in part:
  the qtci LDOS measures the energies from their mean over the Brillouin zone,
  and `kubosimplex.sigma_qtci` divides the transition energies and the weights
  each by its own maximum, which a change of the unit of length showed (8e-3
  before). The rescaling of every field to order one in the qtci DOS and LDOS
  is not done. The pivot search of the cross interpolation is not seeded, so
  the cost of the per-family scale was settled with seeded repeats: the error
  is the same on honeycomb, about 2 times lower on MoS2, and about 10 times
  lower on a metal at $T=0.002$; on a disordered 2x2 supercell it is the same
  on one disorder and about 2 times larger on another, and the time there is
  1.4 times longer in two pairs and not settled in the seeded runs. Record in
  `qtci_optical_response.md`.
- `tests/chi/test_chi_triangle.py:52`: no multiband test compares the
  triangle matrix elements with an independent route. Tested now; the code was
  right, and four wrong copies (a transpose dropped, $U^T$ for $U^\dagger$,
  $\chi_{ij}$ for $\chi_{ji}$, A for B) fail the test. The review added a
  test with two components per block of the source kernel; it is the only one
  that fails on a wrong map from the components to the operators of a block.
- `qtci_optical_response.md:934`: the budget table has a second row above
  4e-14. Corrected.
- `simplexdm.py:645`: `_Mesh` batches $2^{14}$ k-points whatever the matrix
  size. Repaired: the batch is about 64 MB per array, and at $n=50$, `nk=128`
  the peak goes from 2.27 GB to 1.16 GB.
- `densitydensity.py:740`: the comment says the three solvers share one
  convergence criterion; they use three norms. Corrected.
- `superscf.py:433`: `ShellFilling.fill` counts the charge of the emptied
  shell twice in its root equation, which matters when the shell vectors
  carry a fractional charge; `superscf.py:447` and `:353`, an einsum without
  `optimize` and shell vectors kept for the whole mesh. The root equation is
  repaired (0.35 electrons instead of 0.6 on a test site, and no fill at
  $N_0=0.4$, before), and the einsum is one matrix product (3.3 s to 0.53 s).
  The shell vectors of the whole mesh are kept, by decision: dropping them
  needs a second pass over most of the mesh at $T>0$.
- `nonlineardc.py:72`: the sign of the Stark shift in the note on the
  dispersion current is opposite to the one used at third order in the same
  module. Corrected, in the module, `conductivity.py` and
  `nonlinear_response.md`.
- `nonlinear_spin_transport.md:178` and the README entry: still say the
  nonlinear Drude term is 1D and 2D only. Corrected.
- `photocurrent.py:114`, `:162`, `:163`: per-triplet commutators rebuilt,
  fixed batches of 64 k-points, and unused higher derivatives in the
  injection current. Repaired: the injection current builds only the velocity,
  shares its commutators and sizes its batch by bytes, 2.0 to 2.7 times faster.
- `spinon_response.md:25`: the formula of the held amplitude is off by a
  factor $-g/2$ and a conjugation from the code. Corrected.
- `localspinchi.py:218`: the RPA dressing of a spin-degenerate state works on
  the $3N\times3N$ matrix instead of one $N\times N$ block. Repaired: 22.6 s to
  13.2 s at 196 sites; a vertex that tells the components apart keeps the full
  matrix.
- `spinonspectrum.py:165`, `momentchi.py:142`: a full $2N\times2N$
  diagonalization where one spin block is needed, and a $3m\times3m$
  contraction where the routine returns only $\chi^{+-}$. Repaired: the Kondo
  spectrum of a degenerate state from one spin block, 3.9 s to 2.7 s at 196
  sites, and the bare moment response from $S^+$ alone, 2.4 s to 0.93 s. The
  same half block in the k-loops of `cotunneling.py` and of the composite
  spectrum was not claimed by the finding; it was done in the decisions of 14
  November 2026.
- `inplane_valley.py:95`: a vertical first-neighbor bond gives 0/0 and NaN
  entries with no error. Repaired: the vertical bonds are left out, and the
  operators of a stack are those of its layers pulled apart.
- `sharpen.py:11`: `delta=` of the in-plane valley operators is ignored on a
  sparse Hamiltonian. Repaired, for tau_z too.
- `tests/kpmchern/test_kpm_chern_marker.py:11`: quotes the textbook Haldane
  gap $6\sqrt3 t_2$ and critical mass $3\sqrt3 t_2$; with `add_haldane(t2)`
  they are $9t_2$ and $4.5t_2$. Corrected, with the example and
  `sparse_kpm_mean_field.md`.
- `tests/spectrum/test_ev2d_nambu_and_names.py:60`: `nk=2, nsuper=1` puts the
  four mesh points on one physical k-point, so the test cannot see the
  k-point of the state. Repaired with `nk=4`; a copy of the code that applies
  the operator at the first mesh point fails it.

## What the repairs found

- **The embedding LDOS with an operator was one-sided**
  (`embeddingtk/ldos.py:39-42`, `embedding.py:330-331`): $-\mathrm{Im\,diag}(OG)/\pi$
  instead of the convention of the first bug above. For $s_x$, $s_y$ and
  $s_z$ the two agree per site to $10^{-17}$; for a bond operator they differ
  by up to $3\times10^{-2}$, with the same sum. Repaired: `Embedding.get_ldos`
  applies the convention to the plain Green's function, and the pristine
  embedding agrees with `h.get_ldos(mode="green")` to $10^{-10}$.
- **An embedding with any operator at `nsuper>1` crashed** with a matrix size
  mismatch. Repaired: a name, a potential or a callable is built on the
  supercell, and a bond matrix of the 2x2 supercell agrees with the LDOS of
  `h.supercell(2)` to $4\times10^{-16}$.
- **`Embedding.multildos` took `energies=` and ignored it**, returning its
  default grid of 20 energies. It now takes the keyword, as
  `Embedding.multidos` and `h.get_multildos()` do.
- **A complex energy in `mode="triangle"` of the Bloch selfenergy was cast to
  its real part** with only a ComplexWarning, so `Embedding.get_gf` at a
  complex128 energy returned the bulk at the real energy and the Dyson step at
  the complex one. It raised a `ValueError` after the repair, and since the
  decisions of 14 November it is taken as the real energy with the broadening
  Im z, see `qtci_dos_green.md`. No public routine returned a wrong number
  through the cast: the density-matrix contour, the one caller with a complex
  energy, stopped at its first advanced point.
- **The tests of the embedding with an operator had no defect**: the defect
  was selected at $x=0$, where the honeycomb lattice has no site. It is on the
  first site now.
- **The `delta_discard` window of `ldostk/ldoswaves.py:34`** used the lowest
  energy on both edges, which is wrong for `ldosmap` with several energies.
  Repaired, with a test.
- **The first vertex-form kernel of the simplex sum ran on one thread** for
  one band at one energy on meshes above $2^{22}$ points (46 s against 33 s
  for the old kernel). Repaired in its review, with the slices of simplices.
- **The 3D k-mesh of `kpointstk/kmesh.py` was a Python list**, 715 MB and
  3.6 s at `nk=128`, which dominated the peak of the slab sum of the grid
  conductivity. It is now built as an array, 189 MB and 0.04 s, the same
  points in the same order to the last bit.
- **tau_z counted the vertical bonds of a film of two buckled layers** as
  in-plane bonds of length 0, so it fell back to the normalization of a flat
  lattice and had eigenvalues $\pm8/9$ on the Dirac states. Repaired, $\pm1$.
- **Two defects of the new shared mesh of the triangle mean field**, found in
  its review: a search with `keep_mesh` on a Hamiltonian without Bloch matrices
  computed and kept eigenvectors that no density matrix could take, and the
  qtci record kept its simplices, up to 3.2 GB at $R=8$ in 3D, when the next
  call was for another Hamiltonian. Both repaired, each with a test that fails
  without the repair.
- **The generic `Vr` and `Jr` average a function that is not reciprocal**,
  see the reciprocity entry above. Decided, it raises.
- **The Kondo-lattice loop uses the density matrix transposed**
  (`scftk/kondolattice.py`), so a complex hybridization relaxes to a real one.
  The solutions are right, since each $V_j$ can be made real with the phase of
  its own f fermion, but not with a complex hopping between the moments.
  Repaired in the decisions of 14 November 2026.
- **The conftest of the vendored qutecipy tests marked every test of the
  session slow** (`tests/qutecipy/conftest.py`, from `8568da9`), since its
  `pytest_collection_modifyitems` receives the items of the whole session. So
  `pytest tests -m "not slow"`, the command its own docstring gives, selected
  nothing, and a pytest plugin that skips slow tests skipped every test
  collected next to `tests/qutecipy`. Found by the whole-suite run of 14
  November. Repaired: the conftest marks the tests of its own directory only,
  and `tests/guards/test_slow_marker_scope.py` fails on the old conftest.

## The decisions of 14 November 2026

The choices that the repairs left open were decided on 14 November 2026, after
the commit of the repairs (`8518a8e`), in three groups by the files they touch.
Each change of the code has a test that fails on `8518a8e`.

- **The default SHG source kernel stays as it is.** The route of
  `kubosimplex.py` keeps a tuned budget and only makes it cheap with the energy
  bins, so the port is a rewrite of the default kernel and waits for a
  sign-off. The reasons are in `qtci_shg.md`.
- **The first-order term for equal nodes stays.** It is the term the series of
  close nodes already keeps, and a test pins it in both kernels.
- **The scale per family of the qtci conductivity stays.** Five seeds on the
  disordered supercell give an error of $4.3\times10^{-6}$ to
  $2.0\times10^{-5}$ without it and $8.6\times10^{-6}$ to $1.6\times10^{-5}$
  with it, and the fields are resolved as well or better. Reverting it brings
  back the dependence on the unit of length. The scale now lives in the keyword
  `families` of `qtcitk/meshfields.py`, which the LDOS uses for its band
  energies. Its time on an idle machine is not measured.
- **The qtci DOS and LDOS measure each band from the middle of its own range**
  (the low finding on `simplexdos.py:595`), not with one scale per field. The
  qtci LDOS depended on the unit of energy, by $2\times10^{-4}$ for a model in
  meV, and the qtci DOS on the zero of energy, by $3.7\times10^{-4}$ at a shift
  of 100. Now the tolerance of a band is relative to its width, whatever its
  position. The LDOS divides the band energies by the largest of them and leaves
  the site weights as they are, since they have no unit. A flat band is not
  interpolated. A first shape, with the zone mean and a scale of the weights,
  was repaired after a review: it loosened a band far from the mean by 50 times
  and brought weights of rounding noise to order one, so that the LDOS with
  `operator="sz"` on a non-magnetic lattice took the whole mesh.
- **`get_dos(mode="Green", gmode="triangle")` sweeps from one
  diagonalization**, through `green.green_operator_batch`. It is 1.3 times
  faster on four-orbital diamond at `nk=30` and 2.9 times on a 16-orbital cubic
  supercell at `nk=16`, with the same DOS to $7\times10^{-15}$.
- **The tetrahedra take a complex energy as the real energy with its imaginary
  part as the broadening**, so a 3D embedding has a density matrix, which no
  mode gave before. The contour pays one diagonalization of the mesh per point,
  60 to 76 points, and the pristine insulator gives the occupied projector to
  $1.3\times10^{-4}$ at `delta=1e-3`. A negative broadening gives the advanced
  function, and a broadening that is zero to rounding raises. The four modes
  that stop in 3D now say to use `mode="triangle"`, as the `gf_mode` of an
  Embedding. An Embedding of a 3D crystal takes `gf_mode="triangle"` by
  default, by choice of the maintainer: its old default, `"renormalization"`,
  always raised in 3D. At `nk=100` this keeps the eigenvectors of $10^6$
  k-points. A mesh shared by the whole contour is open in
  `qtci_dos_green.md`.
- **The single product of `get_dm` for one direction stays.** Both routes do the
  same number of multiplications there, and the single product is the one that
  BLAS threads. The projector route with several directions on many threads is
  still not measured.
- **The finite-temperature weights take the three nodes of the Gauss rule of
  $-f'$** on a simplex with no vertex within $40T$ of $\mu$, exact to rounding,
  in the weights and in the charge kernel. At $T=10^{-7}$ the Fermi search is 6
  to 34 times faster and the weights 3 to 10. The shell vectors keep no cap,
  since no run has needed one. Record in `tetrahedron_density_matrix.md`.
- **`Embedding.multildos` keeps honouring `energies=`.** `Embedding.multidos`
  and `h.get_multildos()` use the same name for the grid.
- **The generic `Vr` and `Jr` raise too when they are not reciprocal.** What
  reads the interaction does not agree on a function with two values: the
  Hartree-Fock field averages the two ends of a pair, and the BSE kernel and the
  $V(q)$ of the RPA take one end each, which is not Hermitian. One check,
  `specialhopping.require_pair_interaction`, runs in the builder that the mean
  field, the BSE kernel, the density response and `Jr` share, and in the sparse
  builder, and the spinon path has no check of its own any more. It asks that
  every value is finite, real and the same from the two ends of a pair. Before,
  a ramp that is symmetric in its two arguments and not periodic gave no warning
  and a mismatch of 0.4 in the BSE kernel. The review of the decision found that
  an imaginary `Vr` went through and that an infinite value at zero distance
  switched the check off; both raise now. The BSE kernel drops the pair of a
  site with itself before the check, so it still takes a Coulomb tail without a
  soft core. A periodic modulation passes.
- **The Kondo-lattice loop reads the element $[c,f]$ of the density matrix.**
  A complex seed keeps its phase. With a complex hopping between the moments
  the old loop returned a state that was not a solution, with $V$ off by
  $4.5\times10^{-2}$ and the energy 0.036 too high; with a real $V$ nothing
  moves. Record in `kondo_tunneling.md`.
- **The k-loops of `cotunneling.py` and of the composite spectrum diagonalize
  one spin block** of a spin-degenerate state, in one function shared by the
  two loops. With 196 Kondo pairs the cotunneling spectrum goes from 3.12 s to
  0.57 s and the composite spectrum from 3.17 s to 0.65 s, agreeing to
  $2\times10^{-13}$.
- **A field in any direction takes the collinear route of the spin bubble**, in
  its own spin frame, when every hopping depends on the spin through
  $\hat n\cdot\vec\sigma$ alone. At 196 sites in a field $(0.2,-0.1,0.25)$:
  27.6 s and 1.77 GB before, 4.3 s and 0.86 GB after, agreeing to
  $1.3\times10^{-15}$. A field that turns from site to site takes the general
  route.
- **The binned local response of `q=None` is not built.** It covers only the
  bare response, changes the numbers at the level of the bin and needs a new
  option. The docstrings and the guide now say that the cost of `q=None` grows
  as the square of the number of k-points.
- **The origin of the 36-site spinon example stays off the symmetry points.**
  Counted again, the state in the field has 36 different moments with it and 12
  with the origin at the center. Record in `spinon_response.md`.

## What is left open

- The port of the energy bins of `kubosimplex.py` to the default SHG source
  kernel, which waits for a sign-off (`qtci_shg.md`).
- One diagonalization of the mesh for the whole density-matrix contour of a 3D
  embedding, which now pays one per point.
- A memory cap on the shell vectors of the BdG shell filling, and the projector
  route of the density matrix with several directions on many threads.
- The qtci LDOS with an operator takes 20% to 44% more points than before for
  the same tolerance, since the energies are resolved against the width of a
  band, and an operator with a large norm loosens its energies. The weights of
  the qtci DOS that vanish up to rounding are still interpolated. The time of the
  per-family scale of the qtci conductivity on an idle machine is not measured,
  and the raw band energies of the qtci density matrix, which a constant shift
  should loosen as it did the DOS, were read and not run.
- The binned local response of `q=None`, not built by decision
  (`spinon_response.md`), and the default `nk=60` of `get_moment_spinchi` with
  `q=None`, about 6 hours in 2D by extrapolation.
