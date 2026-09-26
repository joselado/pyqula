# Unfolding across the modes of its consumers

`operator="unfold"` (`unfolding.bloch_projector`) is advertised for every
k-resolved observable, and each of those routines computes its answer in
several modes. This records a check of the operator through every mode, the
repairs it led to, the three pieces built after it (orthonormal ARPACK
eigenvectors, unfolding with the KPM, and the primitive-cell mesh for any
supercell), and the one question left open.

## What was checked and found right

The oracle was a brute-force sum
$\sum_n \langle n|O(k)|n\rangle\,\delta/((e-E_n)^2+\delta^2)$ with
$O=P^\dagger P$ built as an explicit matrix, against which every consumer was
compared at the same kpoints, on a non-diagonal honeycomb supercell
($M=[[2,1],[0,1]]$) with and without a point defect, spinless, spinful and
Nambu.

- `get_bands`: full diagonalization and `num_bands` (ARPACK), at a generic
  kpoint.
- `get_kdos_bands`: `mode="ED"` and `mode="green"` agree with each other and
  with the oracle to machine precision, for all three Hilbert spaces.
  `mode="KPM"` refused the operator at the time; it now takes it exactly,
  see "Unfolding with the KPM".
- `get_dos(ks=[k])` with `mode="ED"`, which is what the Fermi surface calls.
  `mode="Green"` and `mode="KPM"` refuse the operator.
- `get_fermi_surface` with `backend="qtci"` agrees with the grid to 1e-16.
- The projector itself (phases, replica count, Hermiticity, rank, 1D/2D/3D,
  diagonal and non-diagonal) is pinned in
  `tests/unfolding/test_unfolding_projector.py` and was not re-derived.

## What was repaired

Pinned in `tests/unfolding/test_unfolding_modes.py` and
`tests/fermisurface/test_fermi_surface_normalization.py`, all of which fail
on the unrepaired source except the five that pin behaviour that was already
right.

- **The Fermi surface changed scale with the mode.** Without an operator,
  `mode="eigen"` returns $\sum_n L(e-E_n)$ with no $1/\pi$, and that is what
  the existing tests pin. With an operator, `eigen` and `lowest` go through
  `h.get_dos`, which applies $1/\pi$, and `mode="full"` takes
  $G^R-G^A$, twice $G^R$. So the same unfolded map came out $1/\pi$ times
  the reference in `eigen`, and $2\pi$ times the `eigen` value in `full`.
  `fermi_surface_generator` (behind `get_multi_fermi_surface` and the
  `mode="pm"` QPI) had the same $1/\pi$. Every mode now returns the
  no-operator scale, so no output without an operator moved. The operator
  maps moved by $\pi$, and `mode="full"` by 1/2 in every case.
- **`get_qpi(mode="response")` dropped `operator` and `nunfold` silently**
  and returned the plain supercell QPI. It now raises.
- **`get_qpi(mode="pm")` unfolded any supercell as if it were
  `nunfold` x `nunfold`**, by dividing the reduced kpoints by `nunfold`. It
  now unfolds any supercell, see "The primitive-cell mesh" below.
- **`get_bands(num_bands=N-1)` raised inside scipy**, since `eigsh` hands a
  complex matrix to `eigs`, which needs $k<N-1$. It found this through the
  `lowest` Fermi surface. The clamp is now `>=`, in the Hermitian and the
  non-Hermitian band structure alike. The no-operator branch of the
  `lowest` Fermi surface calls `eigsh` itself, so it now falls back to
  every state at `num_waves>=N-1` too. Otherwise the same call crashed
  or not depending on whether an operator was passed.
- The `lowest` Fermi surface with an operator shifts the Hamiltonian by
  `shift_fermi(-e)` and looks at zero energy. The identity-operator test
  shows that this lands on the same states as the no-operator branch, which
  looks at `sigma=e` directly. So the sign is right, and the over-counting
  that showed up there came from ARPACK alone (next section).

## Orthonormal ARPACK eigenvectors

At the supercell $\Gamma$-equivalent of a 4x4 triangular cell with a defect
(an 8-fold and a 5-fold level) the `num_bands` bands and the `lowest` Fermi
surface over-counted the unfolded weight, up to 1.5 times the full sum, and
the number changed from run to run. It was first read as ARPACK returning
ghost copies. It is not: the returned vectors have full rank (smallest
singular value about 0.2) and residuals of 1e-15, so they are exact
eigenvectors, but their overlap matrix is 0.65-0.83 away from the identity.
The cause is that ARPACK has no complex Hermitian driver, and scipy's
`eigsh` hands every complex matrix to the non-Hermitian `eigs`, whose
eigenvectors inside a degenerate level are some basis of the eigenspace and
not an orthonormal one. Every sum over states (an operator weight, an LDOS
$\sum_n|v_n|^2$, an occupied manifold) then counted a direction several
times.

`algebra.arpack_eigh` wraps `eigsh` with a Rayleigh-Ritz step on the
returned subspace (QR, the projected matrix, a small `eigh`), which makes the
vectors orthonormal without changing the subspace or the eigenvalues, checks
the residuals, and seeds the start vector as `smalleig` already did. Every
site that uses `eigsh` vectors goes through it: bands with `num_bands`,
`smalleig`, `lowest_bands`, `get_eigenvectors`, the ARPACK LDOS routines,
`density.restricted_density`, the occupied states of the topology code, and
the sparse branch of `spectrum` that sorts states by energy. The sites that
only use eigenvalues were left alone, as were the non-Hermitian `eigs`
calls. Pinned in `tests/algebra/test_arpack_eigh.py`, which takes whole
degenerate levels (where the sum over a level is basis independent) and
fails on the old source: the unfolded weight of the 5-fold level came out
13.49 instead of 13.33, and the LDOS of a Kramers-degenerate island was not
even symmetric between the partners of a pair.

What the step cannot do is complete a degenerate level that `num_bands`
cuts: when the $k$ states nearest the energy end inside a level, the subspace
of it that comes back is arbitrary, and so is its weight. The seeded start
vector makes it the same arbitrary one every run.

## Unfolding with the KPM

$O(k)=P^\dagger P$ has rank $n_0$ (the orbitals of the primitive cell), so
$\mathrm{Tr}[O\,\delta(E-H)]=\sum_{\alpha=1}^{n_0}\langle u_\alpha|
\delta(E-H)|u_\alpha\rangle$ with $u_\alpha$ the columns of $P^\dagger$, the
primitive Bloch states (norm squared $N_\mathrm{rep}$ each). That is $n_0$
Chebyshev expansions from deterministic start vectors and no stochastic
trace, which is how KITE computes the ARPES spectral function (Joao et al.,
arXiv:1910.05194, Sec. 4.4.1, Eqs. 41-43, with Bloch states as the start
vectors). An `Operator` now carries an optional `factor`, a function of $k$
returning $U$ with $O=UU^\dagger$; `bloch_projector` sets it to $P^\dagger$,
a copy keeps it, and a product, multiple or sum drops it (so `unfold*sz` is
refused by the KPM rather than expanded wrongly). `kpm.factored_moments` and
`kpm.factored_dos` run the columns as one batch through
`get_moments_batch`, so the GPU switch reaches them too, and
`get_kdos_bands(mode="KPM")` takes that route for any operator with a
factor. `P` or `frand` with a factored operator raises.

The oracle, in `tests/unfolding/test_unfolding_kpm.py`, is stronger than a
benchmark against KITE: the recursion moments equal
$\sum_m w_m T_j(E_m/s)$ from an eigendecomposition to 1e-10, spinless,
spinful and Nambu, and a clean supercell at $k_S=Mk_0$ gives exactly
$N_\mathrm{rep}$ times the primitive cell's own exact KPM trace at $k_0$ for
the diagonal, the $\sqrt3$ and the unequal-vector cells. The first run of
that test failed in the far tails at the 1e-7 level, which turned out to be
the reference using the full-spectrum energy grid where `kdos_bands` uses
its default `ewindow=4`; with the same window the two agree to 1e-9.

The plain KPM kdos was N times smaller than `ED` and `green`, because
`kpm.pdos` averages over unit random vectors, which is the trace divided by
N. It is now multiplied by N, so the three modes return the same trace and
the unfolded result does not depend on the mode. With `P` or `frand` the
vectors are drawn from a subspace of the caller's choosing and the result
stays an average over it. `get_dos(mode="KPM")` still refuses the unfolding
operator: a k-summed unfolded density of states means nothing.

## The primitive-cell mesh

With `reciprocal=True` a Fermi-surface mesh point is mapped through
`get_k2K`, which normalizes each lattice vector separately, so it is a
Cartesian momentum in units of $2\pi/|\vec A|$ only when
$|\vec A_1|=|\vec A_2|$; the extended-zone unfolded map of a cell like
$M=[[2,1],[0,1]]$ was sheared, and the QPI unfolding worked for
$n\times n$ cells only. `unfolding.get_primal_mesh_map` draws the mesh in the
reciprocal space of the stored primitive geometry, with its own `get_k2K`,
and maps each point with $k_S=Mk_0$, the map `get_unfolded_kpath` uses for
paths. `get_fermi_surface`, `fermi_surface_generator` and so
`get_multi_fermi_surface` take it as `primal_mesh=True`, off by default so
that no existing output moves; turning it on automatically with
`operator="unfold"` was the alternative, and it would have zoomed the
documented `nsuper=n` example out by $n$. The `mode="pm"` QPI now samples
$[0,1]^2$ in primitive reduced coordinates through the same map, and maps
its q-points with the primitive geometry too, since the convolution reads
them in the coordinates of the k-mesh. `nunfold` is read as `get_supercell`
reads a size, `nunfold**2 == |det M|`, so it stays `n` for an $n\times n$
cell and is `np.sqrt(3)` for the $\sqrt3$ one; a supercell built without
`store_primal=True` is refused.

Pinned in `tests/unfolding/test_unfolding_modes.py`: a clean supercell's
unfolded Fermi surface on the primitive mesh is $N_\mathrm{rep}$ times the
primitive one to 1e-10 for the three cells; the $\sqrt3$ QPI is
$N_\mathrm{rep}^2$ times the primitive QPI on the same q-points; and the
$n\times n$ QPI reproduces the numbers recorded from the old
divide-by-`nunfold` implementation to 1e-8.

A float size returns a supercell rotated so that $\vec A_1$ lies along $x$,
and the primitive geometry it stores is rotated with it (150 degrees for
the triangular $\sqrt3$ cell). Reduced coordinates do not see this, but a
Fermi surface drawn with `reciprocal=True` is in that rotated frame, so it
is to be compared with the stored primitive cell's, not with the geometry
the supercell was built from. Separately, `store_primal=True` stores the
primitive copy on the geometry the supercell is built from, so every later
supercell of that same geometry carries it whether it asked or not; this
predates the work here and was left alone.

## User-visible changes

- Fermi surfaces with an operator are $\pi$ times larger in `eigen`,
  `lowest` and `get_multi_fermi_surface`; `mode="full"` is halved with or
  without an operator. Outputs without an operator in the other modes did
  not move.
- `get_kdos_bands(mode="KPM")` without `P` or `frand` is N times larger, the
  trace, as `ED` and `green` return. With `operator="unfold"` it now works,
  exactly, instead of raising.
- `get_bands(num_bands=...)`, `get_eigenvectors(numw=...)`, the ARPACK LDOS
  and the ARPACK occupied states return orthonormal eigenvectors in
  ascending order of energy; sums over a degenerate level changed wherever
  the matrix was complex.
- `get_qpi(mode="response")` raises when given `operator` or `nunfold`;
  `get_qpi(mode="pm",nunfold=...)` needs `store_primal=True` and a
  `nunfold` whose square is the number of primitive cells.
- `get_bands(num_bands=N-1)` and `get_fermi_surface(mode="lowest",
  num_waves=N-1)` fall back to a full diagonalization instead of raising.

## What is open

- **The non-Hermitian unfolded spectral function has two definitions.** The
  `ED` kdos weights each right eigenvector by $\langle R|O|R\rangle$ and
  places a Lorentzian of width $\delta$ at $\mathrm{Re}\,E$. The `green`
  kdos takes $\mathrm{Tr}[O\,G]$, which is the biorthogonal
  $\langle L|O|R\rangle$ with the full complex pole. They agree only when
  the spectrum is real, and they disagree with no operator at all once
  $\mathrm{Im}\,E\neq0$, so this is a property of the non-Hermitian
  `kdos_bands` that unfolding exposes, not an unfolding bug. Which one is
  "the" unfolded spectral function of a non-Hermitian supercell is a
  formalism call that wants a reference. Neither mode was changed.
  `tests/nonhermitian/test_unfolding_non_hermitian.py`'s second docstring
  says it goes through the Green's function, but it runs the default `ED`
  mode.
- Found in passing, not unfolding-specific: `get_dos(mode="adaptive",
  write=False)` raises `TypeError` because `adaptive_dos` forwards `write=`
  to `get_bands` a second time.
