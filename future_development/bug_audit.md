# Bug audit -- four-lens sweep, 2026-09-05

A parallel four-agent sweep over `src/pyqula`, decomposed by bug *class* rather
than by subsystem: Hilbert-space bookkeeping, parallel/backend equivalence,
API contracts in the delegator layer, and physics invariants. The lenses were
seeded with the fixes in `2ed55c0 efdc257 6f0f031 fbee7c9 e996cf3 3b43557
4adc7b0 5ff49f7 1874fad e481ffb`, on the premise that the highest-yield
findings are *sibling* instances of a class the maintainer has just fixed
somewhere else. That premise held: the two largest groups below are siblings
of `fbee7c9` (density-matrix transpose) and `3b43557` (silently ignored
arguments).

29 raw findings, ~26 distinct after dedup. Every entry was reproduced with a
runnable script before being recorded; the structural cause of each was then
re-confirmed by reading the source.

**Not covered** (the agents were barred from the slow suites): `tests/scf`,
`tests/keldysh`, `sctk/superfluidweight.py`, `sctk/pairing.py` internals,
`scftk/spinspin.py`'s sparse density-matrix kernels, and the Nambu handling of
the mean-field SCF kernels. A second sweep should start there. The same two
suites were also not run to verify the fixes -- everything else that touches
the changed modules was.

**One candidate chased and cleared**, recorded so it is not re-chased:
`sctk/spinless.onsite_delta_vev`'s finite-temperature weight `ws[i]*fd` looks
like it squares the occupation, but `f^2 - (1-f)^2 == f - (1-f)` exactly, so
the +-E BdG pair cancellation makes it numerically identical to the linear
form. Not a bug.

Each finding carries a **Status** line: `open`, or `fixed` with the commit and
the test that pins it.

**Where this stands.** Section 1 (silently wrong numbers) and section 3 (hard
crashes) are fixed, in `5f3da27 7087ee6 a39bf68 4d5843d 66df675 2c28e53`, each with a regression test that asserts an
invariant rather than a pinned number. Section 2 is now fixed too, except 2.4;
section 4 is fixed. Section 4 is now fixed
entirely (4.1 in `ccdee4a`). What remains open, deliberately:

- **2.4** `add_sublattice_imbalance` no-oping on triangular and kagome --
  feature work, wants a decision (implement, or refuse in the `3b43557` style)
  rather than a repair.
- **1.11** the parallel RNG; see its entry for why.

The two items that were filed as "wants a decision" and have now been
implemented rather than refused are 2.1 (`get_ldos(operator=)`, which
`dos.get_dos` already honoured, so the library was inconsistent with itself)
and 2.7 (`hs` beyond nearest neighbours, generalized from the nearest-neighbour
path's own convention and pinned against it).

Four of the fixes turn a silent wrong answer into a raised exception, which is
a behaviour change for anyone who was relying on the broken path: an
unrecognised operator name in `topology` (was: the valley operator), a 3D
`set_finite_system(periodic=True)` (was: an open cluster), a `filling`
outside [0,1] in `set_individual_filling` (was: unchecked), and a d-vector
asked of anything but a spinful Nambu Hamiltonian (was: an array of the wrong
length, or nan).

---

## 1. Silently wrong numbers

The worst class: no exception, plausible-looking output, wrong physics.

### 1.1 `topology.py:760` -- `get_operator` returns the valley operator for every string name

```python
if op=="valley": return h.get_operator("valley",projector=True)
else:            return h.get_operator("valley",projector=True)   # <- identical
```

Both branches of the test return the same thing, so
`get_berry_curvature_path(h, operator="sz")` (a.k.a. `topology.write_berry`,
used by `examples/2d/berry_valley` and `examples/2d/berry_green`) computes the
valley-projected curvature. On Kane-Mele: identical to `operator="valley"` to
0.0, and different from the correct answer (an explicit `h.get_operator("sz")`)
by 145 in absolute curvature. Any string -- `"sz"`, `"sublattice"`, a typo --
silently becomes `"valley"`.

Line 762 compounds it: `if type(op)==np.array` is never true (`np.array` is a
function, not a type), so an operator passed as a plain numpy matrix falls off
the end of the function and is silently replaced by `None`, i.e. unprojected.

**Status:** fixed in `5f3da27`. The `else` branch now goes through the ordinary
`h.get_operator(op)` lookup (only `"valley"` accepts `projector=True`), the
matrix test is `algebra.ismatrix`, and an unrecognised type raises `TypeError`
instead of falling through. `tests/topology/test_topology_operator_dispatch.py`.

### 1.2 `spectrum.py:219` -- `real_space_vev` contracts the untransposed density matrix

`densitymatrix.full_dm` returns `dm[i,j] = sum_occ conj(psi_i) psi_j`, the
transpose of rho. `real_space_vev` does `rho = operator(dm)` then
`np.diag(rho)`, i.e. `diag(op @ dm)`, which evaluates `<A*>` instead of `<A>`.
This is exactly the bug `fbee7c9` fixed in `spectrum.ev` and `vev.get_dm_vev`
-- and left in this sibling three lines below.

Real operators (density, `sx`, `sz`, projectors) are unaffected; `sy`, the
valley operator and any `i[H,r]` come back negated:

```
sx  explicit=-0.486664  get_vev=-0.486664  real_space_vev=-0.486664
sy  explicit=-0.811107  get_vev=-0.811107  real_space_vev=+0.811107
sz  explicit=-0.324443  get_vev=-0.324443  real_space_vev=-0.324443
```

`tests/island/test_island_operators.py:52` currently *pins the flipped value*
(-0.8852949826564925 against an explicit occupied-state sum of
+0.8852949826564898) -- the sibling `spectrum.ev` reference in the same test
file was flipped by `fbee7c9`, this one was not. Fixing the code requires
re-pinning that test. Every valley-texture consumer is affected:
`examples/2d/valley_vortex/main.py` and `examples/0d/valley_vortex_vacancy/main.py`
read `valley_x`/`valley_y` maps from this function, so the vortex winding sense
is reversed.

**Status:** fixed in `7087ee6`. The contraction now transposes the density matrix, the same
way `spectrum.ev` does. `test_island_operators.py`'s pin was replaced by an
in-test explicit occupied-state sum so it cannot regress in either direction;
`tests/spectrum/test_real_space_vev.py` covers the rest.

### 1.3 `spectrum.py:220` -- `real_space_vev` double-counts Nambu

`np.diag(op@dm)` runs over the whole 4N-dimensional Nambu space and
`h.full2profile` then sums all four spin x electron-hole components of a site.
Each spin channel contributes `n + (1-n) = 1` from its electron and hole
entries, so every site reads exactly `2.0` whatever the density is -- all
spatial information is destroyed.

```
normal   get_vev          [0.5 1.  0.5]
nambu    get_vev          [0.5 1.  0.5]
normal   real_space_vev   [0.5 1.  0.5]
nambu    real_space_vev   [2.  2.  2. ]
```

`get_vev` restricts the operator to the electron sector (`operators.get_electron`)
for precisely this reason -- the fix in `6f0f031`. `real_space_vev` has no
`has_eh` branch at all. Unambiguous for a density/spin observable; note this
would *not* be a bug for an LDOS, where summing electron+hole is a legitimate
quasiparticle convention.

**Status:** fixed in `7087ee6`, with the `get_vev` electron-sector restriction.
`tests/spectrum/test_real_space_vev.py`.

### 1.4 `chitk/chiAB.py:217` -- the accelerated path runs at half the requested temperature

`chiAB_jit:146` and `chiAB_matrix:174` build occupations as `1/(1+exp(beta*E))`,
Fermi-Dirac at `T`. `chiAB_full_matrix_jit:217` -- reached only through
`chiAB_full_matrix_jit_kmesh` -> `chiAB_matrix_ksum` -> `ij_mode="accelerated"`
(`chiAB.py:110`) -- builds them as `(-tanh(beta*E)+1)/2 = 1/(1+exp(2*beta*E))`,
which is Fermi-Dirac at `T/2`. A user who picks the accelerated loop mode gets
the same-shaped answer at half the temperature they asked for.

Pinned exactly -- running the explicit path at `T/2` reproduces the accelerated
path at `T`:

| T | explicit(T) vs accel(T) | explicit(T/2) vs accel(T) |
|---|---|---|
| 0.1  | 0.0307 | 1.0e-17 |
| 0.05 | 0.0352 | 1.1e-16 |
| 0.01 | 0.1159 | 4.2e-17 |

Against a response scale of `max|chi| = 1.184` at T=0.1, that is a 2.6% error,
growing as T falls.

**Status:** fixed in `2c28e53` -- `0.5*(1 - tanh(0.5*beta*E))`, which is Fermi-Dirac at `T`
and, unlike `1/(1+exp(beta*E))`, does not overflow at low temperature.
`chitk/chijax._occupations` already wrote it that way and its docstring names
this function as the one it was mirroring, so it is the in-repo reference.
`tests/chi/test_chi_backend_agreement.py`.

`chitk/static.py:82`'s `elementchi` carried the same `(-tanh(E/T)+1)/2`, found
while fixing this one and fixed with it. Its default `temp=1e-7` makes the
occupation a step function, so in practice nothing moved -- but the convention
now matches the rest of `chitk/`.

### 1.5 `chi.py:32` -- `elementchi` conjugates the wrong pair of amplitudes

```python
fac  = ws1[i][ii]*ws2[j][ii]
fac *= np.conjugate(ws1[i][jj]*ws2[j][jj])
```

forms `psi_n(i) psi_m(i) conj(psi_n(j) psi_m(j))` instead of the Lehmann matrix
element `conj(psi_n(i)) psi_m(i) conj(psi_m(j)) psi_n(j)`. The two coincide only
when the amplitudes are real (and on the diagonal `i==j`).

This is an internal inconsistency, not a convention question: `chitk/chiAB.py:154-155`
(the modern `chiAB_jit`) already uses the correct
`<psi1_i|A|psi2_j><psi2_j|B|psi1_i>` form. The consequence is that the charge
response is not invariant under a site-local gauge change `H -> U H U^dag`,
`U = diag(exp(i phi_k))`, which leaves the physical density-density response
unchanged: measured `max|chi - chi_gauged| = 0.0144` on values of order 0.2,
where a direct Lehmann reference is invariant to 1e-16. `elementchi_row`, its
batched twin, has the same defect.

**Status:** fixed in `2c28e53`, in both. `chargechi` now reproduces an explicit Lehmann sum
to 1e-17 and is gauge invariant to 2e-15.
`tests/chi/test_chi_backend_agreement.py`.

### 1.6 `ldostk/ldosr.py:29` -- spinless branch overwrites instead of accumulating

```python
if h.check_mode("spinless"):
    yi = calculate_dos(evals,es,delta,w=ds[:,ii])
    yout = yi*ws[i]          # <- all three sibling branches use yout = yout + ...
```

The continuum-space LDOS at a point is therefore only the *last* neighbour's
contribution instead of the weighted sum over the `nn` closest sites. On an
8-site 0d chain with `nn=4`: energy-integrated LDOS 0.1127 where the weighted
sum is 0.9648, and the value equals exactly the last neighbour's term (weight
0.0596 of 1.0). The same geometry spinful gives 1.9296 = 2 x the correct
spinless value, confirming which branch is right.

Worse than the magnitude error: the result depends on the order
`sculpt.get_closest` happens to return, which is an implementation detail, so
the map is not even a smooth function of position. `examples/1d/ldosr/main.py`
builds its ribbon with `has_spin=False`.

**Status:** fixed in `a39bf68` -- the branch accumulates like its three siblings.
`tests/ldos/test_ldosr_accumulation.py` pins spinful == 2 x spinless, an
independent reference for the same quantity.

### 1.7 `kdos.py:98,100` -- surface-DOS operator algebra broken twice over

Found independently by two lenses. Lines 98 and 124:

```python
elif callable(operator): op = callable(op)     # `op` is unbound here
```

`h.get_operator("sz")` returns an `Operator`, which *is* callable, so this
raises `UnboundLocalError: cannot access local variable 'op'` before any
physics happens. Pass a raw numpy matrix instead and lines 100-101 / 126-127
compute `algebra.trace(gs*op)` where `gs` is a plain ndarray -- so `*` is
*elementwise*, and the result is `sum_i gs[i,i]*op[i,i]`, correct only for a
diagonal operator. On a spinful chain with an in-plane exchange field the
sx-projected surface DOS is identically `[-0,-0,-0,-0,-0]` against a correct
`trace(sf@sx)` of `[-0.1038,-0.0506,0,+0.0506,+0.1038]`. `sz`, being diagonal,
happens to come out right.

**Status:** fixed in `a39bf68`. The operator resolution is now a shared
`kdos.get_surface_operator`, hoisted out of the energy loop, and the
contractions are matrix products. `tests/kdos/test_surface_dos_operator.py`.

### 1.8 `transporttk/unitarize.py:26` -- S-matrix off-diagonal blocks swapped

`bmat([[s00,s01],[s10,s11]])` puts `s01` at rows `0:n`, cols `n:2n`. The split
back reads

```python
sout = [[s3[0:n,0:n], s3[n:2*n,0:n]], [s3[0:n,n:2*n], s3[n:2*n,n:2*n]]]
```

so `sout[0][1]` gets `s10` and `sout[1][0]` gets `s01`. Since `check=True` is
the default, `get_smatrix()` returns the two interchanged: on a 3-leg square
ribbon junction with asymmetric central cells, `get_smatrix(check=True)[0][1]`
differs from `get_smatrix(check=False)[0][1]` by 0.255 but matches
`get_smatrix(check=False)[1][0]` to 2.2e-4 (the unitarization residual).

`didv` and `didv_BdG` are immune -- two-terminal unitarity makes
`Tr(t t^dag) = Tr(t' t'^dag)`, and the BdG path only touches the diagonal
blocks -- so this bites any caller using the transmission block itself.
`Heterostructure.get_smatrix` is public and its docstring advises `check=True`.

**Status:** fixed in `4d5843d` -- the split reads the blocks the way `bmat` laid them out,
and it no longer assumes the two leads have the same dimension.
`tests/transport/test_smatrix_block_order.py`.

### 1.9 `hamiltonians.py:1112` -- `set_finite_system(periodic=True)` is unreachable

```python
h.dimensionality = 0
h.geometry.dimensionality = 0
if periodic:
    if h.dimensionality == 1: ...     # false by construction
    if h.dimensionality == 2: ...     # false by construction
```

The dimensionality is zeroed two lines before the branches that test it, so the
wrap-around terms are never added and the function returns silently. A 6-site
ring: `H[0,5] = 0` instead of 1, spectrum `[-1.802,-1.247,-0.445,0.445,1.247,1.802]`
(open chain) instead of the exact `[-2,-1,-1,1,1,2]`. Bit-identical to
`periodic=False`.

**Status:** fixed in `4d5843d` -- the dimensionality is captured before it is zeroed, and a
3D `periodic=True` now raises rather than silently returning an open cluster.
`tests/hopping/test_finite_system_periodic.py`.

### 1.10 `heterostructures.py:242,249` -- left lead built from the right lead's hopping

```python
tr = csc_matrix(h_right.inter)
tl = csc_matrix(h_right.inter)     # should be h_left
```

and identically in the `block_diagonal` branch at line 249
(`tl = h_right.inter.copy()`). Only symmetric junctions are unaffected. This is
the one finding whose agent did not manage a numerical repro; the source lines
read exactly as quoted.

**Status:** fixed in `4d5843d` -- both read `h_left.inter` now.

### 1.11 `paralleltk/multiprocess.py:11` -- `_init_worker` never reseeds the RNG

Forked pool workers inherit the parent's numpy random state, so any function
dispatched through `parallel.pcall` that draws random numbers inside the worker
produces the *same* draws in every worker.

```
parallel.pcall(lambda i: np.random.random(), range(8))
  cores=4 -> 5 distinct values of 8
  cores=1 -> 8 distinct values of 8

pcall(kpm.random_trace(m,ntries=4,n=20), range(8))
  cores=1 -> 8 distinct stochastic estimates
  cores=4 -> 4 distinct stochastic estimates
```

So averaging over k-points does not reduce the KPM variance the way it does
serially: the parallel answer is systematically noisier than the serial one for
the same nominal `ntries`, and the two backends disagree.

**Status:** left open, deliberately, and recorded here instead. Reseeding is
one line (`np.random.seed()` in `_init_worker`), but it makes every parallel
stochastic result irreproducible run to run, and `90a5cf5` establishes that
determinism is wanted here. The fix that keeps both is to draw a base seed from
the parent's own stream and derive per-worker seeds from it plus the worker
index, so the whole parallel run is reproducible given the parent's seed. Until
that is built: **stochastic routines dispatched through `parallel.pcall` do not
average independently when `cores>1`.** In practice that means KPM
(`kpm.random_trace`) run through `pcall` -- note the KPM moment loop's own
batching (`kpmtk/kpmnumba.py`) is a numba kernel, not `pcall`, and is not
affected.

---

## 2. Silently ignored arguments (the `3b43557` class)

### 2.1 `ldos.py:288` -- `get_ldos_tb` converts `operator` and never uses it

```python
if operator is not None: operator = h.get_operator(operator)
```

and the name is dead from there on. Neither the `mode="green"` branch
(`green.bloch_selfenergy`) nor the `mode="arpack"` branch (`ldos_diagonalization`,
which receives only `**kwargs`, from which `operator` was already consumed as a
named parameter) ever sees it. `h.get_ldos(e=1.0, operator="sz")` on a
Zeeman-polarized chain returns an array byte-identical to `h.get_ldos(e=1.0)`.
Because it is a named parameter it is not caught by an unknown-kwarg check.
`dos.get_dos` honours the same argument, so this is an inconsistency inside the
library, not a missing feature. (The header of this file filed it under feature
work; the entry is right and the header was wrong.)

**Status:** fixed in `05e0f17`. Both branches now honour it. `mode="arpack"` forwards it
into `ldos_diagonalization`, whose `ldos_waves_from_eigsystem` already had the
machinery (`<psi|A|psi>` times the local density `|psi(i)|^2`) and is what
`ldosmap` uses. `mode="green"` contracts the Green's function with the
operator through a new `ldos.green2ldos`, as `-Im(diag(G A + A G))/(2 pi)`.
The Hermitian combination is not cosmetic: `diag(G A)` and `diag(A G)` are
site-by-site complex conjugates, so each alone carries a piece the other
cancels -- large enough to give a nonzero `sy` map on a purely real
Hamiltonian, with a sign set by which matrix goes first. That is 1.2's
question again, and the sum rule `sum_i LDOS_A(i) = -Im Tr(G A)/pi` (what
`green.green_operator` returns) holds for either ordering, so it does not
catch it; the `sy`-is-zero test does.

The two modes weight the operator differently -- global expectation times
local density, versus the local matrix element -- and coincide only when the
eigenstates are eigenstates of the operator. Both integrate over sites to the
same operator-resolved DOS. Switching `ldos_waves` to the local form as well
would make them agree everywhere, but it changes a working path that
`ldosmap` and `multi_ldos` use, so it is left alone and documented instead.
`tests/ldos/test_ldos_operator.py`.

Found while fixing this, and fixed with it: inside the same `mode="green"`
branch the `nk is not None` sub-branch computed `-Im(gb[i,i])` with no
`1/pi`, where its adaptive sibling three lines below and the whole
diagonalization path (`ldos_waves_jit`) divide by pi. The same LDOS came out
a factor pi apart depending on which sub-branch ran. It now agrees with the
diagonalization result to 1e-15.

### 2.2 `bandstructure.py:152` -- `ewindow` applied only when an operator is given

The `if callable(ewindow): if not ewindow(e): continue` test lives inside
`getek`'s `else` (operator is not None) branch. The `operator is None` branch
and the batched fast path (taken when `num_bands is None and operator is None
and not h.is_sparse`) never consult it.

```
h.get_bands(nk=20, ewindow=lambda e: abs(e)<0.5)
  operator=None  -> 80 bands, max|e| = 3.0      <- filter ignored
  operator="sz"  -> 16 bands, max|e| = 0.382
```

**Status:** fixed in `05e0f17`. The filtering moved into a shared `kes2rows` helper that
every path packs its output through, so the three branches cannot drift
apart again. `callback` still sees the full unfiltered set of energies at
each k-point, which is what the operator branch always did.
`tests/bandstructure/test_ewindow.py`.

### 2.3 `hamiltonians.py:668` -- `add_hamiltonian` drops terms in new lattice directions

```python
for i in range(len(self.hopping)):
    d = tuple(self.hopping[i].dir)
    if d in hd: self.hopping[i].m = self.hopping[i].m + hd[d]
```

iterates over the *target's* existing directions, so any key of `hd` absent
from `self.hopping` is never merged and never reported. Only the `(0,0,0)`
block always lands. `h.add_hopping_matrix(fm)` with `fm` a second-neighbour
hopping on a 1D chain leaves the Hamiltonian bit-identical (multihopping norm
1.414214 before and after, directions still `[(-1,0,0),(1,0,0)]`), where the
same `fm` through `g.get_hamiltonian(mgenerator=fm, is_multicell=True, nc=3)`
correctly produces `(+-2,0,0)` with norm 0.7071.

In fairness: `add_kekule`/`add_chiral_kekule` on a first-neighbour honeycomb
stay within the existing directions, so the common uses are unaffected.

**Status:** fixed in `05e0f17`. `add_hamiltonian` now merges the two `MultiHopping`
dictionaries and calls `set_multihopping`, i.e. it goes through the same
machinery the `+` operator (`algebratk/hamiltonianalgebra`) already used
correctly -- which is also what the test compares against. `add_hopping_matrix`
gained a `**kwargs` passthrough so `nc`, the neighbour cutoff of the generated
Hamiltonian, can be raised above its default for a long-range `fm`; without it
the generated Hamiltonian has no long-range directions for the merge to find.
`tests/hopping/test_add_hamiltonian_directions.py`.

### 2.4 `hamiltonians.py:514` -- `add_sublattice_imbalance` is a silent no-op off the beaten path

```python
if self.geometry.has_sublattice and self.geometry.sublattice_number==2:
    add_sublattice_imbalance(self,mass)
else: pass
```

On triangular (`has_sublattice=False`) the bands are unchanged, so a caller
building "a gapped semiconductor" the way `documentation/user_guide.md` does
gets a gapless metal and no warning. On kagome (`sublattice_number==3`) it also
no-ops, even though the neighbouring `add_antiferromagnetism` routes
`sublattice_number>2` to `magnetism.add_frustrated_antiferromagnetism` -- so
two adjacent methods disagree about what a 3-sublattice geometry supports.

**Status:** open

### 2.5 `kpointstk/kmesh.py:39` -- `endpoint` dropped for dimensionality 2

`kmesh` forwards `endpoint` to `np.linspace` in the 1D and 3D branches but
calls `kmesh2d(nk,nsuper)`, whose signature takes no `endpoint` and whose two
`np.linspace` calls are pinned to `endpoint=False`. `kmesh(2,nk=3,endpoint=True)`
returns the same mesh as `endpoint=False`; `kmesh(1,...)` correctly differs.
No in-tree caller passes `endpoint=True` today, so the impact is latent.

**Status:** fixed in `05e0f17` -- `kmesh2d` takes `endpoint` and `kmesh` forwards it, like
the 1D and 3D branches. `tests/geometry/test_kmesh_endpoint.py`.

### 2.6 `greentk/rg.py:192` -- the numba backend swallows `nite` and `error`

`green_renormalization_jit` (and `_jit_batch`) take only
`(intra,inter,energy,delta,**kwargs)` and recompute `nite = max(int(100/|delta|),100000)`
and `error = |delta|*1e-6`, dropping whatever the caller passed --
for exactly the two arguments `green_renormalization_python` documents as
caller-supplied (`nite`: "asks for a *truncated* decimation"). With `nite=2`
the two backends' surface Green's functions differ by 1.05 on an O(1) quantity:
the numba path returns a fully converged answer where the Python path returns
the requested 2-step truncation. The `error` divergence is milder (4e-7 at
`error=0.01`).

**Status:** fixed in `05e0f17`, in both `green_renormalization_jit` and
`green_renormalization_jit_batch`. Both take `nite`/`error` as named
parameters and only fall back to the derived values when they are absent, and
the two numba kernels take a `truncate` flag so a caller-supplied `nite` means
exactly what it means in the Python path: that many iterations, no convergence
test, and -- this is the part that would otherwise have undone the fix -- no
`_fix_green_renormalization` afterwards, since a deliberately truncated result
fails the Dyson residual by construction and would have been silently
re-converged. The three backends now agree to 1e-12 or better with `nite` or
`error` given, and bit-identically without.
`tests/green/test_rg_truncation_and_error.py`.

### 2.7 `greentk/kchain.py:29` -- `hs` discarded beyond nearest neighbours

For Hamiltonians with hoppings beyond NN, `green_kchain` dispatches to
`green_kchain_NNN`/`green_kchain_LR`, which forward `**kwargs` into
`dysonNNN`/`dysonLR` and finally into `green_renormalization_python`'s
swallowing `**kwargs` -- so the `hs` surface-onsite matrix is accepted and
silently dropped.

**Status:** implemented rather than refused, in `05e0f17`. The decimation solves
`gs = (ez - ons - hop gs hop^dag)^(-1)`, and `hop gs hop^dag` is the
selfenergy of everything attached below the surface cell, which does not
depend on that cell's own onsite matrix -- so replacing the onsite and
re-solving is exactly what `green_kchain_NN` already does. Beyond nearest
neighbours the chain is a chain of supercells and only the first of the
sub-cells is the actual surface, so `hs` replaces that block alone
(`greentk/dyson.surface_onsite_dyson`). `green_kchain_NNN`/`_LR` resolve a
callable `hs` at their own k-point, where k is known, so the `dyson*` routines
only ever see matrices. Two checks pin it, neither of which needs a new
convention to be trusted: on a nearest-neighbour Hamiltonian the long-range
solver collapses to a one-cell supercell and reproduces `green_kchain_NN`
(1e-11), and for any range, passing the surface cell's *actual* onsite matrix
returns the Green's function the decimation already gives (1e-16), because
that is the equation `gs` solves. A third check settles the convention
itself rather than assuming it: a 400-cell chain with a non-trivial onsite
matrix on its first cell alone, inverted directly, agrees to 1e-12 -- as
accurately as the no-`hs` path does against the same reference, so the
one-Dyson-step formula is exact and not an approximation inherited from
`green_kchain_NN`.
`tests/green/test_kchain_surface_onsite_longrange.py`.

### 2.8 `spectrum.py:221` -- `real_space_vev`'s `nrep` is ignored

The signature declares `nrep=3`; the body calls
`h.geometry.write_profile(rho, nrep=5, name=name)`. Also, the documented
default `operator=None` dies in `operators.Operator(None)` on a bare `raise`
-> `RuntimeError: No active exception to reraise`.

**Status:** fixed in `7087ee6`, alongside 1.2/1.3 -- the same four lines. `nrep` is forwarded
and `operator=None` now means the density.

---

## 3. Hard crashes on live code paths

### 3.1 `dos.py:219` -- `dos2d_ewindow` misses an import

`e481ffb` routed the H(k) stacking through `hk_matrix_batch` but, unlike the
sibling edits in `calculate_dos_hkgen` (line 133) and `dos1d_ewindow` (line
242), left the local import on line 217 as `from .htk.eigenvectors import
peigvalsh`. `hk_matrix_batch` is not a module-level name in `dos.py`, so
`dos.dos_ewindow(h, use_green=False, ...)` on any 2D Hamiltonian raises
`NameError: name 'hk_matrix_batch' is not defined` before doing any work. The
1D sibling works. Reachable as `from pyqula import dos; dos.dos_ewindow(...)`
-- not exposed as a `Hamiltonian` method.

**Status:** fixed in `66df675` -- the local import names both, like its two siblings.
`tests/densitymatrix/test_dead_code_paths.py`.

### 3.2 `filling.py:106` -- `set_filling(average=False)` is dead

`full_dm`'s smearing parameter is named `T` but it forwards it as `delta=T` to
`full_dm_accumulate`, so any caller who also passes `delta` collides.
`set_individual_filling` does so unconditionally:

```
filling.py:106     out = hi.get_vev(delta=1e-2,**kwargs)
densitymatrix.py:40  return full_dm_accumulate(h,delta=T,**kwargs)
TypeError: full_dm_accumulate() got multiple values for keyword argument 'delta'
```

raised inside the first `fsolve` residual evaluation, so the whole non-averaged
filling path cannot run. `examples/2d/decorated_triangular/main.py:9` calls
exactly this.

Secondary, *not* verified because the crash comes first: even once callable,
`fmin` compares `get_vev`'s per-site occupancy (0..2 for a spinful Hamiltonian)
against a `filling` the rest of the module treats as 0..1.

**Status:** both fixed in `66df675`. `full_dm` accepts `delta` as an explicit alias for `T`
-- they are the same smearing under two names -- and raises if given both.
`set_individual_filling` now aims at `filling * (2 if has_spin else 1)`, the
per-site occupancy `get_vev` actually returns, and calls `check_filling`. The
secondary defect turned out to be real: at `filling=0.5` on a spinful chain the
solver had been aiming at 0.5 electrons per site where half filling is 1.0.

### 3.3 `chitk/chiAB.py:106` -- the GPU branch does not densify

```python
np.array([hk(k) for k in ks])
```

For an `is_sparse` Hamiltonian `hk(k)` returns a `csc_matrix`, so this gives a
`dtype=object` array, fed straight to `chijax.chi_matrix_kmesh_gpu`:
`TypeError: Dtype object is not a valid JAX array type`. The CPU path on the
same Hamiltonian works and agrees exactly with the dense result (0.0
difference), so this is purely a backend asymmetry -- `hk_matrix_batch`
(`htk/eigenvectors.py:7`), added by `e481ffb` precisely to close this hole, was
not applied here. Lines 271-275 (`lg.eigh(m1)` on the raw `hk(k)`) have the
same shape of problem.

**Status:** fixed in `2c28e53`, in both places, with `algebra.todense`. A third instance of
the same class turned up while testing it and is fixed too: the numba kernels
multiply the operator against complex wavefunctions and numba's `@` refuses a
mixed-dtype product, so a real-valued named operator (`A="sz"`) made
`ij_mode="accelerated"` fail to compile. All four backend combinations
(dense/sparse x explicit/accelerated/GPU) now agree to 1e-15.
`tests/chi/test_chi_backend_agreement.py`.

### 3.4 `greentk/kchain.py:12` -- `np.identity` with numpy never imported

`green_kchain_NN`'s `hs` branch calls `np.identity`, but `greentk/kchain.py`
imports only `.rg.green_renormalization` and `..algebra`. Asking for a modified
surface onsite matrix raises `NameError` instead of returning the surface
Green's function.

**Status:** fixed in `66df675` -- the module imports numpy.
`tests/green/test_kchain_surface_onsite.py`.

### 3.5 `hamiltonians.py:1067` + `vev.py:4` -- `get_dm_vev` has never worked

```python
def get_dm_vev(self,*args,**kwargs):
    from . import get_dm_vev        # a package attribute, not a module
```

`src/pyqula/__init__.py` deliberately leaves every submodule import commented
out (documented in CLAUDE.md), so nothing will ever populate that name:
`ImportError: cannot import name 'get_dm_vev' from 'pyqula'`, for any argument
on any Hamiltonian. Behind it sits a second, independent break: `vev.py:4` is
`from operators import Operator`, a Python-2 style absolute import that fails
under Python 3 (`ModuleNotFoundError: No module named 'operators'`) -- so
`pyqula.vev` cannot be imported at all.

That matters beyond the dead method: `fbee7c9` applied its density-matrix
transpose fix in two places, and one of them was `vev.get_dm_vev`, i.e. the fix
is recorded in a module that cannot run. Either repair both layers or delete
`vev.py` so the fix is not filed in an unreachable place.

**Status:** fixed in `66df675`, both layers -- `from .vev import get_dm_vev` and
`from .operators import Operator`. The transpose fix filed there turns out to
be correct: `get_dm_vev` now reproduces an explicit occupied-state sum for
`sx`, `sy` and `sz` alike. `tests/densitymatrix/test_dead_code_paths.py`.

---

## 4. Aliasing and missing Hilbert-space guards

### 4.1 `hamiltonians.py:543-545` -- `get_supercell` returns `self`

```python
if nsuper is None: return self
if nsuper==1: return self
if self.dimensionality==0: return self
```

Three early returns inside a method whose contract everywhere else is "return a
new, larger Hamiltonian". A parameter sweep

```python
for n in [1,2]:
    hn = h.get_supercell(n)
    hn.add_zeeman([0,0,0.5])
```

corrupts the base `h` -- bandwidth 3.618034 -> 4.618034, because the `n=1`
iteration's Zeeman field leaks back while `n=2` returned a fresh object. The
corruption is silent. `geometry.py`'s `get_supercell` has the same hole
(`if self.dimensionality==0: return self`), which additionally makes a 0D
supercell request a silent no-op rather than an error.

**Status:** fixed in `ccdee4a`. Every no-op branch returns `self.copy()`, in
both `hamiltonians.py` and `geometry.py`. The 0D case keeps its no-op meaning
rather than raising: `geometry.replicate_array` already collapses `nrep` to 1
for dimensionality 0, and internal callers pass a default `nrep>1` to
geometries that may be 0D (`ldos.get_ldos_tb`'s `nrep=5`), so raising would
break paths that are correct today. Moving the scalar `nsuper==1` test inside
the "not a tuple" branch additionally lets `nsuper` arrive as a numpy array,
which used to raise "the truth value of an array with more than one element is
ambiguous". `htk/mode.py`'s `reduce_hamiltonian` was folded in as the one
other live instance of the class -- it returned a fresh Hamiltonian when the
spin could be dropped and `self` when it could not, so whether `h.reduce()`
aliased its input depended on the answer. `sctk/extract.py:45` and
`htk/hamiltonianmodify.py:18` were checked and are fine (both `self.copy()`
first); `embedding.py:85,87` are `Embedding` accessors with no mutating
callers. `tests/geometry/test_supercell_returns_new_object.py`.

### 4.2 `sctk/dvector.py:134` -- `dvector_non_unitarity` has no Hilbert-space guard

`extract.extract_triplet_pairing` assumes a 4x4 spin x electron-hole block per
site (`nr = m.shape[0]//4`). On a plain spinful (non-Nambu) Hamiltonian with 4
sites the Hilbert space is 8-dimensional, so `nr=2` and
`h.get_dvector_non_unitarity()` returns a `(2,3)` array of zeros for a 4-site
geometry -- not a d-vector, not even the right length, no error. On a 1-site
cell it returns an empty array, which `dvector_non_unitarity_map` then hands to
`np.savetxt` against nrep-replicated positions. `average_hamiltonian_dvector`
has `if not h.has_eh: raise` but no `spinless_nambu` check, so a spinless Nambu
Hamiltonian (2x2 per cell, `nr=0`) sails through and returns `[nan nan nan]`
from a mean over an empty slice.

**Status:** fixed in `05e0f17`. One `sctk/dvector.check_spinful_nambu` helper raises a
`ValueError` naming what the Hamiltonian actually is, called from all four
public entry points -- `extract_dvector_from_hamiltonian` (which covers
`dvector_non_unitarity` and `average_hamiltonian_dvector`, whose weaker
`if not h.has_eh: raise` it replaces) and the two `*_map` routines, which
bypass it by going through `h.get_hopping_dict()` and `matrix2dvector`.
`tests/superconductivity/test_dvector_hilbert_space.py`.

---

## 5. Not a library bug: a golden-value test that had gone red

### 5.1 `tests/moire/test_twisted_bilayer_graphene_bands.py` -- the pinned sum

The test pinned `np.sum(e) == -13.738703648103538` and failed on master at
`-13.735331753001255`. It is not a regression from any of the audit fixes:
every commit from the `0.0.94` release (`2832f36`) forward returns the same
`-13.7353...`, which is also the value the test was *originally* written with.
`6c8cfbc` replaced it with a number produced by some other environment, and
the test has been red since.

Dense `eigvalsh` at each k-point of the path, taking the eight eigenvalues
nearest zero, reproduces the sparse ARPACK output to 5e-15 -- exactly the
check `6c8cfbc`'s message describes doing by hand. Fixed in `ccdee4a` by
making the test perform that comparison instead of pinning a number, so it
cannot go stale against an environment again.

The general lesson for the 72 golden-value tests added in `22e35ff`: a pinned
sum that has already failed to port once will fail again. Where an
independent code path computes the same quantity, assert the agreement.

---

## Method note

The sweep was run as a four-agent workflow, one lens each, with these rules:
read-only on the repo; every finding backed by a repro script actually executed
from the scratchpad against `PYTHONPATH=src`; targeted single-file pytest runs
only, never piped; physics uncertainty reported as a question naming both
candidate conventions rather than asserted as a bug. Decomposing by bug class
seeded from recent commits, rather than by subsystem, is what made 4 agents
tractable over 408 files -- worth repeating for the uncovered areas listed at
the top.

One caveat on any timing that appears in this file or in the sweep's logs: the
machine was running two other sessions' jobs throughout (one benchmark at ~8
cores of 12), so every wall-time measured here is inflated and none of it
should be recorded as a baseline. The correctness results are unaffected.
