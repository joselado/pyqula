# Open decisions left by the second audit sweep

Everything in [`bug_audit_2.md`](bug_audit_2.md) that was **not** fixed, plus
the calls that were made one way and could reasonably be made the other. Three
findings and three judgement calls. Each is written up with what was measured,
so picking one up does not mean re-deriving the measurement.

Nothing here is a bug waiting to be squashed. Each is a question with a real
trade-off behind it, which is exactly why the fix pass stopped rather than
guessing.

Section 4, added on 27 September 2026, holds two more calls, raised by a
downstream caller rather than by the sweep; the maintainer decided both the
same day.

On 2026-09-24 the maintainer decided three of the ones still open: the AAA
`converged` flag (1.2), the factor of pi (2.1) and the weakened assertion
(2.3); each section says what was chosen and what it changed. GPU Tier 2
(1.3) is the one left open.

---

## 1. Three findings that are decisions, not repairs

### 1.1 `topologytk/qgt.py` -- batching the per-k `eigh` is not output-equivalent

`_qgt_over_kpoints` is a serial per-k-point `eigh` list comprehension, and it
looks like the same batching that gave 1.6x-6.0x elsewhere. It is not.

**What was measured.** `algebra.eigh` and numba's batched `eigh` agree on the
*eigenvalues* to 5e-15, but choose **different eigenvectors within degenerate
subspaces**, and the quantum geometric tensor is built from the eigenvectors.
The batched form therefore returns a different QGT wherever bands are
degenerate. The agent that looked at this wrote the batched version, measured
the disagreement, and reverted it rather than ship a faster wrong answer.

**The decision.** Either (a) leave it serial; (b) fix a gauge explicitly before
contracting, which means deciding *which* gauge is canonical for the QGT and is
a physics choice, not a refactor; or (c) batch only the non-degenerate case,
which is the kind of conditional gate this project has already rejected once as
"complicated and system-dependent".

Note the QGT's *Abelian* (single-band) contractions are gauge invariant and
could be batched safely; the non-Abelian multiband ones are the problem. Nobody
has checked whether the hot path is actually the multiband one.

**Decided and done (16 September 2026).** The premise of option (b) turned
out to be the real issue: the band-indexed Q_ij^{mn} is only gauge covariant
(it goes to U^dag Q U under a rotation of the subspace), so no choice of
canonical gauge makes its entries physical, and the old spin-block test passed
only because LAPACK happened to return spin-pure vectors for the degenerate
pair. `non_abelian=True` now returns the tensor in the orbital basis,
sum_{m,n} |u_m> Q^{mn} <u_n| = P dP dP P, which depends on the projector
alone, so the diagonalization is batched (`_qgt_batch`, chunks of 256
k-points). Checked against the serial code to 1e-13 (the old band tensor
sandwiched back into the orbital basis), against a finite difference of the
projector to 5e-8 on a spin-mixed model, and the Chern and analytic two-band
tests are unchanged. The speedup on an nk=30 mesh was 2.5x at 36 orbitals and
14x at 4, measured with another test run on the machine, so read those as
indicative. The non-Abelian output changed shape, from
(dim,dim,nocc,nocc) to (dim,dim,n,n).

**Orbital positions, decided 25 September 2026.** A check of the tensor
against the literature found that every oracle in its tests took P(k) from the
same `hk_gen`, whose Bloch phase drops the orbital positions, so the suite
could not see that the lattice gauge is not the physical one. On Haldane with
a sublattice mass the Berry curvature at three C3-related k-points came out
0.0266, 0.0265 and -0.0406 (identical spectra), the BZ average of Tr g went
from 0.693 to 1.989 under a 2x1 supercell instead of doubling, and PythTB
2.0.2, which the module cited as its reference and which puts tau_j - tau_i
in the phase, differed pointwise by up to a factor of 20. The maintainer chose
`gauge="atomic"` as the default, with `"lattice"` kept, as in the superfluid
weight; the atomic term is the commutator 2 pi i [H, F_i] with the fractional
positions, added in the lattice eigenbasis as 2 pi i <m|F_i|l>, and the new
tests pin C3 symmetry, pointwise supercell unfolding in 2D and 3D, the SSH
spread (d^2/4)(1+r^2)/(1-r^2) and PythTB agreement through the atomic
projector. The Chern number is the same in both gauges. The same pass removed
`topologytk/quantumgeometry.py`, whose "quantum geometry" came from an
integrand antisymmetric in x and y and so never contained the metric (its one
live routine, the real-space Berry map, moved to `topologytk/green.py`), added
3D, and sized the k-batch from the number of orbitals (a fixed 256 cost 4.8 GB
at 400 orbitals). The output is still in reduced coordinates; a Cartesian
option was not added.

### 1.2 `aaatk/selfenergy_aaa.py` -- what `converged=True` should mean

`SelfenergyAAA` reports `converged=True` while the local relative error is 7.7%,
because its tolerance is normalized by the **window-maximum** |Sigma| rather
than by the local value. Near a node of Sigma the absolute error is small and the
relative error is not.

**Why it was left.** This is exactly what the class's own docstring specifies
("against `tolerance` (relative to the largest sampled |Sigma|)"), and the audit
entry itself classifies it as within contract. Changing it changes what a
documented flag means for every existing caller, and would make fits that
currently pass start failing -- a behaviour change with no bug behind it.

**The decision.** Keep the contract and document the caveat at the call sites
that care, or switch to a mixed absolute/relative criterion
(`err < atol + rtol*|Sigma_local|`, the usual shape) and accept that some fits
now need more support points. If the latter, the AAA suites will need their
tolerances revisited together, not one at a time.

**Decided on 2026-09-24: the mixed criterion.** A fit is converged when at
every validation energy the largest entry of |Sigma_fit - Sigma| is below
`atol + tolerance*|Sigma|`, with |Sigma| the largest entry of the true
self-energy at that energy. `atol` defaults to `tolerance*delta`, the
broadening: a Green's function with broadening delta has |G|<=1/delta, so an
error in Sigma below `tolerance*delta` moves it by less than a fraction
`tolerance` of itself, and below that asking for relative accuracy means
nothing. `validation_error` is now that local relative error,
max |Sigma_fit - Sigma|/(atol/tolerance + |Sigma|).

Changing the validation alone was not enough. On the audit's two-orbital
lead the fit then never converged: refinement ran to `ncand_max=20000` with
the local error stuck at 15%, because `aaa()` stops once its residual is
below its tolerance times the largest |F| of the entry, so where |Sigma| is
small the fit is never asked to be accurate. `aaa()` now takes an optional
`scale`, one positive number per sample point, which replaces max|F| both in
the greedy choice of the next support point and in the stopping test (the
least-squares step is unchanged, and `scale=None` is the standard
algorithm), and `SelfenergyAAA` passes the same floored local |Sigma| the
validation uses.

Measured on the leads of the reproduction, tolerance 1e-3, delta 1e-4, the
local relative error on a 401-point grid the fit never saw:

| Lead | Before | After | True solves before/after |
| --- | --- | --- | --- |
| single-orbital chain | 8.6e-5 | 6.9e-5 | 2385 / 2385 |
| two-orbital, the audit's (seeds 1, 2) | 1.05e-1, converged=True | 9.0e-5 | 2385 / 3481 |
| two-orbital, seeds 3, 4 | 2.4e-4 | 6.7e-5 | 3481 / 3481 |

The audit's lead costs 46% more true solves and its build went from 1.7 s to
13.7 s, the price of a flag that now means what it says. The 40 AAA-related
tests (`tests/keldysh/test_selfenergy_aaa*.py`, `test_shared_selfenergy_sweeps.py`,
`test_kappa_finite_temperature.py`, `test_current_jax.py` and
`tests/green/test_rg_batch_thread_safety.py`) all pass unchanged, in 600 s
against 386 s before, 1.55x. The cost falls on the superconducting leads: the
catastrophic-cancellation dI/dV test went from 29 s to 69 s and the
`nmax_max=40` current sweep from 32 s to 68 s, while
`test_build_selfenergy_aaa_matches_direct_dc_current` went from 50 s to 37 s.
The margin between the fit's own tolerance and the validation's
(`aaa_tolerance=0.1*tolerance`) was kept at 10x; it is the first knob to
revisit if the build cost matters more than that margin.

### 1.3 GPU Tier 2

`documentation/gpu_porting_plan.md` requires explicit per-tier sign-off, and
Tier 2's entry asks for a size/batch crossover sweep that needs a GPU to
measure. Tier 1 (KPM) landed in `35b7a43`.

**What the sweep added to the picture.** Every dense k-mesh path now funnels
through two functions (`htk/eigenvectors.py`'s `hk_matrix_batch` +
`parallel_diagonalization`), which is the shape the plan wanted before a GPU
port -- so the port is now a smaller change than when the plan was written. The
one concrete obstacle noted: `full_dm_accumulate`'s `batch_size=16` would starve
a GPU, and that constant would have to become size-aware.

**The decision.** Yours, per the plan's own rule. Nothing should start without
it.

---

## 2. Three calls made one way that could be made the other

### 2.1 `fermisurfacetk/spinsplitting.py` keeps its factor of pi

The sweep found and fixed a family of missing `1/pi` normalizations: the
adaptive DOS, both `dos_ewindow` routines, `get_multildos`'s maps and its
`DOS.OUT`, the atomic multi-LDOS, and the real-space LDOS. `calculate_dos`
returns pi times a sum of unit-normalized Lorentzians, and every routine that
reports a density of states has to divide.

`spin_splitting_density` (`spinsplitting.py:52`) is the last member of that
family that does not divide, and it was **deliberately left alone**. Unlike the
others it does not report a density of states: it reports a
splitting-*weighted* spectral density, `sum_n (e_up - e_dn)^2 * L(E - e_avg)`,
whose absolute scale is a convention rather than a sum rule. No finding claims
it is wrong, and there is no oracle saying which normalization is right --
changing it would move numbers on the strength of an aesthetic argument about
consistency.

**If it should change**: dividing by pi makes its energy integral the mean
squared splitting per cell, which is at least an interpretable quantity. That is
the argument for doing it. It is a one-line change plus whatever pins it.

**Decided on 2026-09-24: divide.** `spin_splitting_density` now divides by pi,
so its integral over energy is the squared splitting summed over bands and
averaged over the zone, and every value it returns is pi times smaller than
before. `test_spin_splitting_density_integrates_to_the_squared_splitting`
pins that integral on the square altermagnet to 1%, the tails beyond the
window holding 0.3% of it, and gives 3.13 times it on the unfixed source.

### 2.2 `multicell.turn_multicell` still aliases

`ccdee4a` established the contract that a `get_*` returning a new object must
not hand back its receiver, and the sweep closed the `turn_no_multicell`
sibling (#13) -- that one was reproducible: `h2 = h.get_no_multicell();
h2.add_onsite(1.0)` lifted the *original* chain's bandwidth from 2.0 to 3.0.

`turn_multicell` has the identical shape and was **not** changed, on the fixing
agent's argument:

- `htk/kchain.detect_longest_hopping` calls `h.get_multicell()` unguarded, once
  per energy, inside `greentk/selfenergy`'s decimation;
- three call sites (`conductivitytk/kubo.py:38`,
  `sctk/superfluidweight.py:307`, `topologytk/qgt.py:64`) already work around
  the alias with an explicit `.copy()` and a comment saying why;
- turning an O(1) short-circuit into an unconditional deepcopy is a performance
  change that could not honestly be benchmarked while thirteen agents shared the
  machine.

So the public `h.get_multicell()` currently contradicts `ccdee4a`'s stated
contract. **The clean resolution**, if it is wanted, is to split the two: keep
`multicell.turn_multicell`'s fast path for internal callers and have the public
`Hamiltonian.get_multicell` copy. That also lets the three explicit `.copy()`
workarounds be removed. Needs an idle-machine measurement of the decimation
first.

**Done (17 September 2026), the clean resolution.** `Hamiltonian.get_multicell`
returns a copy when the receiver is already multicell, and
`multicell.turn_multicell` keeps its O(1) return for internal read-only
callers, which the per-energy `detect_longest_hopping` and the per-k-point
`current.derivative` now call directly; the three `.copy()` workarounds are
gone. Instead of a timing, the Hamiltonian copies were counted, which does
not depend on what else runs on the machine: 401 before and 402 after for a
whole `kdos.surface` call on the honeycomb lattice (20 energies, 10
k-points), and none for 50 `current.derivative` calls, so the decimation
gained nothing per energy. The regression test is
`tests/hopping/test_hamiltonian_method_contracts.py`, which compares the
spectrum rather than the bandwidth, since a uniform onsite shift leaves the
bandwidth unchanged and a bandwidth test passes on the aliasing code too.

### 2.3 One assertion was deliberately weakened

In `tests/scf/test_spinspin_rotational_symmetry.py`,
`assert scf2.converged` became `assert scf2.hamiltonian is not None`.

The reasoning is a real consequence of finding 21 (the magnetism constraints
used to rewrite only the onsite block, so they were silent no-ops for any
intersite interaction). Once the constraint is enforced on the bond mean field
too, the `no_inplane_magnetism` run has no magnetic channel left and settles
onto a complex bond order whose **phase is an exact flat direction**: on a chain
`t1 -> t1*exp(i*phi)` is a rigid shift of the dispersion in k, so every phase has
the same energy at fixed filling, and the SCF cannot converge in phase even
though it has converged in everything observable.

The fixing agent flagged this as the single edit it most wanted a maintainer to
look at, and that judgement is carried here rather than left in a diff. **If the
weakening is not acceptable**, the alternative is to fix the gauge explicitly in
the test (pin the phase of one bond and assert convergence of the rest), which
is more code but restores a real convergence assertion.

**Decided on 2026-09-24: pin the phase.** The test passes a `callback_mf` that
makes the whole bond, bare hopping plus mean field, real at its largest entry
after every mixing step, which is exactly the gauge freedom and nothing more,
and it asserts `scf2.converged` again: the run converges to E=-0.697426 with no
moment, where without the pin it stops unconverged at -0.695556. `SxSx` and
`SySy` could not take a `callback_mf` before, since they pass their own to
`SzSz` for the constrains and a second one collided with it as a `TypeError`;
they now accept one, applied in the laboratory frame after the constrains, as
`SzSz` does.

---

## 3. Where a third sweep should start

**Done on 24 September 2026.** Every area below except the bond pairing
prefactor, already closed, was swept in [`bug_audit_3.md`](bug_audit_3.md): 18
findings, 16 fixed, the qtci backend kept by the maintainer's decision. That
file's last section says where a fourth sweep should start. The list is kept
as it was written, as the record of what the third sweep was aimed at.

The eight lenses each recorded what they did not reach, in section 4 of
[`bug_audit_2.md`](bug_audit_2.md). The largest gaps named there, in rough order
of how much is unexamined:

- **The jax SCF solvers.** `scftk/densitydensity_jax.py` (856 lines) and
  `scftk/vjinteraction_jax.py` (604 lines) were read only for their entry
  points. None of the solvers (newton, newton_krylov, fsolve, error_gradient,
  linear_mixing, broyden_mixing) was ever run by the sweep.
- **3D superfluid weight.** Everything exercised was 1d/2d.
  `twist_directions` returns three axes for `dim==3`, and the off-diagonal
  four-point finite-difference stencil and the `nd=3` einsum loop have never
  been run. `tests/superfluid/` is 1d/2d only.
- **`integration="qtci"` as an SCF density-matrix backend** -- untouched.
- ~~**The bond (`d != 0`) anomalous pairing prefactor**~~, closed on 17
  September 2026 by `tests/scf/test_bond_pairing_prefactor.py`: a mean-field
  energy functional written by hand (Hartree, Fock and pairing contractions,
  sharing no code with the SCF kernels) is stationary at the self-consistent
  extended s-wave and triplet states of a doped V1=-2.5 chain, the slope
  falling as eps^2 to 5e-10, while a factor of 2 or 1/2 on the pairing energy
  gives a slope of order 0.1. The functional also reproduces
  `scf.total_energy` to 7e-14.
- **`scftk/broydenmixing.py`** -- not exercised at all.
- **The per-site (array) filling branch** of `_run_anisotropic_scf`, whose
  Fermi handling is a separate code path from the one finding 19 repaired.

One method note worth carrying forward, because it cost a full wrong ranking
pass once already: grepping `tests/` for a module basename is wrong in **both**
directions. Resolve each module to its public entry point first, then grep
`tests/` *and* `examples/`.

---

## 4. Two calls raised by a downstream caller

On 27 September 2026 the guiqula session, which wraps pyqula calls in a GUI,
sent seven notes. Five were repairs and were made: `kdos_bands` and
`get_bands` honour `write=False` (the k-path files included), clearer errors
for `add_valley_exchange`, `add_crystal_field` and the K label on a
non-hexagonal lattice, the `shift`/`rotate` docstrings, `LatticeGas` and
`LatticeIsing` no longer setting `g.nrep` on the caller's geometry, and
`SpinModel.energy()` returning a float with `minimize_energy(silent=True)`;
a sixth, that a path through K and M on the honeycomb lattice relies on
`closest_path` because the two labels are not neighbours in reduced
coordinates, went into `get_kpath_labels`'s docstring. The two below change behaviour across the package, so they were put to the
maintainer, who decided both on 27 September 2026.

### 4.1 jax's double precision depends on what was imported first

Eleven modules call `jax.config.update("jax_enable_x64",True)` at import
(`htk/eigenvectorsjax.py`, `dmtk/fulldmjax.py`, `kpmtk/kpmjax.py`,
`chitk/chijax.py`, `chitk/pairchijax.py`, `bsetk/screeningjax.py`,
`scftk/densitydensity_jax.py`, `scftk/vjinteraction_jax.py`,
`graphenetk/relax.py`, `keldyshtk/current_jax.py`,
`transporttk/kappa_jax.py`). The jax users that do not set it --
`classicalspin.py`, `classicalspintk/align.py`, `graphenetk/elastic.py`,
`graphenetk/gsfe.py`, `fermisurfacetk/swarmfs.py`,
`symmetrytk/localsymmetry.py`, and the jax branch of `htk/bloch.py`, which
asks for `jnp.float64` and gets float32 without it -- run in single
precision or in double depending on whether one of the eleven happened to be
imported earlier in the same process.

The reproduction, a 3x3 triangular Heisenberg model minimized with
`np.random.seed(1)` and `tries=3`:

| imported first | `jax_enable_x64` | E per site | `magnetization[0]` |
| --- | --- | --- | --- |
| nothing | False | -3.0000002384 | (-0.431, 0.868, 0.246) |
| `scftk.densitydensity_jax` | True | -3.0000000000 | (-0.180, -0.620, -0.764) |

The 120-degree state is degenerate, so which texture the minimizer lands on
flips with the rounding. `gpu_rpa_spin_response.md` already records that
without x64 every complex128 request is silently truncated to complex64.

**The candidate fix.** Every jax module already imports `pyqula.gpu` and
calls `gpu.apply()` at import, so setting x64 once there (at `gpu.py`'s
import, or inside `apply`) and deleting the eleven scattered lines would make
the precision the same whatever the import order. The single-precision
routes (`kpm_prec`, `chi_prec`, `eigh_prec`) cast to complex64 explicitly
and already run with x64 on, since their own modules set it, so they are
unaffected.

**Why it is a decision.** The seven modules above move from float32 to
float64 when imported on their own, so their numbers (and the classical-spin
texture in a degenerate case) change, and they get slower on a GPU, where
FP64 is the expensive precision. Nothing in `tests/` pins the float32
behaviour. guiqula works around it meanwhile by turning x64 on in its engine
and in the scripts it exports.

**Decided: double precision everywhere.** `gpu.py` switches x64 on at import
and the eleven scattered lines are gone; `htk/bloch.py`,
`graphenetk/elastic.py` and `graphenetk/gsfe.py`, the jax users that did not
import `gpu`, now do. The reproduction above gives E per site
-2.9999999999996 and the same texture, (-0.180, -0.620, -0.764), in both
import orders, which
`tests/classicalspin/test_precision_independent_of_imports.py` pins by
running it in two fresh interpreters. The notebooks that minimize a
classical spin model were executed in single precision and may land on a
different, degenerate texture when re-run.

### 4.2 Files written to the working directory with no way to turn it off

`topology.chern` (and so `h.get_chern`) writes `CHERN.OUT` and
`BERRY_CURVATURE.OUT`, `h.get_spin_chern` the same,
`topology.get_berry_curvature_path` writes `BERRY_CURVATURE.OUT` (and the
k-path files), `topology.z2_invariant` writes `WANNIER_CENTERS.OUT`, and
`h.get_qpi` writes `DOS.OUT` and a `MULTIQPI` directory and returns `None`.
`berry_phase`, `chern_density` and `get_bands` already take `write=`.

**The candidate fix.** A `write=True` keyword on each, the way `get_bands`
has one, and a return value for `get_qpi`. It is an API pass over several
routines rather than a repair, and what `get_qpi` should return is a design
question of its own.

**Decided: a keyword on each, and a global switch over all of them.**
`mesh_chern`, `precise_chern`, `chern_qtci`, `wannier_centers` (and through
it `wannier_winding`, `z2_invariant`, `chern(integration="wannier")`),
`get_berry_curvature_path`, `topologytk.topologicalsector.spin_chern` and
`get_qpi` take `write=`. `get_berry_curvature_path` and `kdos.interface`
used to read their own output file back, so with `write=False` they read a
stale file or none; both now build their result in memory. `get_qpi`
returns `(q, energies, qpi)`, with `qpi` of shape (nenergies, nq) as
`get_qpi_impurity` returns it, and no longer empties `MULTIQPI/` when it is
not writing.

The maintainer also asked for one switch over every routine:
`filewrite.set_write(True|False|None)` in `src/pyqula/filewrite.py`. Every
function that takes `write=` now defaults to `None` and resolves it first
thing with `filewrite.resolve(write, <its old default>)`: the call's value
if given, else the switch, else the old default. The internal calls that
pass `write=False` on purpose (kdos into `get_bands`, the Fermi surface
into `get_dos`...) are intermediate computations and keep it.
`tests/filewrite/test_global_write.py` checks, for eight routines, that the
switch set to False writes nothing and returns the same numbers as
`write=True`, and it resets the switch around every test, so that one test
cannot leave the rest of the suite silenced.

**What the switch does not reach.** Functions whose job is writing
(`g.write()`, `h.write_hopping()`, `h.write_magnetization()`,
`h.write_non_unitarity()`, the `write_*` helpers, `states.*`,
`scftypes.extract`) are not meant to, nor are the cluster job scripts of
`paralleltk` or the Wannier90 input, which `wannierpy` writes to a
temporary directory. The first pass left about thirty computational
routines that wrote with no keyword at all; the maintainer asked for them
too, the same day, and they now take `write=None` resolved the same way.
Among them are the mean-field loop, whose `MF.pkl` restart file is the
most common file the package writes (`generic_densitydensity` and its KPM
twin; `VJinteraction`, the spinful path, writes nothing and accepts
`write` so that every mean-field entry point does), `h.get_density()`,
`h.get_multildos()` in both projections, the older DOS, LDOS and k-list
helpers, the embedding DOS, LDOS and k-DOS, `real_space_chern`,
`real_space_vev`, `Omega_rmap`, `dOmega_dE_kmap`, the three band-structure
writers of `bandstructure.py`, `selected_bands2d`, `ev2d`, the legacy
Hubbard and Coulomb loops (their per-iteration files go to `os.devnull`),
the `massive_green` disk caches (which still read an existing cache), and
the geometry that `TMDC_MX2` and the island builders of `skeleton.py`
wrote as a side effect of building a Hamiltonian.

Twenty-five of them used to return nothing and only write, so with
`write=False` they would have done nothing; they now return what they
compute: `get_multildos` and `atomicmultildos.multi_ldos` return
`(x, y, energies, ldos)`, as does the embedding `multildos`; the older
`dos0d`, `dos0d_kpm`, `dos0d_sites`, `dos1d_sites`, `dos1d_ewindow`,
`dos2d_ewindow` and `dos_ewindow` return `(energies, dos)`; `dos_surface`
returns the energies with the surface and bulk DOS; `berry_bands`,
`current_bands` and `lowest_bands` return the columns of their
`BANDS.OUT`; `selected_bands2d` returns one array per band and `ev2d` one
array; `conduction_texture`, `chargechi_reciprocal`,
`magnetic_response_map`, `diagram2d`, `evolve_local_state`,
`dOmega_dE_kmap`, `ldos0d_wf`, `kdos1d_sites` and the embedding
`get_kdos` return their arrays too. Three small bugs went with them:
`lowest_bands` wrote whole k-vectors, brackets included, into
`BANDS.OUT` (it writes the k index now), `klist.default_v2` computed its
path and returned `None`, and the embedding `multildos` never closed its
index file. `tests/filewrite/test_global_write.py` covers eight of the
newly converted routines in the same off-versus-on sweep.
