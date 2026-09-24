# Open decisions left by the second audit sweep

Everything in [`bug_audit_2.md`](bug_audit_2.md) that was **not** fixed, plus
the calls that were made one way and could reasonably be made the other. Three
findings and three judgement calls. Each is written up with what was measured,
so picking one up does not mean re-deriving the measurement.

Nothing here is a bug waiting to be squashed. Each is a question with a real
trade-off behind it, which is exactly why the fix pass stopped rather than
guessing.

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
