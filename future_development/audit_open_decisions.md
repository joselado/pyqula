# Open decisions left by the second audit sweep

Everything in [`bug_audit_2.md`](bug_audit_2.md) that was **not** fixed, plus
the calls that were made one way and could reasonably be made the other. Three
findings and three judgement calls. Each is written up with what was measured,
so picking one up does not mean re-deriving the measurement.

Nothing here is a bug waiting to be squashed. Each is a question with a real
trade-off behind it, which is exactly why the fix pass stopped rather than
guessing.

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

---

## 3. Where a third sweep should start

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
- **The bond (`d != 0`) anomalous pairing prefactor**, verified structurally and
  by SU(2) equivariance but never against an independent numeric gap equation.
  A uniform prefactor error on the bond channel alone would have survived every
  check that was run.
- **`scftk/broydenmixing.py`** -- not exercised at all.
- **The per-site (array) filling branch** of `_run_anisotropic_scf`, whose
  Fermi handling is a separate code path from the one finding 19 repaired.

One method note worth carrying forward, because it cost a full wrong ranking
pass once already: grepping `tests/` for a module basename is wrong in **both**
directions. Resolve each module to its public entry point first, then grep
`tests/` *and* `examples/`.
