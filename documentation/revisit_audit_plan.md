# Revisit-and-audit plan for the 2026-07-19 → 2026-08-15 development window

Maintainer-facing. Written 2026-08-15 after a survey of the last four weeks of work
(228 commits, 814 files, +48420/-12558, 266 new files, 91 new test files).

The goal is **not** to re-review everything. It is to target the places where the
development pattern of the last month plausibly left a wrong result that nothing
currently checks, plus optimizations that were found and then not applied
consistently.

**Headline: the work holds up well.** An entry-point-level audit of the new features
found most of them validated against genuine external oracles (analytic values,
independent code paths, brute-force references, published physical invariants). The
list of real gaps is short, and is in §3. Two systematic sweeps in §4 are the highest
value-per-effort items and are the main reason to spend Opus time here at all.

One loose end: a full-suite run faulted fatally during this survey (§3.5). It was not
reproduced, the machine was heavily loaded, and a second pytest process completed
cleanly in the same window — so it may well be environmental. One clean re-run closes it.

---

## 0. Method note, and a warning about how to survey this repo

An earlier pass of this survey ranked five features as "thin coverage" that turned out
to be well tested. The cause is worth recording, because it will mislead the next
survey too:

**Grepping `tests/` for a module basename gives wrong answers in both directions.**
`chern_qtci` appears nowhere in `tests/`, but `topology.chern(..., integration="qtci")`
is tested against trivial and topological Haldane models in
`tests/topology/test_haldane_chern.py` and demoed in two `examples/`. Likewise
`grep -w plasmon_bands` misses `h.get_plasmon_bands` entirely, because `_` is a word
character.

**Resolve each module to its public entry point first** — the `Hamiltonian`/`Geometry`
method, the string-dispatch keyword, or the top-level function — then grep `tests/`
**and** `examples/` for that. Everything in §2/§3 below was redone that way.

---

## 1. Framing: what a green test suite would prove, and what it wouldn't

Assume the suite is green (status note at the end of this section). Two structural
caveats bound how much that is worth:

1. **At least 48 of 165 test files assert against values recorded from the code's own
   earlier run** ("must match the values recorded from a known-good run"). These pin
   current behavior. If a sign or normalization was already wrong when the reference
   was recorded, the test locks the error in and will actively resist the fix. This
   repo fixed *four* normalization/sign bugs in this window (`d0cd47d` kdos π,
   `ed190a5` dos_kpm mesh + Green 1/π, `9c7b472` chern_density sign, `1ce2894` KPM
   operator-DOS scaling) — the failure mode is demonstrated, not hypothetical.

   *(Counted two independent ways: 54 files whose docstrings admit a recorded
   reference, 49 files asserting an 8+-digit float literal, 48 in the intersection.
   The docstring count alone over-counts — `test_chern_density_plateau_matches_wilson_
   chern_sign_and_scale` matches it but genuinely validates against the Wilson loop.
   Treat 48 as a floor and re-check any individual test before calling it
   self-referential.)*
2. **The entire Keldysh/transport suite is single-orbital 1D chains** (§4.1), which is
   provably blind to the class of bug found there in `1b82095`.

So: do not open any item below with "the suite passes, so this is probably fine."

**Suite status: not verified during this survey.** One full run faulted (§3.5) and a
rerun was inconclusive under heavy machine load. Treat "green" as this repo's documented
steady state, not as something confirmed here. Both caveats above stand either way.

---

## 2. Settled during the survey — do not re-derive

**`qutecipytk` is a verbatim vendor.** Diffed the whole tree against
`joselado/qutecipy` HEAD (`6df39e7`). Same file set; **11 differing lines total**, all
import-path rewrites in `__init__.py` plus two stale path strings in comments
(`matrix/rrlu.py:96`, `tci1.py:351`). No logic was modified during vendoring.

This collapses the audit scope for the largest new mass in the repo from **4272 LOC to
the ~415 LOC of `qtcitk/`** (pyqula's *use* of it). Do not review the port itself.

**Confirmed empirically during this survey:** upstream ships 16 test files that were
not vendored; run against the vendored copy (imports rewritten `qutecipy` →
`pyqula.qutecipytk`), **all 100 tests pass** (389s, 10 benign warnings). Import path
verified to resolve to `src/pyqula/qutecipytk`, not a stale PyPI copy. What this
establishes precisely: **vendoring introduced no defect.** Upstream's own numerical
correctness remains upstream's problem — the `divide by zero` / `invalid value` warnings
from `matrix/aca.py:208` are upstream behavior, not something the vendoring created. The
only remaining action here is to vendor those tests so it stays that way (§6.1).

**Features confirmed well validated — no audit needed.** Each of these checks against
something outside itself, which is the standard the rest of the plan is measured by:

| Feature | Oracle it checks against |
|---|---|
| `topology.chern(integration="qtci")` | analytic Chern on trivial + topological Haldane |
| `get_dm_qtci` | the full dense density matrix, and a symmetry-protected-zeros case |
| QGT (`chern_from_qgt`, metric) | independent Fukui-Hatsugai-Suzuki Wilson loop, **plus** model-independent geometric bounds (Roy, PRB 90) |
| `qpitk` FFT self-convolution | the O(n⁴) brute-force double loop, to 1e-8 |
| `qpitk` impurity QPI | physical invariants (clean supercell ⇒ no weight off Γ; impurity ⇒ weight off Γ; ARPACK-seed independence) |
| `graphenetk` relax/gsfe/elastic | arXiv:1805.06972 (Carr *et al.*), and physical invariants — AA is the GSFE max, AB/BA degenerate minima, bond collapse forbidden, relaxation amplitude monotonic in twist angle |
| `chern_density` / `dOmega_dE` | Wilson-loop Chern, for sign *and* scale |

`graphenetk`'s test file is the model the rest of the repo should copy: it states
explicitly *why* it tests invariants rather than a reference number (no external
DFT/LAMMPS available) and then tests the invariants that actually discriminate.

**Documentation is in good shape.** `user_guide.md` covers QGT, plasmon, pointgroup,
relax, QPI, latticeising, Kondo, spinon, Broyden, Wannier, Keldysh, AAA. **`qtci` is
the single gap: 0 mentions**, despite `h.get_chern(integration="qtci")` and the
`get_dm_qtci` SCF backend being public and having runnable examples.

**MEMORY.md is stale on one point:** `functionality_notebooks_plan` says transport and
Wannier notebooks remain, but `ceaf8fc` and `96f8dac` added both.

---

## 3. Tier A — the genuine correctness gaps, ranked

### A1. `topologytk/operatorberry.py` / `topology.operator_berry` — the clearest physics gap

Reached by users via `h.get_bands(operator=<berry-family>)` and
`operators.get_operator_berry`. **Every test that touches it is self-referential.**
Verified: `test_berry_valley.py` (`assert np.isclose(np.sum(y), 159.90994226520772)`),
`test_berry_valley_spin.py`, `test_quantum_geometry.py`
(`13.95734966711326`), `test_berry_curvature_disentangle_strained.py` — all recorded
constants. Nothing checks this path against an independent computation.

Note the contrast with `chern_density`, whose test *does* compare against the Wilson
loop for sign and scale (`test_berry_density_functions.py:36`). That test exists
because a sign bug was found there (`9c7b472`). `operator_berry` never got the
equivalent.

Three compounding reasons this is first:
- Two Berry sign/composition bugs were fixed in this same window (`9c7b472`,
  `0649957`) — precedent for this exact failure mode in this exact area.
- The normalization is a hand-tuned `return b*np.pi*np.pi*8` (`topology.py:440`,
  `:457`) compensating for `multicell.derivative`'s missing `2π/order` factor — a
  coupling already flagged in memory as fragile
  (`multicell_derivative_2pi_convention`). Two conventions cancelling by hand is
  exactly where an off-by-2π hides, and the recorded-constant tests cannot see it.
- It is user-reachable through a common, documented call.

**Check against an external oracle.** For a Haldane model with `operator=identity`,
the BZ-integrated `operator_berry` must give the analytic Chern number and agree with
`h.get_chern()` (independent Wilson-loop path). Do the same for the spin-Berry path
against Kane-Mele. **If a discrepancy appears, the recorded constants in those four
tests are suspects to be rederived, not references to be preserved.**

### A2. Zero-coverage public entry points

Smaller surface, but nothing at all checks them:

- **`h.get_densitychi_RPA`** — 0 tests. (`h.get_plasmon_bands`, layered on top of the
  same module, has one — so the module is partly exercised, but the charge-response
  entry point itself is unchecked.)
- **`operatortk/inplane_valley.py` `get_inplane_valley`, `sharpen.get_sharpen`** — 0
  tests on the entry point; only `add_valley_exchange` is covered. Before touching:
  read `0649957` (fixed a composite spin+valley Berry operator bug) and `53f58d4`
  (documents why a sublattice-diagonal in-plane valley operator *cannot* work) — the
  design constraints here are already understood and written down.

### A3. `qtcitk/gkintegrate.integrate_robust` — the shared primitive

Both qtci entry points are tested end-to-end (§2), which exercises this indirectly.
What is *not* checked is its robustness contract: tolerance handling and
`gkorder_from_nk` behavior across `nk`/tolerance settings. Worth a targeted test
since it is the single shared primitive behind both a topological invariant and an
SCF backend. Lower priority than A1 precisely because the end-to-end oracles pass.

### A4. Extend the external-oracle standard

Cross-cutting, and the durable version of this whole plan. Produce a list of features
whose **only** validation is a recorded constant or two internal paths agreeing, and
mark each as needing an outside reference — two internal paths can share a wrong
assumption and both pass. The ~48 self-referential test files from §1 are the raw
input; the §2 table is the standard to bring them up to. This does not need doing all
at once, but new features should not ship below that bar.

---

## 3.5. Open observation — one full-suite run faulted, unreproduced

Not ranked with the gaps above, because unlike them it is not established. Recorded so
it is not silently forgotten.

A full `python -m pytest tests -q` run on 2026-08-15 died with a **fatal interpreter
fault**, not a test failure — a C-level stack dump topped by

```
numba/np/ufunc/workqueue.cpython-314-x86_64-linux-gnu.so
  ← numba/_dispatcher.cpython-314 ...
  ← pthread_kill / gsignal
```

**Why it might be real:** `parallel.py:11` sets `numba.config.THREADING_LAYER =
'workqueue'`, with a comment explaining that the default `tbb` layer deadlocks when
`paralleltk/multiprocess.py`'s `multiprocess.Pool` forks after a `parallel=True` call has
initialized tbb's pool. Fork-vs-numba was diagnosed and mitigated once already — and this
fault is inside the layer chosen as the fix. (Override confirmed live: a bare interpreter
reports `tbb`, so pyqula is genuinely running `workqueue`.) The repo now mixes numba
`parallel=True` kernels with a forking pool throughout.

**Why it might be nothing:** load average was **19.6-22.5 on 14 cores** — this survey had
a second pytest process running *and* an unrelated session was running another project's
suite. Thread oversubscription alone can produce a workqueue fault. Notably, **the
vendored-qutecipy suite ran to completion (100/100, 389s) inside that same window**, so
the load was not uniformly fatal. The Python traceback identifying the test was lost to
the `| tail` truncation, so it is not even established that a pyqula code path was
involved rather than teardown. A verbose rerun reached 21 tests (0 failures) in ~15 min
under the same load and was stopped as not useful. This is also Python **3.14** with numba
and MKL, where a toolchain issue is a live alternative. Per-directory runs
(`pytest tests/scf`, `pytest tests/keldysh`) do not trigger it, which is why three weeks
of development never surfaced it.

**Close criterion, one command:** run `pytest tests -v` unpiped on an idle machine. If it
completes, close this as environmental. If it faults, take the last test that started and
bisect — does it reproduce in that directory alone? With `parallel.set_enabled(False)`?
With `numba.set_num_threads(1)`? Only then is it worth deciding between a pyqula bug, a
numba/3.14 bug to pin around, and a test-ordering interaction.

The cause-independent half of this finding is mechanical and lives in §6 D7.

---

## 4. Tier B — two systematic sweeps, each proven in one module and not applied elsewhere

**These are the highest value-per-effort items in the plan.** In both cases the bug or
the win is already demonstrated in this repo; the only open question is where else it
applies.

### B1. Single-orbital test blindness

`1b82095` found `_dense_hlist` stored the right bond daggered opposite to
`enlarge_hlist`'s convention — **invisible on a single-orbital chain, 97.7% wrong on
two orbitals**, because a single-orbital coupling is trivially Hermitian.

Surveyed the fixtures: across `tests/keldysh/` and `tests/transport/`, **62 of 63
geometry constructions are `geometry.chain()`** (the 63rd is one `dimer`). The only
uses of `has_spin=True` or a supercell are in `test_central_ij*.py` — written by the
work that found the bug. The blind spot is the suite, not one test.

Note that `has_spin=True` alone does **not** fix this: a spin-degenerate block is still
Hermitian-coupled, so it cannot discriminate a dagger convention either. The
discriminating fixture is specifically a **non-Hermitian inter-cell coupling on ≥2
orbitals**; nothing weaker catches a convention flip. Apply to `transporttk/central.py`,
`heterostructures.py`, `supercell.py`, the block-chain assembly in
`keldyshtk/current.py`, and the spinspin supercell path.

### B2. Python-level bookkeeping around cheap numerics

The 2026-08-15 profiling found `builtins.round` for self-energy cache keys was **2.3s
of a 7.7s call** (325500 calls) and `algebra.todense` on already-dense input another
~1.2s — against 1.27s for the actual batched Sancho-Rubio it was wrapping.

**The fix was applied to `keldyshtk/current.py` only.** Verified this survey: that file
now uses `np.round(es, 10).tolist()` (`current.py:250`), but the two sibling modules on
the same hot path still use per-item Python `round()` in comprehensions:

- `aaatk/selfenergy_aaa.py:276, 281, 284, 285` — `round(e, 12)` per energy, including
  inside `np.array([solved[round(e, 12)] for e in es])`, on the AAA build path that is
  now the **unconditional default for all three sweep entry points**.
  **DONE** (2026-08-15): vectorized to one `np.round(es, 12).tolist()`. Keys verified
  bit-identical (0 mismatches over 45002 values across linspace and random grids —
  note the comparison must round *ndarray elements*, as the old code did, not Python
  floats, or numpy-vs-Python banker's rounding shows spurious last-digit diffs).
  Measured 28–32x on key handling (ncand=20000, 20 rounds: 6.88s → 0.25s).
- ~~`qtcitk/selfenergy_qtci.py:187`~~ — **verified NOT the same pattern, no change
  needed.** Its `full_matrix(e)` is a *scalar* cache called once per energy
  (`full_matrix(e)[i,j]` at :210), so there is one `round()` per invocation, not a
  per-item comprehension over an array. Nothing to vectorize.

Also sweep the 160 `algebra.todense` call sites for ones whose input is already dense —
though note `algebra.todense` already early-exits cheaply for non-sparse input
(`np.array(m, dtype=complex128)`), so the remaining win here is small.

This lands directly on the maintainer's stated preference from the AAA thread — attack
build cost rather than gate on sweep length — and on the standing memory note to
**prefer root-cause fixes over conditional gates**. Carry the other generalized lesson
too: **re-profile after each round**; the bottleneck in this module has moved to
Python-level bookkeeping three times running.

---

## 5. Tier C — optimization backlog: two items needing your go/no-go

Both are documented, scoped, and deliberately parked waiting on a maintainer decision.
A plan-writing moment is the right time to surface them rather than leave them parked.

### C1. SCF loop redundancy — `scftk/scftypes.py` (Tier 3 of the perf plan)

The last un-started tier of the 4-tier performance plan (Tiers 1, 2, 4 all landed).
What it would change:

- `iterate`/`update_hamiltonian` **deep-copy the whole Hamiltonian + geometry every SCF
  iteration** and unconditionally rebuild the hopping list via
  `multicell.collect_hopping`, even though only the mean-field hopping matrix changes.
- `update_occupied_states` **regenerates the k-mesh twice per iteration**, though
  `nkgrid` never changes across iterations.

This was held back because it touches iteration behavior more directly than the other
tiers — a subtle change here alters SCF convergence paths, not just speed. It affects
every mean-field calculation in the library, so it is the largest remaining
single-target speedup. **Line numbers will have moved** since the survey (the
`selfconsistency/` → `scftk/` merge in `bf7a721` came after) — re-locate before
starting. Needs an explicit go/no-go.

### C2. GPU porting — Tiers 2-4 of `documentation/gpu_porting_plan.md`

Tier 1 (KPM batched GPU path) **is done** — `35b7a43` completed it and wired
`kpm_cpugpu` through the public API. The roadmap's own status line is accurate; what
is stale is MEMORY.md's `gpu_porting_plan` entry, which still records the whole plan
as not started — the second stale memory found by this survey (§6.6 is the other).

The roadmap's next-ranked candidate is **batched dense diagonalization in
`htk/eigenvectors.py` (`peigh`/`peigvalsh`)** — attractive because Tier 1 of the *CPU*
perf plan already routed `dos.py`, `spectrum.py`, and `bandstructure.py` through those
two functions, so a GPU backend behind them would reach every one of those call sites
without touching them again. Worth a decision on whether this is still the right next
move, or whether the CPU numba path is fast enough that the remaining GPU tiers should
be closed out rather than pursued.

---

## 6. Tier D — hygiene, cheap and mechanical

1. **Vendor upstream `qutecipy`'s 16 test files under `tests/qutecipy/`** (imports
   rewritten `qutecipy` → `pyqula.qutecipytk`). Already run once during this survey —
   100/100 pass (§2) — so this is purely about keeping the port validated against
   future edits. Note they take ~6.5 min, so consider a marker to keep them out of the
   default run.
2. **Add a LICENSE and pin the upstream commit** (`6df39e7`) in
   `qutecipytk/__init__.py`. Upstream has no LICENSE either — worth adding there
   first, since it is also the maintainer's repo.
3. **`user_guide.md` entry for qtci** — the one documentation gap (§2).
4. **Fix two stale path strings** in the vendored copy (`matrix/rrlu.py:96`;
   `tci1.py:351` still says `qutecipy.tensortrain...`).
5. **`keldyshtk/current_jax.py` (~700 LOC) has no caller** anywhere in `src/` — only
   its own test imports it. Decide: wire `keldysh_didv_jax` to a public entry point,
   or annotate it as a removal candidate alongside the eight modules `015b776` already
   flagged (`alloy.py`, `effective.py`, `estimators.py`, `fitting.py`, `mullen.py`,
   `numbaneighbor.py`, `reciprocalmap.py`, `slabs.py`). Those eight are still present
   — either finish that cleanup or drop the annotations.
6. **Correct the stale MEMORY.md notebooks entry** (§2), and the `gpu_porting_plan`
   entry that still says "not started" though Tier 1 landed in `35b7a43` (§5.2).
7. **Stop piping pytest output** — this one is confirmed regardless of §3.5's cause.
   `pytest ... | tail` reports the *pipe's* exit status, so a crashed suite exits 0. Hit
   twice during this survey: once on the fatal fault, and once on a plain
   `unrecognized arguments` argument error that also "exited 0" while running nothing at
   all. Either drop the pipe, or `set -o pipefail`, or use `--tb=short -q` and let pytest
   own the output. Any "the suite passes" claim made through a pipe is unverified.
8. **Update CLAUDE.md's test-suite figures** — it documents "~7.5 min for 406 tests";
   the suite now collects **667 tests** and takes correspondingly longer. Worth
   refreshing since CLAUDE.md is what sets the next session's expectations about
   whether a long-running suite is hung or just slow.

---

## 7. Already triaged — do not re-litigate

Each was investigated and closed deliberately. Reopening them is rediscovery, not
progress. Read the linked note before touching anything nearby.

- **`keldyshtk/boundary.py`** — known broken, hard-guarded with `RuntimeError`. The RGF
  closure primitive is validated to 1e-16 and reusable; the *sum*-truncation redesign
  around it is wrong by 10-90%, because the DC current is a genuinely unbounded
  sideband sum. Do not resurrect without new evidence.
- **BdG `total_energy` anomalous dc-energy gap** — known, documented, unfixed by choice.
- **AAA short-sweep length gate** — implemented (`afef383`), **explicitly rejected by
  the maintainer** as too system-dependent, reverted (`65c8499`). Attack build cost
  instead; all three queued build-speedup ideas are landed (`e0de870`, `d67d662`,
  `4a086f5`). B2 above is the next increment of that same approach.
- **AAA acceleration of `finite_T_didv`'s native-temperature path** — deliberately not
  done; needs its own arXiv consult first, since direct finite-T is structurally
  different from the T=0 branch AAA was validated against.
- **`quadrature="fixed"` as default** — shelved; fixed's value is determinism, not
  speed, and adaptive is now batched.

---

## 8. Suggested execution order and model routing

Run each item in a **fresh session** — they are independent and context-heavy.

**Your two decisions first** — C1 (SCF loop) and C2 (GPU tiers 2-4) are blocked on a
go/no-go, and C1 is the largest remaining single-target speedup in the library. Nothing
else in the plan depends on them, but they gate the biggest work.

| # | Item | Model | Why |
|---|---|---|---|
| 0 | **§3.5** re-run the suite unpiped on an idle machine | Sonnet | One command; either closes the fault as environmental or localizes a real one |
| 1 | **B2** `round`/`todense` sweep | Fable / Sonnet | Mechanical; exact sites listed in §4.2 |
| 2 | **A1** Berry operator vs analytic oracle | **Opus** | Sign/normalization derivation, two hand-cancelling conventions, four recorded constants may need rederiving |
| 3 | **B1** multi-orbital non-Hermitian fixtures | **Opus** | Judgment about which conventions actually discriminate |
| 4 | **A2** zero-coverage entry points | **Opus** | Valley-operator design constraints are subtle; read `0649957`/`53f58d4` first |
| 5 | **D1-D6** hygiene | Fable / Sonnet | Mechanical |
| 6 | **A3** `integrate_robust` robustness | Sonnet | Contained; end-to-end oracles already pass |
| 7 | **C1** SCF loop redundancy | **Opus** | *If green-lit.* Touches SCF convergence behavior, not just speed |
| 8 | **C2** GPU tiers 2-4 | **Opus** | *If green-lit.* Read `gpu_porting_plan.md` first per CLAUDE.md |
| 9 | **A4** oracle-standard audit | **Opus** | Judgment-heavy, ongoing rather than one-shot |

Item 1 is cheap and independent of everything else — worth doing first regardless.

**Routing caveat, from experience on this repo:** phrase any Fable-assigned item as
*"verify X against the code"*, never *"what is the default for X"*. Advisor/Fable
claims about defaults in this codebase have gone stale mid-thread before — one consult
cited the old convolution mode's evaluation count after `keldysh_thermal_mode="direct"`
had already become the shipped default. Verify against the code path.

**Rule for the whole plan:** when an item turns up a discrepancy in a feature covered
by one of the ~48 self-referential tests, the recorded constant is a *suspect*, not a
*reference*. Rederive it.
