# Floquet sideband decimation for keldysh.dc_current — research idea

Status: **not started, not validated**. This is a single untested hypothesis, written up
for record and for a second opinion before anyone spends implementation time on it. Nothing
here should be implemented without explicit user sign-off, same process as
`gpu_porting_plan.md` and the CPU `perf_optimization_plan`.

**Update after independent review (Claude Fable, see below): the original framing below is
partly wrong** — a literal Sancho-Rubio *doubling* indeed doesn't apply (correct), but an
*exact* tail closure still exists via plain backward recursion, and the more promising,
lower-risk target is not the tail at all but the wasteful full re-solve on every `nmax`
growth step. See "Review findings" below before reading the original analysis as anything
more than motivating context.

## Context

`keldyshtk/current.py`'s `dc_current` (San-Jose, Cayao, Prada, Aguado, NJP 15, 075019 (2013),
arXiv:1301.4408) computes the multiple-Andreev-reflection/AC-Josephson DC current by summing
over Floquet sidebands: the (block, sideband) lattice is exactly two decoupled 1D chains
(see that module's docstring), each solved with an O(ns) recursive Green's-function (RGF)
sweep, `ns = 2*nmax+1`. `nmax` is grown adaptively (`nmax += 2`, re-running the *entire*
quasienergy quadrature from scratch each time) until the result stops changing by more than
`tol` for `min_consecutive` steps in a row, capped at `nmax_max`.

An arXiv/literature search (see conversation) turned up nothing describing a fundamentally
faster closure for this specific problem — RGF, AAA rational self-energy fitting (already
implemented, `aaatk/selfenergy_aaa.py`), and the adaptive-`nmax` truncation itself all match
what's already the standard approach in this literature (Cuevas/Martín-Rodero/Levy-Yeyati,
Averin-Bardas, San-Jose et al.). A profiling-driven pass already removed a real, verified
Python-dispatch bottleneck in the chain-assembly loop (see the git history of
`keldyshtk/current.py` and `aaatk/selfenergy_aaa.py`, `SelfenergyAAA.call_batch`/
`_assemble_chain_jit`) — that work is done and landed. What follows is a separate,
*unverified* idea for reducing how much work the adaptive `nmax` loop itself does.

## The idea

Each chain is currently solved with an **open boundary condition**: it is truncated at a
hard cutoff `nmax`, and every sideband beyond it is simply absent from the Hamiltonian. When
the result hasn't converged, `nmax` is grown and the *entire* chain — including everything
already converged near the AC bond at the chain's center — is rebuilt and re-solved from
scratch.

Semi-infinite *real-space* leads elsewhere in this codebase (`greentk/rg.py`,
`green_renormalization`/Sancho-Rubio) are handled differently: instead of truncating, an
**absorbing self-energy** is computed once (via iterative doubling/decimation) that exactly
represents the effect of the infinite remainder of the lattice, attached to a small
"device" region. The same trick, applied to the sideband index instead of real space, would
replace the hard `nmax` cutoff with two small self-energies — one per chain end — capturing
the effect of every sideband beyond some small, fixed `n0`. If that self-energy converges
(as a function of how far out the decimation is iterated) much faster than the direct
sideband sum converges as a function of `nmax`, a single small, fixed-size chain dressed with
these boundary self-energies could replace the current adaptively-regrown one, similar in
spirit to how `keldyshtk.current.build_selfenergy_aaa`/`build_shared_selfenergy` already
amortize a expensive per-energy computation across many calls.

## Why this is NOT a straightforward reuse of Sancho-Rubio

Real-space semi-infinite leads are exactly periodic (every unit cell is identical), which is
what makes Sancho-Rubio's doubling trick exact and fast (log convergence in iteration count).
The Floquet sideband chain is **not** translationally invariant in the same way:

- The "onsite" energy at sideband `n` is `quasienergy + n*voltage`, growing without bound —
  there is no fixed unit cell to repeat.
- The lead self-energy `Sigma_r(e)` entering each site depends on that same growing energy,
  not a fixed value.

So a literal Sancho-Rubio doubling does not apply. Two distinct regimes are plausible for the
chain's far tail (large `|n|`), and they'd need to be checked, probably numerically, before
trusting any of this:

1. **If `Sigma_r(e) -> 0` for `|e|` beyond the lead's bandwidth** (no lead states there, which
   is generically true for a bounded-bandwidth tight-binding lead), the chain's tail becomes a
   *bare*, real-onsite-energy 1D chain with no dissipation — its own Green's function has a
   closed form (or converges to one via a much cheaper fixed-point iteration than a growing
   dense solve), and could be used to build the tail self-energy analytically rather than by
   brute-force sideband growth.
2. **If `Sigma_r(e)` does not decay** (e.g. a gapless/metallic lead with a bounded but
   nonzero self-energy at all reachable energies), there is no obvious periodicity to exploit,
   and this idea likely does not help.

## The bigger open question: is the tail even the bottleneck?

MAR/AC-Josephson literature (confirmed via the same search) reports `nmax ~ V0/V` — i.e. the
sideband count needed scales with how many orders of Andreev reflection are physically
required to describe transport at a given voltage/gap ratio. That is a **physical**
requirement (resolving that many multi-photon/multi-Andreev processes), not necessarily a
numerical truncation artifact. If most of the convergence difficulty comes from needing many
resonances *within* the physically relevant sideband range (not from an artificially reflecting
hard edge far beyond it), a smarter tail closure would only trim the last few, cheap
iterations of the adaptive loop, not the `O(V0/V)` scaling itself — a real but modest win, not
the qualitative improvement the "GPU/tensor-method" framing might suggest.

**This has to be checked empirically before assuming it's worth building**: e.g. instrument
`dc_current`'s adaptive loop to see how much of the current a single sideband contributes as a
function of `n`, for a few representative (voltage, transparency, gap) points, and see whether
truncation error is dominated by the last few sidebands near `nmax` (favorable — an absorbing
boundary would help) or spread across the whole range (unfavorable — it wouldn't).

## Review findings (Claude Fable consultation)

An independent review (Claude Fable, prompted with this document plus `current.py`/`rg.py`)
raised four corrections/refinements worth recording before anyone acts on this:

1. **The tail closure doesn't need doubling at all — plain backward recursion is exact and
   converges by itself.** The tail is a Wannier-Stark-tilted chain (onsite detuning grows as
   `n*voltage`); that growing detuning is exactly what suppresses sensitivity to the
   recursion's starting seed (factorially, in the size of the detuning relative to the
   hopping) — the standard continued-fraction/minimal-solution property of such ladders. So
   `Sigma_tail(n) = tau^dagger @ [E_(n+1) - Sigma_tail(n+1)]^-1 @ tau`, iterated inward from
   any reasonable seed far out, converges to the *exact* tail self-energy — no doubling trick,
   no closed form, no periodicity assumption needed. This is a real correction to the
   "Sancho-Rubio doesn't apply here" framing above: doubling doesn't apply, but the underlying
   goal (an exact absorbing boundary) is still reachable, just via ordinary backward recursion.

2. **Regime 2 (non-decaying self-energy) is the *favorable* case, not the unfavorable one --
   the opposite of what this document originally claimed.** A non-decaying `Im Sigma_r` is
   itself absorbing (it's exactly what makes the RGF backward sweep contract), so the tail
   recursion converges *faster* there, not worse. The genuinely delicate case is a self-energy
   that decays to a purely real value outside the lead's band (case 1): decay there is driven
   by the growing Wannier-Stark detuning, not by `Sigma -> 0` (which happens only
   algebraically, `~t^2/e`) — still fine, but for a different reason than originally written.

3. **A Keldysh-specific subtlety the original writeup missed**: this is a lesser (`Sigma^<`),
   not just retarded, closure. In regime 1 (energy outside the lead band), `Sigma^r_lead` is
   Hermitian there, so `Sigma^< = -f*(Sigma^r - Sigma^{r dagger}) = 0` and the tail injects
   nothing -- only the retarded closure is needed. This should be verified numerically, not
   assumed, since the finite broadening `delta` used everywhere in this module adds a uniform
   anti-Hermitian piece at every tail site that could spoil the exact cancellation.

4. **The likely bigger, lower-risk win is elsewhere**: `dc_current`'s adaptive loop re-solves
   the *entire* chain from scratch at every `nmax += 2` step (full quadrature, full RGF sweep),
   discarding everything already converged near the AC bond at the chain's center. Restructuring
   this as an incremental extension -- caching each quasienergy point's inner-boundary
   self-energy and only extending the recursion outward as `nmax` grows, rather than rebuilding
   the whole chain -- captures most of the achievable win with **no new physics and an exact
   (machine-precision-verifiable) result**, unlike the tail-closure idea which still needs the
   validation in point 1 above. For the observed `nmax` growth in this module's own tests
   (roughly 6 -> 64 in some slow cases), that's on the order of a 10x reduction in redundant
   chain-solve work, independent of whether the tail-closure idea pans out at all.

5. This reframes point 3 of the "bigger open question" section above: hard-wall truncation
   error decays factorially once `nmax` exceeds the physically-required window (`~few * gap /
   voltage`), so an absorbing boundary can only shave the adaptive loop's O(1) safety margin --
   it cannot remove the `gap/voltage` scaling itself, which is physical. This matches (and
   sharpens) the original document's own suspicion.

**Net effect on next steps**: prioritize point 4 (incremental chain extension across the
`nmax` loop) over the tail-closure/absorbing-boundary idea -- it is simpler, exact, easier to
validate, and plausibly captures most of the win. The tail closure (points 1-3) is still
worth prototyping afterward if more headroom is wanted, using backward recursion (not
doubling) as the correct method, with the Keldysh-lesser cancellation in point 3 checked
numerically before relying on it.

## Update: the small-fixed-window absorbing-boundary design was built and FAILS

The backward-recursion boundary closure from points 1-3 was implemented and validated
(`src/pyqula/keldyshtk/boundary.py`, kept for reference/reuse -- not wired into `dc_current`):
`converged_boundary`'s (retarded + lesser) closure matches a large-`nmax` hard-truncation
reference to **machine precision (~1e-16)**, at a fixed window as small as `nmax0=6`, across a
sweep of voltages (deep subgap / mid-gap / above-gap) and quasienergies -- so the recursion
itself, the Keldysh-lesser handling, and the parity/indexing bookkeeping are all correct.

But wiring it into a full `dc_current`-equivalent (`dc_current_boundary`, same module) gives
**badly wrong currents** -- 10-90% relative error against the validated `dc_current`, not a
small residual. Root cause, confirmed directly (compare each sideband's own contribution to
the trace sum at `nmax=60`, `voltage=0.1*delta`, `T=0.5`, `quasienergy=0.005`): **80% of the
current's value comes from sidebands with `|n|>8`** -- entirely outside any modest fixed
window. This is a structural property of the formula, not a resolution issue: the DC current
is `sum_isb Tr{[Gr00[isb]*Sigma_L^<[isb] + Gless00[isb]*Sigma_L^a[isb]]*tauz}`, a *direct sum
of every sideband's own local contribution* -- via the Floquet-unfolding identity `energy =
quasienergy + n*voltage`, this sum (over sidebands, at fixed quasienergy in `[0,voltage)`) is
literally a discretized integral over an *unbounded real-energy range*. `nmax` in the original
code is not a boundary-condition artifact around a localized region that can be closed off --
it is directly truncating that energy-integration range. The absorbing boundary correctly
makes the WINDOW's own Green's functions exact (dressing/embedding is right), but it silently
*drops* every sideband's direct contribution outside the window, and those are not negligible
-- they are most of the answer.

This settles the "is the tail even the bottleneck" question from earlier in this document,
definitively and in the opposite direction from what points 1-4 assumed: growing `nmax` is
not wasted boundary-condition churn, it is doing genuinely required work (visiting more terms
of a sum that does not truncate). **The tail-closure/absorbing-boundary idea, as designed
here, does not work and should not be pursued further in this form.**

**What's still salvageable**: `boundary.py`'s closure primitive (`converged_boundary`,
`_rgf_chain_boundary_jit`) is correct and cheap (~tens of ms per quasienergy point after JIT
warmup) -- what's wrong is using it to permanently truncate the *sum*, not the primitive
itself. A corrected design would use it to make each *additional* sideband term's evaluation
cheap as the summation range is extended -- i.e. genuinely incremental growth of the sum
(visit each new outer term once, using the closure so its own Green's function doesn't need a
full chain re-solve), instead of the current re-solve-the-whole-chain-from-scratch-at-every-
nmax pattern. That is a smaller, real, but still separate piece of work from what's built
here, and has NOT been attempted -- see the process note below before picking it up.

## Proposed validation plan (if picked up)

**Recommended starting point (per the review)**: on one representative slow deep-subgap case,
in one pass, log (a) the current's truncation error vs `nmax` (how far from the final
converged value each intermediate `nmax` step is) and (b) wall-clock time spent per adaptive
`nmax` step. This single experiment answers both open questions at once: whether truncation
error is tail-concentrated (informing whether the absorbing-boundary idea is worth building)
and how much of total wall-clock is spent on redundant early-`nmax` re-solves (quantifying the
upper bound on what incremental chain extension, point 4 above, could actually save). Do this
before either prototype below.

1. **Characterize the tail** on a small representative case: plot `|Sigma_r(quasienergy+n*
   voltage, lead)|` vs `n` for both leads, at a few `(voltage, delta)` combinations spanning
   the regimes `dc_current` is slow for (deep subgap, near-gap, above-gap). Confirm whether it
   decays, plateaus, or oscillates — this decides whether case 1 or 2 above applies.
2. **Measure where truncation error actually comes from**: for a case that needs a large
   converged `nmax`, compute the current contribution per sideband and check whether it's
   concentrated near `n=nmax` (tail-dominated, worth pursuing) or spread out (not worth
   pursuing).
3. Only if both checks are favorable: prototype a fixed-`n0` chain with an iteratively-refined
   boundary self-energy, validate it against the existing dense/RGF reference to machine
   precision (same bar `_rgf_chain_jit`/`floquet_hamiltonian` were validated to), and benchmark
   wall-clock against the current adaptive-`nmax` path on the same representative cases before
   claiming any speedup.

## Process notes

- Do not implement any of this without explicit sign-off — it is speculative, unlike the
  profiling-driven optimization already landed.
- If step 1/2 above show the tail isn't the bottleneck, the right conclusion is to document
  that and stop, not to force the idea through anyway. (This is exactly what happened, see
  above.)
- `src/pyqula/keldyshtk/boundary.py` is a validated-but-insufficient prototype, not wired
  into any production path (`dc_current` is untouched). Its `converged_boundary`/
  `_rgf_chain_boundary_jit` primitive is trustworthy and reusable; its `dc_current_boundary`/
  `floquet_green_functions_boundary` wrapper is NOT — it silently drops most of the current's
  value and must not be used as-is. Any future incremental-sum redesign should reuse the
  primitive, not the wrapper.

## Update: incremental-extension premise confirmed, but the remaining win is now modest

Followed the "Proposed validation plan" starting point above, on the same representative slow
case boundary.py used (`voltage=0.1*delta`, deep-subgap SC-SC two-lead junction, `T=0.5`,
`quasienergy=0.005`), plus one new check point 4's incremental-extension idea specifically
needs and the original validation plan didn't cover:

1. **Interior-site freezing is real and fast**: at a FIXED physical sideband `n`, the combined
   Green's function `G[n]` converges to *machine precision* (`0.0` relative error, `float64`)
   once the window `nmax` exceeds `|n|` by roughly a factor of 2 — e.g. `n=10` is still off by
   `0.46` (relative) at `nmax=10` (the true open boundary) but already `6.7e-14` by `nmax=20`.
   This confirms the "old interior sites don't need re-solving once nmax has grown a bit past
   them" premise behind point 4/the review's correction 1 (Wannier-Stark-driven contraction),
   directly, numerically, on the actual production chain (not a toy case).
2. **The sum itself still needs the full range**: only 52.9% of the converged current comes
   from `|n|<=20`, 90.9% from `|n|<=40`, confirming (again, on a fresh measurement) that `nmax`
   growth is doing genuinely required summation work, not fighting a slow-converging boundary
   — consistent with the "Update: ... FAILS" section above.
3. **`scipy.integrate.quad`'s node set is, in the common case, IDENTICAL across nmax-varying
   calls of the same integrand over the same domain**: instrumented `current_integrand` to
   record every quasienergy it was called at, across separate `dc_current`-internal `quad()`
   calls for `nmax=6,8,10,20,22`. Whenever the base 21-point Gauss-Kronrod rule was sufficient
   (no subdivision), all 21 nodes matched exactly (bit-for-bit) across every nmax tested,
   including `nmax=6` vs `nmax=20`. This matters because it means a per-quasienergy incremental
   cache, keyed the same way `_batch_selfenergy` already keys self-energies (`round(e,10)`),
   would get a high hit rate for free, *without* needing to replace `quad`'s adaptive scheme
   with a fixed quadrature rule (a bigger, riskier change this note originally assumed would be
   necessary) — a genuinely low-risk way to wire point 4's idea in, if it's worth building.
4. **But the *absolute* remaining cost is now small enough that the ROI has dropped**. Two
   safe, targeted fixes landed straight from this profiling pass (no incremental-cache
   machinery, no new physics, `keldyshtk/current.py`'s `_prepare_chain_consts`/`_batch_
   selfenergy` — see git history):
   - `_floquet_green_functions` was rebuilding the AC-bond hoppings and onsite blocks
     (`todense`/projector matmuls) from `system` on *every* quasienergy x nmax-step call
     (1050+ times in the profiled case) instead of once per `dc_current` call, unlike `system`
     itself, which already got this treatment. Hoisted into `_prepare_chain_consts`, computed
     once in `dc_current`/on demand elsewhere.
   - `_batch_selfenergy` was computing `round(e,10)` cache keys twice per energy per call (once
     in the old `_prefetch_selfenergies_batch` to check for cache misses, again in `_cached_
     selfenergy` to gather results). cProfile attributed ~22% of total wall time to `round()`
     itself (214200 calls); a follow-up (Claude Fable) review correctly flagged that figure as
     cProfile's own per-call instrumentation overhead on a cheap C builtin, not real cost --
     confirmed directly (`timeit`: 214200 bare `round()` calls cost ~30ms, not the ~300-500ms
     cProfile reported). The fix is still real, though: an isolated, un-profiled A/B of the
     old (double round+dict-lookup) vs new (deduped) gather logic over 2100 x 129-energy arrays
     showed a genuine ~0.46s difference -- the cost was never `round()` in isolation, it was
     doing the tuple-construction+dict-lookup pair *twice* per element, ~540000 times. Also
     worth noting (missed in the first pass): this path only runs in `selfenergy_method=
     "direct"`/fallback mode -- the default `"aaa"` path returns via `interp.call_batch(es)`
     and never touches the round()-keyed cache at all.

   Net effect on the same representative case (`nmax_max=64`, `selfenergy_method="direct"`,
   cProfile'd, warmed-up numba): wall time ~2.4s -> ~2.1-2.5s (round() calls halved,
   107100 -> from 214200; modest, real, run-to-run noise comparable to the gain). The dominant
   remaining cost is `_floquet_green_functions`'s own per-call overhead (~1.1-1.3s, `numba`
   dispatch + O(ns) sweep/combine work redone at every nmax step) — exactly what a full
   incremental-chain-extension (point 4, using the quad-node-reuse cache from point 3 above)
   would target. But a single `dc_current` call on this genuinely worst-case (deep-subgap,
   `nmax` maxing out at 64) scenario is now ~2-2.5s total, not the tens of Python-dispatch-bound
   seconds the original "~58% of wall time"/"~150000 Python calls" finding was measured against
   before the batching work landed. Building the full splicing cache (validate a decaying
   forward/backward-sweep buffer near each moving edge, freeze/reuse deep interior state, keep
   a self-correcting fallback to a full resolve when the buffer doesn't verify — a real, several
   -hundred-line addition with the same correctness-critical-indexing risk profile as
   `boundary.py`'s closure) is worth doing only if a caller's *aggregate* cost across many such
   calls (e.g. a large `iv_curve` sweep or a finite-temperature quadrature over many voltages)
   still matters at this reduced per-call cost — not clearly true by default. Recommendation:
   ship the two safe fixes (already done), and only build the full incremental-splice cache if
   a concrete slow workload at this new, lower per-call baseline is identified.

## Update: Claude Fable review found the "no concrete slow workload" framing was wrong, landed a cheap fix instead

A follow-up review (Claude Fable, prompted with this document plus `current.py`/`boundary.py`)
pushed back on the "shelve, no concrete slow workload" framing above: `current.py`'s own
`build_shared_selfenergy` docstring already documents one (a single `finite_T_didv` call makes
147 independent `dc_current`-family evaluations; `iv_curve` sweeps compound further), and the
Σns≈16x-redundant-work estimate for a `nmax: 6->64` run is derivable directly from the fixed
`nmax += 2` stepping, not just measured. It also correctly called out that the round()-cost
figure above was cProfile-instrumentation-inflated (see the correction above) and proposed a
cheap intermediate step this document hadn't considered: **grow `nmax` geometrically instead of
by a fixed `+= 2` per adaptive step**, since the per-step cost (a full chain re-solve, no
incremental reuse -- see the "FAILS" section above for why that's still true) scales with `ns`,
so fewer, larger steps directly cuts the total redundant work with no new physics and no
correctness risk beyond the existing `min_consecutive`/`nmax_max` safety net.

**Implemented and validated** (`dc_current`'s new `nmax_growth=1.5` parameter, default on):
`nmax = max(nmax+2, ceil(nmax*nmax_growth))` per adaptive step instead of `nmax += 2`, capped at
`nmax_max` as before. `nmax_growth<=1` recovers the old fixed-step behavior exactly, for anyone
who wants it.

- **Correctness**: full `tests/keldysh` suite (67 tests) and `tests/transport/test_kappa_jax.py`
  (9 tests) still pass. Directly compared `nmax_growth=1.0` vs `1.5` on two representative cases:
  the deep-subgap worst case above (hits `nmax_max=64` without converging either way) returned
  the identical value to 10 significant figures; a case that genuinely converges before the cap
  (`voltage=0.6*delta`) returned values agreeing to 9 significant figures (`7.1554430529e-02` vs
  `7.1554430520e-02`), both far inside `tol=1e-3`. The min_consecutive convergence guard is, if
  anything, a *stronger* test with widely-spaced steps (a coincidental near-agreement between two
  nearby small-Delta-nmax evaluations -- the failure mode that guard exists for -- is far less
  likely between two much-more-different windows).
- **Speed**: the same deep-subgap single-call benchmark: 2.76s (`nmax_growth=1.0`) -> 0.57s
  (`nmax_growth=1.5`), ~4.8x. At sweep scale (the workload the "no concrete slow workload"
  framing above missed): an 8-voltage `iv_curve` spanning deep-subgap to above-gap, same
  junction, `nmax_max=64`: 83.3s -> 34.1s, ~2.4x, with bit-identical output across all 8
  voltages.

**Net effect**: the full incremental-splice cache (freeze/reuse deep interior chain state
across nmax growth) remains shelved -- this cheaper fix captures a large fraction of its
theoretical benefit (Σns redundancy down from ~16x to ~3x for the representative case) for a
~15-line, zero-new-physics change, at a much smaller correctness-risk profile. Revisit the full
splice cache only if a workload is still slow after this fix.

## Update: shared-nmax finite difference landed (T=0 path); surfaced a separate, more
## important accuracy issue in the default AAA self-energy path

Per a follow-up (Claude Fable) review prioritizing what's next after the geometric-nmax-growth
fix above, the maintainer asked to prioritize the T=0 (zero-temperature) path specifically over
the finite-temperature thread (`finite_T_didv`'s ~294-`dc_current`-calls-per-point structure,
which needs an arXiv-consult before any native-temperature-routing change per CLAUDE.md -- not
started). That review's P2 item: `transporttk.didv.keldysh_didv`'s `Ip`/`Im` finite-difference
pair each independently re-ran `dc_current`'s adaptive-nmax search, even though the two biases
differ by only `2*dv` (~1-2% of voltage) and converge to the same nmax in practice.

**Implemented**: `dc_current` gained `fixed_nmax` (skip the adaptive search, solve once at an
explicit nmax) and `return_nmax=True` (return `(current, nmax)` instead of just `current`).
`keldysh_didv` now runs `Ip` with `return_nmax=True`, then `Im` with `fixed_nmax=` that same
value, instead of two independent adaptive searches -- unless the caller already passed their
own `fixed_nmax` explicitly, in which case both branches just use it directly (unchanged
behavior).

**Correctness**: verified bit-identical (not just "close") to the pre-change independent-search
behavior, isolating the self-energy method (`selfenergy_method="direct"`, no AAA involved) so
only the nmax-sharing itself is being tested: two representative points (deep-subgap
voltage=0.03 and near-gap voltage=0.18, same delta=0.3/T=0.5 SC-SC chain as this doc's earlier
benchmarks) both gave exactly 0.0 relative difference between old (`Ip`,`Im` independently
adaptive) and new (`Im` at `Ip`'s converged nmax) results -- expected, since both branches
happened to converge to the same nmax (14 and 64 respectively) even independently, and `dc_current`
with `fixed_nmax=N` calls the exact same internal `integral(N)` closure the adaptive loop's own
final step would have called. Full `tests/keldysh` (67 tests) still passes unchanged.

**Speed**: same deep-subgap point (voltage=0.03, nmax_max=64, `selfenergy_method="aaa"`
default): independent Ip+Im search 11.3s -> shared-nmax `keldysh_didv` 3.6s, ~3.1x (on top of
the geometric-growth win already landed -- this is now cumulative with that fix, since each of
Ip/Im's own adaptive search is itself geometric).

**Separate finding, NOT caused by this change, found while validating it**: comparing
`selfenergy_method="aaa"` (default) against `"direct"` on the *same* near-gap case
(voltage=0.18, delta=0.3, T=0.5, HT.delta=1e-4) turned up per-branch self-energy fit errors of
~7-8% (`Ip`: direct=6.740e-2 vs aaa=6.246e-2; `Im`: direct=6.173e-2 vs aaa=5.677e-2) -- each
individually larger than the 5% bound `tests/keldysh/test_selfenergy_aaa.py`'s existing
AAA-vs-direct tests assert, though those tests exercise a different system (a shallow-gap
LocalProbe, delta=0.1, `nmax_max<=12`, `tol=5e-2`) that doesn't reach this regime. Because
`keldysh_didv`'s finite difference divides by `Ip-Im` (here ~0.0057, much smaller than either
`Ip` or `Im` themselves), that ~7-8% per-branch error is amplified by catastrophic cancellation
into a 37-60% error in the resulting dI/dV depending on whether the AAA fit is built shared
(one interpolant for both `Ip`/`Im`, `keldysh_didv`'s current default) or independently (each
call builds its own) -- confirmed present identically on unmodified master, i.e. **this is a
pre-existing accuracy gap in the shipped default (`selfenergy_method="aaa"`), not something this
change introduced**, but it was found as a direct side effect of validating this change (the
"apples-to-oranges" comparison in an early version of this validation, before pinning
`selfenergy_method="direct"` on both sides, is what surfaced it).

**Not yet resolved -- flagged to the maintainer, no fix attempted**: whether the right fix is
tightening `SelfenergyAAA`'s convergence tolerance specifically when it will feed a
finite-difference derivative (since the failure mode is error amplification, not the raw
self-energy fit being unreasonable in absolute terms), widening test coverage to catch this
regime (deep SC gap + tight broadening + near-gap bias, not currently exercised by
`tests/keldysh/test_selfenergy_aaa.py`), or something else -- this needs a decision, not a
guess, before any code changes.

## Update: first attempted fix (validation-resolution gate) was miscalibrated and reverted;
## the real error mechanism is a continuous, window-size-dependent bias, not a threshold effect

Asked to investigate and fix the accuracy gap above. First hypothesis: `SelfenergyAAA`'s
held-out validation offsets (`rng.uniform(0.1*step, 0.9*step, ...)`) are always drawn within one
candidate-grid step of an existing sample, so validation can never probe finer resolution than
the candidate grid already has -- a real feature narrower than the grid step (e.g. a gap-edge
singularity of width ~delta on a grid with step >> delta) would be invisible to both the fit
and its own validation check, explaining a false `converged=True`. Implemented a gate
(`resolution_factor`, requiring `step <= resolution_factor*delta` in addition to the existing
`maxerr <= tolerance`) plus an early-bailout check (skip straight to `converged=False` when even
`ncand_max` candidates can't reach that resolution, to avoid burning the full escalation budget
on a doomed fit -- confirmed necessary: without it, the gated build took ~10s before giving up
vs ~1s with it).

**This was reverted after broader testing showed it doesn't match the real error behavior.**
Directly measuring relative current error (`selfenergy_method="aaa"` vs `"direct"`, same system,
`voltage=0.18`, `delta=1e-4`) across a range of `nmax_max` (hence window width) found the error
grows *continuously* with window size -- 0.5% at `nmax_max=4`, 0.6% at 8, 1.4% at 16, 3.3% at 24,
4.8% at 32, 9.8% at 40 -- with no sharp resolution threshold anywhere in that range, and
`validation_error` staying deceptively flat (~1e-7-5e-7) throughout regardless of the actual
error's 20x growth. The `resolution_factor` gate, calibrated against only the single worst-case
point that motivated it, correctly rejected that one point (by coincidence of it having an
extreme step/delta ratio) but had no principled relationship to this actual, continuous trend --
and independently, it broke 3 existing tests (`test_shared_selfenergy_for_branch_only_builds_
for_both_sc_leads`, `test_iv_curve_shares_one_interpolant_across_the_voltage_sweep`,
`test_finite_T_didv_shares_one_interpolant_across_its_thermal_quadrature`) whose small-`nmax_max`
fixtures (also `delta=1e-4`, transparency=0.3) turned out to be genuinely accurate (rel. error
2.9e-7 and 1.2e-3, confirmed by direct measurement) despite similarly "unfavorable" step/delta
ratios -- proof the gate's threshold logic doesn't track the real failure boundary.

Tightening `tolerance` alone (1e-6 -> 1e-12, at `nmax_max=40`) helps but doesn't cleanly fix it
either: 9.8% -> 7.4% -> 6.6% -> 0.8% -- a real ~12x improvement, but not converging to
correctness, and `aaa_tolerance` (0.1*tolerance, the AAA fit's own internal residual target)
turns out to be doing most of that work rather than added candidate density (`ncand` stayed
fixed at 432 from tolerance=1e-6 through 1e-10, only escalating to 863 at 1e-12) -- i.e. the same
432 candidates support a materially better fit once AAA is told to work harder fitting them,
independent of any resolution/coverage argument.

**Working hypothesis, not yet confirmed**: error compounding through the RGF chain across many
sidebands (each `_rgf_chain_jit` step recursively combines Green's functions from adjacent
sites, potentially amplifying even a small ~1e-7-level per-energy self-energy error many times
over as `nmax_max`/window size grows -- roughly consistent with the observed error trend scaling
with window size rather than with any single evaluation's local accuracy) rather than a missed
narrow feature. Not yet verified directly (would need e.g. injecting a known small perturbation
into a single self-energy evaluation and measuring how much the final current changes as a
function of chain length/nmax, to see if the amplification factor actually matches).

**Status: unresolved, reverted to the pre-fix `aaatk/selfenergy_aaa.py` (the accuracy gap
documented in the update above is real and still present) -- needs the maintainer's input on
how to proceed** (deeper investigation into the compounding-error hypothesis; a blunter interim
mitigation like always using `selfenergy_method="direct"` above some `nmax_max`/domain-width
threshold at the cost of losing the AAA speedup exactly where it's needed most; or deprioritizing
this and shipping only the independently-validated nmax-sharing change from the update above).
The nmax-sharing (`fixed_nmax`/`return_nmax`, `keldysh_didv`) change itself is unaffected by this
revert -- it was validated with `selfenergy_method="direct"` specifically to isolate it from this
issue, and remains correct and landed.

## Update: chose the blunt mitigation -- "direct" is now the default everywhere, "aaa" is opt-in

Maintainer's choice, given the compounding-error mechanism above is still not understood well
enough to fix confidently: make `selfenergy_method="direct"` the default at every entry point
that could reach the Floquet-Keldysh path, so nothing silently returns the AAA accuracy gap's
wrong answer unless a caller explicitly asks for it. Implemented:

- `dc_current`'s own `selfenergy_method` default: `"aaa"` -> `"direct"`.
- `keldysh_didv`'s `use_aaa` default: `True` -> `False`.
- `iv_curve`'s auto-share-a-fit check: was `kwargs.get("selfenergy_method", "aaa") == "aaa"`,
  now defaults to `"direct"` in the same `.get(...)` (so it only shares when the caller passed
  `selfenergy_method="aaa"` explicitly).
- `thermaldidv.finite_T_didv`'s auto-share (previously unconditional whenever both leads were
  superconducting, regardless of any `selfenergy_method` kwarg): now additionally gated on
  `kwargs.get("selfenergy_method") == "aaa"`.
- `kappa._with_shared_selfenergy` (get_kappa's zero-temperature path) and `kappa._shared_
  selfenergy_for_branch` (the finite-temperature path): same new gate, `kwargs.get(
  "selfenergy_method") == "aaa"` required before building/sharing anything.

Every one of these previously built (and shared) an AAA fit *by default* whenever the junction
was Floquet-Keldysh-eligible, with no way for a caller to have hit the "direct" path without
explicitly asking for it at every single call site -- the gates above close that gap in one
consistent sweep rather than leaving some entry points silently still defaulting to "aaa".

**Test fallout, all expected and fixed**: several `tests/keldysh` tests were specifically
testing the AAA-sharing plumbing (call counting, interpolant-object identity across a sweep) and
relied on the old AAA-by-default behavior without passing `selfenergy_method="aaa"` themselves --
these now explicitly opt in (`tests/keldysh/test_shared_selfenergy_sweeps.py`'s `_CHEAP` dict,
`test_kappa_finite_temperature.py`'s `test_shared_selfenergy_for_branch_only_builds_for_both_
sc_leads`, `test_selfenergy_aaa.py`'s `test_build_selfenergy_aaa_matches_direct_dc_current`/
`test_keldysh_didv_use_aaa_matches_use_qtci_and_direct`) so they keep testing what they were
built to test rather than silently comparing "direct" against "direct". Full `tests/keldysh`
(67 tests) passes with these updates.

**Cost of this choice**: every Floquet-Keldysh call is back to the pre-AAA per-energy solve cost
by default -- the AAA speedup work (aaatk/selfenergy_aaa.py, this doc's earlier updates) is not
lost, just no longer reached without an explicit `selfenergy_method="aaa"`/`use_aaa=True`, and a
caller who has verified it's accurate for their own system can still opt back in. `documentation/
user_guide.md`'s Floquet-Keldysh section was updated to describe "direct" as the default and
"aaa" as a checked-it-yourself opt-in, with a pointer to this document for the accuracy gap.

## Update: direct finite-T Keldysh evaluation replaces thermal convolution by default

A follow-up consult (Claude Fable, asked for "wild ideas" to speed up Keldysh transport broadly)
raised `finite_T_didv`'s thermal-convolution approach as a likely large win for `finite_T_didv`/
`iv_curve`-at-finite-T/`kappa.py`'s finite-temperature path. `transporttk.thermaldidv.
finite_T_didv` computed finite-temperature dI/dV by evaluating `zero_T_didv` (a T=0
`dc_current`-pair finite difference) at ~150-400 energies spanning `+-THERMAL_WINDOW*temp` around
the bias, convolved with a Fermi-derivative kernel -- while `keldyshtk.current.dc_current` already
has a native `temperature` parameter that broadens each Floquet sideband's own occupation directly
(`_fermi_scalar`/`_assemble_chain_jit`), reachable in 2 `dc_current` calls instead of ~150-400.

**Corrected framing (important, and initially gotten wrong in this thread's own first draft)**:
these are NOT two routes to the same number, so "replace the slow approximation with the fast
exact one" is the wrong mental model. `finite_T_didv`'s convolution smears **bias voltage** --
for a Floquet ladder, shifting the bias by `dV` moves sideband `n` by `n*dV`, an n-dependent
displacement of the whole ladder. `dc_current`'s native `temperature` broadens each sideband's own
occupation by a fixed `temp`, independent of `n`. These are structurally different operations and
only have to agree as `temp->0` (both reduce to the T=0 curve); do not expect agreement away from
that limit, and none was found in testing (see below). The convolution approach is exact for
ordinary elastic/Landauer transport with energy-independent transmission (which is presumably why
`thermaldidv.py` was written that way generically -- it is also the only mechanism available for
the non-Keldysh `"smatrix"` path, where the identity genuinely holds and stays the default there).

**Blocking gate, since `dc_current`'s `temperature` parameter had ZERO test coverage anywhere in
the repo before this**: added `tests/keldysh/test_normal_junction_finite_temperature.py`,
extending `test_normal_junction_gauge_invariance.py`'s pattern (`turn_nambu`, zero pairing) to
finite T -- an independent, non-Floquet finite-temperature Landauer reference (`T(E) =
HTb.didv(energy=E)` at T=0, integrated against the standard thermal window `f_T(E-V/2)-
f_T(E+V/2)` over a window widened by `+-20*temp`) versus `dc_current(voltage, temperature=temp)`.
19 parametrized cases (3 transparencies x 3 signed voltages x 2 temperatures, plus a `temp->0`
sanity check) all pass, relative error <=8e-4 throughout -- comfortably inside the 2e-2 margin the
zero-temperature sibling test uses. This confirms `dc_current`'s existing `temperature` machinery
is itself correct.

**Wired in**: `finite_T_didv` gained `keldysh_thermal_mode` ("direct", new default, or
"convolution", the prior behavior), used only when `_both_leads_superconducting(self)` -- a
non-Keldysh junction is unaffected either way. "direct" calls `keldysh_didv(self, voltage=energy,
temperature=temp, **kwargs)` directly. `keldysh_didv`'s `dv` default
(`max(abs(voltage)*1e-2,1e-3)`) was tuned for the T=0 finite difference, where there is no thermal
scale to resolve; at finite T it is now clamped to `min(dv, 0.1*temp)` when `temp>0` so the central
difference doesn't smooth away the thermal structure this mode exists to capture.

**Speed and divergence, measured directly** (transparency=0.1, `HT.delta=1e-2`, `delta_sc=0.3`,
`fixed_nmax` used on both sides to isolate the thermal-averaging cost from the unrelated adaptive-
nmax search, which both modes already share via other fixes in this doc): at `temp=0.02`, direct
took 1.33s (`fixed_nmax=2`) versus convolution's 138.4s for the *same* nominal call (399
`zero_T_didv` evaluations) -- **~104x**, on a deliberately mild case chosen to even *finish* in a
reasonable time for this measurement. On the doc's own "worst case" deep-subgap SC-SC parameters
(`delta_sc=0.3`, transparency=0.6, `HT.delta=1e-4`), convolution did not complete within 500s at
even `fixed_nmax=4`, while direct mode's own per-call cost there is sub-second (consistent with a
single `dc_current(fixed_nmax=4)` call measured at 0.02-0.05s warm) -- i.e. the win is qualitatively
larger, not smaller, exactly where this optimization effort's own worst cases live. As expected from
the corrected framing above, direct and convolution do **not** numerically agree away from `temp->0`
(rel. difference ~104% at `temp=0.02` on the mild case) -- this is not a bug, it's the two
formalisms computing genuinely different quantities.

**Default**: per the maintainer's explicit choice, "direct" is now the default for every
Keldysh-eligible caller (`finite_T_didv`, `kappa.py`'s `get_conductances_finite_temp`/
`get_kappa_finite_temperature_energies`, `LocalProbe.didv(temp=...)`) -- this changes the returned
numbers for all of them (intended: convolution and direct are different quantities, and direct is
the more physically direct one for this formalism). One existing test
(`tests/keldysh/test_shared_selfenergy_sweeps.py::test_finite_T_didv_shares_one_interpolant_across_its_thermal_quadrature`)
specifically exercised the convolution-mode AAA-sharing plumbing and needed
`keldysh_thermal_mode="convolution"` added explicitly to keep testing what it was built to test,
matching this doc's own established pattern for prior default flips. Full `tests/keldysh` (87
tests, including the 19 new ones) and `tests/transport/test_kappa_jax.py` (9 tests) pass.

**`temp->0` convergence, measured** (same mild case, transparency=0.1, `HT.delta=1e-2`,
`delta_sc=0.3`, voltage=0.3, `fixed_nmax=2` both sides):

| temp  | direct   | convolution | rel. diff | convolution wall time |
|-------|----------|-------------|-----------|------------------------|
| 0.05  | 0.035716 | 0.014227    | 1.51      | 301.0s                 |
| 0.02  | 0.036878 | 0.018033    | 1.05      | 83.4s                  |
| 0.005 | 0.038962 | 0.031230    | 0.25      | 32.5s                  |
| 0.001 | 0.039761 | 0.038439    | 0.034     | 37.1s                  |

Relative difference shrinks monotonically toward 0 as `temp->0` (1.51 -> 1.05 -> 0.25 -> 0.034),
confirming the two formalisms converge to the same T=0 curve as expected, while staying genuinely
different away from that limit -- exactly the corrected framing above, not a bug. Convolution's own
wall time does NOT shrink monotonically with `temp` (301.0s at temp=0.05 down to 32.5s at
temp=0.005, back up to 37.1s at temp=0.001) -- see the mechanism below for why.

**Why convolution is this expensive, root-caused (Claude Fable consult, prompted with this
section's own numbers plus the actual current `thermaldidv.py`/`didv.py`/`keldyshtk/current.py`
text)**: it is not merely "~150-400x more `dc_current` calls than direct mode," which was the
original, incomplete framing above. `fixed_nmax` only freezes the Floquet sideband count per
quasienergy point -- it does NOT make `dc_current`'s own inner adaptive quasienergy quadrature
(`quad(f, 0, abs(voltage), epsrel=1e-3)`) fixed. Convolution's outer thermal sweep visits
`energy+e` across the whole `+-THERMAL_WINDOW*temp` window, so (a) the inner quad's own domain
width scales with `|energy+e|`, up to ~14x wider than a single-voltage baseline, and (b) that bias
sweep crosses the MAR resonance ladder (`2*delta_sc/n`), and each crossing makes the inner quad
subdivide heavily (the same 21->735-node effect `documentation/keldysh_sideband_decimation_plan.md`'s
item-2a instrumentation already measured) -- so total RGF-chain-solve count is (outer thermal-quad
nodes) x (inner quasienergy-quad nodes at that specific bias) x 2, not just the outer count. Ruled
out as contributing causes: per-call setup (`cache={}`/`_prepare_system`/`_prepare_chain_consts`) is
paid identically in an isolated single call too, so it can't explain a *ratio*;
`_prepare_bias_target` doesn't copy a two-lead `Heterostructure` at all (only for a `LocalProbe`);
numba's kernels specialize on dtype/ndim, not array shape, so nothing recompiles mid-sweep. The
non-monotonic wall-time-vs-temp trend above is explained the same way: `scipy.integrate.quad`
(QAGS) bisects whichever subinterval has the largest error estimate, so how many of the outer
sweep's `e` values land near a MAR onset (and hence trigger expensive inner subdivision) depends on
where the `+-THERMAL_WINDOW*temp` window happens to sit relative to the `2*delta_sc/n` ladder at
that specific `temp`, not on `temp` monotonically. A further, separate mechanism likely responsible
for the non-terminating (>500s) deep-subgap/`HT.delta=1e-4` case earlier in this section: the outer
quad demands `epsrel=1e-4` on an integrand built from `(Ip-Im)/(2*dv)`, but `Ip`/`Im` are each only
resolved by the inner quad to `epsrel=1e-3` -- amplified by `|Ip|/|Ip-Im|`, that can put the outer
integrand's noise floor above the tolerance the outer quad is demanding of it, so QAGS keeps
bisecting toward its `limit=60` ceiling (~2500 outer nodes, ~5000 `dc_current` calls) chasing noise
it cannot resolve rather than genuinely converging -- consistent with, though not separately
re-confirmed beyond, the observed non-termination.

This is now understood well enough that no further characterization sweep is planned -- the
mechanism (nested adaptive quadratures, one of which scales with bias and crosses resonances,
occasionally compounded by chasing finite-difference noise below its own resolvable floor) fully
accounts for both the size and the noisiness of convolution mode's cost, and is exactly the kind of
compounding cost that made this an obvious target for the "direct" replacement in the first place.

## Update: `dc_current`'s `quadrature="fixed"` node solve batched over the quasienergy-node axis

Follow-up to `keldysh_sideband_decimation_plan`'s `quadrature="fixed"` work above (that work's own
report calls itself "item 2b"; this is its planned "item 2c"): `quadrature="fixed"`'s node set
(`_fixed_quasienergy_nodes`) is known in full before any integrand evaluation, so the per-node chain
solve `current_integrand` -> `_floquet_green_functions` -> `_assemble_chain_jit`/`_rgf_chain_jit` --
previously called once per node via a plain Python list comprehension inside `dc_current`'s
`integral(nmax)` closure -- was batched over an added leading node axis: `current_integrand_batch`
-> `_floquet_green_functions_batch` -> `_assemble_chain_batch_jit`/`_rgf_chain_batch_jit`, both
numba `@jit(parallel=True)` with a `prange` loop over quadrature nodes (mirroring the existing
sideband-axis batching pattern in `greentk/rg.py:green_renormalization_jit_batch_core`, since each
node's chain is fully independent of every other node's -- different quasienergy, no coupling).
Self-energies are still funneled through the existing `_batch_selfenergy`/per-energy `cache` dict,
just called once over the flattened `(nq*ns,)` energy array per lead per `integral(nmax)` call
instead of `nq` separate per-node batched calls -- this keeps the cache's cross-`nmax`-step reuse
(see the "quad-node reuse" update above) working exactly as before, since batching only changes how
many Python/numba dispatches the *solve* costs, not what gets cached or when. `quadrature="adaptive"`
is untouched: its node set is discovered one `scipy.integrate.quad` callback at a time and cannot be
known in advance, so this batching does not apply there.

**Correctness**: per-node integrand values from the batched path are bit-identical (`np.array_equal`,
not just close) to the pre-batching per-node-loop path, checked on the doc's own deep-subgap case
(`delta_sc=0.3`, `transparency=0.5`, `voltage=0.031`, `nmax=20`) and two cases from the "fixed"
validation sweep (`delta_sc=0.1`, `transparency=0.3`, `voltage=0.15` and `voltage=0.55`, `nmax=20`).
Full `dc_current` output (adaptive-nmax loop included) on the same three cases at their
`nmax_max` differs from `quadrature="adaptive"` by 1.6e-3/1.1e-3/1.9e-3 relative -- consistent with
(not a regression from) the accuracy already established for `quadrature="fixed"` itself in the
update above. `tests/keldysh` (86 tests) and `tests/transport/test_kappa_jax.py` (9 tests) both pass
unchanged.

**Speed**, same three representative cases as the `quadrature="fixed"` update above
(`selfenergy_method="direct"`, median of 5 uncontended runs, numba JIT warmed up first, node-chunked
at the shipped default `_BATCH_CHUNK_NODES=256`):

| case | old per-node loop | new batched+chunked | speedup | adaptive (same run) |
|---|---|---|---|---|
| deep-subgap SC-SC (`delta_sc=0.3`, `T=0.5`, `V=0.031`, `nmax_max=64`, 96 nodes) | 0.757s | 0.606s | 1.25x | 0.920s |
| hardest SC-SC (`delta_sc=0.1`, `T=0.3`, `V=0.55`, `nmax_max=40`, 1472 nodes) | 5.566s | 1.466s | 3.80x | 1.068s |
| cheap normal-normal (`T=0.6`, `V=0.3`, `nmax_max=20`, 800 nodes) | 1.726s | 1.195s | 1.44x | 0.211s |

Note the old-per-node-loop/adaptive columns above were measured together, in this session, NOT
copy-pasted from item 2b's own numbers (~1.4x/~1.9x/~45x slower than adaptive on these same three
shapes of case) -- the two measurement sessions ran under different load and do not reproduce each
other (this session's own old-fixed-vs-adaptive ratio on the deep-subgap case is already ~on par
before batching, and its normal-normal ratio is ~8x, not ~45x), so only the ratios *within* this
table (old -> new -> adaptive, all three columns from the same run) should be read as a consistent
before/after comparison; item 2b's numbers remain a valid, separately-reported data point from a
different environment, not a baseline this table extends.

Batching (with the 256-node chunking the memory fix below requires) turns "fixed" from clearly
slower than adaptive on the two SC-SC cases in this session's own baseline into faster than adaptive
on the deep-subgap case and much closer (~1.4x slower, down from ~3.8x more work) on the hardest
SC-SC case, and cuts the cheap normal-normal case's own cost by 1.4x even though it stays ~5.7x
slower than adaptive there. The normal-normal gap is structural, not a batching
shortfall: `_fixed_quasienergy_nodes` sizes its panel count off `|voltage|` alone (no gap
introspection, by design -- see `_FIXED_QUAD_PANEL_WIDTH`'s comment), so a normal junction with no
gap-edge singularity still pays for the same 800 nodes a singular case would need, while "adaptive"
discovers the integrand is smooth and stops at ~21 points.

**Net assessment**: batching is a real, validated win for `quadrature="fixed"` itself (up to 3.8x
here even with the 256-node chunking the memory fix below requires, more for larger `nmax_max`/
voltage where node count grows), and closes most of the gap to "adaptive" for the singular (SC-SC)
cases this optimization effort's own worst cases live in. It does NOT make "fixed" a clear
replacement for "adaptive" as the default: adaptive remains faster on the normal-junction and
hardest-SC-SC cases (2 of 3 representative cases, including the structurally-unfavorable
normal-normal case), and
"fixed"'s main advantage was always determinism/cacheability for a future batched pipeline, not raw
per-call speed on an isolated call. `dc_current`'s default remains `quadrature="adaptive"`; `"fixed"`
stays available as an explicit opt-in, now backed by the batched solver from this update rather than
the plain per-node loop item 2b shipped it with.

**Memory**: solving all `nq` quadrature nodes' chains in one unchunked batched call holds ~13-18 live
`(nq,ns,dim,dim)` complex128 arrays at once across `_floquet_green_functions_batch`/
`_assemble_chain_batch_jit`/`_rgf_chain_batch_jit` (`sigR0`/`sigR1`/`Gr00`/`Gless00`/`sigL_less`/
`sigL_a`/`Es`/`SigLess`/`taus`/`sl_less`/`sl_a`/`G`/`Gless`, briefly more while the second
`start_block`'s set is built before the first is released). Measured directly (not just estimated) on
a deliberately large-node-count/long-chain shape (`delta_sc=0.1`, `transparency=0.3`, `voltage=1.0`,
`nmax_max=64` -> nq=2672, ns=129, dim=4, `resource.getrusage(...).ru_maxrss`, single process): an
unchunked call peaks at **~2.1GB** RSS; the same call through `dc_current` with the shipped
node-chunking (below) peaks at **~380MB** -- roughly a 5.6x reduction on this shape, and would scale
further apart at larger `dim` (more orbitals, or a LocalProbe) since chunking caps memory at a
constant set by `chunk_size` regardless of `nq`, while the unchunked figure scales linearly with it.
`iv_curve` fans the unchunked cost out per `parallel.pcall` worker independently, so this matters more
there, not less. `current_integrand_batch` therefore solves in node chunks of `_BATCH_CHUNK_NODES`
(256, in `keldyshtk/current.py`) rather than all `nq` nodes at once: only the final `(nq,)` float
array of per-node integrand values survives across chunks, and the weighted sum in `dc_current`'s
`integral(nmax)` still runs as a single `np.dot(weights, vals)` over the complete, correctly-ordered
array, so chunking does not change the summation order or the result (verified bit-identical to an
unchunked call, independent of `chunk_size`, in
tests/keldysh/test_batched_fixed_quadrature.py).

**Threading caveat**: the `prange` parallelism in `_assemble_chain_batch_jit`/`_rgf_chain_batch_jit`
only delivers multi-threaded scaling in a single-process call. Inside a `parallel.pcall` worker (e.g.
`iv_curve` with `cores>1`), `parallel.set_num_threads()` clamps numba to 1 thread per worker to avoid
oversubscribing the process pool, so batching's benefit there is limited to fewer Python/numba
dispatch round trips (still real) rather than the thread-parallel speedup the single-process numbers
above show.

## Update: items 2b/2c closed out (fixed quadrature shipped as opt-in, now batched); item 3
## (AAA accuracy-growth diagnosis) investigated and does NOT proceed to a fix

Wrap-up of the three items queued after the previous two updates. All three ran against the repo
state left by the "direct finite-T Keldysh evaluation" update above (unrelated to this thread --
that was a separate, already-validated, already-tested default flip in `finite_T_didv`, not touched
by any of items 2b/2c/3, but still sitting uncommitted in the same working tree at the time of this
writing -- worth landing as its own changeset rather than folded into this one).

**Item 2b -- deterministic fixed-node quasienergy quadrature.** Added `_fixed_quasienergy_nodes`
(composite Gauss-Legendre over `[0,|voltage|]`, panel width `_FIXED_QUAD_PANEL_WIDTH=0.006`, floor
`_FIXED_QUAD_MIN_PANELS=6` panels, order `_FIXED_QUAD_ORDER=16` -- so node count scales with
`|voltage|` rather than being fixed) and a `quadrature` kwarg on `dc_current` (`"adaptive"`,
unchanged default, vs. opt-in `"fixed"`). Accuracy is fine: worst relative error 5.8e-4 against a
tight reference across a 34-case SC-SC/normal sweep (delta_sc in {0.1,0.3}, transparency in
{0.3,0.6,1.0}, voltage in {0.05,...,1.0}), independently re-checked at verify time with a
from-scratch reference script on 7 different cases (including near-unity transparency) at 2.1e-6
(adaptive) / 6.8e-6 (fixed) worst-case -- comfortably under the 1e-3 bar, no sign of the doc's own
known truncated-sum failure mode. Speed, as shipped, was NOT a win: 1.94x slower than adaptive on
the hardest SC-SC validation case, 1.39x slower on the doc's own deep-subgap representative case,
45x slower on a cheap normal-normal case with no gap-edge singularity to resolve -- a fixed
location-blind grid has to be dense enough to catch a singularity wherever it lands, which costs
more than adaptive quadrature discovering there's nothing to refine around. Shipped anyway as
`quadrature="fixed"`, non-default, explicitly built as the known-node-set foundation item 2c needed
-- `dc_current`'s default behavior (`quadrature="adaptive"`) is unchanged, verified byte-identical
to the pre-change code path. Two process gaps the verify pass flagged at the time are now partially and fully closed,
respectively: there was no committed pytest test for `quadrature="fixed"`/
`_fixed_quasienergy_nodes` at all (only ad-hoc validation scripts) -- item 2c's
`tests/keldysh/test_batched_fixed_quadrature.py` (below) does call `_fixed_quasienergy_nodes`
directly and asserts against it (integrand-vs-per-node-loop agreement, chunk-size independence),
so the function is no longer untested, but no committed test asserts the specific determinism
property (same `voltage` -> bit-identical node/weight arrays across repeated calls) that was 2b's
own justification for calling the node set "known in advance" -- that narrow property still has no
direct regression test. `documentation/user_guide.md` did not mention the new `quadrature` kwarg --
that gap is now closed, in this update, next to the existing `selfenergy_method` paragraph.

**Item 2c -- batch the per-node chain solve over quadrature nodes.** Built to give item 2b's known,
fixed node set something to pay for itself with: `_assemble_chain_batch_jit`/`_rgf_chain_batch_jit`,
numba `@jit(parallel=True)` mirrors of the existing per-node `_assemble_chain_jit`/`_rgf_chain_jit`
with an added leading quadrature-node axis run over `prange` (same batch-over-an-independent-axis
pattern as `greentk/rg.py`'s sideband batching -- each node's chain is a different quasienergy with
no coupling to any other node, so this is a pure vectorization, not a truncation of any sum).
Self-energies still route through the existing per-energy `cache` dict, just flattened over
`(node,sideband)` per call, so cross-`nmax`-step cache reuse is unchanged. An advisor review during
the task caught a real memory issue before it shipped: unchunked batching held ~13-18 live
`(nq,ns,dim,dim)` complex128 arrays at once, measured at ~2.1GB peak RSS on a large-node-count case
(`nmax_max=64`, `voltage=1.0` -> nq=2672, ns=129); fixed by chunking the *solve* (not the final
weighted sum, which stays one `np.dot` over the complete array) into groups of
`_BATCH_CHUNK_NODES=256`, cutting peak RSS to ~380MB on the same case, verified bit-identical to an
unchunked call independent of chunk size. Net speed on the same three representative case shapes
benchmarked in the section immediately above (deep-subgap, hardest-SC-SC, cheap normal-normal):
batching+chunking closes most of the gap to adaptive quadrature that item 2b alone left open --
the deep-subgap case goes from 1.39x-slower-than-adaptive to faster than both the unbatched loop
and adaptive itself, the hardest SC-SC case goes from 1.94x-slower to only ~1.4x slower, and the
structurally-unfavorable normal-normal case (no gap-edge singularity for a location-blind fixed
grid to size itself around) improves 1.44x but stays ~5.7x slower than adaptive regardless (full
numbers in the "Speed" table above). Net: `quadrature="fixed"`+batching is no longer a clear loss
everywhere (as item 2b alone was), but adaptive still wins 2 of the 3 representative shapes, so
`dc_current`'s default stays `quadrature="adaptive"` -- "fixed" remains opt-in infrastructure for
callers that need a deterministic/cacheable node set, not a general speed upgrade. Shipped with `tests/keldysh/test_batched_fixed_quadrature.py` (8 tests: integrand-level
bit-identical checks against the unbatched loop, chunk-size independence, and `dc_current`-level
fixed-vs-adaptive agreement including a finite-temperature and a normal-junction case). Full suite
after landing: `tests/keldysh` 94 passed (86 pre-existing + 8 new), `tests/transport/
test_kappa_jax.py` 9 passed. Verify pass independently reran the suite (103 passed total), and
independently re-checked correctness on cases the implementer's own tests didn't reach: a fresh
finite-temperature case with a deliberately non-divisor chunk size (chunk_size=97 against 992
nodes) came back bit-identical to both chunk_size=1 and a single unchunked chunk; the hardest SC-SC
case at `voltage=0.55` (only checked bit-identical at the integrand level by the implementer, not
end-to-end through `dc_current`'s adaptive-nmax outer loop) came back at 1.88e-3 relative
difference against adaptive when checked end-to-end -- consistent with, not a regression from, the
fixed quadrature's own already-established ~1e-3-scale accuracy bound, no blow-up; and a
`parallel.pcall`-worker invocation (`cores=2` vs `cores=1`) was confirmed to give identical results
with no hang, closing the one caveat item 2c's own report had left unverified. Verdict: SOUND, no
correctness concerns raised.

**Item 3 -- diagnose the AAA self-energy accuracy growth (0.5% at `nmax_max=4` up to 9.8% at
`nmax_max=40`, first found and left unresolved in the update above).** Two candidate mechanisms
were tested against a strict decision rule (proceed to a fix only if passivity violation tracks the
error growth AND plain RGF-chain amplification alone does not already explain it); both mechanisms
failed to survive direct measurement, so **no fix was attempted, and this item does not proceed.**
The representative case (voltage=0.18, delta_sc=0.3 via `add_swave`, transparency=0.5,
`HT.delta=1e-4`) reproduced the doc's own numbers almost exactly first (0.515%, 0.555%, 1.367%,
3.308%, 4.758%, 9.777% at `nmax_max` in {4,8,16,24,32,40}, vs. the doc's 0.5/0.6/1.4/3.3/4.8/9.8%),
confirming the diagnosis was measuring the same effect.

*Passivity check*: built the exact `SelfenergyAAA` interpolant `dc_current(aaa)` uses at each
`nmax_max`, sampled `-Im(Hermitian part of Sigma_fit(E))` eigenvalues at fixed density per unit
energy across the whole sweep. Violations are real in absolute terms (worst negative eigenvalue
-0.16 to -1.08 against a typical `-Im(Sigma)` eigenvalue scale of ~17-55) but their magnitude does
**not** track the error growth: worst-violation depth was -0.156, -0.447, -0.228, -0.137, -0.848,
-1.077 and violating-region measure (as a fraction of the fit window) was 0.0056, 0.0043, 0.0039,
0.00067, 0.0035, 0.0134, for `nmax_max`=4,8,16,24,32,40 -- both non-monotonic, and decisively so at
`nmax_max=24`: the smallest violation of the entire sweep occurs exactly where the error (3.3%,
already 6x its `nmax_max=4` value) is well underway. A mechanism whose own magnitude bottoms out
partway through the growth it's supposed to explain cannot be that growth's driver. Passivity
violation is present but is a roughly constant background feature of the AAA fit, unrelated to
window size.

*Amplification check*: injected a known `PERTURB=1e-7` perturbation (matching `SelfenergyAAA`'s own
validation-error scale) into one self-energy evaluation (lead 0, sideband n=0, the chain's center
site) inside the RGF chain, at a fixed quasienergy=0.05 (isolated from the outer quadrature so only
chain-length dependence is measured), across the same `nmax_max` sweep (chain length ns =
2*nmax_max+1 = 9,17,33,49,65,81). Result: the resulting current's sensitivity to that perturbation
(`|current(perturbed)-current(base)|/PERTURB`) was flat at 1.430e-2 to 1.433e-2 across a 9x growth
in chain length -- not merely insufficient to explain the error, but the opposite of amplification:
the RGF sweep is ~70x *less* sensitive to a local perturbation than the perturbation itself, and
that sensitivity does not grow as the chain lengthens (consistent with the doc's own
interior-freezing/Wannier-Stark finding elsewhere in this document -- the recursive sweep contracts
local perturbations rather than compounding them). A back-of-envelope bound assuming every one of
the ~81 sites in the largest window carries an independent, fully-coherently-summed 1e-7 error at
that flat 1.4e-2 sensitivity gives ~1e-7 absolute against a ~6.4e-2 current -- about 1e-6 relative,
six orders of magnitude short of the observed 9.8%.

*Decision*: both legs of the decision rule fail -- passivity violation does not track the error
(condition 1 false), and RGF amplification is not merely insufficient but actively contradicted
(attenuation, not amplification; condition 2's premise doesn't hold either). Per the task's own
rule and its explicit allowance that "this doesn't help" is a legitimate outcome, `proceed_with_fix
= false`; no code was changed for item 3, and there is no fix report or verify report for it
(both null). `selfenergy_method="aaa"` remains opt-in, non-default, with the same known accuracy
gap already documented in the "Update: chose the blunt mitigation" section above --
`selfenergy_method="direct"` is still the shipped default and is unaffected by any of this. The
real driver of the 0.5%->9.8% growth remains an open question: most likely a systematic bias in the
AAA fit itself that scales with window size in a way the fit's own 32-point held-out validation
check doesn't detect (`validation_error` stays flat at ~1e-7-5e-7 across the same sweep where the
actual current error grows 20x, already noted in the "shared-nmax finite difference" update above)
-- but confirming that specific mechanism needs its own follow-up measurement (e.g. characterizing
the fit's error as a function of position within the window and correlating it with which sidebands
dominate the current) and was outside this task's scope.

**Suite status after all three items**: `tests/keldysh tests/transport/test_kappa_jax.py
--import-mode=importlib -q` reconfirmed green (all passing, no failures) on the final combined repo
state.
