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
