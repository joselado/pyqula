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
     selfenergy` to gather results) — `round()` alone was 214200 calls / ~22% of total wall
     time in one profiled `dc_current` call. Now computed once and reused for both.

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
