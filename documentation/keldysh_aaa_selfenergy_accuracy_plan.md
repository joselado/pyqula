# Improving `selfenergy_method="aaa"` accuracy for Keldysh transport — arXiv-informed plan

Status: **not started, not validated — needs maintainer sign-off**, same process as
`keldysh_sideband_decimation_plan.md` and `gpu_porting_plan.md`. This document proposes a
*diagnostic-first* plan, reviewed once already (Claude Fable, see "Review findings" below)
before any code changes are made.

## Context

`keldysh_sideband_decimation_plan.md`'s later updates ("Update: first attempted fix...",
"Update: chose the blunt mitigation...") document an unresolved accuracy gap in
`aaatk/selfenergy_aaa.py`'s `SelfenergyAAA`: relative Keldysh-current error of
`selfenergy_method="aaa"` vs `"direct"` grows continuously with fitting-window size —
0.5% at `nmax_max=4` up to 9.8% at `nmax_max=40` — while `SelfenergyAAA`'s own held-out
`validation_error` stays flat at ~1e-7–5e-7 across the *entire* sweep, i.e. the validation
check never detects the growing true error. Two candidate mechanisms were tested and ruled
out by direct measurement: passivity violation (doesn't track the error) and RGF-chain
amplification of a local perturbation (chain *attenuates*, doesn't amplify). The maintainer's
interim mitigation: `"direct"` (no AAA) is now the default everywhere; `"aaa"` is opt-in only.
That mitigation is unaffected by anything below.

This document was requested directly: "aaa could work better for keldysh, check on arxiv how
to make it better and make a plan. consult with fable."

## ArXiv / literature findings

Searched for (a) how AAA-family methods handle accuracy degrading as the fitting domain
grows, (b) physical (passive/causal/stable) constraints on rational fits, (c) prior work
applying AAA-style fitting to lead self-energies specifically. No paper does (c) — this
remains a general numerical-analysis technique applied to this problem, not something with an
existing domain-specific reference to mirror (consistent with the prior decimation doc's own
conclusion). Candidates found for (a)/(b), reviewed below in light of Fable's critique of how
well each actually explains the *measured* symptom:

1. **AAA-Lawson** (Nakatsukasa & Trefethen, [arXiv:1908.06001](https://arxiv.org/abs/1908.06001))
   — adds an IRLS/Lawson-reweighting phase converting AAA's least-squares fit into a minimax
   fit (over the *sample set*). **Deprioritized**: if the real failure is under-resolution
   *between* samples (see diagnosis below), Lawson reweighting is exactly as blind to it as
   plain AAA — it only helps once the candidate grid already resolves the features.
2. **AAA rational approximation on a continuum** (Driscoll/Nakatsukasa/Trefethen line,
   [arXiv:2305.03677](https://arxiv.org/abs/2305.03677)) — motivated the idea of validating
   error over the whole domain rather than a finite point set. **Its literal fix (broad
   random validation sampling) does not close this specific gap** — see diagnosis below;
   the useful piece of this line of work is adaptive *sample* refinement targeted at
   high-error regions, not broader-but-still-random validation.
3. **stabAAA** (convex stability-constrained AAA,
   [arXiv:2312.16978](https://arxiv.org/abs/2312.16978)) — forces the fit to be
   passive/stable by construction. **Dropped**: passivity was already measured not to track
   the error (decimation doc, item 3); this fixes an already-ruled-out symptom.
4. **Vector Fitting + iterative passivity enforcement** (Gustavsen & Semlyen; the mature,
   independently-derived power-systems/microwave-engineering method for the same underlying
   problem). Open-source, BSD-licensed implementation in `scikit-rf`
   (`skrf.vectorFitting.VectorFitting`), usable per CLAUDE.md's "benchmark against an existing
   open-source implementation" guidance. **Kept, reframed as a differential diagnostic** (see
   plan below), not a replacement to adopt outright.
5. **Gauss–Christoffel-quadrature / Hankel-rank rational Green's-function representation with
   structurally-guaranteed positivity** (arXiv:2605.26887, many-body Green's functions
   specifically — id format unconfirmed independently beyond the fetch that returned this
   summary; verify before citing further). A fundamentally different architecture
   (moment-matching, not greedy barycentric fitting). **Out of scope near-term** — no evidence
   yet that a different representation is needed rather than better sampling/validation of the
   current one.

## Review findings (Claude Fable consultation)

An independent review (prompted with this document's draft, the decimation doc, and
`aaatk/selfenergy_aaa.py`/`keldyshtk/current.py`) found the originally-drafted first
diagnostic step circular, and — critically — surfaced a concrete, quantitative mechanism the
prior investigation thread hadn't checked directly:

- **The fitting window grows linearly with `nmax_max`** (`(nmax_max+1)*|voltage|`, per
  `build_selfenergy_aaa`), while **`default_ncand`'s candidate-grid size grows only
  logarithmically** (`scale*ceil(log2(erange/delta))`). At the decimation doc's own worst case
  (`nmax_max=40`, `voltage=0.18`): candidate-grid step ≈0.034 vs. the gap-edge feature width
  `HT.delta=1e-4` — a **factor ~340 under-resolved**, ~7x worse than at `nmax_max=4`. This is a
  continuously-growing under-resolution, exactly matching the observed continuous 0.5%→9.8%
  error growth (no threshold effect expected or observed).
- **The escalation loop that would normally catch this never fires**: `SelfenergyAAA`'s
  `validation_error` (~1e-7) passes `tolerance` (1e-6) on round 1 in these cases, so `ncand`
  never doubles — it silently stays at its logarithmically-small starting value for the entire
  window, regardless of how badly under-resolved the actual gap-edge features are.
- **This is *not* refuted by the earlier reverted `resolution_factor` fix.** That fix was a
  hard threshold gate (`step <= resolution_factor*delta`) and was reverted because the true
  error trend is continuous, not a threshold effect — but a continuous trend is exactly what a
  resolution *mechanism* (grid step growing linearly against a fixed feature width) predicts.
  The gate's failure was in its threshold framing, not in "resolution" as the underlying cause.
  Supporting evidence already in the decimation doc: its own largest single accuracy jump
  (6.6%→0.8%, ~8x) happened exactly when `ncand` finally doubled (432→863) during a tolerance
  sweep.
- **Continuum-style broad-random validation (arXiv:2305.03677's literal fix) would not close
  the detection gap**: 32 validation points drawn uniformly over a window of width ~14.8 have
  ~0.1% probability of landing within `1e-4` of a gap edge. Broad random sampling is exactly as
  blind to a narrow feature as the current offset-from-grid sampling. What's needed instead is
  **feature-targeted** sampling/validation — points clustered near known singular energies
  (`±Δ_sc`), the fit's own poles, and the actual sideband-ladder energies `dc_current` evaluates
  — not merely a wider or more numerous random set.
- The originally-drafted diagnostic ("inject the AAA fit's own residual at every sideband site
  and see if it reproduces the error curve") was flagged as **tautological** — it's
  term-for-term identical to running the AAA fit itself, so it would reproduce the curve by
  construction without isolating a mechanism. A back-of-envelope check using the decimation
  doc's own measured flat per-site sensitivity (~1.4e-2) shows the observed current error is
  consistent with an ordinary per-site *relative* Σ error of only ~1e-4–3e-4 (three orders of
  magnitude above the 1e-7 validation figure, but unremarkable as plain off-grid fit error) —
  i.e. no new "coherent propagation" physics needs to be invoked or tested at all. The real
  open question reduces to a pure Σ-comparison: is the fit's true error at the energies
  `dc_current` actually evaluates ~1e-4-relative rather than the ~1e-7 the validation reports?

## Plan (revised per the review above)

**Step 1 (first, cheap, decisive): dense pointwise error map, no `dc_current` involved.** For
each `nmax_max` in the decimation doc's own sweep ({4,8,16,24,32,40}), build `SelfenergyAAA` as
today, then compare `Σ_fit(E)` against `Σ_direct(E)` (a µs-scale Sancho-Rubio solve, so a
10^5-point scan costs about a second) densely across the window, and specifically at the actual
`quasienergy + n*voltage` ladder energies `dc_current` evaluates. This directly distinguishes:
(a) a genuinely smooth, broad bias (would validate a "coherent" story) vs. (b) localized large
error concentrated at gap-edge singularities (`±Δ_sc`) and/or spurious poles (Froissart
doublets) sitting between candidates — the mechanism the review's analysis above predicts.
Cross-check spike locations, if any, against the fit's own pole set (available directly from
its barycentric form).

**Step 2 (only if Step 1 is ambiguous or to confirm attribution): hybrid ablation inside
`dc_current`.** Patch `Σ_direct` back in over just the high-error region(s) Step 1 identifies,
`Σ_fit` everywhere else, and check whether the current error collapses. Confirms the error's
*current-side* attribution without yet committing to a fix.

**Step 3 (the fix, chosen by Step 1/2's diagnosis — expected outcome: localized
under-resolution):** a feature-aware candidate grid — cluster extra candidates at spacing
~`delta` around the lead's known singular energies (a few hundred extra cheap solves, not
global densification, so cost stays modest even as the window grows) — **plus** feature-targeted
validation points in the same locations, so the existing escalation loop (`ncand`
doubling/`mmax` escalation) actually fires when the fit is bad there, instead of silently
passing on a flat but blind validation check.

**Step 4 (differential diagnostic, can run in parallel with Step 1 if useful): fit
scikit-rf's Vector Fitting on the *same* sample set `SelfenergyAAA` used** (same `Z`,
`ncand=432` at the worst case) and compare its current error. If VF matches AAA's error growth
on the identical samples, the sample set is the problem (confirms Step 1's diagnosis
independent of AAA's own algorithm specifics); if VF is materially more accurate on the same
samples, that points at something AAA-specific in how it selects/weights support points rather
than pure under-sampling.

**Step 5 (optional, only if Step 3 alone doesn't fully close the gap): AAA-Lawson minimax
refinement** on top of an already-adequately-sampled fit. Not expected to be needed if Step 3
succeeds, since Lawson reweighting can't fix under-resolution, only redistribute error across an
already-resolved sample set.

**Not pursued**: stabAAA (fixes an already-ruled-out symptom, passivity), the
quadrature/Hankel-rank architecture (arXiv:2605.26887 — no evidence a different representation
is needed rather than better sampling of the current one).

## Housekeeping noted during this review, not yet acted on

- `keldyshtk/current.py`'s `dc_current` docstring (~line 1087) still attributes the accuracy
  gap to "error compounding through the RGF chain" — the decimation doc's own item-3
  measurement showed *attenuation*, not amplification, so this text is stale and should be
  corrected as part of whatever fix lands.
- `SelfenergyAAA`'s docstring describes validation as "relative, entrywise," but the code
  normalizes error by a single global `denom = max|Σ|` over the validation points — not
  believed to be the driver of the accuracy gap, but worth fixing for docstring accuracy while
  the validation logic is being touched anyway (Step 3).

## Acceptance bar for any fix

Per the review's closing point: whichever fix lands, the acceptance test must be direct
AAA-vs-direct **current** agreement across the `nmax_max∈{4,...,40}` sweep (the curve that
needs to flatten near 0), not `SelfenergyAAA.validation_error` — this whole investigation
exists because that quantity was shown not to be a reliable proxy for the thing that actually
matters.

## Update: Step 1 (dense Σ error map) run — confirms localized under-resolution, refines the mechanism

Ran the dense pointwise `Σ_fit` vs `Σ_direct` scan (200,000 points per `nmax_max`, batched via
`ht.get_selfenergy_batch`, ~2 minutes total) on the exact item-3 case
(`delta_sc=0.3`, `transparency=0.5`, `HT.delta=1e-4`, `voltage=0.18`), across
`nmax_max∈{4,8,16,24,32,40}`. For each `nmax_max`, computed: (a) max relative error over the
whole dense window, (b) max relative error restricted to a `±20*HT.delta` neighborhood of the
lead's known singular energies (`0`, `±delta_sc`), (c) max relative error *away* from those
neighborhoods ("bulk"), (d) max relative error specifically at the actual
`quasienergy+n*voltage` sideband-ladder energies `dc_current` evaluates, and (e) `SelfenergyAAA`'s
own reported `validation_error`.

| nmax_max | ncand | grid step / HT.delta | validation_error | max error (whole window) | max error (near ±Δ_sc / 0) | max error (bulk, away from features) | max error (actual ladder energies) |
|---|---|---|---|---|---|---|---|
| 4  | 719  | 25.1  | 2.5e-7 | 8.8e-2 | 8.8e-2 | 1.8e-5 | 8.7e-2 |
| 8  | 360  | 90.3  | 2.6e-7 | 3.3e-1 | 3.3e-1 | 3.9e-3 | 3.3e-1 |
| 16 | 767  | 79.9  | 1.7e-7 | 3.1e-1 | 3.1e-1 | 1.5e-3 | 3.1e-1 |
| 24 | 1629 | 55.3  | 2.8e-7 | 2.2e-1 | 2.2e-1 | 7.7e-4 | 2.2e-1 |
| 32 | 815  | 145.9 | 1.7e-7 | 4.4e-1 | 4.4e-1 | 7.8e-3 | 4.4e-1 |
| 40 | 863  | 171.2 | 1.8e-7 | 4.8e-1 | 4.8e-1 | 1.3e-2 | 5.0e-1 |

**This is a decisive confirmation of the review's core diagnosis, with one refinement.** Every
single dense-window error maximum lands exactly at `e=±delta_sc` (`argmax_e` was `±0.3` in all
six rows, never anywhere else) — the gap-edge coherence-peak singularity, whose physical width
is set by `HT.delta=1e-4`, the same scale the review's grid-step-vs-feature-width argument used.
The candidate grid step is 25–170x coarser than that width at every `nmax_max` tested (never
close to resolving it), which is why the local error there is enormous (9–48%) — but that error
is confined to an extremely narrow energy neighborhood and does *not* track the observed
0.5%→9.8% *current* error trend on its own (it's already 8.8% at `nmax_max=4`, where the
measured current error is only 0.5%).

**The trend that does track the current-error growth is the "bulk" column** — the error away
from the singularities, which is what most of the sideband ladder actually samples: it grows
from 1.8e-5 (`nmax_max=4`) to 1.3e-2 (`nmax_max=40`), a ~700x increase over the same sweep where
the measured current error grew ~20x (0.5%→9.8%) — same order of magnitude, same growing trend,
plausible as the dominant contribution once you account for it not being a coincidence that both
grow roughly together, even if the two ratios aren't identical (a single bulk-error scalar isn't
expected to linearly predict an aggregate trace-sum error without going through the actual
chain). Both quantities grow non-monotonically with `nmax_max` (driven by the escalation loop's
uneven `ncand` doubling — e.g. `ncand` is 1629 at `nmax_max=24` but only 815 at `nmax_max=32`,
because the loop only re-escalates when `validation_error` happens to fail, which — per the
review's core point — it essentially never does), consistent with the observed non-monotonic
noise in the original current-error measurements too.

**Refined picture**: this is not purely "the singularity is under-resolved" (that error, while
huge, is too narrow in energy to move the current integral much on its own) nor purely a
diffuse "coherent bias" (there's no such thing spread smoothly across the whole window — away
from the singularities the fit is quite good, ~1e-5 at the narrowest window). It is the
**growing background/bulk fit error** — itself still a consequence of the grid step growing
faster than `default_ncand`'s logarithmic scaling can compensate for, exactly the review's
mechanism, just measured now to be the "bulk" component rather than the singularity spike
itself — that best explains the current-error trend. The `ladder_maxerr` column confirms the
actual sideband-ladder energies *do* land close enough to the singularities to inherit their
huge local error at several `nmax_max` values (nearly identical to the whole-window max in most
rows) — so Step 2's ablation (patching `Σ_direct` back in near the singularities vs. patching it
in across the bulk) is still worth running to cleanly separate how much of the current error
each component contributes, before committing to Step 3's fix design.

**Implication for Step 3**: the fix should address both found here — (a) cluster extra
candidates at spacing `~HT.delta` around the lead's known singular energies (`0`, `±delta_sc`),
as originally planned, to close the huge but narrow spike, and (b) make the *background*
candidate density scale with window width itself (not just `log2(erange/delta)`), or make
`nvalidate` scale similarly, so the escalation loop is actually exercised by bulk under-
resolution and not just gap-edge features — `default_ncand`'s log-scaling assumption (that only
counting resonance-width scales, not raw window width, is enough) looks likely to be the piece
that needs revisiting, not just adding gap-edge-targeted points on top of it unchanged.

Script: not yet committed (ad hoc, run from `/tmp/.../scratchpad/aaa_error_map.py`); should be
turned into a proper `tests/keldysh` diagnostic or benchmark script before Step 2/3 land, so this
measurement is reproducible and re-runnable as a regression check once a fix is in place.

**Next step needs sign-off before proceeding**: Step 2 (ablation) is cheap and would sharpen the
attribution above; Step 3 (an actual fix to `SelfenergyAAA`'s candidate-grid/validation logic) is
a real code change to production self-energy-fitting logic and should not proceed without
explicit maintainer approval, per this document's own process notes.

## Update: Step 3 implemented and validated — accuracy gap closed, growth trend eliminated

Implemented directly on the maintainer's go-ahead (skipping Step 2's ablation, since Step 1's
measurement already isolated the mechanism clearly enough to design a fix). Three changes to
`aaatk/selfenergy_aaa.py`'s `SelfenergyAAA`:

1. **Domain-independent validation.** Held-out validation points used to be constructed as an
   offset within one grid step of an existing candidate (`Z[base] + offsets`) — exactly as blind
   to under-resolution as the grid itself. Replaced with two independent sources every round: (a)
   points drawn uniformly at random across the *entire* `[emin,emax]` window, for generic/bulk
   coverage; (b) points drawn *inside* the currently roughest intervals (same curvature signal
   used for refinement, see below), at fractions distinct from any future bisection midpoint, so a
   narrow unresolved feature is actually tested rather than only the wide gaps between features.
   Pure domain-uniform sampling alone was tried first and made things *worse* (see below) — only
   the combination catches both failure modes.
2. **Curvature-driven adaptive grid refinement (`_refine_grid`).** Escalation used to bisect
   *every* candidate interval uniformly each round, which reaches `ncand_max` after only ~2-3
   rounds regardless of `maxrounds` (`ncand` roughly doubles each round) — meaning uniform
   doubling structurally *cannot* resolve a feature narrower than the initial spacing by a large
   factor, no matter how many rounds are allowed, since the whole budget gets spent on already-fine
   regions before the narrow feature's own spacing shrinks enough. `_refine_grid` instead bisects
   the roughest intervals first (largest jump in already-sampled Σ between adjacent candidates),
   so a genuine singularity's own interval — and, after each bisection, whichever of its two new
   children is worse — keeps getting selected round after round, converging its local spacing
   exponentially instead of sharing the budget uniformly with already-resolved regions.
3. **Loosened tolerance/raised caps to match what's actually needed.** The default `tolerance`
   (self-energy fit accuracy) was `1e-6` — far tighter than warranted: chasing that on a genuine
   near-singularity took ~4-9 minutes to build one lead's interpolant (`ncand` growing to ~16,600),
   for no measurable benefit to the quantity that actually matters (current accuracy was already
   ~0.2% at a self-energy tolerance 1000x looser). Changed the default to `1e-3` — matching
   `dc_current`'s own `tol=1e-3` current-convergence target, so the self-energy isn't fit to a
   precision the physics calculation itself doesn't resolve — and raised `ncand_max`
   (2500→20000) and `maxrounds` (8→20) so genuinely hard targets have enough budget/rounds to
   reach real convergence rather than silently plateauing at an under-resolved fit (as the old
   `ncand_max=2500` cap did for the deep-subgap case). `default_ncand`, `mmax`/`mmax_max`
   escalation, and `aaa()` itself were untouched.

A first attempt using only the domain-uniform validation fix (without the curvature-targeted
points) was tried and directly measured to be *worse* than the original blind-grid-tied
validation — 32 random points spread over a window as wide as ~15 have too little statistical
power to reliably land near either the narrow gap-edge singularity or the somewhat-localized bulk
degradation, so escalation triggered even less often than before. Adding the curvature-targeted
component (reusing the exact signal `_refine_grid` already computes) fixed this.

**Validated directly against the acceptance bar this whole investigation converged on — current
agreement, not `validation_error`** — on the exact item-3 case (`delta_sc=0.3`,
`transparency=0.5`, `HT.delta=1e-4`, `voltage=0.18`), across the full `nmax_max` sweep:

| nmax_max | direct time | AAA build time | ncand | AAA eval time | current rel. error |
|---|---|---|---|---|---|
| 4  | 0.71s | 7.04s  | 2733  | 0.08s | 2.05e-3 |
| 8  | 0.67s | 6.54s  | 4099  | 0.17s | 7.62e-3 |
| 16 | 1.28s | 15.26s | 6561  | 0.29s | 3.54e-4 |
| 24 | 1.97s | 17.38s | 6967  | 0.46s | 2.38e-3 |
| 32 | 3.00s | 16.97s | 6967  | 0.64s | 2.90e-4 |
| 40 | 2.47s | 31.06s | 11070 | 0.84s | 2.18e-3 |

**No more growth trend**: relative current error is now flat in a ~2e-4–8e-3 band across the
entire sweep, vs. the old continuously-growing 0.5%→9.8%. Eval-only time is consistently 5-9x
faster than a single `"direct"` call at every window size (0.08-0.84s vs 0.67-3.00s); build time
(7-31s) does *not* undercut a single isolated call — the win is entirely in reuse, exactly matching
the module's designed use case (`build_shared_selfenergy`, an `iv_curve` sweep, `finite_T_didv`'s
per-point thermal quadrature).

**The discriminating check an advisor review flagged before declaring this done**: the global
`tolerance` default change (1e-6→1e-3) affects every `SelfenergyAAA`/`build_selfenergy_aaa`
caller, not just plain `dc_current` — including `didv.keldysh_didv`'s finite-difference dI/dV,
which the earlier decimation-doc investigation found amplifies a per-branch self-energy error by
`|Ip|/(Ip-Im)` (~12x on this exact case) via catastrophic cancellation, and had previously measured
a 37-60% dI/dV error from that amplification. Validated directly on that same case
(`voltage=0.18`, `delta_sc=0.3`, `transparency=0.5`, `HT.delta=1e-4`, `nmax_max=40`,
`fixed_nmax=40` on both branches to isolate the self-energy-method effect):

| configuration | Ip | Im | Ip−Im | dI/dV | rel. error vs. direct |
|---|---|---|---|---|---|
| direct | 2.792117e-2 | 2.518027e-2 | 2.7409e-3 | 7.613601e-1 | — |
| aaa, shared interpolant (keldysh_didv's own default) | — | — | — | 7.608092e-1 | 7.24e-4 |
| aaa, independent fits per branch | 2.793847e-2 | 2.517045e-2 | 2.7680e-3 | 7.688940e-1 | 9.90e-3 |

Both configurations are now comfortably accurate (well under 1%), a ~500-800x improvement over the
pre-fix 37-60% figure for exactly the failure mode that motivated tightening (not loosening)
tolerance in the first place — confirming the fix's mechanism (candidate-grid resolution, not raw
fit tolerance) was the right lever, and that the looser `1e-3` default does not reopen this
amplification-sensitive path. The shared-interpolant configuration (what `keldysh_didv` actually
uses by default) is markedly more accurate than independent fits, consistent with the original
decimation doc's own observation that shared vs. independent fits differ materially here — Ip and
Im's correlated fit errors partially cancel in the difference when they come from the same
interpolant, an additional (unplanned) benefit of interpolant sharing beyond its original
build-cost-amortization motivation.

**Regression coverage**: `tests/keldysh/test_selfenergy_aaa_accuracy.py` (new) asserts (1)
AAA-vs-direct `dc_current` agreement (`<2e-2` relative) at `nmax_max∈{8,24,40}`, and (2)
AAA-vs-direct `keldysh_didv` agreement (`<5e-2` relative) on the catastrophic-cancellation case
above — deliberately checking current/dI/dV agreement rather than `validation_error`, since the
latter was shown not to be a reliable proxy for either. Full `tests/keldysh` +
`tests/transport/test_kappa_jax.py`, run together including the new file: 107 passed (103
pre-existing + 4 new), no regressions. `documentation/user_guide.md`'s AAA self-energy section and
the stale "error compounding through the RGF chain" docstring in `keldyshtk/current.py`'s
`dc_current` were both updated to describe the actual root cause and the fix.

**Still opt-in, on purpose**: `selfenergy_method="aaa"` remains non-default. The fix closes the
specific accuracy gap this investigation found on the cases tested; it doesn't change the module's
own documented bounded-effort philosophy (`converged=False` + safe fallback to `"direct"` for a
target that genuinely resists this ansatz within its budget) or constitute a blanket accuracy
guarantee for every possible system/parameter range. A caller should still check agreement with
`"direct"` for their own case before relying on `"aaa"` for anything unattended.

## Process notes

- Do not implement Step 3 (or any fix) without explicit sign-off — Step 1 is cheap and
  low-risk enough to run first regardless, since it changes no production code and directly
  decides whether the rest of this plan's ordering is right.
- If Step 1 shows a smooth broad bias rather than localized under-resolution (contrary to this
  plan's expectation), stop and re-consult before proceeding to Step 3 as drafted — the fix
  for that case would look different (closer to the originally-drafted coherent-bias framing)
  and hasn't been designed here.
