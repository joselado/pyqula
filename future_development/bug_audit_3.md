# Bug audit 3 -- five-area third sweep, 2026-09-24

The third sweep started where [`audit_open_decisions.md`](audit_open_decisions.md)
section 3 said it should: at the five areas the first two sweeps named as
never exercised. One auditing agent per area ran real calculations against an
oracle independent of the code under test, and a second agent per area then
tried to refute every finding by re-running it. The fixes were made afterwards,
one agent per file-disjoint group of findings, each in its own git worktree,
and each group was checked by a further agent that re-ran the original
reproductions *and* every oracle the area had passed before the fix, and ran
each new test against the unfixed source to confirm it fails there.

| Area | What it covers |
| --- | --- |
| jax SCF | `scftk/densitydensity_jax.py`, `scftk/vjinteraction_jax.py`: every solver, both routes |
| 3D superfluid weight | `sctk/superfluidweight.py` with `dim==3` |
| qtci SCF backend | `integration="qtci"`, `qtcitk/densitymatrix_qtci.py` |
| Broyden mixing | `scftk/broydenmixing.py` through both engines |
| Per-site filling | the array-filling branch of `spinspin._run_anisotropic_scf` |

**18 findings; 17 confirmed by the refuting agent, one rated plausible.**
Every run was on the CPU. The reproduction scripts were written in a session
scratchpad and are not preserved here; each entry says what was run and what
it printed.

## Status

**16 fixed, one decided, one left as it is, and two more fixed after the
sweep (#19, #20), as was the resolution limit #11 had left.** Every fix has a
regression test that asserts the finding's oracle, and each of those tests
fails on the unfixed source. The full suite on the merged fixes was stopped
at 78% with no failure, and the qtci energy fix (#11) was checked with the qtci
tests only; a complete run on the merged result is still owed, and it now also
has to cover the jax side of #16, the qtci Fermi level of #11, the finite
`maxite` default of #19 and the `fsolve` handover of #20, which were checked
with their own tests only.

- **Decided -- keep the qtci backend.** Its tensor cross interpolation
  compresses nothing in 2D (see #13). The maintainer's call on 2026-09-24 was
  to keep it; the measurements are below so the question does not have to be
  re-asked from scratch.
- **Left as it is -- a spinless Nambu `get_mean_field_hamiltonian` returns
  `None`** (#18). It is the package-wide `NotImplemented -> None` contract that
  `_mean_field_scf_result` documents, not a qtci slip.

## User-visible changes

Read these before upgrading. Most of them turn a silently wrong number into a
right one; two turn a silently ignored argument into an exception.

- `use_jax=True` with a filling target holds the requested electron count when
  the Fermi level falls inside a degenerate multiplet (#1).
- The spinless `use_jax=True` route returns a Hamiltonian measured from the
  Fermi level, with `.fermi` set, and the right energy at a fixed nonzero `mu`
  (#2).
- `solver="newton"` and `"newton_krylov"` converge at a fixed filling where
  they used to stop with `converged=False` (#3), and so does `"fsolve"`,
  which continues with Newton where it stalls (#20).
- Both `use_jax=True` routes accept the same nine solver names, and an unknown
  name raises a `ValueError` listing them (#4).
- **Breaking:** `solver=`, `gmres_tol=` or `gmres_restart=` on a spinful
  mean-field call without `use_jax=True` raise `NotImplementedError`; they used
  to be ignored (#5).
- `solver="fixed_point"` / `"linear_mixing"` under jax stop on the largest
  residual, so they can take a few more iterations (#6).
- A per-site (array) filling converges at the default temperature, converges
  quickly in gapped states, is validated up front, and is refused with a named
  error by the routes that cannot take one (#7-#10). Its **total** count is now
  rounded to a whole number of k-states, as a scalar filling's is; only the
  site-resolved differences are held to `maxerror`.
- `integration="qtci"` locates the Fermi level on the nodes its density matrix
  is integrated on, holding the filling exactly at any T>0 by occupying the
  level at the Fermi energy in part, sums the total energy there too, and
  returns the complete density matrix in `scf.dm` (#11, #12).
- **Breaking:** every numpy mean-field loop stops at `maxite=1000` by default
  and returns `None` past it, where it used to run until it converged;
  `maxite=None` restores no limit (#19).
- `solver="broyden_mixing"` honours `mix=`, under both engines, is silent at
  `verbose=0`, and warms up with `lam=0.5`, taking 2.5-4.5x fewer
  density-matrix evaluations (#15, #16).

## 1. jax SCF

### 1. A filling target holds the wrong electron number when the Fermi level cuts a degenerate multiplet, and reports converged=True

**Cause.** `densitydensity_jax.py` (and `vjinteraction_jax.py`) put the Fermi
level at the midpoint of the last occupied and first empty eigenvalue. When
both belong to one degenerate multiplet -- generic for a metal on a symmetric
k-mesh; the honeycomb at nk=6 has sixfold k-stars -- that midpoint is the level
itself, and `sigmoid(0)=0.5` half-fills every member.

**Oracle.** `Tr dm(0,0,0)/n` must equal the requested filling; the numpy
engine, which inverts the finite-T count, gets it right.

**Before.** Bare honeycomb at filling 2/72: requested 0.02778, numpy 0.02778,
jax 0.05556. At filling 0.45 the jax newton run converged at `Tr/n=0.368`.

**Status.** Fixed in `603d4ce`. A new `mu_for_filling`, mirroring
`spectrum.get_fermi_energy_T`, keeps the midpoint when it already holds the
count and otherwise solves the smeared count by bisection, with one Newton step
on the live spectrum to carry the implicit-function derivative (checked against
finite differences).

### 2. The spinless use_jax route returns a Hamiltonian not shifted to the Fermi level, and its energy at fixed mu is off by mu*N

**Cause.** `generic_densitydensity_jax` never called `shift_fermi` or set
`.fermi`, and summed the energy on the unshifted spectrum. The spinful sibling
`vjinteraction_jax` had already been fixed for exactly this.

**Oracle.** The same public call with `use_jax=False` and `True` must return the
same bands and energy; their mean fields agree to 1e-9.

**Before.** At `mu=0.5` the energies differed by 0.35 and the bands by 0.5; all
seven solvers gave -3.2100 against numpy's -4.0000 at `mu=0.4`.

**Status.** Fixed in `bdfe6d9` by giving it `vjinteraction_jax`'s tail.

### 3. The default solver="newton" stalls with a filling target

**Cause.** A soft mode of `J-I` with singular value 3e-6 survives
`lstsq(rcond=1e-8)`, the Newton step comes out at |dx|~1.4e4, and 30 halvings
of backtracking never reach descent. The in-code comment claimed `lstsq`
"degrades gracefully".

**Status.** Fixed in `0bab582`. When backtracking fails, `newton_solve` tries
Levenberg-Marquardt steps of growing damping with the Jacobian it already has,
and if the point is a stationary point of the merit with a nonzero residual it
takes up to five bursts of 60 linear-mixing steps before resuming.
`newton_krylov` gets the same fallbacks matrix-free. Over 12 seeds newton now
converges 12/12 (3/6 before) and newton_krylov 12/12 (1/12). Truncating at
`rcond=1e-5`, which the finding suggested, made it worse (7/12).

**Open.** The burst length of 60 was tuned on one system (biased AF chain, U=3,
nk=10). Since 25 September 2026 it is the keyword `kick_steps` of
`VJinteraction` and `Vinteraction` with `use_jax=True`, forwarded to
`newton`, `newton_krylov` and the Newton handover of a stalled `fsolve`
(`tests/scf/test_newton_kick_steps.py`), the maintainer's call over a
measured sweep, so a stalled case can be tuned without editing the source;
the default of 60 was not re-measured. `scf.iterations` counts outer iterations only. Over seeds 0 to 11 on
that chain newton takes 23 to 119 outer iterations and newton_krylov 24 to
365, the slow ones crawling on damped steps. A trust region was thought to be
the principled follow-up, on the grounds that `fsolve`'s dogleg is never
trapped here; that was measured on seeds 0 to 5 only, and it is not true (see
#20). The one trust region in the package gets stuck at the soft mode that
the Newton fallbacks leave, so replacing those fallbacks with a trust region
would move the Newton solvers toward the behaviour that fails, and it is not
the follow-up any more.

### 4. The unknown-solver error lists no names, and the two routes accept different names

**Status.** Fixed in `fd0b2e1`: one registry (`_JAX_SOLVERS` plus aliases) that
both routes import, an error built from it, and names resolved before any work.

### 5. solver= is silently ignored without use_jax

**Status.** Fixed in `0f61591`. `solver`, `gmres_tol` and `gmres_restart` on
`VJinteraction`'s numpy path raise `NotImplementedError` naming the jax solver
names, like the kpm-only keywords next to them.

### 6. fixed_point reports converged=True while the largest residual is ~8x maxerror

**Cause.** It stopped on the mean change, every other solver on the largest.

**Status.** Fixed in `06a5605`.

## 2. Per-site (array) filling

### 7. It cannot converge at the default T when the target needs a partly filled k-shell, and with maxite=None it never returns

**Cause.** The chemical-potential update `lam += mix*(filling-occ)` in
`spinspin.py` overshoots the steep step of `occ(lam)` at T~0, where the scalar
path simply puts the Fermi level on the level.

**Oracle.** A uniform array must reproduce the scalar-filling result; two
decoupled parts with different fillings must equal the parts solved separately.

**Status.** Fixed in `618949d`. The uniform part of `lam` is now set by the same
Fermi search the scalar path uses; only the site-resolved part is iterated.
The decoupled-parts oracle holds to 1e-9.

### 8. It crawls in gapped states at finite T

**Before.** 15788 iterations to reach 1e-5; 58 on the scalar path.

**Status.** Fixed by #7's change: the uniform component, the one that sits in
the gap, is now found by bisection. 225 iterations on the same case.

### 9. Routes without per-site filling crash with "numpy.ndarray doesn't define __round__"

**Status.** Fixed in `76f79f9`: each route raises a named error pointing at
`VJinteraction`.

### 10. An array filling is not validated

**Status.** Fixed in `b1f10d3`: a wrong length or a value outside [0,1] raises a
`ValueError` before any diagonalization.

**Side effect.** Two spinon Zeeman tests moved to `mix=0.1` (`015bbf4`), the
value the other spinon tests already use: with the uniform array now following
the scalar path exactly, the chain at seed 0 and `mix=0.3` falls into the same
period-7 limit cycle the scalar path has always had there. `maxite=None` was
still unbounded on both paths when the sweep closed; see #19.

## 3. qtci SCF backend

### 11. filling is not honoured in a metal

**Cause.** The Fermi level came from the uniform nk mesh, the density matrix
from the Gauss-Kronrod nodes, and in a metal the two hold different charges at
one Fermi level.

**Status.** Fixed in `35a88eb` (Fermi level located on the GK nodes, by the new
`get_fermi4filling_qtci`) and `a8596b6` (the total energy summed on the same
nodes). Low-symmetry metals now hold the filling (Rashba plus exchange, nk=8:
0.589 before, 0.597 now, for 0.6), and at finite T it is exact to 1e-6.

**Resolution limit, fixed after the sweep.** At T~0 the charge was right only
to the weight of one level on the node grid, and the nodes come in
symmetry-related groups: on the bare square lattice at filling 0.3 and nk=8
the level at the Fermi energy is 8 states holding 0.087 electrons, and the
nearest cut holds 0.6405. On the maintainer's call of 2026-09-24,
`get_fermi4filling_qtci` now keeps the whole-level cut only when it already
holds the filling, and otherwise bisects the Fermi-Dirac count at T, which is
continuous for any T>0, to the filling itself, so that the level at mu is
partly occupied, as the uniform mesh does when its cut falls inside a
degenerate multiplet. The spinless square lattice at filling 0.3 now holds
0.300000 at nk=8 and 16 (0.320 and 0.3095 before), with mu=-1.030 and -0.969
against -1.059 on a dense mesh (-0.762 and -0.828 before). For an attractive
U=-2 on the spinful square lattice at filling 0.3 (`mf="swave"`), the pairing
gap, twice the smallest |E| on a 60x60 mesh, is 0.430 with `ed` at nk=80
(0.429 at nk=40); qtci gave 0.513 at nk=8 and 0.554 at nk=16 before, moving
away from it, and gives 0.362 and 0.419 now, converging on it. What is left
is the quadrature error of a step, which nk converges away.
`test_qtci_fermi_level_holds_the_filling_in_a_high_symmetry_metal` fails on
the unfixed source.

**The energy.** With the Fermi level on the nodes but the band energy still on
the uniform mesh, a bare square metal's energy missed the exact band energy of
the charge it held by 1.7e-2 at nk=8. Summed on the nodes, with the un-shift
using the charge the density matrix holds, it misses by 4.8e-3 (1.7e-3 at
nk=16) -- the quadrature error of a discontinuous integrand, smaller than the
uniform mesh's own 8e-3 at nk=8. A gapped antiferromagnetic honeycomb still
agrees with `ed` to 4e-8.

### 12. scf.dm is zero in every entry the mean field does not read

**Status.** Fixed in `13e35ed`: `full_dm_gk` recomputes the whole matrix on the
same nodes once the loop is done, as the `ed` path already did on its mesh.

### 13. The cross interpolation compresses nothing in 2D (decided: keep)

`get_dm_qtci` makes 169/289/441 distinct diagonalizations at nk=8/16/32, which
is 13^2/17^2/21^2, the full GK tensor grid. A direct tensor sum over that grid
(`full_dm_gk`) reproduces it to 1e-8 and is 25-177x faster. Against an nk=80
reference on a gapped honeycomb, qtci against the uniform mesh: 8.4e-5 vs
1.6e-4 at nk=8, 5.1e-5 vs 3.4e-6 at nk=16, 5.2e-6 vs 1.9e-10 at nk=32. A full
AF honeycomb SCF took 9 s with `ed` and 100-127 s with qtci, same answer.
`gkintegrate.gkorder_from_nk`'s docstring, "matches the accuracy of an nk-point
mesh", does not hold for smooth periodic integrands. **Kept by the
maintainer's decision**; routing the 2D case through the direct tensor sum is
the obvious speedup if it is ever wanted.

### 14. (coverage) The qtci tests covered no metal, no Nambu, no tolerance

**Status.** Fixed in `0ca6e89`: gapped against an nk=80 mesh to 1e-3, frozen
BdG, metal filling frozen and through the loop, finite T, completeness of
`scf.dm`, and the energy test of #11.

## 4. Broyden mixing

### 15. The warm-up spends 85-96% of the evaluations and loses to plain mixing

**Cause.** A fixed `lam=0.1` linear warm-up against an absolute `warmup_tol`.

**Status.** Fixed in `20998a5` by warming up with `lam=0.5`, after checking it
on the Lieb-flake case the module's own benchmark rests on: all 3 seeds still
converge, in 59/69/134 evaluations against 198-271 before (plain `mix=0.8`:
89-103). Loosening `warmup_tol` or skipping the warm-up broke convergence on
that flake, which is why the warm-up stays.

### 16. The numpy engine ignores mix= and prints two "ERROR" lines per iteration at verbose=0

**Status.** Fixed in `c100e2f` for the numpy engine, which forwards `mix` as
`lam`. The jax engine went on warning that `mix` had no effect on
`broyden_mixing` and dropped it, except at `mix=0.1`, which it could not tell
apart from its own default and so dropped in silence. It now forwards it too,
on both the Vinteraction and the VJinteraction route: `mix` defaults to `None`
there (meaning not given, so the warm-up keeps its own `lam=0.5`, and
`fixed_point` its 0.1), and one `warn_if_mix_unused` serves both routes. On
the antiferromagnetic honeycomb (U=3, nk=6, maxerror 1e-7) `mix=0.1` now takes
144 evaluations and `mix=0.5` or no `mix` 32, all on the same energy to 1e-7;
before, all three took 32.

## 5. 3D superfluid weight

### 17. (coverage) Only a cubic lattice was tested in 3D, where every off-diagonal is zero

**Status.** Fixed in `90d8777` with `tests/superfluid/test_three_dimensional.py`:
a rotated anisotropic cubic model with nonzero off-diagonals at T=0 and 0.2,
rotation covariance, decoupled 2D layers that must give `D_zz=0` and the 2D
in-plane weight (square, honeycomb with Rashba, triangular with Zeeman), and
multi-orbital cubic cells. Scaling the off-diagonal of the diamagnetic loop by
1.01 fails the new test and passes the old one. No source bug was found.

## 6. Left as it is

### 18. A spinless Nambu get_mean_field_hamiltonian returns None

Reproduced on both backends, but it is the documented `NotImplemented -> None`
contract rather than a defect. It does make `get_dm_qtci`'s own spinless-Nambu
error unreachable.

## 7. Closed after the sweep

### 19. maxite=None is unbounded on every numpy SCF loop

A mean field that never converges never returned: `generic_densitydensity`
(Vinteraction, hubbard, SzSz, SxSx, SySy), VJinteraction's own loop,
`Jinteraction`, the kpm loop and the spinless attractive Hubbard loop all
defaulted to `maxite=None`, while `hubbardscf` and the coulomb loop stopped at
1000 and the jax engine and the Kondo lattice at 2000. The user guide
documented the unbounded default and told the reader to set `maxite` by hand.

**Status.** Fixed on the maintainer's call of 2026-09-24, which picked 1000
over 2000 and over keeping `None`. Every numpy loop now defaults to
`maxite=1000` and returns `None` (not converged) past it, with the existing
"No convergence has been reached" print. `maxite=None` passed explicitly still
means no limit. VJinteraction tells "not given" from `None` with a sentinel,
as it already did for `T`, so its `use_jax=True` route keeps the jax engine's
2000. The guide's convergence paragraph and its `maxite` bullet say so. One
side effect: the numpy engine forwards `maxite` to `broyden_mixing_solve`
whenever it is not `None`, so that solver's budget there is now 1000 by
default instead of its own 500.
`tests/scf/test_maxite_default.py` drives the guide's non-converging chain
(J1=-2, filling 0.2, nk=8) through the SzSz loop and through VJinteraction,
and the spinless bichain of the fourth-sweep list through Vinteraction, and
each returns `None` after 1000 iterations; on the unfixed source each of them
runs past the test's cap of 1100 iterations.

**Not yet measured.** No full-suite run has exercised the finite default, so
a test that needed more than 1000 plain-mixing iterations without saying so
would now fail on a `None`, and that run is the way to find it.

### 20. fsolve stops unconverged at the soft mode of #3 on a third of the seeds

On the chain of #3 (U=3, nk=10, filling 0.5, T=1e-4, maxerror 1e-9)
`solver="fsolve"` converged from seeds 0 to 5, which is where the claim that
its dogleg is never trapped came from, and stopped from seeds 6, 7, 8 and 11
with `ier=5` ("not making good progress") at E~-0.9964 and
`converged=False`, identically on the source before this session. Where it
stops, |r|~4e-2 but |J^T r|~3e-4 and J-I has a singular value of 3e-4 to
7e-4, a near-stationary point of the merit |r|^2, which is the stall #3 met
in the Newton loops.

**Status.** Fixed on the maintainer's call of 2026-09-24. When MINPACK stops
with `ier` 4 or 5 the rest of the evaluation budget goes to `newton_solve`
from that point, whose Levenberg-Marquardt steps and kicks leave it: all 12
seeds now reach E=-1.459374, the four that stalled in 38 to 179 evaluations.
Newton's own kicks, bolted onto `fsolve` with a restart after each, did not
work: after every kick of 60 steps at mix 0.1 it walked back to the same
point. 60 steps at mix 0.5, or 300 at 0.1, did escape, but that is tuning on
one system, the weakness #3 already records, and the maintainer picked the
handover over it. `scf.iterations` is then `nfev` plus Newton's outer
iterations. `test_vjinteraction_jax_fsolve_leaves_the_soft_mode_at_fixed_filling`
fails on the unfixed source.

## Where a fourth sweep should start

- The open items above: the kick length of the hand-rolled Newton loops, tuned
  on one system and now exposed as `kick_steps` but not re-measured, and
  newton_krylov's crawl on damped steps (#3). The qtci
  resolution limit in high-symmetry metals (#11) is closed. The jax/numpy
  disagreement on `mix` for `broyden_mixing` (#16) and the unbounded
  `maxite=None` (#19) are closed.
- The jax `fixed_point` solver that did not converge on a spinless two-site
  chain (`geometry.bichain()`, V1=3, `mu=0`, nk=6, `mf="CDW"`) is not a jax
  defect, and nothing was changed for it. The numpy engine's plain mixing does
  exactly the same on the same guess: after 2000 iterations at `mix=0.1` the
  two stand at -1.18992 and -1.18982, and at `mix=0.5` both sit on the same
  cycle at -0.768880. The reason is that at fixed `mu=0` the Hartree shift holds
  the chain at a filling of about 0.32, not at half filling, so it is a metal,
  and at T~0 a level at `mu` flips its occupation from one iteration to the
  next; both engines converge at nk=7 and 8, stall again at nk=9, and converge
  at nk=6 once T=1e-3. At that T there is one more thing to know: `newton` and
  `fsolve` converge to E=-0.85422 with a real bond and filling 0.291, while
  `fixed_point` and the numpy engine reach E=-0.87719 with a complex bond and
  filling 0.333. Both are genuine fixed points (residuals 1e-7 and 1e-5), the
  bond phase of the second shifting which k-points the nk=6 mesh occupies, so
  a Newton solver can land on the higher one, which is a caveat on the solver
  rather than a bug.
- Nothing here ran on a GPU.
