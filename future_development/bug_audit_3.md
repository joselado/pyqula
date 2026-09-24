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

**16 fixed, one decided, one left as it is.** Every fix has a
regression test that asserts the finding's oracle, and each of those tests
fails on the unfixed source. The full suite on the merged fixes was stopped
at 78% with no failure, and the qtci energy fix (#11) was checked with the qtci
tests only; a complete run on the merged result is still owed.

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
  they used to stop with `converged=False` (#3).
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
  is integrated on, sums the total energy there too, and returns the complete
  density matrix in `scf.dm` (#11, #12).
- `solver="broyden_mixing"` honours `mix=`, is silent at `verbose=0`, and warms
  up with `lam=0.5`, taking 2.5-4.5x fewer density-matrix evaluations (#15,
  #16).

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
nk=10). `scf.iterations` counts outer iterations only. One newton_krylov seed
took 779 outer iterations crawling on damped steps; a proper trust region is
the principled follow-up, since `fsolve`'s dogleg is never trapped here.

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
period-7 limit cycle the scalar path has always had there. `maxite=None` is
still unbounded, on both paths.

## 3. qtci SCF backend

### 11. filling is not honoured in a metal

**Cause.** The Fermi level came from the uniform nk mesh, the density matrix
from the Gauss-Kronrod nodes, and in a metal the two hold different charges at
one Fermi level.

**Status.** Fixed in `35a88eb` (Fermi level located on the GK nodes, by the new
`get_fermi4filling_qtci`) and `a8596b6` (the total energy summed on the same
nodes). Low-symmetry metals now hold the filling (Rashba plus exchange, nk=8:
0.589 before, 0.597 now, for 0.6), and at finite T it is exact to 1e-6.

**Resolution limit, not fixed.** At T~0 the charge is right only to the weight
of one level on the node grid, and the nodes come in symmetry-related groups:
on the bare square lattice at filling 0.3 and nk=8 the level at the Fermi
energy is 8 states holding 0.087 electrons, and the nearest cut holds 0.6405.
A BdG run on that lattice overshoots the gap accordingly (0.764 against 0.70
converged). Finite T, or a fractional occupation of the degenerate level, are
the two ways out.

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

**Status.** Fixed in `c100e2f`. The jax engine still warns that `mix` has no
effect on `broyden_mixing`, while the numpy engine now forwards it as `lam`; the
two engines disagree there.

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

## Where a fourth sweep should start

- The open items above: a trust region for the hand-rolled Newton loops (#3),
  the qtci resolution limit in high-symmetry metals (#11), the jax/numpy
  disagreement on `mix` for `broyden_mixing` (#16), `maxite=None` being
  unbounded on every SCF path.
- The jax `fixed_point` solver did not converge on a spinless two-site chain
  with V1=3 at `mu=0`, nk=6, within 2000 iterations at `mix` 0.1 or 0.5. Hit
  while writing a test, not investigated.
- Nothing here ran on a GPU.
