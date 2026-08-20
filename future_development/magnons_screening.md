# Screening and magnons -- a measured "no"

Whether the static RPA screened interaction added in `510f056`
(`bsetk/screening.py`, `h.get_screened_interaction()`) should also be used
in the magnon RPA kernel, the way it is used in the BSE direct term.

**Conclusion: no, not on its own.** Screening the magnon kernel while the
mean field keeps its own interaction destroys the Goldstone mode, at first
order in the mismatch and with the Stoner splitting as prefactor. Screening
applied *consistently* (same interaction in the mean field and in the
kernel) is fine and changes only the spin stiffness. Nothing was
implemented; this file records the measurements so the question does not
have to be reopened from scratch.

## What exists today

`chitk/spinchi.py`'s `magnon_bands` / `h.get_magnon_bands()` builds the RPA
vertex from `h.V` via `_full_spin_U`, i.e. from the interaction the mean
field was converged with. That is already the Goldstone-correct choice, and
it is not an accident -- `tests/chi/test_magnon_goldstone_doped_chain.py`
says so explicitly: the RPA "renormalize[s] the bare bubble with the SAME U
that produced the mean-field splitting, which is what the Goldstone/Ward
identity requires".

## The Ward identity, and how sharp it is

For a Hubbard mean field the identity is exact:

```
U_MF * chi0(q=0, w=0) = 1
```

so a kernel built from a *different* interaction leaves a residual
`1 - U_kern/U_MF` and the acoustic magnon acquires a real gap. Measured on
a saturated ferromagnetic chain (`geometry.chain()`, `filling=0.2`,
`mf="ferro"`, `nk=300`, `|m_z| = 0.4` at every U below), scaling only the
kernel interaction:

| U_kern/U_MF | q=0 kernel residual | w(q=0), U=10 | verdict |
|---|---|---|---|
| 1.00 | delta/4 -- pure broadening | **0.00000** | gapless |
| 0.99 | 0.01122 | 0.040 | GAPPED |
| 0.95 | 0.05029 | 0.200 | GAPPED |
| 0.90 | 0.10017 | 0.400 | GAPPED |
| 0.75 | 0.25008 | 1.000 | GAPPED |
| 0.50 | 0.50004 | -- | GAPPED |

"gapless" vs "GAPPED" is the delta-scaling test
`test_magnon_goldstone_doped_chain.py` uses: at `scale = 1` the residual is
exactly `delta/4` at both `delta = 0.02` and `delta = 0.005`, i.e. a pure
Lorentzian-broadening artifact; at every other scale it is
delta-independent, i.e. a real gap.

The gap obeys, exactly, over six measurements at three couplings:

```
w(q=0) = U * |m_z| * (1 - U_kern/U_MF)
```

| U | U\*\|m_z\| | scale=0.90 | scale=0.75 |
|---|---|---|---|
| 10 | 4.000 | 0.4000 | 1.0000 |
| 8 | 3.200 | 0.3200 | 0.8000 |
| 6 | 2.400 | 0.2400 | 0.6000 |

(`U*|m_z|` is half the up-down Stoner splitting `Delta = U*(n_up - n_dn)`
in this convention.)

**This is why it matters.** The violation is *first order* in the mismatch
with the Stoner splitting as prefactor -- there is no small parameter
protecting it. A 1% error in the kernel interaction puts the acoustic
magnon at 1% of the Stoner splitting, which for a real magnet
(`Delta ~ 2 eV`) is ~20 meV, i.e. the whole magnon bandwidth. Realistic
screening (`eps ~ 2-10`, so `U_scr/U ~ 0.1-0.5`) would put it at 50-90% of
the splitting: not a magnon at all, but the Stoner continuum edge.

## Dispersion: the power was never the thing at risk

Linear versus quadratic is fixed by Goldstone counting, not by the
interaction. A ferromagnet conserves `S^z`, so its two broken generators
pair into a single type-B mode with `w ~ q^2`; an antiferromagnet's
staggered order parameter does not, giving type-A and `w ~ q`. Both
measured here with a consistent kernel:

| model | fitted power | expected |
|---|---|---|
| FM chain (U=10, filling=0.2, `mf="ferro"`) | **q^2.006** | 2 |
| AFM chain (U=4, half filling, `mf="antiferro"`, 2-site cell) | **q^0.994** | 1 |

The AFM case has `w/q = 3.05` flat from `q=0.02` to `q=0.10` -- a clean
spin-wave velocity.

Screening applied *consistently* (same U in mean field and kernel) keeps
both the gaplessness and the power, and moves only the stiffness:

| U (both) | w(q=0) | fitted power | stiffness D |
|---|---|---|---|
| 10 | 0.000000 | 2.006 | 14.81 |
| 8 | 0.000000 | 2.015 | 11.08 |
| 6 | 0.000000 | 2.063 | 4.87 |

D changes by 3x while the power stays 2. So the answer to "would screening
give the right dispersion" is: the *power* is right either way, because it
is symmetry and was never at risk. What screening breaks is
**gaplessness**, and once the mode is gapped the linear-versus-quadratic
question is moot -- the dispersion is `w = Delta_spurious + D q^2`.

Careful with the pole finder below `U ~ 5` on this model: the stiffness
collapses towards the Stoner threshold and the acoustic branch drops below
the frequency-grid spacing, while `_poles_from_chi_matrix`'s zero-crossing
interpolation still reports spurious ~1e-4 poles. Taking `min(w)` there
gives a meaningless power fit (0.767 in one exploratory run). Filter the
poles by `w > few*grid spacing`, or widen the grid, before believing a
number in that regime.

## What the ab initio literature does, and what it costs them

Worth knowing, because it looks at first glance like a counterexample.
First-principles GW-BSE magnon codes *do* put the screened `W` in the
magnon ladder, with the bare interaction in the exchange term -- exactly
the same split this repo's BSE uses for excitons. They then pay for it with
precisely this Goldstone violation. In
[arXiv:2502.06598](https://arxiv.org/html/2502.06598) (magnons in chromium
trihalides from an ab initio BSE) the acoustic magnon of CrI3 comes out at
**1.25 eV** at q=0, and the published dispersion is obtained by shifting
the whole thing down by that amount by hand. Muller, Friedrich and Blugel,
[PRB 94, 064433](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.064433),
is the dedicated study of the violation; their remedy is a renormalized
Green function, not a screened kernel.

The reason the exciton case is different is that there is no Goldstone
theorem to satisfy there -- nothing forces the lowest exciton to any
particular energy, so an inconsistency between the mean-field interaction
and the kernel shows up as a quantitative error rather than as a
qualitatively wrong spectrum.

## Two blockers specific to this codebase

1. **`get_magnon_bands` would reject a screened W outright.**
   `_full_spin_U` calls `_require_onsite_only_V`, which raises for any
   `h.V` with support beyond `(0,0,0)`. A `ScreenedInteraction` is
   long-ranged in real space, so the screening machinery cannot be fed to
   the magnon path at all as things stand.
2. **The charge channel is the wrong dielectric function here anyway.**
   `bsetk/screening.py` screens with the density-density `chi0`. The
   transverse spin channel is not screened by charge bubbles at RPA level;
   the ab initio ladder-with-W is a different resummation (a T-matrix), and
   its Goldstone violation is exactly the problem above.

## If someone wants to do it anyway

The only defensible route is the consistent one:

1. build `W = h.get_screened_interaction(...)`
2. `W.get_dict()` back to a real-space interaction
3. re-converge the mean field with `get_mean_field_hamiltonian(V=...)`
4. use the SAME `W` in the magnon kernel

Step 4 needs the non-onsite spin-channel RPA verified first -- which is
exactly the caveat `_require_onsite_only_V`'s docstring records as not yet
done. That verification, not the screening, is the piece of work here.
