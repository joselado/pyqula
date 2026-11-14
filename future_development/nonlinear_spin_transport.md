# Nonlinear spin transport in X-wave magnets: what is built, and what is not

## What is built

`conductivity.nonlinear_drude_conductivity` and friends
(`conductivitytk/nonlineardrude.py`), plus the X-wave lattice models in
`specialhamiltoniantk/xwave.py`. This is the **electric** channel of Ezawa,
*Phys. Rev. B* **111**, 125420 (2025), [arXiv:2411.16036][e2411]: the l-th
order nonlinear Drude spin/charge conductivity, whose lowest nonvanishing
order reads off the X-wave index (d:1, f:2, g:3, i:5). The p-wave magnet has
no spin current at any order, since its two spin bands are one band shifted
rigidly in k (Ezawa, Sec. V), and the l = 0 entry is zero for every
Hamiltonian, its integrand being a total derivative. Where no point-group
symmetry forces them, both zeros are reached only exponentially in nk (for
the p-wave the shift is not a translation of the mesh): its even orders are
3.2e-4 at nk = 48 and 1.1e-11 at nk = 384, at J = 0.3 and T = 0.02.
That mesh error was once read as a p-wave threshold at l = 0 (in the
docstrings, the guide, the example, the notebook and a test asserting it);
`test_pwave_magnet_has_no_spin_current_at_any_order` now pins the zero. See
the user guide section "Nonlinear spin current as a measurement of
altermagnetic order".

Validated against the paper's own analytic results: the selection-rule
table (whose entry for the p-wave magnet is "None"), the component relations `sigma^{yyyyy;x} = sigma^{xxxxx;y} =
-sigma^{xxxyy;y}`, and the absolute scale `360 V^F J`, recovered to 0.2% by
extrapolating the lattice model into its continuum limit.

## Not built, deliberately: the thermal channel

The companion paper, Ezawa, [arXiv:2602.19034][e2602], derives the
temperature-gradient analogues — nonlinear spin-Seebeck in f-wave,
third-order spin-Nernst in g-wave, and a **linear transverse spin-Nernst in
i-wave**. Its formula is

```
j^(x^l ; b) = e (-tau/hbar d_x T)^l Int d^Dk (d eps/d k_x)^l (d eps/d k_b)
                                             d^l f^(0)/dT^l
```

which is a *different* structure from the electric case — products of l
velocity factors and one more, weighted by a temperature derivative of the
Fermi function, rather than a single high-order derivative of the band
energy. It would reuse this module's band-derivative machinery but is not a
Mott-relation wrapper around it.

It is unbuilt because **the i-wave claim in that paper does not reproduce**,
and building the channel would mean shipping a formula whose headline
consequence we could not confirm.

The argument against it, and the measurements:

1. A linear thermal response is a symmetric rank-2 tensor
   `M_ab = Int v_a v_b W(eps)` for any weight `W` that is a function of the
   band energy alone. In 2D such a tensor has only `m = 0, +-2` angular
   components, so under a C6-invariant dispersion it is isotropic and its
   transverse part vanishes identically. Fermi-surface warping does not
   rescue it — that was the first thing tried, and it is wrong, because the
   warping cannot supply an `m = 6` component to a rank-2 object.
2. The i-wave form factor **is** C6 invariant. Measured on Ezawa's own
   six-sine product: `max|F(Rk) - F(k)| = 1.1e-15` under a 60 degree
   rotation. Each 60 degree rotation flips the sign of exactly one sine in
   the nearest-neighbour triple and one in the next-nearest triple, and the
   two flips cancel.
3. Integrating `M_xy` over the correct Brillouin zone — a periodic cell
   spanned by the reciprocal lattice vectors — gives `1e-18` for each spin
   channel and hence for the spin combination. Exact zero, not small.

**The trap that produced the false positive**, recorded so it is not walked
into again: integrating over a square `[-pi,pi]^2` instead of the true
Brillouin zone. That domain is not C6 symmetric, breaks the symmetry that
forces the cancellation, and yields a plausible-looking nonzero `M_xy/M_xx`
of `1.5e-3` that flips sign with spin — exactly what a real spin-Nernst
signal would look like. Any future work on a triangular or honeycomb lattice
must integrate over a genuine periodic cell.

This is not a claim that the paper is wrong. Its result is a
high-temperature continuum expansion and there may be a term in its setup
that the argument above misses. But the discrepancy has to be resolved
before the thermal channel is worth implementing.

## jax autodiff for the band derivatives: evaluated and rejected

The obvious alternative to hand-deriving the Rayleigh-Schroedinger recursion
is to let jax do it -- nest `jax.jacfwd` six times over
`jnp.linalg.eigvalsh`. jax is already a hard dependency, so this was
measured rather than argued about. Two reasons it is not used:

1. **It returns NaN at exact degeneracies.** jax's eigendecomposition
   derivative divides by eigenvalue differences. Verified directly: nesting
   `jacfwd` twice over `eigvalsh` of a matrix with two identical eigenvalues
   gives `[nan nan nan nan]`. That is not a corner case here -- a C3 cell has
   two-dimensional irreps and hence exactly degenerate bands at Gamma, K and
   K', which is the whole reason the block (Kato) treatment exists. Autodiff
   fails precisely where the method is needed.
2. **It is ~24x slower where it does work.** On an 8-band gapped honeycomb
   over a 12x12 mesh: the RS recursion produces the complete Taylor
   expansion to order 6 -- all bands, all 28 coefficients -- in 0.11 s,
   while a jitted, vmapped `jacfwd^6` takes 0.33 s warm for a *single* band
   (~2.6 s for all eight) plus ~2 s of compilation each. Nesting forward
   mode six times over a 2-component momentum evaluates a rank-6 tensor with
   64 entries of which only 7 are independent, and differentiates the
   eigendecomposition at every level.

`jax.experimental.jet` would avoid the 2^N blow-up by propagating a
truncated Taylor series directly, but it has no rule for `eigh`, so it does
not apply.

**What jax was kept for**: as an independent check. Where the spectrum is
non-degenerate, autodiff and the recursion agree to 5e-15 at sixth order on
a real two-orbital Hamiltonian, which is a far stronger validation of the
hand-derived recursion than any internal consistency check -- the two routes
share nothing but the Bloch series. See
`test_band_derivatives_match_jax_autodiff`.

**Still open: jax as an execution backend, not as a differentiator.** The
recursion is a sequence of batched matrix products over `(nk, nb, nb)`
arrays, which is the shape the GPU roadmap identifies as favourable (see
`documentation/gpu_porting_plan.md` and the RPA chi measurements). Porting
the inner loop to `jnp` with `jit`, keeping the hand-derived recursion and
using jax only to execute it, is untried and unmeasured -- there is no GPU
on the development machine. That is a different proposition from autodiff
and is not ruled out by anything above.

## A trap in the spin channel: degenerate is not the same as compensated

A compensated magnet is not necessarily spin split. A Neel state on a
bipartite lattice is PT symmetric, its bands are exactly spin degenerate, and
it is an antiferromagnet rather than an altermagnet -- so it has no spin
response at all.

That failure does **not** show up in the answer. On a multiorbital cell the
two spin channels are diagonalized independently, so near a degeneracy their
eigenvector gauges differ and the high-order derivatives drift apart far more
than the band energies do. Measured on a 44-site honeycomb antidot Neel
state: bands agreeing to `1.2e-14` gave `sigma_up` and `sigma_dn` a factor of
8 apart and a fifth-order "spin response" of `1.3e-3`. Inspecting
`sigma_spin/max(|sigma_up|,|sigma_dn|)` does not catch it either -- that
ratio came out at 0.3-0.6, nowhere near a cancellation floor.

The diagnostic that does work is the band splitting itself, which is
unambiguous and already computed:
`nonlineardrude._spin_splitting_guard` warns when
`max|eps_up - eps_dn|` falls below `1e-10` of the bandwidth. This cost real
time to find, and it was found only because a peer session challenged a
claim about the fixtures -- the fixtures had been reported as carrying
`l = 0, 3, 4` order, which was `argmax` on an all-noise array.

**Reality is itself a selection rule.** With real hoppings and no spin-orbit
coupling, `H_s(k)* = H_s(-k)`, hence `Delta(phi+180) = Delta(phi)` and every
odd harmonic is forbidden. C3 kills every `l` not divisible by 3. Together:
a real, SOC-free, C3-symmetric cell is necessarily **i-wave**, and no vacancy
pattern can make it f-wave. Confirmed on the models here -- p-wave and f-wave
require *imaginary* hoppings (measured `max|Im H| = 0.5`) while d, g and i
wave are exactly real. Consequence for fixtures: an f- or g-wave test case
cannot be built from a real C3 model.

## Also not built in this module

- **Ezawa's second-order charge response**, [arXiv:2409.09241][e2409], is
  built since 7 November 2026, not here but in `conductivitytk/nonlineardc.py`
  (`h.get_nonlinear_dc_conductivity()`): the nonlinear Drude (tau^2), Berry
  curvature dipole (tau^1) and quantum-metric (tau^0) terms of any multiband
  Hamiltonian, spin-orbit coupling included, from the band velocities, the
  inverse masses and the quantum geometry of the Fermi surface. The fixture
  is his d-wave altermagnet with Rashba coupling built by hand with a
  d_{x^2-y^2} form factor (hopping -J n.sigma along x and +J n.sigma along
  y), since `specialhamiltonian.dwave_altermagnet` is d_xy and moves the
  zeros of his law. `tests/conductivity/test_nonlinear_dc.py` checks the
  cos(Phi) Neel-angle law of the Drude and quantum-metric terms at J = 0.01,
  the exact zeros for n along x and z, the linearity and the sign flip in J,
  and the cos(3 Phi) harmonic, which is third order in J (11 % of the
  amplitude of the quantum-metric term at J = 0.05). His absolute
  prefactors are not used: his Eq. (HallAna) carries a tau in a tau^0
  quantity, and his Berry curvature dipole is a factor 2 larger than Sodemann
  and Fu's by a double-counted symmetrization. The Drude term agrees with
  `nonlinear_drude_conductivity` at l = 2 in the charge and spin channels.
  The details are in `nonlinear_response.md`.
- **3D.** This module is 3D since `5b55594`: the Taylor recursion of the
  band energies runs over three indices (p,q,r), the Bloch series uses the
  third lattice vector, and a 3D Hamiltonian is accepted. The second-order
  terms come from `nonlineardc` as well, and its third-order Drude term
  reuses this recursion, so it is 3D too (`nonlinear_response.md`). The
  sweep of `nonlinear_drude_components` and `nonlinear_drude_orders` ran
  over x and y only until the seventh bug audit, see "Repairs of the
  seventh bug audit" below. What is still missing is a 3D model of every
  wave from Ezawa in `specialhamiltoniantk/xwave.py`.
- **Restricting the perturbation expansion to bands near the Fermi level.**
  The cost is `~nb^2.8 nk^2` and is dominated by expanding every band, but only
  bands within a few `T` of the chemical potential carry weight. Rewriting
  `sum_n f_n D^a eps_n` as `sum_n (f_n - 1) D^a eps_n` (legitimate, since
  `sum_n D^a eps_n = D^a Tr H` integrates to zero over the zone) lets one
  choose whichever of `f` or `f - 1` has fewer active bands. For a
  half-filled system that is only a factor ~2, which is why it was not done;
  it would be worth more for a heavily doped or nearly empty band.

## Performance, measured

C3 antidot superlattices (honeycomb supercell with a vacancy cluster,
imposed staggered mz), full order sweep `l = 0..5`, at `nk = 12`. Measured
single-threaded (`MKL_NUM_THREADS=1 OMP_NUM_THREADS=1`,
`parallel.set_enabled(False)`) with CPU time within 1% of wall time, so the
numbers are not contaminated by the shared machine:

| bands/spin | wall | cpu |
|---|---|---|
| 16 | 4.2 s | 4.1 s |
| 32 | 28.2 s | 28.0 s |
| 44 | 71.6 s | 70.9 s |
| 66 | 221.1 s | 218.8 s |

The four points fit `nb^2.8` (not the naive `nb^3`), and the mesh enters as
`nk^2`. Extrapolated to a realistic superlattice at `nk = 36`: ~25 min at 60
bands, ~1.2 h at 88, ~1.5 h at 96.

Pin the threads before believing any timing here. An earlier pass let MKL
take 12 threads per process on a 14-core machine at load 26; it happened to
cost only ~6% at these matrix sizes (28.2 s pinned against 30.0 s), but that
was luck rather than a reason to skip the pinning: oversubscription can
dominate every other performance effect, and a timing taken while other jobs
are running says nothing.

The whole sweep costs what a single component costs, because the
perturbation expansion is computed once to the top order and cached, and
each of the 42 tensor components is then a weighted sum over the mesh. This
was not free -- the cache was initially keyed by exact order, so a sweep
recomputed the expansion for every order and cost ~6x more. If that caching
is ever refactored, keep the property that an expansion to order N serves
every request below N.

**Give it the primitive magnetic cell.** A redundant cell is not merely
wasteful here, it is wasteful to nearly the third power: at `nb^2.8` a
3-fold redundant 180-band cell costs ~22x its 60-band primitive, turning
half an hour into most of a day.

## A trap in interpreting the result

With the chemical potential in a gap the response is **exactly zero at every
order**, the threshold order included: `f` is 1 on every valence band and 0
on every conduction band, so the integrand is a pure k-derivative of
`Tr(P H)` with `P` the valence projector, which is smooth and periodic, and
the zone integral of a derivative of a smooth periodic function vanishes.

So a band insulator reproduces the "nothing below fifth order" pattern
trivially. The absence of the low orders identifies i-wave order **only**
when the fifth order is simultaneously shown to be present, which requires a
metallic state. Anyone using this as a fingerprint for a gapped material has
to dope it first — and then show that the magnetic order survives the
doping, which for a Lieb-mechanism half-filled bipartite magnet is not
automatic.

## Repairs of the seventh bug audit

The 3D lift of `5b55594` let a 3D Hamiltonian in, but the sweep of
`nonlinear_drude_components` and `nonlinear_drude_orders` still ran the
field powers over x and y and the current over x and y. So a spin splitting
odd in k_z gave exactly zero at every order. On the cubic lattice with
hopping -1, onsite 0.5 and the splitting s J sin kx sin kz (J = 0.3,
nk = 40, T = 0.1), the old sweep gave at most 2.8e-17 at every order up to 3,
while the single component sigma^{x;z} is 4.4e-3: a d-wave magnet read as
"no response". In 3D the sweep now runs over x, y and z, both for the
field powers and for the current; in 1D and 2D the keys are the ones it
always had. The keys of a 3D call are `"x^l1 y^l2 z^l3;b"`, 9 at l = 1 and
30 at l = 3. The order expansion was already done to the full 3D order, so
the sweep now reads the z coefficients it was computing anyway.
`tests/conductivity/test_nonlineardrude_3d_sweep.py` pins it with a
symmetry: on the cubic lattice the xz magnet is the xy one with y and z
exchanged, so the two sweeps must agree, and with the old sweep they do not
(2.8e-17 against 4.4e-3 at l = 1). It also checks that the keys up to
third order are the expected ones and that each equals
the single-component call, and that the largest component with the current
along z equals the largest one with the current along x, which the x and z
symmetry of that magnet demands. The review broke the sweep twice in a
scratch copy, with a current that stops at y and with a field label that
has too many powers of z, and the file fails for both; the symmetry test of the
orders alone passes the second one, since it only compares maxima.
A third break, a sweep over z in every dimension, passed the two older test
files of this module, so nothing pinned that the 1D and 2D keys stay the
ones of x and y. The file now has a test of the keys of a chain and of a
square lattice up to third order, and it fails for that break.

[e2411]: https://arxiv.org/abs/2411.16036
[e2602]: https://arxiv.org/abs/2602.19034
[e2409]: https://arxiv.org/abs/2409.09241
