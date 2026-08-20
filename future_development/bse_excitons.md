# Excitons / Bethe-Salpeter -- future development

Status of the BSE implementation added in `6f81888`, what is deliberately
not built yet (oscillator strengths, an iterative solver), and a *measured*
feasibility study of the tensor-network route to large k-meshes. The numbers quoted below were all measured on this
codebase; they are recorded so nobody has to re-derive them.

## What exists today

`src/pyqula/bsetk/` (+ public `bse.py`), reachable as `h.get_bse()`,
`h.get_exciton_energies()`, `h.get_exciton_binding_energies()`,
`h.get_exciton_states()`.

- Full non-Tamm-Dancoff BSE at arbitrary center-of-mass momentum `Q`,
  following the localized-orbital formalism of the Xatu code
  ([arXiv:2307.01572](https://arxiv.org/abs/2307.01572)).
- Kernel read from `h.V` by default (time-dependent Hartree-Fock on top of
  Hartree-Fock, no double counting), or supplied via `V=`. The direct term
  can instead take the static RPA screened interaction, `screening="rpa"`
  or `"crpa"` (Phase 3 below);
  `bsetk.interaction.density_interaction` builds a `U`/`V1`/`V2`/`V3`/`Vr`
  screened interaction.
- `kernel="full"|"direct"|"exchange"|"none"`, `tda=True/False`,
  `nv`/`nc` band windows, `max_memory` guard.
- Tests in `tests/bse/`, example in `examples/2d/excitons_bse/`.

### Measured cost of the dense solver

Honeycomb, spinful, 4 orbitals per cell, all bands:

| nk | N_pair | matrix | time | peak RSS |
|---|---|---|---|---|
| 8 | 256 | 512² | 1.2 s | 0.42 GB |
| 12 | 576 | 1152² | 2.7 s | 0.58 GB |
| 16 | 1024 | 2048² | 7.8 s | 1.07 GB |
| 20 | 1600 | 3200² | 21.6 s | 1.91 GB |

The default `max_memory=2.0` refuses nk=24. This is the wall everything
below is about.

### Why the wall matters physically

The same system's lowest exciton energy over that mesh sequence:
1.85244 / 1.83308 / 1.84119 / 1.84012 (gap 2.1648). Converging, but only
because that exciton is fairly tightly bound. A shallow Wannier-Mott
exciton whose Bohr radius spans many unit cells has an envelope `A(k)`
squeezed into a small neighbourhood of the band edge, and needs nk in the
hundreds -- unreachable densely.

## Phase 2 -- observables (partially done)

Straightforward, no new numerics needed.

1. **Oscillator strengths and optical conductivity.** Velocity matrix
   elements from `current.derivative(h,k,order=...)` (already used by
   `operators.get_velocity`); the exciton's is
   `V^a_X = sum_{cvk} A_X(k) v^a_{vc}(k)`, and bright/dark classification
   follows. Gives the excitonic absorption spectrum, which is the point of
   computing excitons at all.
2. **Exciton bands `E_X(Q)`** scanned along a q-path. DONE:
   `bsetk/bands.py`, `h.get_exciton_bands`, flat `(qs, es)` return in the
   `get_bands`/`plasmon_bands` convention, `parallel.pcall` over the path,
   `examples/2d/exciton_bands/`, `tests/bse/test_bse_bands.py`. It is a
   thin loop over the existing fixed-Q solver, so the cost is nq dense
   diagonalizations and the table above still sets the reachable mesh.

   Building it turned up one trap that is invisible at Q=0: an `nv`/`nc`
   window that cuts a **degenerate multiplet** in half keeps an arbitrary
   state out of the degenerate subspace, and the exciton energies inherit
   that arbitrariness. Measured on the spinful ionic chain, `nv=nc=1` (so
   half of each spin-degenerate pair) breaks `E_X(Q) = E_X(-Q)` by 0.15 in
   the full kernel and 0.17 in the direct one, where `nv=nc=2` gives 3e-15;
   the `kernel="none"` spectrum is symmetric either way, so only the
   interacting terms expose it. `bsetk/pairbasis.select_bands` now warns
   when the window splits a multiplet. Any later work that truncates the
   pair basis (the iterative solver of Phase 4 especially) has to respect
   the same rule.
3. **Envelope visualization**: `|A_vc(k)|²` over the BZ, and its real-space
   transform. `BSE.pairs.labels` already carries the `(ik,iv,ic)` of every
   pair index, so this is presentation work.

## Phase 3 -- the screened interaction W from the mean field (DONE)

DONE: `src/pyqula/bsetk/screening.py` (+ public `screening.py`),
`h.get_screened_interaction()`, `h.get_polarizability()`,
`h.get_bse(screening="rpa"|"crpa", channel="charge"|"orbital")`,
`tests/bse/test_bse_screening.py` (27 tests),
`examples/2d/screened_bse/`. The plan below is what was built;
the "What the implementation measured" section at the end records what
came out of it that the plan did not anticipate.

Before it, the BSE kernel was built from a *bare* interaction: `h.V` (the
one the SCF converged with) or whatever `V=` supplies. Both the direct and
the exchange term used the same matrix, which makes the whole thing exactly
time-dependent Hartree-Fock. That is internally consistent, but it is not
what binds excitons in a real material: the electron-hole attraction is
the *screened* interaction `W = eps^-1 v`, with `eps` built from the
polarizability of the very bands the mean field just produced. Xatu
([arXiv:2307.01572](https://arxiv.org/abs/2307.01572)) does not compute
this -- it substitutes a phenomenological Rytova-Keldysh form. Since the
mean-field step already hands us `{e_n(k), C^n(k)}` on a k-mesh, computing
the RPA `W` instead of postulating it is a small amount of extra work, and
it is the natural next physics step after the mean field.

The plan below is a static-RPA screened interaction in the same
point-like-orbital convention the rest of `bsetk/` uses, wired into the
direct kernel only.

### The formalism

Everything stays in the orbital (really spin-orbital) basis that
`bsetk/interaction.py` already uses for `W_ab(d)`, and in the cell gauge
(`interaction_at_q`'s docstring explains why no intracell-position phases
appear). With point-like orbitals the density form factor of a transition
is diagonal in the orbital index,

```
rho^{nm}_a(k,q) = conj(C^{n,k}_a) C^{m,k+q}_a
```

which is exactly the object `kernel.exchange_block` already consumes as
`Fr = conj(pb.el)*pb.ho`. The static Adler-Wiser polarizability is then a
sum of outer products of those length-`norb` vectors:

```
chi0_ab(q) = (1/N) sum_k sum_{n,m} (f_nk - f_{m,k+q})
                 * rho^{nm}_a(k,q) conj(rho^{nm}_b(k,q))
                 / (e_nk - e_{m,k+q})
```

**Sum both orderings explicitly** -- (n occupied at k, m empty at k+q)
*and* (n empty at k, m occupied at k+q). The familiar "factor of 2 times
one ordering" shortcut is only valid with time-reversal symmetry, and this
codebase's SCF routinely converges to magnetic mean fields where it is
false. This is not pedantry, it is what makes the next paragraph work.

Then, per q,

```
eps_ab(q) = delta_ab - sum_c v_ac(q) chi0_cb(q)
W(q)      = eps^-1(q) v(q)
```

`W` is Hermitian (it is the series `v + v chi0 v + ...`, each term
Hermitian), so symmetrize away the roundoff after the inversion.

### Why this resolves the W(-Q) trap rather than reopening it

`kernel.build_blocks` computes the antiresonant exchange block from
`WmQ = np.conj(WQ)`, which is only correct because a real-space `W(d)` with
real entries gives `W(-q) = conj(W(q))`. A numerically tabulated `W(q)` has
no such guarantee *a priori*. With both orderings summed, `chi0(q)` is
Hermitian and obeys `chi0_ab(q) = chi0_ba(-q)`; carrying that through the
inversion with a Hermitian `v` gives `W(-q) = conj(W(q))` again. So the
existing kernel code stays valid -- but only under that condition, which
makes it an **invariant to assert in a test, not a derivation to trust**.
A magnetic (Zeeman-split or SCF-magnetized) chain is the case that
separates it from the TRS-symmetric one; check `W(-q)` against
`conj(W(q))` there directly.

### Direct screened, exchange bare

The direct (ladder) term takes `W`, the exchange (Hartree / local-field)
term takes the **bare** `v`. This is the standard GW-BSE split, and it is
not optional: screening the exchange term too would resum the same bubbles
twice, and among other things would destroy the `kernel="exchange"`
cross-check against `chitk/rpa.py` in `tests/bse/test_bse_rpa.py`.

Concretely `build_blocks(pb, W, kernel=...)` becomes
`build_blocks(pb, W, Wx=None, kernel=...)` with `Wx=None` meaning "same as
W". Every existing call keeps its current behavior bit for bit.

### Which channel the dielectric matrix lives in

This is the part the plan did not foresee at all, and it turned out to
matter more than anything else in the section. It is written up in full as
point 5 of "What the implementation measured" below: the short version is
that the dielectric matrix must be built in the CHARGE channel, on site
indices, or the result is not spin-rotation invariant. `channel="charge"`
is the default; `channel="orbital"` keeps the plain spin-orbital RPA the
plan described.

### RPA vs cRPA -- and where the real double counting is

Bubbles inside `W` and ladders in the BSE are different diagram classes,
so full-RPA `W` with a BSE ladder is the standard construction, **not**
double counting. Offer both anyway:

- `screening="rpa"` (default): all transitions enter `chi0`.
- `screening="crpa"`: transitions inside the `nv`/`nc` BSE window are
  *excluded* from `chi0`, following Miyake-Aryasetiawan
  ([arXiv:0710.4013](https://arxiv.org/abs/0710.4013)), the reference for
  a screened interaction evaluated in a localized basis. This is the right
  choice when the BSE window is being treated as a downfolded model solved
  exactly afterwards.

**The double counting that does matter is upstream of both.** A Hubbard
`U` fitted to reproduce a material is *already* an effective screened
interaction; screening it again with this machinery is wrong, and the
result will be a badly underbound exciton. This feature is for a *bare*
interaction -- a long-range Coulomb tail entered through
`interaction.density_interaction(Vr=...)`, or a genuinely bare model `V1`,
`V2`, ... . The default `V=h.V` path is the dangerous one, because a
Hubbard-only `h.V` is exactly the case that must not be screened. Say this
loudly in the docstring and in the user guide; it is the most important
documentation in the feature.

Note also the mean field itself is left as unscreened Hartree-Fock, whose
gap is too large. That is the usual GW-BSE inconsistency and is acceptable
here, but see the follow-up tier below.

### Where the code goes

- `src/pyqula/bsetk/screening.py` -- new.
  - `static_polarizability(h, nk=..., nv=None, nc=None, exclude_window=False)`
    returns `(qs, chi0)` with `chi0` of shape `(nq, norb, norb)`.
  - `screened_interaction(h, V=None, nk=..., screening="rpa", ...)` returns
    a small object holding `qs`, the tabulated `W(q)`, and `.at(q)` /
    `.get_dict()`.
  - `epsilon_at_q` / an eigenvalue-based instability guard (below).
- `src/pyqula/bsetk/kernel.py` -- the `Wx` argument above, and accepting a
  tabulated-`W` object wherever `interaction_at_q(W,g,q)` is called now.
- `src/pyqula/bsetk/solve.py` -- `BSE(..., screening=None)`. `None` keeps
  today's bare behavior exactly; `"rpa"`/`"crpa"`/a precomputed object
  switch the direct term to `W`.
- `src/pyqula/screening.py` (public facade) + `h.get_screened_interaction()`,
  following the thin-delegator convention.
- `documentation/user_guide.md` section + "Main functions and methods"
  entry; `README.md` FUNCTIONALITIES.

### The mesh, and why it is free

`geometry.get_kmesh` returns a Gamma-centered uniform mesh in fractional
coordinates, so `k+q` folds back onto the *same* mesh by index arithmetic
-- one diagonalization per mesh point, no second Bloch solve. And
`kernel.qdifference_map` already establishes that the direct term only
ever needs `W` at the `nk` distinct mesh differences, which are precisely
the mesh points themselves. So the screening q-grid and the q-grid the
kernel wants are the same set: **`W` can be tabulated at exactly the
points it is consumed at, with no interpolation anywhere.**

Allow a denser screening mesh `nkW = m*nk` (a Gamma-centered mesh of
`m*nk` contains every point of the `nk` one), since screening converges
faster in `nk` than the exciton envelope does. Reject a non-multiple
rather than interpolating.

**Off-mesh Q is the one hard edge.** A tabulated `W(q)` exists only at
mesh points, but `h.get_exciton_bands` scans arbitrary `Q` along a path,
and the exchange term wants `W(Q)` at that `Q`. The `screening=` knob
sidesteps this entirely (exchange uses the real-space bare `v`, which
Fourier transforms at any `q`). But a user who takes a tabulated `W` and
hands it back in as plain `V=` would hit it -- so `.at(q)` must **raise a
clear error on an off-mesh q**, naming the mesh, rather than silently
snapping to the nearest point. Alternatively `.get_dict()` inverse-Fourier
transforms the tabulated `W(q)` back to a truncated real-space dictionary,
which *is* usable at arbitrary q (and is separately useful: that dict can
be fed to `get_mean_field_hamiltonian(V=...)`, plotted, or inspected).
Document the truncation error that route carries.

### Cost

`chi0` is `nq * nk * (n_occ*n_emp) * norb^2` outer-product accumulations,
with `nq = nk`. That is the same cost class as the direct kernel build and
belongs in a numba `prange` kernel over q (mirroring `direct_block_jit`),
not in `parallel.pcall`. Reuse the mesh eigendecomposition rather than
recomputing it. Add a `max_memory`-style guard for the `(nq,norb,norb)`
tensor, in the style of `solve.check_memory`.

Do **not** route this through `chitk.chiAB`: that path is site-resolved
(not spin-orbital-resolved), scans frequencies with a finite broadening
`delta`, and would be both wrong-basis and far more expensive than a
direct static sum. It is a cross-check, not an implementation.

### Instability guard

If an eigenvalue of `eps(q)` crosses zero the RPA screening diverges --
a charge-density-wave instability of the mean field, at the wavevector
where it happens. Detect it (cheapest: `min(abs(eigvals(eps(q))))` over
the grid, or the sign of `det`) and raise with the offending `q` named and
a message pointing at the mean field, mirroring what
`chitk.rpa.rpa_kernel_poles` reports. Cheap to build in now, ugly to
discover as a silent garbage number later.

### Tests -- and the benchmark obligation

`CLAUDE.md` asks for a benchmark against an existing open implementation.
There is not one here: Xatu, the reference this BSE follows, uses
phenomenological Rytova-Keldysh screening rather than an RPA `W`, and no
directly comparable tight-binding TB-RPA-`W` code was found. **State that
explicitly** and substitute internal cross-checks, in roughly this order
of value:

1. **q=0 uniform-mode null (exact, cheap, do it first).** At `q=0` the
   form factor sums to `sum_a rho^{nm}_a = <n|m> = delta_nm`, which
   vanishes for every `n != m` transition entering `chi0`. So
   `chi0(0) @ ones = 0` identically: the uniform charge mode is unscreened
   at `q=0`. Any indexing or conjugation error breaks this.
2. **Supercell folding.** Historically the only test that caught the
   finite-Q bugs (see the traps section). An n-cell supercell's `W` at
   Gamma must fold onto the base cell's `W(q)` at every q folding onto it.
3. **Weak-coupling series.** For small `v`, `W` must match
   `v + v chi0 v` to the expected order. Tests the inversion and the
   matrix ordering independently of `chi0` itself.
4. **Hermiticity and reciprocity.** `W(q) = W(q)^dag` everywhere, and
   `W(-q) = conj(W(q))` -- the latter tested on a *magnetic* model, per
   the section above.
5. **Static limit against `chitk.chiAB`.** Independent code path.
   Site-resolved and spinless only, and it carries a finite `delta`
   broadening -- so use a loose tolerance or extrapolate `delta -> 0`
   rather than expecting agreement to 1e-12.
6. **Screening reduces binding.** `E_binding(screened) < E_binding(bare)`
   on a model with a long-range tail. A physics sanity check, not a
   precision one.
7. **`screening=None` is bit-for-bit today's answer.** Regression guard on
   the whole refactor.

### What the implementation measured

Recorded because none of it was obvious from the plan, and two of them
would cost real debugging time to rediscover.

**1. Every invariant the plan proposed holds to machine precision.** On
both the gapped ionic chain and the gapped honeycomb, nk=8:

| check | measured |
|---|---|
| `chi0` Hermitian | 0.0 (exact) |
| `chi0` negative semidefinite (largest eigenvalue) | 0.0 / 7e-18 |
| q=0 uniform-mode null, `chi0(0)@1` | 1.4e-17 |
| `chi0(-q) = conj(chi0(q))`, magnetic model | 5.6e-17 |
| `W(-q) = conj(W(q))`, magnetic model | 1.1e-16 |
| supercell folding of the screened BSE, nsuper=2/4 | 5.6e-15 / 6.7e-15 |
| `get_dict()` round trip on the mesh | 3.4e-16 |
| real-space `W` imaginary part | 6.9e-17 |

The reciprocity identity in particular is *confirmed*, not assumed: with
both (n,m) orderings summed, `W(-q) = conj(W(q))` holds on a model with a
Zeeman field and a non-collinear exchange field, so `kernel.build_blocks`'
`WmQ = np.conj(WQ)` stays valid with a tabulated screened `W`.

Cross-check against `chitk/chiAB.py` (independent code, frequency scan
with finite broadening, site-resolved): agreement to 4 decimals at
delta=0.02 and improving as delta falls. Weak-coupling series: scaling the
interaction by 1/4 three times gives `|W-(v+v chi0 v)| / |W-v|` =
0.127 -> 0.030 -> 0.0074, i.e. the neglected `O(v^3)` remainder falling
like `v` relative to the term it is compared against, as it must.

**2. `v` in this representation is necessarily indefinite, so screening
can ENHANCE.** This is the finding worth keeping. A density-density
interaction matrix that excludes self-interaction has a zero diagonal --
`density_interaction` does exactly that with
`np.fill_diagonal(v[(0,0,0)],0.)`, and a Hubbard `U` enters only in the
up-down channel, so a site's spin block is `[[0,U],[U,0]]`. Either way the
matrix is traceless, hence has negative eigenvalues, hence `eps(q)` sits
*below* one in those channels: the charge channel is screened and the spin
channel is Stoner-*enhanced*. That is the same physics `chitk/rpa.py`
reports as a magnetic instability, not a bug.

Measured on the gapped honeycomb, nk=6, Coulomb tail `e2=0.6`, lowest
exciton binding energy:

| onsite U | min eig `eps` | bare | screened | change |
|---|---|---|---|---|
| 0.0 | 0.651 | 0.210039 | 0.215554 | **+0.0055** |
| 0.6 | 0.713 | 0.210039 | 0.211306 | +0.0013 |
| 1.2 | 0.774 | 0.210039 | 0.211224 | +0.0012 |
| 2.0 | 0.671 | 0.210039 | 0.217824 | +0.0078 |

i.e. screening *increased* the binding in every one of those. It is not a
sign error: fed an explicitly positive-definite `v = U0 * identity`
(unphysical -- it has self-interaction -- but a clean sign probe),
`epsmin` is exactly 1.000 for every `U0`, and `|W|/|v|` at q=0 falls
0.962 / 0.929 / 0.876 for `U0` = 0.5 / 1.0 / 2.0. Pure screening, right
direction, monotone in the coupling.

The expected reduction does reappear on a model with a realistic onsite
term: `examples/2d/screened_bse/` (honeycomb, `U=1.0` plus an `e2=0.6`
tail, nk=8) gives binding 0.224373 bare -> 0.213018 screened. So the rule
of thumb is that a *point-orbital Coulomb tail with no onsite term*
antiscreens, and the fix is to include the onsite `U` the point-orbital
approximation drops.

A charge-channel-only screening would fix this too -- see point 5, where
the same construction turns out to be needed for a much more serious
reason.

**3. A vacuous cRPA window is the easy trap.** With the default
`nv=nc=None` the BSE window is the whole spectrum, so cRPA excludes every
transition, `chi0` is identically zero and `W` is just `v` -- silently, if
the check only asks whether the exclusion mask has any `True` entries (its
diagonal always does). The check has to be on occupancy-*changing* pairs.
It now raises. Note also that a genuine `nv`/`nc` subset is hard to come
by in the usual test models: every band of a spinful non-magnetic model is
two-fold degenerate and a supercell folds bands onto each other, so both
trip `select_bands`' degenerate-multiplet warning.
`tests/bse/test_bse_screening.py` builds a spinless chain with four
*different* onsite energies for this.

**5. The plain spin-orbital RPA breaks spin-rotation invariance. The plan
did not anticipate it; the charge-channel construction below fixes it.** On a non-magnetic spinful
reference the Bloch states are spin-diagonal, so `chi0` comes out
proportional to the identity in spin. The bare interaction's spin
structure is spanned by `{1, sigma_x}` -- a site pair couples every spin
combination equally, and a site's own block is the up-down-only Hubbard
term -- and that algebra is commutative, so `eps` and `W` stay inside it.
But the same-spin and opposite-spin entries, equal in `v`, are no longer
equal in `W`. Measured on the gapped honeycomb (`U=1.0`, `e2=0.6` tail,
nk=6, one q-point):

| site pair | bare same / opposite | screened same / opposite |
|---|---|---|
| (0,1) | +1.6867 / +1.6867 | +1.5646 / +1.4897 |

Splitting `W` into those two parts,

```
A n_iu n_ju + B n_iu n_jd + ... = (A+B)/2 n_i n_j + 2(A-B) Sz_i Sz_j
```

so the screened interaction has picked up an **Ising Sz-Sz coupling**,
which is not SU(2) invariant. The consequence is visible in the spectrum:
a lowest transition four-fold degenerate to 1e-14 with the bare
interaction (1.789961 x4) comes out as 2+2, split by 4.3e-3, once
screened (1.789194 x2, 1.793456 x2). A 2+2 split is not a
triplet/singlet resolution -- that would be 3+1 -- so it is the artifact,
not physics.

This is not a coding error. RPA in the density channel generically
generates Ising-like effective spin couplings; treating charge and spin
channels with a single density-density kernel in the spin-orbital basis
is what does it. `tests/bse/test_bse_screening.py::
test_screening_breaks_spin_rotation_invariance` pins both halves (the
matrix elements and the multiplet split) so it cannot change silently.

**The fix, applied.** `channel="charge"` (now the default) builds the
dielectric matrix in the **charge channel alone** -- a matrix over SITE
indices -- because that is what screening physically is: what polarizes
the medium is the total density, and what the induced charge acts back on
is again the total density.

```
chi^c_ij = sum_{s,s'} chi0_{(i,s)(j,s')}      (= chitk/chiAB.py's <n_i;n_j>)
v^c_ij   = (1/4) sum_{s,s'} v_{(i,s)(j,s')}
eps_ij   = delta_ij - sum_k v^c_ik chi^c_kj
W_{(i,s)(j,s')} = v_{(i,s)(j,s')} + [v^c chi v^c]_ij,  chi = chi^c(1-v^c chi^c)^-1
```

Two things about this that the earlier sketch in this file got wrong and
that are worth not re-deriving:

1. **It is `v + v^c chi v^c`, not `eps^-1 v`.** Left-multiplying the
   spin-orbital `v` by a site-space `eps^-1` is not Hermitian -- the usual
   argument that `eps^-1 v = v + v chi v` is a symmetric form only works
   when `eps` and `v` live on the same index space, which is exactly what
   is not true here. Going back to the diagram settles it: the second
   order term is `v^c_ik chi^c_kl v^c_lj`, with `v^c` on BOTH sides,
   because the coupling of a spin-orbital to the medium's total density is
   `v^c` at either end. Resumming gives `v^c chi v^c`, which is
   manifestly Hermitian and is what is implemented. (The two forms are
   related by `(1-AB)^-1 A = A(1-BA)^-1`; the identity
   `(eps^-1 - 1)v^c = v^c chi v^c` is how the code computes it.)
2. **`v^c_ii = U/2`, not `U`.** This was the convention question that
   stopped the first pass, and the derivation is short: the Hartree
   potential an electron at `(i,s)` feels from a spin-symmetric density
   fluctuation `dn_j` is `sum_{s'} v_{(i,s)(j,s')} dn_j/2`. Off-site that
   is `V_ij`; on-site the up-down-only Hubbard block `[[0,U],[U,0]]` gives
   `U/2`, because only half of the site's own density couples to a given
   electron. Both cases are the single spin-average formula above.

Why it works, measured:

| | orbital channel | charge channel |
|---|---|---|
| same-spin vs opposite-spin, `\|A-B\|` | 7.97e-02 | **0.0 exactly** |
| exciton multiplet spread (bare: 1.6e-14) | 4.26e-03 | **8.22e-15** |
| Hermiticity of W | 0.0 | 0.0 |
| spinless model vs orbital channel | -- | agrees to 2.2e-16 |

The SU(2) argument is structural rather than numerical luck: the
correction `[v^c chi v^c]_ij` is spin *independent* and is added equally
to every spin combination, so the same-spin minus opposite-spin part of
`W` -- the Ising term -- is left exactly as the bare interaction had it.

It also half-fixes point 2. `v^c` picks up a positive diagonal `U/2` where
the spin-orbital matrix has none, so ordinary screening is restored
whenever the model has a realistic onsite `U`. With `U = 0` it does not
help, because then `v^c` is traceless too. Honeycomb, `e2=0.6` tail, nk=6,
change in binding energy against bare:

| onsite U | orbital | charge |
|---|---|---|
| 0.0 | +0.0055 | +0.0055 |
| 0.6 | +0.0013 | +0.0005 |
| 1.2 | +0.0012 | **-0.0020** |
| 2.0 | +0.0078 | **-0.0030** |

`channel="orbital"` is kept, not deleted: it is the honest full-matrix
RPA and is what quantifies the error above. It is simply not the default.

**4. Cost.** Screening is cheap next to the kernel build when it reuses
the `PairBasis` diagonalizations (the `nkW == nk` path, which is the
default): honeycomb nk=6, the screened BSE took 0.2 s against 3.0 s for
the bare one on the same call -- the difference being numba compilation,
not screening. The real cost is indirect: `eps^-1` is dense, so a screened
`W` destroys the orbital sparsity `kernel.nonzero_pattern` exploits and
the direct term pays full `O(norb^2)` per matrix element again. On a model
whose bare interaction was nearly diagonal that is a large factor.

The `chi0` build itself is `nk^2 * n_occ * n_emp * norb^2`, jitted with a
`prange` over q. That is fine for the model sizes here but grows as
`norb^4` at fixed mesh, so a large unit cell will feel it.

### Follow-up tier (noted, not scoped)

Two plan items were deliberately not built. A `max_memory` guard on the
`chi0` tensor is unnecessary -- it is `nq*norb^2`, always dwarfed by the
BSE matrix `solve.check_memory` already guards. A standalone
`epsilon_at_q` helper was not needed either; the example computes
`eps = 1 - v(q) chi0(q)` inline from the two arrays the
`ScreenedInteraction` already exposes.

Re-converging the mean field with the screened `W` (via
`.get_dict()` into `get_mean_field_hamiltonian(V=...)`) gives a
screened-exchange / static-COHSEX reference, and the BSE on top of *that*
is once again exactly TDHF with a consistent interaction -- removing the
GW-BSE inconsistency noted above. Whether to iterate the screening
alongside it (recomputing `chi0` from the new bands) is the usual
self-consistent-GW question and should not be entered into lightly.

## Phase 4 -- scaling (not started)

**Do this before the tensor-network route.** A matrix-free iterative
eigensolver (Lanczos/LOBPCG) that applies the BSE kernel without ever
materializing it removes the O(N_pair²) memory wall outright and returns
the few lowest excitons at large nk. The kernel-apply is cheap to write:
the exchange term is already a rank-≤n_orb factorization
(`kernel.exchange_block` is pure matmuls), and the direct term is the
existing numba contraction restructured to act on a vector.

Expect this to be roughly a day's work and to move the reachable mesh by
an order of magnitude. Everything below is the step *after* it.

## The quantics tensor-train route -- measured feasibility

The idea: encode the electron-hole amplitude `A(k)` in a quantics tensor
train (binary-encode the k index; a smooth, peaked `A(k)` is low-rank in
that encoding), represent the BSE kernel as an MPO, and solve variationally
with DMRG. Cost then scales with the number of quantics bits, i.e.
*logarithmically* in nk.

### Both halves of the toolchain already exist in this ecosystem

- `src/pyqula/qutecipytk/` (bundled qutecipy): TCI1/TCI2, tensor trains
  with bond-dimension compression, MPO-MPS contraction
  (`contract_naive`/`zipup`/`TCI`), quantics grids.
- `dmrgpy/pyitensor` (pure Python, numpy/scipy only, JAX optional): `MPO`
  built from an arbitrary list of tensors, `dmrg(psi,H,sweeps)`,
  `dmrg_excited` (overlap-penalty, several excitons), `dmrg_generalized`,
  TDVP, KPM energy truncation. Same dependency profile as the bundled
  `wannierpy`, so it can be vendored or imported lazily the way
  `pyqula2dmrgpy.py` already imports `dmrgpy`.

They are complementary, not alternatives: **qutecipy builds the MPO,
pyitensor solves it.**

### Measurement 1 -- the solver works on a real BSE matrix

A real pyqula TDA BSE matrix (ionic chain, nk=8, N_pair=32) was decomposed
exactly into a 5-qubit MPO and handed to `pyitensor.dmrg`:

```
DMRG lowest exciton = 1.6694777390
exact (eigh)        = 1.6694777390     difference = 4.00e-15
```

So the plumbing -- hand-built MPO into DMRG -- is not in question.

### Measurement 2 -- gauge is the whole obstacle, and it is fixable

Exact quantics TT ranks of the TDA BSE matrix, 1D spinless two-band ionic
chain with `V1=0.8`, `nv=nc=1` (so N_pair = nk and every qubit is a k-bit):

| nk | qubits | raw `eigh` gauge | phase-fixed gauge | max possible |
|---|---|---|---|---|
| 128 | 7 | [4, 16, 64, 64, 16, 4] | [4, 16, **30**, 24, 13, 4] | [4, 16, 64, 64, 16, 4] |
| 256 | 8 | [4, 16, 64, **256**, 64, 16, 4] | [4, 16, **30**, 24, 17, 13, 4] | (= raw) |
| 512 | 9 | -- | [4, 16, **30**, 22, 15, 13, 10, 4] | [4, 16, 64, 256, 256, 64, 16, 4] |
| 1024 | 10 | -- | [4, 16, **29**, 21, 15, 14, 10, 9, 4] | [4, 16, 64, 256, **1024**, 256, 64, 16, 4] |

Two conclusions:

1. **In the raw gauge the matrix is exactly incompressible** -- maximal
   bond dimension at every cut, at every mesh size. The MPO *is* the dense
   matrix. Bloch coefficients come out of `algebra.eigh` with an arbitrary
   phase per k-point, so `C^{nk}` is a discontinuous function of k even
   where the physics is perfectly smooth, and that discontinuity is what
   destroys the rank.
2. **A trivial gauge fix already bounds the rank.** Making one fixed
   reference component of each eigenvector real and positive holds the peak
   bond dimension at ~30 while the mesh grows 8x and the dense matrix grows
   64x. Bounded rank under mesh refinement is exactly the quantics
   signature.

The spectrum is unchanged to 1e-14 across the gauge change (it is a
diagonal unitary on the pair index, `U_m = exp(i(theta_m - phi_m))`), so
this is a pure representation choice with no physics content -- but it is
the difference between "impossible" and "logarithmic".

### What is still missing

1. **Build the MPO without the dense matrix.** The measurement above built
   it by SVD-ing the full matrix, which defeats the entire purpose. A real
   implementation must construct the MPO directly: either analytically from
   the convolution structure of `W(k-k')` (convolution kernels have compact
   quantics representations) or via TCI from a matrix-element oracle. This
   is the bulk of the work, and it is what `qutecipytk` is for.
2. **Full A/B does not fit ground-state DMRG.** The spectrum of `S@K` is
   unbounded below, so plain `dmrg` would chase `-E_max`;
   `dmrg_generalized` needs a positive-definite metric and
   `S = diag(1,-1)` is indefinite. **TDA fits directly** (the `A` block is
   Hermitian and bounded below) and is what large-scale BSE codes use for
   the same reason. `tests/bse/test_bse_physics.py` already shows TDA
   converging to full at weak coupling, which is the regime bound excitons
   live in. Keep the dense full-A/B path for smaller meshes and for
   quantifying the TDA error.
3. **A smooth gauge in the general case.** The phase fix above works for a
   non-degenerate band in 1D. Degenerate subspaces need a full unitary, not
   a phase -- i.e. genuine parallel transport or Wannierization.
   `h.get_wannier_hamiltonian` (`wanniertk/`) already produces exactly such
   a gauge and is the natural supplier.
4. **Bit ordering in 2D/3D.** The k-bits of different reciprocal directions
   have to be interleaved by scale, not concatenated, or the rank will not
   saturate.

### Next decision point (go/no-go)

Repeat Measurement 2 on a **2D model with multiple, possibly degenerate
bands, using a Wannier gauge** from `wanniertk/`. If the peak bond
dimension still saturates as nk grows, the route is viable and the work
becomes engineering. If it does not, stop at Phase 4's iterative solver.

This is a contained experiment -- it needs no new solver, only the rank
measurement applied to a differently-gauged matrix.

Before implementing, check arXiv for existing quantics/tensor-network
treatments of BSE-type eigenproblems (per this repo's CLAUDE.md policy on
new formalisms) rather than deriving the MPO construction from scratch.

### Side benefit

A tensor-network solver returns the few lowest excitons, not the whole
spectrum -- so optical absorption (a sum over *all* excitons) would need a
Chebyshev/KPM treatment of the BSE operator instead. `pyitensor` ships a
KPM energy-truncation module and this repo has `kpmtk/`, so the two fit
together naturally. Worth planning for rather than discovering late.

## Conventions and traps worth not re-discovering

Recorded because each of these cost real debugging time and each is
invisible at Q=0.

- **`h.V` is W/2, not W.** The SCF stores a halved interaction as a
  decoupling convention. `bsetk.interaction.bare_interaction` doubles it,
  and `tests/bse/test_bse_interaction.py` verifies this against
  `scftk.densitydensity.get_mf_normal` rather than asserting the algebra.
  The same factor holds for the spin-exchange channels, which are
  density-density matrices in the spin-orbital basis.
- **`h.V` is incomplete for anisotropic exchange runs** (stores the z
  channel only) and is left in the internally-rotated spin frame after
  `SxSx`/`SySy`. Pass `V=` explicitly in those cases.
- **The lower-left BSE block is `-B^dag`, not `-B^*`.** The textbook Casida
  form assumes B symmetric, true at Q=0 and false at finite Q.
- **The antiresonant exchange block takes `W(-Q)`, not `W(+Q)`.** Identical
  for an onsite Hubbard U, different for any extended interaction.
- **Diagonalize via Cholesky of `S@H`, not general `eig`.** `S@H` is exactly
  Hermitian; the direct route gave 5e-5 absolute error where the structured
  one gives 3e-14 on a case with a known exact answer.
- **`supercell folding` is the test that catches finite-Q errors.** An
  n-cell supercell at Q=0 must reproduce the base cell at every Q folding
  onto it. Both bugs above were caught by it and by nothing else.

## References

- Xatu code, [arXiv:2307.01572](https://arxiv.org/abs/2307.01572) -- the
  localized-orbital BSE formalism this implementation follows. Note it
  uses a *phenomenological* (Rytova-Keldysh) screening, not an RPA W, so
  Phase 3 goes past it rather than reproducing it.
- Miyake and Aryasetiawan, [arXiv:0710.4013](https://arxiv.org/abs/0710.4013)
  -- screened Coulomb interaction (RPA and constrained RPA) evaluated in a
  maximally localized Wannier basis; the reference for Phase 3's W and for
  the cRPA window partitioning.
- Linear-scaling BSE with maximally localized Wannier functions,
  [arXiv:2309.06834](https://arxiv.org/abs/2309.06834) -- the same
  localized-basis electron-hole kernel (local fields + screened
  attraction) built from Wannier functions.
- Dynamical screening in the BSE,
  [arXiv:2302.07948](https://arxiv.org/abs/2302.07948) -- what the static
  approximation in Phase 3 costs, and when it starts to matter (strongly
  bound excitons).
- `tests/bse/test_bse_rpa.py` -- the exchange-only BSE reproduces the poles
  of `chitk/rpa.py`'s independent frequency-scan RPA.
- `tests/bse/test_bse_tdhf_cluster.py` -- 0D cluster against a
  molecular-orbital TDHF built from explicit four-index Coulomb integrals.
