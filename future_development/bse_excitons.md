# Excitons / Bethe-Salpeter -- future development

Status of the BSE implementation added in `6f81888`, what is deliberately
not built yet (oscillator strengths), and what the scaling work of Phase 4
measured. The numbers quoted below were all measured on this codebase; they
are recorded so nobody has to re-derive them.

The headline of Phase 4, since it changes how the rest of this file should
be read: the BSE kernel is **exactly** a diagonal plus a fixed number of
rank-one terms, with that number set by the interaction and independent of
the k-mesh. The dense matrix the "wall" sections below are about was never
necessary. Two solvers now exploit that -- an exact matrix-free one and a
quantics tensor-train one whose cost grows like log(nk) -- and the mesh
sizes quoted in the dense tables are no longer limits.

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
- `solver="dense"|"iterative"|"qtt"` (Phase 4 below): the dense matrix, the
  exactly factorized matrix-free operator, or a quantics MPO solved by
  DMRG. The latter two are Tamm-Dancoff only.
- Tests in `tests/bse/`, examples in `examples/2d/excitons_bse/`,
  `examples/1d/exciton_qtt/` and `examples/2d/exciton_qtt/`.

### Measured cost of the dense solver

Honeycomb, spinful, 4 orbitals per cell, all bands:

| nk | N_pair | matrix | time | peak RSS |
|---|---|---|---|---|
| 8 | 256 | 512² | 1.2 s | 0.42 GB |
| 12 | 576 | 1152² | 2.7 s | 0.58 GB |
| 16 | 1024 | 2048² | 7.8 s | 1.07 GB |
| 20 | 1600 | 3200² | 21.6 s | 1.91 GB |

The default `max_memory=2.0` refuses nk=24. This is the wall everything
below is about -- and which Phase 4 removed; `solver="iterative"` and
`solver="qtt"` never allocate this matrix at all.

### Why the wall matters physically

The same system's lowest exciton energy over that mesh sequence:
1.85244 / 1.83308 / 1.84119 / 1.84012 (gap 2.1648). Converging, but only
because that exciton is fairly tightly bound. A shallow Wannier-Mott
exciton whose Bohr radius spans many unit cells has an envelope `A(k)`
squeezed into a small neighbourhood of the band edge, and needs nk in the
hundreds -- unreachable densely, and routine with the Phase 4 solvers
(measured up to nk = 262144 in 1D).

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
   pair basis (the Phase 4 solvers especially) has to respect the same
   rule -- and both of them do, in their own way: `iterative._request`
   grows the eigenvalue request past a degenerate multiplet of the
   diagonal, and the quantics solver's `gauge="projection"` exists
   precisely because a degenerate subspace has no well-defined phase to
   fix.
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

Extending the screened interaction to the **magnon** RPA kernel was asked
about and ruled out; see `magnons_screening.md` for the Goldstone/Ward
identity measurements that settle it. The short version is that the magnon
kernel must use the same interaction that produced the mean-field exchange
splitting, and screening only the kernel opens a gap
`U|m_z|(1 - U_kern/U_MF)` -- first order in the mismatch, with the Stoner
splitting as prefactor. The exciton case is different only because no
Goldstone theorem constrains it.

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

## Phase 4 -- scaling (DONE)

Both halves landed together, because the same structural finding powers
them: `bsetk/factorize.py`, `bsetk/iterative.py`, `bsetk/gauge.py`,
`bsetk/oracle.py`, `bsetk/qtt.py`, reachable as
`h.get_bse(solver="iterative"|"qtt")`. Tests in
`tests/bse/test_bse_factorize.py`, `test_bse_iterative.py`,
`test_bse_gauge.py`, `test_bse_qtt.py`; examples in
`examples/1d/exciton_qtt/` and `examples/2d/exciton_qtt/`.

### The finding the plan did not have: the kernel is EXACTLY low rank

The plan for this phase said "restructure the numba contraction to act on
a vector". That is not necessary, and the reason is worth more than the
solver it enabled. A real-space interaction dictionary Fourier transforms
as

```
W_ab(k-k') = sum_d W_ab(d) phi(d,k) conj(phi(d,k'))
```

so the direct term's only k-dependence *separates*, and the block becomes

```
A = diag(dE) + (1/N)[ F W(Q) F^dag - sum_t W_ab(d) |U_t><U_t| ]
U_t[m] = conj(el[m,a]) ho[m,b] phi(d,k_m),   t = (a,b,d)
```

one rank-one term per non-zero `(a,b,d)` of the real-space interaction --
**36 for a 4-orbital cell with U+V1+V2, independent of nk**. The
antiresonant and coupling blocks separate identically. Verified against
`kernel.build_blocks` at finite Q: `|A - A_rec| = 4.4e-16`.

So the matrix-vector product is `O(nterm*npair)` and needs
`O(nterm*npair)` memory. The dense wall was never a property of the
problem, only of how it was being written down.

The one exception, and it matters: a **tabulated** screened interaction
has no fixed-rank real-space form. Inverse transforming `W(q)` over the
mesh gives nk lattice vectors, so `R = nk*norb^2` and the rank
independence is gone. `factorize.KernelFactorization` refuses one and
points at `ScreenedInteraction.get_dict()` for a real-space truncation.

### The eigensolver: two traps, and why the fast option lost

The obvious matrix-free eigensolver is a Jacobi-preconditioned LOBPCG --
the BSE block is strongly diagonally dominant, its diagonal being the
O(gap) transition energy against an O(1/N) kernel. **The preconditioner is
a trap, and it fails worse the finer the mesh**, i.e. exactly where the
solver exists. Spinless ionic chain, against the exact 1.5023327683:

| nk | unshifted Jacobi | shifted Jacobi | no preconditioner |
|---|---|---|---|
| 1024 | 3.8e-05 | 3.8e-12 | 3.8e-12 |
| 4096 | **2.9e-01** | 3.8e-12 | 3.8e-12 |

Refining the mesh packs the transition energies near the band-edge minimum
into an ever denser cluster -- the spread scales like 1/nk^2 -- so
`1/(dE - min dE)` becomes enormous and nearly constant across hundreds of
states, preconditioning nothing and wrecking the conditioning. Shifting
the denominator by a fixed fraction of the diagonal's own spread fixes it
completely, and costs nothing.

**ARPACK is 100x faster, its eigenvalues are exact, and it still cannot be
used.** A single-start-vector Lanczos builds a Krylov space containing
each *distinct* eigenvalue once, so it cannot resolve eigenvalue
MULTIPLICITY: a degenerate level comes back once and the remaining slots
are filled from higher up, with rounding noise deciding how much of the
multiplet survives. On the spinful ionic chain at nk=16, whose lowest
exciton is four-fold degenerate, repeated runs returned either the correct
1.65611456 x4 or 1.65611456 x3 followed by 1.88540220 -- a 0.16 error,
nondeterministically. It first showed up as a test failing on
`kernel="none"` in one run and on `kernel="full"` in the next.

**This repo already knew this, elsewhere.** `qpitk`'s impurity QPI carries
the same warning for the same reason -- its `num_waves` is grown until the
partial diagonalization "never cuts a degenerate manifold in half, since
summing over a partial degenerate manifold isn't basis-independent", and
`tests/fermisurface/test_qpi_impurity.py::
test_ldos_is_independent_of_arpack_starting_vector` pins it. Worth
remembering the next time a partial diagonalization is added anywhere:
ARPACK's start vector is a hidden parameter wherever degeneracies are, and
lattices with symmetry have degeneracies everywhere. Exciton
spectra are degenerate as a rule (a spinful model with no spin-orbit
coupling makes every transition a four-fold multiplet, and the
singlet/triplet structure *is* the physics), so this cannot be documented
away.

**Seeding LOBPCG from ARPACK looks like the best of both and is not.** It
was implemented and measured: the seeded block inherits ARPACK's inability
to span a degenerate eigenspace, and one run in three still came back
0.23 off while the others were exact to 4e-15. It was also *slower* (88 s
against 24 s at nk=4096, since the seed has to be a whole block).

What is implemented is shifted-Jacobi LOBPCG from a **deterministic**
starting block -- unit vectors on the smallest diagonal entries, widened
until it does not cut a degenerate multiplet of the diagonal
(`iterative._block_size`, the same rule `select_bands` applies to band
windows). Measured: 3.8e-12 at nk=256/1024/4096, and bit-for-bit identical
across repeated runs on the degenerate case (3.33e-15 every time). It
costs 1.6 / 3.6 / 23.9 s at those meshes where ARPACK alone would take
~0.2 s. That is the price of a reference implementation being a reference:
the same answer every time.

## The quantics tensor-train route -- BUILT

The two measurements below supersede Measurement 2 of the earlier plan and
answer its go/no-go. Everything is at cross-interpolation tolerance 1e-6
unless said otherwise, and the tolerance matters: at 1e-10 the 2D ranks
still creep, at 1e-4...1e-8 they saturate. **Quote the tolerance with
every rank number.**

### Gauge: confirmed, and "phase" is not enough

Max tensor-train rank of the kernel's stacked factor tensor as the mesh is
refined 16-fold:

| model | npair | raw gauge | fixed |
|---|---|---|---|
| 1D chain, spinless | 128 / 512 / 2048 | 16 / 32 / 64 | **8 / 8 / 8** (phase) |
| 2D honeycomb, spinless | 1024 / 4096 / 16384 | 96 / 192 / 383 | **57 / 62 / 63** (phase) |
| 2D honeycomb, spinful | 4096 / 16384 / 65536 | 256 / 512 / 1024 | **182 / 248 / 274** (projection) |
| 1D chain, spinful | 4*128 / 4*512 | 16 / 32 | phase: 16 / 32; **projection: 7 / 7** |

The last row is the one to remember. **A phase fix does exactly nothing on
degenerate bands** -- it reproduces the raw-gauge rank digit for digit --
because what is arbitrary inside a degenerate subspace is a full unitary,
not a phase. Every band of a spinful Hamiltonian with no spin-orbit
coupling and no magnetic order is two-fold degenerate, so this is the
common case, not the exotic one. `gauge="projection"` (project the band
subspace onto trial orbitals, `U = A(A^dag A)^-1/2`, Wannier90's first
step) is therefore the default.

The plan proposed `wanniertk/` as the gauge supplier. That would not work:
Wannierization is mesh-global, so it needs every k-point and puts the
O(nk) scaling straight back. The projection gauge is **k-local** -- each
k-point is gauged using only its own eigenvectors, against references
fixed once on a coarse submesh -- which is what lets it run inside the
cross-interpolation oracle.

### Bit ordering: the plan's assumption was backwards

The plan asserted that the k-bits of different reciprocal directions "have
to be interleaved by scale, not concatenated, or the rank will not
saturate". Measured, the opposite is true on every 2D case tried here.
`dE(k)` on the gapped honeycomb, tolerance 1e-6, grouped (all kx bits then
all ky bits) against interleaved:

| nk | grouped | interleaved |
|---|---|---|
| 32^2 | 15 | 23 |
| 64^2 | 16 | 25 |
| 128^2 | 16 | 25 |

Both saturate; grouped saturates lower. `unfolding="grouped"` is the
default and `"interleaved"` is kept as a knob. Grouped also happens to
make the quantics grid index *identical* to the existing flat pair index
`m = (ix*nk + iy)*nband + ib`, so no permutation appears anywhere.

### How the MPO is built -- and the trap in building it

Not by decomposing a dense matrix (the earlier measurement did that, which
defeats the purpose) and not from the rank-one factors either. Two
separate cross interpolations, following the same pattern
`fermisurfacetk/singlefs.py` and `qtcitk/` already use:

1. the **interaction alone**, `X - D`, over the fused (row,column)
   quantics index, giving the kernel MPO directly at its true rank;
2. `dE(m)` over the row index alone, which becomes a diagonal MPO.

They are summed exactly at the end, with no truncation. **Interpolating
`A = dE + X - D` in one go would be a silent disaster**: the diagonal is
O(gap) while each kernel element is O(1/N), so on a fine mesh a relative
tolerance discards the entire interaction and hands back the
independent-particle spectrum, looking perfectly converged. This is the
single most important implementation decision in the module.

Going through the factors instead was considered and rejected on a
measurement: summing `R` rank-one MPOs reaches bond dimension
`sum_t chi_t^2` before any compression -- 18 * 21^2 ~ 8000 on the 2D
honeycomb -- where cross-interpolating the matrix element function lands
directly on the true rank.

### Measured: the cost really does grow like log(nk)

Spinless gapped chain, `V1=0.8`, tolerance 1e-8, lowest exciton, as
`examples/1d/exciton_qtt/` runs it. `ndiag` is how many k-points were
actually diagonalized:

| nk | E_X | ndiag | npair | time |
|---|---|---|---|---|
| 1024 | 1.5023327696 | 1011 | 1024 | 21.5 s |
| 4096 | 1.5023327707 | 2657 | 4096 | 14.2 s |
| 16384 | 1.5023327696 | 5721 | 16384 | 22.3 s |
| 65536 | 1.5023327789 | 6592 | 65536 | 20.6 s |
| 262144 | 1.5023327904 | 8757 | 262144 | 23.1 s |

The mesh grows by a factor of 256, the work by a factor of 8.7, the wall
time not at all. For reference the dense solver stops near
`npair ~ 2000`. The energy drifts in the ninth digit, which is the cross
interpolation tolerance, not the mesh; the converged value from the dense
and matrix-free solvers is 1.50233276830.

Below nk ~ 1024 the quantics solver is *slower* than the exact matrix-free
one -- cross interpolation visits a fixed few thousand points regardless.
It wins by not growing.

**Two dimensions costs much more per mesh point**, and the example says so
rather than burying it: on the gapped honeycomb the three solvers agree to
ten digits (1.5300762424 at 8x8, 1.5075436487 at 16x16) but a 16x16
quantics run takes ~30 s where a 1D run of 16384 points takes ~22 s. The
MPO bond dimension is what differs -- ~63 against ~8 for the factors --
and it enters the DMRG cost multiplied by the square of the MPS bond
dimension.

### Two dimensions: measured, and it does NOT pay off

The 1D table above is the good case. In 2D the quantics solver is slower
than the exact matrix-free one at every mesh reachable here, and the
reason is the denominator. Gapped honeycomb, spinless, `V1=0.6` plus a
Coulomb tail, lowest exciton:

| mesh | exact (iterative) | qtt tol=1e-4 | qtt tol=1e-6 | qtt tol=1e-8 |
|---|---|---|---|---|
| 16x16 | **0.9 s** | 8.6 s (err 9.6e-09) | 9.7 s | 9.7 s |
| 32x32 | **0.7 s** | 103 s (err 7.7e-06) | 222 s | 264 s |
| 64x64 | **3.1 s** | 234 s (err 9.6e-06) | -- | -- |

`ndiag` was the whole mesh at every one of these, so none of the
asymptotic advantage is in play yet. Loosening the cross-interpolation
tolerance to 1e-4 buys about 2.5x and costs surprisingly little accuracy
(1e-5 rather than the 1e-4 the name suggests), but 2.5x does not close a
factor of 76-150. The ratio does improve with mesh (150x at 32^2, 76x at
64^2), so a crossover presumably exists; it is beyond what was measured.

The 1D crossover existed because the exact solver got slow there (24 s at
4096 pairs). In 2D the same 4096 pairs take 3.1 s. **The quantics solver
did not get worse in 2D; its competitor got better.**

### Supercells are the wrong shape for this method

Asked what happens on a 3x3 supercell of the same honeycomb (18 orbitals,
9 valence, 9 conduction bands), and the answer turned out to be a usage
rule worth writing down.

First, an encoding bug the question exposed. The band pair was originally
one tensor-train site of dimension `nv*nc`, which the MPO fuses to
`(nv*nc)^2`. Fine for a primitive cell (1 or 4, fused 1 or 16);
catastrophic for the supercell, where it is 81 and **6561** -- cross
interpolation has to sample that many states at that site per pivot, and
the full-window case simply never completed. Valence and conduction are
now separate variables, each prime-factorized across sites
(`qtt.prime_factors`), so 9 becomes two sites of 3 and the widest MPO
site is 9. Any `nv` works, the largest site being the largest prime
factor, with no padding. Verified neutral on small windows: 9.3 -> 9.9 s
and 341.9 -> 353.7 s with bit-identical energies, which is the right
signature for a pure change of representation.

That fix was necessary and **not sufficient**. Profiling the full-window
supercell at nk=4 (npair=1296, sites `[2,2,2,2,3,3,3,3]`):

| stage | time |
|---|---|
| diagonal TCI | 0.1 s |
| kernel TCI | **450.7 s** |
| MPO build | 1.0 s |
| DMRG | 69.0 s |

with **MPO bond dimension 729, which is the maximum possible at that cut
(9^3)**. The operator is exactly incompressible. The reason is structural
rather than a tuning failure: quantics compression works because k is a
smooth coordinate whose binary digits are a genuine multi-scale
decomposition. A band label is not -- "digit 0 of the valence index" has
no physical meaning -- so splitting it lowers the local dimension without
buying any rank.

**The rule.** The quantics solver wants a NARROW band window on a FINE
k-mesh. A supercell is the opposite trade: a 3x3 supercell at nk is the
same physics as the primitive cell at 3nk by folding, but it buys its
resolution in the band index (81 incompressible pairs) where the
primitive cell buys it in k (compressible, nv*nc = 1). Use the primitive
cell and a fine mesh; if a supercell is unavoidable, use `nv`/`nc` to keep
the window narrow, or `solver="iterative"`, which does not care -- it did
the full window in 9.1 s against dense's 96.5 s.

Reference numbers, 3x3 supercell, nk=8x8, tolerance 1e-4:

| window | dense | iterative | qtt |
|---|---|---|---|
| nv=nc=2 (npair 256) | 0.9 s | 0.7 s | 9.9 s (err 1.5e-03) |
| all bands (npair 5184) | 96.5 s | 9.1 s | did not finish in 35 min |

### What was built vs. what the plan expected

- **Full A/B remains out**, as the plan said: `S@K` is unbounded below and
  `dmrg_generalized` needs a positive-definite metric, which `diag(1,-1)`
  is not. Both new solvers require `tda=True` and say so rather than
  silently switching.
- **The band index is one extra tensor-train site** of dimension `nv*nc`,
  via `InherentDiscreteGrid`'s per-dimension `base` -- no separate
  handling needed.
- **Binding energies do not need a mesh scan** -- but not the way it
  looked like they would. The lowest independent-particle transition is
  the ground state of the diagonal MPO alone, which is exactly what
  `kernel="none"` means, and running DMRG on it would be elegant and
  logarithmic. **It does not work: a diagonal Hamiltonian is the
  pathological case for DMRG**, since every basis state is already an
  eigenstate and the local eigenproblem at each bond leaves the sweep
  nothing to descend. Measured, it came back 0.042 high on the gapped
  chain at nk=32. What is there instead is a binary descent on the mesh
  index (`PairOracle.lowest_transition_energy`), seeded from the k-points
  the cross interpolation already visited and therefore nearly free: exact
  to 1e-15 on the same case, `O(dim*log nk)` extra diagonalizations, and
  honest about being a local refinement rather than a global minimum.
- **`pyitensor` is imported lazily from `dmrgpy`**, not vendored, so the
  quantics tests skip where dmrgpy is absent.

### Benchmark obligation -- and why it is unmet

`CLAUDE.md` asks for a comparison against an existing open implementation.
There is none for this construction, and this was checked rather than
assumed: Xatu (arXiv:2307.01572), whose formalism `bsetk/` follows, uses a
dense solver, and **TensorBinding** (arXiv:2607.00991) -- the closest
match, quantics TCI MPOs for tight-binding Hamiltonians in the same
ecosystem, and the paper whose abstract does mention excitonic physics --
has no exciton or BSE module in the copy available. The substitutes are
internal, in descending order of value: agreement with `solver="iterative"`
(exact, and independent of every tensor-train choice) past the dense wall;
agreement with the dense solver below it; gauge invariance of the
spectrum; supercell folding.

Literature that does cover the construction, and is cited in the module
docstrings: arXiv:1602.02646 (BSE eigenproblem by low-rank kernel
factorization plus QTT eigenvectors and ALS -- the same idea, in a quantum
chemistry basis), arXiv:2410.22975 (QTT-TCI operations for
Bethe-Salpeter-type equations), arXiv:2607.00991.

### Left undone

- **Excited excitons from the quantics solver do not work, and are
  refused rather than approximated.** `pyitensor.dmrg_excited`'s
  overlap-penalty objective does not converge on this problem. Gapped
  chain at nk=32, dense reference 1.813914 / 1.831623 / 1.917174: a
  penalty weight well above the bandwidth gave 1.787139 / 1.829526 /
  2.019560, and the default weight gave 1.659272 / 1.666385 / 1.797612 --
  *below* the true second eigenvalue, so not even variational.
  `dmrg_excited`'s own docstring records the same class of stationary
  point on an unrelated model, so this is a property of the penalized
  objective rather than of the plumbing; `run_dmrg`'s excited branch is
  kept so a better driver (block DMRG, or shift-invert on the MPO) can
  slot in. Until then `solver="qtt"` accepts `neig=1` only and points at
  `solver="iterative"`, which is exact and equally matrix-free.

  There was a real bug underneath it, worth not rediscovering:
  `dmrg_excited` does not leave the state normalized, so `<psi|H|psi>`
  reports an energy scaled by `<psi|psi>` -- which lands BELOW the true
  excited energy and therefore does not look wrong at all. Use the
  Rayleigh quotient. Fixing it improved the numbers and did not rescue
  the method.
- **A matrix-free full A/B solver** would need a Lanczos run in the `S@H`
  inner product. Not built.
- **Amplitude reconstruction is O(npair)**, the one non-logarithmic step,
  done because every other solver returns amplitudes. It returns an empty
  array above four million pairs rather than making the allocation the
  solver existed to avoid.
- **Optical absorption** still wants all excitons, so it needs a
  Chebyshev/KPM treatment of the BSE operator rather than a few-lowest
  solver. The matrix-free apply built here is exactly what such a KPM
  expansion would need.
- **A spinful 2D quantics run is not tested end to end.** The rank
  measurement covers it (bond dimension ~274 against ~63 spinless, for
  the same physics twice over), and the machinery has no 2D-specific or
  spin-specific branch that the 2D spinless and 1D spinful tests do not
  already exercise between them -- but the combination is expensive
  enough that it is left to the user guide's caveat rather than run in
  the suite.

### Side benefit realized

`bsetk/gauge.py` is independently useful: it is a k-local smooth-gauge
utility with a spectrum-invariance test, and nothing about it is specific
to the BSE.


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
- Benner, Dolgov, Khoromskaia and Khoromskij,
  [arXiv:1602.02646](https://arxiv.org/abs/1602.02646) -- fast iterative
  solution of the BSE eigenvalue problem by low-rank factorization of the
  kernel plus QTT-compressed eigenvectors. The same construction as
  Phase 4, in a quantum-chemistry basis.
- [arXiv:2410.22975](https://arxiv.org/abs/2410.22975) -- quantics tensor
  trains for Bethe-Salpeter-type equations (the Matsubara/parquet family),
  including how convolutions and Fourier transforms are done in compressed
  form.
- TensorBinding, [arXiv:2607.00991](https://arxiv.org/abs/2607.00991) --
  quantics tensor cross interpolation of tight-binding Hamiltonians into
  MPOs, the closest match to Phase 4's MPO construction. It has no exciton
  or BSE module, so it could not serve as a numerical benchmark.
- Dynamical screening in the BSE,
  [arXiv:2302.07948](https://arxiv.org/abs/2302.07948) -- what the static
  approximation in Phase 3 costs, and when it starts to matter (strongly
  bound excitons).
- `tests/bse/test_bse_rpa.py` -- the exchange-only BSE reproduces the poles
  of `chitk/rpa.py`'s independent frequency-scan RPA.
- `tests/bse/test_bse_tdhf_cluster.py` -- 0D cluster against a
  molecular-orbital TDHF built from explicit four-index Coulomb integrals.
