# Excitons / Bethe-Salpeter -- future development

Status of the BSE implementation added in `6f81888`, what is deliberately
not built yet, and a *measured* feasibility study of the tensor-network
route to large k-meshes. The numbers quoted below were all measured on this
codebase; they are recorded so nobody has to re-derive them.

## What exists today

`src/pyqula/bsetk/` (+ public `bse.py`), reachable as `h.get_bse()`,
`h.get_exciton_energies()`, `h.get_exciton_binding_energies()`,
`h.get_exciton_states()`.

- Full non-Tamm-Dancoff BSE at arbitrary center-of-mass momentum `Q`,
  following the localized-orbital formalism of the Xatu code
  ([arXiv:2307.01572](https://arxiv.org/abs/2307.01572)).
- Kernel read from `h.V` by default (time-dependent Hartree-Fock on top of
  Hartree-Fock, no double counting), or supplied via `V=`;
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
   pair basis (the iterative solver of Phase 3 especially) has to respect
   the same rule.
3. **Envelope visualization**: `|A_vc(k)|²` over the BZ, and its real-space
   transform. `BSE.pairs.labels` already carries the `(ik,iv,ic)` of every
   pair index, so this is presentation work.

## Phase 3 -- scaling (not started)

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
becomes engineering. If it does not, stop at Phase 3's iterative solver.

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
  localized-orbital BSE formalism this implementation follows.
- `tests/bse/test_bse_rpa.py` -- the exchange-only BSE reproduces the poles
  of `chitk/rpa.py`'s independent frequency-scan RPA.
- `tests/bse/test_bse_tdhf_cluster.py` -- 0D cluster against a
  molecular-orbital TDHF built from explicit four-index Coulomb integrals.
