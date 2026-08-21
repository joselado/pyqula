# Magnons beyond the site-basis RPA -- what landed, and what did not

The spin response of an interacting mean field has two implementations in
this repo, and they cover different interactions. This file records why,
what was measured, and the two things still open: relaxing the RPA's
non-onsite gate (which is broader than the physics requires) and carrying
the transverse exchange rung in the TDHF kernel.

## The two routes

`chitk/spinchi.py` -- **site-basis RPA**. Dresses the N x N site-resolved
spin response with a site-separable vertex, `chi@(1-V@chi)^-1`. Works for
metals and insulators alike, needs only a frequency grid, and is exact for
an onsite Hubbard U, where the transverse ladder rung lives on a single
site. It is gated to onsite-only `h.V` (`_require_onsite_only_V`).

`bsetk/spinflip.py` -- **time-dependent Hartree-Fock** in the spin-flip
electron-hole pair basis, i.e. the BSE of `bsetk/` restricted to pairs
whose electron and hole have opposite spin (arXiv:2502.06598 does the same
thing ab initio for the chromium trihalides). Handles any density-density
interaction, onsite or not. Needs a gapped mean-field reference and the
same k-mesh the mean field was converged on.

## What the site-basis RPA does and does not miss

`V2K_matrix` extracts the coefficient of `Sz_i Sz_j` from the interaction
matrix, and for a spin-independent `V_ij` that coefficient is **exactly
zero** (`W_uu - W_ud - W_du + W_dd = 0` when all four are equal). So
whatever an extended density-density interaction contributed to the
magnetic order through its Fock term is simply absent from the vertex.

When that contribution is real, the consequence is total. Measured on the
`V1`-only ferromagnetic chain of `tests/scf/test_rpa_nononsite_ferro_chain.py`
(filling 0.1, V1=1.1, no U, converged to a genuine moment mz=-0.133): the
vertex comes out identically zero, the RPA kernel is the identity, and its
smallest eigenvalue at q=0, w=0 is **1.0** where the Goldstone theorem
requires 0. No magnon of any kind.

**But it is not automatic, and the cases where it does not happen are the
common ones.** Measured smallest eigenvalue of `1 - V chi0(q=0,w=0)`, on a
honeycomb Neel mean field with the SCF and the response on the same nk=6
mesh, delta=1e-4:

| interaction | min eigenvalue | verdict |
|---|---|---|
| U=3 | 2.2e-9 | Goldstone intact |
| U=3, V1=0.5 | 3.0e-9 | Goldstone intact |
| J1=3, isotropic | 4.3e-10 | Goldstone intact |
| U=2, J1=1 | 1.1e-9 | Goldstone intact |
| V1=1.1 alone, FM chain | 1.0 | **broken** |

The eigenvalue scales as delta^2 and the null vector is the staggered
transverse pattern `[+1,-1]` in Sx or Sy, i.e. the rotation of the Neel
vector -- it is the Goldstone mode, not an accident. The controls behave:
the same state with the response on a mismatched mesh gives 5.6e-3, at
finite q=0.1 gives 0.10, and a non-magnetic reference with the same vertex
gives 0.50.

Two reasons the vertex can be incomplete and the Ward identity survive
anyway:

- **V1 on a Neel state.** Its Fock term renormalizes the hopping
  spin-INdependently: the two sublattices swap under a spin flip, so both
  spins carry the same bond charge. It never enters the exchange
  splitting, so a vertex without it is still the consistent one. (On a
  ferromagnet the bond charges do differ by spin -- hence the failure
  above. A *saturated* ferromagnet is the exception again: a filled
  majority band has no bond charge at all, so V1 does nothing there, and
  U=10 vs U=10,V1=1 give bit-identical moment and gap.)
- **isotropic exchange J.** `VJinteraction` builds all three channel
  matrices -- `vz = _build_v(J1+J1z,...)`, `vx = _build_v(J1+J1x,...)`,
  `vy = _build_v(J1+J1y,...)` at `scftk/spinspin.py:580` -- and
  `_run_anisotropic_scf` decouples the x and y ones by rotating the
  density matrix into the frame where that axis is the computational z.
  The mean field is therefore the genuine SU(2)-symmetric Hartree-Fock
  state, transverse Fock terms included. And `_full_spin_U` replicates
  the z vertex across all three channels, which is exactly right when the
  three couplings are equal. Vertex and mean field match; Goldstone holds.

So the gate in `chitk.spinchi._require_onsite_only_V` is more conservative
than the physics requires -- see the next section.

## What the Goldstone theorem measures

A magnetic mean field without spin-orbit coupling breaks SU(2), so the
exact response has a zero mode at Q=0. TDHF inherits it exactly, provided
the same interaction generates the mean field and the kernel. Measured as
`||M v||` with `v` the spin generator in the pair basis (not as "the
eigenvalue nearest zero" -- that eigenvalue is defective, so it only
converges as the square root of the same error):

| state | interaction | residual |
|---|---|---|
| honeycomb Neel | U=3 | 1.8e-10 |
| honeycomb Neel | U=3, V1=0.5 (5 keys in `h.V`) | 2.0e-10 |
| honeycomb Neel, tilted off the z axis | U=3 | 1.8e-10 |
| fully polarized Hubbard chain | U=10 | 1.7e-16 |
| triangular 120-degree spiral (3x3 cell) | U=8 | 2.9e-9 (at maxerror 1e-9) |

all at SCF `maxerror=1e-10` except where noted. The residual tracks the SCF tolerance
linearly (1.8e-6 / 1.8e-8 / 1.8e-10 at 1e-6 / 1e-8 / 1e-10), i.e. nothing
but the convergence of the mean field contributes to it. The corresponding
eigenvalue sits at 4e-5, the square root, which is why the assertion is on
the generator and not on the spectrum.

Independently of the symmetry argument, the two routes agree numerically
where both are valid. On the honeycomb Neel Hubbard state at nk=6 the
acoustic magnon comes out at 0.4917 (q=0.1) and 0.9037 (q=0.2) from both
the site-basis RPA frequency scan and the pair-basis TDHF eigenproblem,
which share no code below the Hamiltonian
(`tests/magnon/test_rpa_crosscheck.py`).

Two things break the Goldstone mode, both documented rather than worked
around:

- **a mismatched k-mesh.** A mean field converged at nk=20 and a magnon
  solved at nk=4 gives a Goldstone gap of 0.38 instead of 1e-5. The Ward
  identity is between a mean field and a kernel on the same mesh.
- **a screened kernel.** See `magnons_screening.md`; that is also why the
  ab initio BSE magnons of arXiv:2502.06598 miss the Goldstone mode by
  1.25 eV and are shifted by hand, while this construction does not need
  to be.

## The gate on non-onsite h.V -- relaxed for exchange (done)

`chitk.spinchi._require_onsite_only_V` used to reject every non-onsite
`h.V`. The measurements above showed that was too broad, and the reason it
could not simply be lifted was that **`h.V` cannot distinguish the cases
that work from the ones that do not**: an isotropic `J1` and an
anisotropic `J1z` leave the *same* z-channel Ising matrix in it
(`vz = _build_v(h1, J1+J1z, ...)`), so replicating that across the three
spin channels is exactly right for the first and wrong for the second,
with nothing on the Hamiltonian to say which one ran.

That is now fixed by recording the information instead of guessing it.
`_run_anisotropic_scf` stores the three exchange channels it already
builds, plus the density-density part, on the converged Hamiltonian as
`h.Vchannels = {"x": vx, "y": vy, "z": vz, "d": vd}` (`h.V` keeps its old
meaning -- other consumers read it -- so this is a second attribute, not a
change to that one). `chitk.spinchi._channel_spin_U` builds the vertex per
channel from it, `_full_spin_U` prefers that route when it is available,
and `_require_onsite_only_V` lets a Hamiltonian through when it is. For an
isotropic interaction the per-channel vertex reproduces the old replicated
one exactly, so nothing that worked before changed.

The per-channel vertex is not only bookkeeping: it is what lets an
easy-axis anisotropy do the physical thing. With `J1=3` fixed and `J1z`
turned up, the q=0 kernel eigenvalue goes 4.3e-10, 3.2e-2, 9.1e-2, 2.5e-1
at `J1z` = 0, 0.1, 0.3, 1.0 -- the continuous symmetry is broken
explicitly, so the Goldstone mode gaps out, by more and more. A replicated
vertex reports the isotropic answer for all four, since `h.V` is identical
in every one of them.

`chitk.spinchi._transverse_spin_K` does the same for the S+/S- ladder,
which lives in the transverse channel and therefore wants the x (=y)
coupling rather than the z one. Those coincide for everything that worked
before; `Kx != Ky` is refused rather than averaged, since S+/S- is then
not an eigen-channel of the interaction at all. The two paths are
cross-checked against each other rather than trusted separately: at q=0.1
on the honeycomb Neel state the ladder's chi_+- peaks at 0.4900 against
the full kernel's acoustic pole at 0.4917 (U=3), and at 1.3300 against
1.3296 (J1=3) -- the 0.005 spacing of the energy grid both are read off.

Both engines record the channels: `scftk/spinspin.py`'s numpy
`_run_anisotropic_scf` and `scftk/vjinteraction_jax.py`'s
`generic_vjinteraction_jax`. They have to, or the same physics would be
accepted through one and refused through the other.

Still refused, deliberately:

- a **non-onsite density-density** interaction, even in the same
  Hamiltonian as an exchange one. Its rung is a Fock term on the
  electron-hole pair index, which no site-separable vertex can carry, and
  whether dropping it matters is a property of the converged state rather
  than of the interaction (it cancels on a Neel state, it is fatal on a
  V1-ordered ferromagnet). `_channel_spin_U` returns None in that case and
  the old gate fires. `chitk/pairchi.py` (below) and the TDHF route are
  the answers there.
- a **hand-built `h.V`** with no `h.Vchannels` alongside it, for the
  original reason: nothing says which interaction it came from.

## The pair-basis ladder (done)

`chitk/pairchi.py` is the third route, and the one that removes the
density-density limitation from the frequency-resolved response rather
than routing around it. The rung
`K_{(ij),(kl)} = -W_{(i up),(j dn)} delta_ik delta_jl` is diagonal in the
PAIR index, so the ladder is summed in the basis of pair operators
`A_P = sum_r c^dag_{i up,r} c_{j dn,r+R}`, one per non-zero entry of the
real-space interaction, and the physical response read off the diagonal
pairs. The cost is set by the interaction's support, not by N^2: only
pairs in it enter the inversion, giving N(z+1) -- linear in N, eight pairs
for a honeycomb cell with a nearest-neighbour V.

One formula covers the two cases that look different in this basis: the
coupling is always the up-down element `W[2i,2j+1]`, which is `V_ij` for
an extended spin-independent interaction and `U` for an onsite Hubbard
one (whose same-spin entries are zero, since n^2 = n for one orbital).
That is what makes the ladder reduce to the familiar chi0/(1-U chi0).

Measured:

| state | interaction | min eigenvalue of 1 + V chi0 at q=0 |
|---|---|---|
| honeycomb Neel | U=3 | 2.2e-9 (both spin channels) |
| honeycomb Neel | U=3, V1=0.5 (8 pairs) | 3.2e-9 |
| metallic V1-ordered chain | V1=1.1, no U | proportional to delta exactly (4.614e-3, -4, -5, -6 at delta 1e-3 ... 1e-6) |

and it agrees with the independent TDHF pair basis to five decimals
(0.49165 at U=3, 0.59492 with V1=0.5, both at q=0.1), and with the
closed-form saturated-ferromagnet dispersion in a METAL to five decimals
(0.00173, 0.01291, 0.07756 at q = 0.02, 0.05, 0.1). No gap is required
anywhere in this route.

What it does NOT do is exchange, for the same reason the TDHF kernel does
not -- see the next section.

## The open case: the transverse rung in the TDHF kernel

`Jinteraction`/`VJinteraction`'s `J1/J2/J3/Jr` and `SzSz` are refused by
`bsetk/spinflip.py`'s `check_su2_interaction`. The reason is entirely on
the kernel side:

1. `scftk/spinspin.py:_build_v` writes the exchange coupling as the
   `+-1/4` sign pattern of `Sz_i Sz_j`, which is a density-density matrix
   in the spin-orbital basis and therefore storable in `h.V`.
2. The transverse rung of the isotropic interaction,
   `J_ij/2 (S+_i S-_j + S-_i S+_j)`, is a spin-flip two-body term. It has
   no density-density representation, so it is not in `h.V`, and
   `bsetk/interaction.py` cannot carry it.
3. Solving the Ising part alone returns a magnon spectrum gapped by of
   order J: the smallest eigenvalue of the Casida matrix at Q=0 is 1.81
   for J1=3 alone on the honeycomb Neel state, and 1.88 for J1=1 with U=3
   alongside, where both should have been zero.
   `check_su2_interaction` rejects it rather than returning that number,
   and `tests/magnon/test_interaction_guard.py` pins both the rejection
   and the gap that motivated it.

The mean field is **not** the problem -- see the previous section: the SCF
already decouples the x and y channels by rotation, so the converged state
is the genuine SU(2)-symmetric one. That means adding the rung to the
kernel alone is expected to be sufficient, and the work is contained:

- carry a second, spin-flip interaction alongside the density-density `W`,
  built from the same bond coefficients (`V2K_matrix` of the stored
  z-channel matrix gives them);
- add a `direct_block`-shaped contraction with the orbital index
  spin-flipped on each side,
  `conj(u1[m,i_up]) u2[n,i_dn] J_ij conj(w1[n,j_dn]) w2[m,j_up]`, to A,
  Abar and B;
- distinguish isotropic from anisotropic exchange first, which needs the
  stored-channels change of the previous section -- reconstructing a
  transverse rung from a z-channel matrix is only legitimate when the
  interaction really was isotropic.

The acceptance test is already written: on a J1-ordered state the Q=0
magnon has to drop from 1.81 to zero and `goldstone_residual` to the SCF
tolerance, exactly as they already do for U and V.

In the meantime, an isotropic-exchange model whose magnons are wanted
should use `get_magnon_bands(method="rpa")`, whose Goldstone mode is
intact for exactly this case (4.3e-10 above) and whose gate no longer
stands in the way.

## Metals (done)

`PairBasis(metal=True)` drops the global band window and decides the
occupied and empty sets per k-point; `spinflip.occupancy_masks` applies
them, separately for the two halves of the Casida matrix (a resonant pair
needs v occupied at k and c empty at k+Q, its antiresonant partner needs v
occupied at k+Q and c empty at k). Nothing downstream needed touching --
the flattened arrays already carry a `kindex` per pair and `kernel.py`
indexes the interaction through it, so a varying number of pairs per
k-point was always fine. The exciton path keeps `metal=False` and is
unaffected; for a gapped reference the filter is a no-op and the answer is
bit-identical either way.

This covers the case neither route could do: a ferromagnet ordered by a
neighbour-shell V1 alone is metallic (so the TDHF route used to refuse it)
and has no site-basis vertex at all (so the RPA gives it no magnon).
Measured Goldstone residual on the V1=1.1 chain at filling 0.1: 2.6e-16.

Validated against an exact reference rather than only a symmetry argument.
For a SATURATED ferromagnet the single-magnon sector is a two-body problem
with a separable interaction, whose dispersion solves
`1 = (U/N) sum_k 1/(dE_k - E)` over the occupied k. The TDHF magnon
reproduces that to five decimals (0.00173, 0.01291, 0.07756 at
q = 0.02, 0.05, 0.1), and with a symmetric occupied set the site-basis RPA
agrees with both to the same five decimals.

Two things are genuinely different in a metal:

- **the magnon is inside the Stoner continuum**, so it is not the lowest
  mode and cannot be read off by energy. `magnon_spectrum` returns the
  spectral weight |<generator|mode>|^2 per mode, and
  `magnon_bands_tdhf(by="weight")` selects branches with it. The weight
  also measures Landau damping directly: 1.00, 0.96, 0.78, 0.44 at
  Q = 0, 0.02, 0.05, 0.1 on the V1=1.5 chain.
- **E(q) is even in q only if the occupied set is**, which on a finite
  mesh is not automatic. With an even number of occupied points around
  k=0 the +q and -q magnons genuinely differ -- 0.02413 against 0.00559 at
  q=0.05 on one such mesh -- and the two routes then disagree because they
  weight the two differently (TDHF resolves +q; the RPA's (Sx,Sy,Sz) block
  mixes them). This looks exactly like a bug in whichever method is
  checked second, and is not one;
  `tests/magnon/test_metal.py` pins both halves of it.

One more trap, unrelated to metals but easiest to hit there: a persistent
exchange seed field on h (as opposed to on the initial mean-field guess)
breaks SU(2) explicitly, and the Goldstone residual then comes out at
exactly the Zeeman gap -- 2e-2 for a 1e-2 seed. That is the right answer,
not a failure.
