# Magnons beyond the site-basis RPA -- what landed, and what did not

The spin response of an interacting mean field has three implementations
in this repo, and they cover different interactions. This file records
why, what was measured, how the transverse rung of an exchange interaction
was carried into the two pair-basis kernels (the item that was left open
here), and what is still open after it.

## The three routes

`chitk/spinchi.py` -- **site-basis RPA**. Dresses the N x N site-resolved
spin response with a site-separable vertex, `chi@(1-V@chi)^-1`. Works for
metals and insulators alike, needs only a frequency grid, and is exact for
an onsite Hubbard U, where the transverse ladder rung lives on a single
site. It is gated to onsite-only `h.V` (`_require_onsite_only_V`).

`chitk/pairchi.py` -- **the same ladder in the interaction's pair basis**.
Keeps the frequency scan and the metal support of the site basis and gains
the interaction coverage of the one below, with no collinearity
requirement. See "The pair-basis ladder" below.

`bsetk/spinflip.py` -- **time-dependent Hartree-Fock** in the spin-flip
electron-hole pair basis, i.e. the BSE of `bsetk/` restricted to pairs
whose electron and hole have opposite spin (arXiv:2502.06598 does the same
thing ab initio for the chromium trihalides). Handles any density-density
interaction, onsite or not, and an exchange interaction whose spin
channels the SCF recorded, collinear or not, metallic (metal=True) or
gapped. Needs the same k-mesh the mean field was converged on.

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
  (At finite q the site vertex still misses the Fock rung of the exchange
  bonds, and its magnons differ from the pair-basis ones by a few percent;
  see "At finite q the site-basis RPA is a different number for exchange".)

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

Independently of the symmetry argument, the routes agree numerically
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

No collinearity assumption, and getting there is the instructive part.
The first version restricted the basis to one spin-flip sector, which lets
the Hartree rung be dropped (every pair has a != b there) and works
perfectly for a state with a global spin quantization axis. It leaves a
non-collinear state's Goldstone mode gapped by 0.41, independent of the
broadening -- the two sectors mix, and summing them separately breaks the
SU(2) Ward identity. That is not a contradiction of the Goldstone theorem,
which constrains the EXACT response: an approximation inherits it only if
it is conserving, and that truncation is not.

Two fixes were tried. Rotating the state into a global axis works for
collinear states along any axis and cannot work for a spiral; it is also
where the angle convention of global_spin_rotation bit (a partially
rotated state is neither along z nor detectably wrong, and gapped the mode
by 0.145 for a 54-degree tilt -- rotating the eigenvector SPINOR
COMPONENTS with a matrix written down from the axis avoids the convention
entirely). The real fix is the one in place: keep every spin-orbital pair
and both rungs. The 120-degree triangular spiral then gives 9.4e-8,
9.4e-10, 9.4e-12 at delta 1e-3, 1e-4, 1e-5 -- proportional to delta^2 and
limited by nothing else.

Exchange was the one interaction it did not carry, and it did not say so:
it read the Ising part from h.V and returned a kernel with no Goldstone
mode (smallest eigenvalue 0.288 on the J1=3 Neel honeycomb). The next
section is how that was closed.

## The transverse rung of an exchange interaction (done)

`Jinteraction`/`VJinteraction`'s `J1/J2/J3/Jr` used to be refused by
`bsetk/spinflip.py`'s `check_su2_interaction`, and handled wrongly without
a word by `chitk/pairchi.py`. The reason was entirely on the kernel side.
What `scftk/spinspin.py:_build_v` writes is the `+-1/4` sign pattern of
`Sz_i Sz_j`, a density-density matrix in the spin-orbital basis, so it can
live in `h.V`; the rest of the isotropic interaction,
`J_ij/2 (S+_i S-_j + S-_i S+_j)`, is a spin-flip two-body term with no
density-density form. Solving the Ising part alone put the Q=0 magnon at
1.8895 for J1=3 on the honeycomb Neel state at nk=6 and maxerror 1e-10
(1.81 was the number recorded here before), and at 1.46 on the U=4, J1=1
Neel chain, where zero is required.

### How the rung enters the kernel

The mean field was never the problem. `_run_anisotropic_scf` decouples
`J Sx_i Sx_j` and `J Sy_i Sy_j` by rotating the density matrix into the
frame where that axis is the computational z, applying the density-density
decoupling of the same Ising matrix there and rotating the mean field
back, and it records the three matrices as `h.Vchannels`. So the mean
field is the Hartree-Fock one of

    H_int = sum_c  H_dd[ W_c ; states rotated by R_c ]

with `c` running over the laboratory frame (`h.V`, which holds the z
exchange channel plus the density-density part), the x frame and the y
frame. The TDHF kernel is the derivative of that mean field with respect
to the density matrix, which is linear in the interaction, so it is the
sum of the density-density kernels of the three channels, each built from
the state coefficients rotated into its own frame (`psi -> R_c psi`, with
`R_c` the SCF's own `_AXIS_ROTATION` matrices). Nothing new had to be
derived or jitted: `bsetk/interaction.py:interaction_channels` returns the
list of `(W_c, R_c)`, and `spinflip.masked_blocks` calls
`kernel.exchange_block` and `kernel.direct_block` once per channel with
rotated `el/ho/elA/hoA` arrays. Which rotation takes z into x, and with
which sign, does not matter, since an Ising matrix is quadratic in the spin
operator.

Written in the laboratory frame the rung has two pieces, and it is worth
knowing which one does what, because the plan sketched here before
(a `direct_block`-shaped contraction with the spin flipped) would have
built only the one that does not restore the collinear Goldstone mode. By
the Fierz identity
`J S_i.S_j = (J/2) sum_st c+_is c_it c+_jt c_js - (J/4) n_i n_j`, and the
`s != t` part is the transverse term. Its Hartree piece couples the onsite
spin-flip pair `(i, s->t)` to `(j, t->s)` with `J_ij(Q)/2`: this is the
`J(Q) S+ S-` bubble, it acts inside the collinear spin-flip block, and it
is what puts the Goldstone mode back. Its Fock piece couples
spin-conserving bond pairs and so only matters once the pair basis is not
restricted, for a tilted or non-collinear state. Checked on the U=4, J1=1
chain at Q=1/6: summed over the x and y frames, the `direct_block`
contribution to the spin-flip block has norm 2e-18 and the
`exchange_block` one 0.15.

`chitk/pairchi.py` gets the same construction in its own basis. The pair
operator `A_(a,b) = c+_a c_b` of a rotated frame is `T A` with
`T[(is,jt),(is',jt')] = conj(R[s,s']) R[t,t']` (`pair_rotation`), so a
kernel `K'` built by `ladder_kernel` in that frame is `T^dag K' T` in the
laboratory pair basis, and the channels are summed. The pair basis has to
be closed under the rotation: every site pair a channel touches gets all
four spin combinations, and every onsite block gets all four, since that
is where the Hartree rung of a rotated frame lands
(`_complete_spin_blocks`). For a Hamiltonian with no rotated channel (U, V,
or a VJinteraction with J=0) the basis and the numbers are what they were.

No arXiv reference was used: this is the functional derivative of the
existing mean field rather than a new formalism, and the brute-force
reference below checks it more directly than a paper would.

### Measured

The reference, which shares nothing with the kernel: the Casida matrix of
a Gamma-only ring of N=6 cells of the two-site chain, with
`A_ia,jb = (e_a-e_i) delta + <aj|ib> - <aj|bi>` and
`B_ia,jb = <ab|ij> - <ab|ji>`, the interaction a four-index tensor written
from the Pauli matrices (`U n_up n_dn` on every site and
`sum_a J_a S^a_i S^a_j` on every bond), and the orbitals those of the
converged pyqula mean field on the same ring. Its excitation energies are
the union over the mesh of the TDHF energies at every Q, so what is
compared is the whole spectrum, 144 energies with the charge excitations
included (`channel="all"`). Largest deviation:

| interaction | state | deviation |
|---|---|---|
| U=4, before the change (control) | Neel along z | 5e-11 |
| U=4, V1=0.5, before the change (control) | Neel along z | 9e-11 |
| U=4 | Neel along z / tilted | 2e-10 / 4e-11 |
| U=4, J1=1 | along z / tilted | 4e-10 / 4e-10 |
| J1=3 | along z / tilted | 1e-10 / 4e-11 |
| U=2, J1=1.5, V1=0.3 | along z / tilted | 4e-10 / 9e-10 |
| U=4, J1=1, J1z=0.5 (XXZ), `check_su2=False` | along z | 9e-14 |
| U=4, J1=1, J1x=0.4 (x != y), `check_su2=False` | along z | 2e-11 |
| SzSz J1=3, `check_su2=False` | along z | 6e-14 |
| U=4, J1=1, before the change (Ising kernel only) | along z | 1.46 |

The two controls validate the reference itself against the kernel as it
was. The tilted states are the converged state rotated by
`global_spin_rotation` about a generic axis, which exercises the Fock piece.

Goldstone residual `||M v||/||v||` (TDHF) and smallest eigenvalue of
`1 + K chi0` at q=0 (pair basis), SCF and magnon on the same mesh:

| state | interaction | TDHF | pair basis, delta 1e-3 / 1e-4 |
|---|---|---|---|
| honeycomb Neel, along z and tilted, nk=6, maxerror 1e-10 | J1=3 | 9.7e-11 | 6.3e-8 / 4.1e-10 |
| honeycomb Neel, along z and tilted, nk=6, maxerror 1e-10 | U=3, J1=1 | 6.6e-11 | 7.0e-8 / 5.8e-10 |
| saturated FM chain along z (n2=0), nk=6 | U=10, J1=-1 | 1.8e-17 | 9.1e-5 / 9.1e-6 |
| the same, converged from a guess along (0.3,-0.5,0.8) | U=10, J1=-1 | 6.3e-16 | 9.1e-6 at 1e-4 |
| metallic FM chain along (0.3,0.4,0.5), nk=200, `metal=True` | U=3, J1=-1 | 2.0e-16 | 1.1e-3 / 1.1e-4 |
| triangular 120-degree spiral, 3x3 cell, nk=3 | U=6, J1=1 | 2.0e-10 (2.3e-3 with the rung off) | 2.8e-8 / 2.8e-10 |

The pair-basis column goes as delta^2 on the gapped antiferromagnets and
the spiral, and as delta on the metal and on the saturated ferromagnet,
which is how U and V1 behaved too. The delta^2 law stops at a floor set by
the SCF tolerance: on the J1=3 honeycomb at maxerror 1e-10 the values are
6.35e-6, 6.33e-8, 4.1e-10, 2.2e-10 at delta 1e-2 to 1e-5, and at maxerror
1e-12 the ratio stays at 100 down to delta 1e-4 (6.33e-10); the U=3 state
without exchange does the same (2.2e-9 against 2.6e-9 at delta 1e-4 for
the two tolerances).

Between the two pair-basis routes, the acoustic magnon on the honeycomb
Neel state is 1.3686781 (TDHF) against 1.3686780 (pair) at q=0.1 and
2.5074012 against 2.5074008 at q=0.2 for J1=3, and 0.7551428 against
0.7551427, 1.4317632 against 1.4317631 for U=3, J1=1. The site- and
spin-resolved `get_transverse_spinchi` of the U=2, J1=1.5 chain peaks at
1.3904 at q=1/6 on a 0.005 grid, against 1.39027 from TDHF, for the state
along z and tilted alike.

### At finite q the site-basis RPA is a different number for exchange

The site-basis RPA's per-channel vertex keeps its Goldstone mode for an
isotropic J (the table further up), but at finite q it does not agree with
the two pair-basis routes: 1.3296 against 1.3687 at q=0.1 and 2.4488
against 2.5074 at q=0.2 for J1=3, and 0.7408 against 0.7551, 1.4047
against 1.4318 for U=3, J1=1. The TDHF number is the one the brute-force
reference confirms on the chain. The difference is the Fock rung of the
exchange bonds, which the site vertex has no place for, the same limitation
that makes it drop V1: rebuilding the TDHF kernel with the direct
(`direct_block`) term restricted to onsite blocks in every frame, Hartree
terms untouched, gives 1.329616 and 2.448808 for J1=3 and 0.740740 and
1.404663 for U=3, J1=1, the site-basis RPA numbers to within 3e-5 (its pole
search runs at delta 5e-3). On a Neel state that rung cancels at q=0 and not at
finite q. This was measured and not changed, and the user guide's
statement that the routes agree wherever more than one applies was
narrowed to say so.

### What is refused, and why

- An Ising bond coupling in `h.V` with no `h.Vchannels`. That is what
  `SzSz` leaves, what `SxSx`/`SySy` leave (in a rotated frame), and what a
  hand-built exchange matrix or a Hamiltonian that lost its channels looks
  like, and nothing tells a genuine Ising interaction, whose Ising kernel
  is exactly right and whose gap is real, from the Ising half of an
  isotropic exchange. `check_su2_interaction` refuses it with a message
  naming the cases, and `pairchi` refuses it unless the interaction is
  passed explicitly as `W=`. With `check_su2=False` the TDHF route solves
  the Ising kernel of `h.V`, which the reference confirms is the TDHF of a
  laboratory-frame SzSz (6e-14 above) and which is wrong for SxSx/SySy.
- An anisotropic exchange with recorded channels (`J1x`, `J1y`, `J1z`
  unequal, or `J1z` alone) is refused by `check_su2_interaction` with a
  different message: the kernel carries every channel and its spectrum is
  right (the XXZ and x != y rows), but the symmetry is broken explicitly,
  the gap is real, and there is no Goldstone mode to measure.
  `check_su2=False` computes it. When the x and y channels differ, Sz is
  not conserved and the spin-flip block is not a block of the kernel, so
  `channel="auto"` keeps the whole pair basis and `channel="spinflip"`
  raises.
- `transverse=False` on `magnon_matrix` drops the rotated channels. It is
  only there so the 1.89 gap stays pinned in
  `tests/magnon/test_interaction_guard.py`.

Tests: `tests/magnon/test_exchange_rung.py` (the reference for isotropic,
XXZ, x != y and SzSz; Goldstone on the seeded ferromagnet, the honeycomb
Neel states, the metal and the spiral), `tests/magnon/test_interaction_guard.py`
(rewritten for the narrower refusal), and three new tests in
`tests/chi/test_pair_basis_rpa.py` (Goldstone with delta^2 scaling,
agreement with TDHF, the SzSz refusal).

### Still open

- `SzSz`, `SxSx` and `SySy` do not go through `_run_anisotropic_scf` and
  record no channels, which is why they are refused. Recording
  `{"z": v, "x": 0, "y": 0}` in `SzSz`, and the corresponding rotated
  assignment in `SxSx`/`SySy`, would let them through with a correct
  kernel. Not done, since it touches `scftk/spinspin.py`, which another
  session was editing at the time.
- The site-basis RPA's finite-q disagreement for exchange, above. Nothing
  in `chitk/spinchi.py` was changed; whether to warn there or to point
  exchange users at `method="pair"` is a decision for the maintainer.
- A Hamiltonian whose channels were recorded and which is then rotated
  with `global_spin_rotation` keeps its old `h.Vchannels`. For an isotropic
  exchange that is still exact, since the three channels are equal; for an
  anisotropic one it is not, and nothing catches it.
- A `Jr` function is evaluated at zero distance as well, so its channels
  carry onsite `Sa_i Sa_i` terms. The kernel is consistent with the SCF
  either way, but no `Jr` case was checked against the reference.
- BdG (Nambu) Hamiltonians are refused by `PairBasis` as before.

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
