"""Magnons as the spin-flip channel of the Bethe-Salpeter equation.

A magnon is the same kind of object as an exciton -- a bound two-particle
excitation of a mean-field state -- but built from electron-hole pairs
whose electron and hole have OPPOSITE spin, rather than the same spin an
optical exciton is made of. That is the standard many-body-perturbation
route to magnons (arXiv:2502.06598 computes the magnons of the chromium
trihalides this way), and it means the machinery is already here: the BSE
of bsetk/solve.py, restricted to the spin-flip block of its own pair
basis.

Why bother, when chitk/spinchi.py already has magnon_bands? Because that
one dresses an N x N site-basis response with a site-separable vertex,
chi@(1-V@chi)^-1, whose vertex is extracted from h.V as the coefficient of
Sz_i Sz_j -- and for a spin-INDEPENDENT V_ij that coefficient is exactly
zero. So whatever an extended density-density interaction contributed to
the magnetic order through its Fock term is absent from the vertex there.
When that contribution is real, the consequence is total rather than
small: a ferromagnetic chain ordered by V1 alone (no U) gets a vertex that
is identically zero, an RPA kernel equal to the identity, and no magnon of
any kind (measured: smallest kernel eigenvalue 1.0 where it should be 0).

That failure is not automatic, and it is worth knowing when it does not
happen. On a honeycomb Neel state, V1's Fock term renormalizes the hopping
in a spin-INdependent way -- the two sublattices exchange roles under a
spin flip, so the two spins carry the same bond charge -- and the U-only
vertex is still the consistent one: the site-basis RPA keeps its Goldstone
mode there (measured 3e-9 with U=3, V1=0.5). The same holds for an
isotropic exchange J: VJinteraction builds and decouples all three channel
matrices vx, vy, vz (the x and y ones by rotating the density matrix into
the frame where that axis is the computational z), so the mean field is a
genuine SU(2)-symmetric one, and replicating the z vertex across the three
channels -- which is what chitk.spinchi._full_spin_U does -- reconstructs
exactly the right vertex. Its Goldstone mode is intact too (2e-10 on a
J1=3 Neel honeycomb).

What this route adds, then, is not a replacement but a second, structurally
complete one: the pair basis is where the Fock rung of an extended
interaction actually lives, so it is right by construction rather than by
a cancellation that has to be checked state by state.

An exchange interaction needs one more thing. Its Ising part Sz_i Sz_j is
a density-density matrix and sits in h.V, but its transverse part
J/2 (S+_i S-_j + h.c.) is a spin-flip two-body term and does not. The
exchange SCF decouples Sx_i Sx_j and Sy_i Sy_j as the same Ising matrix in
two rotated spin frames and records the three channels in h.Vchannels, so
the kernel here is built the same way: one density-density kernel per
channel, from the states rotated into that channel's frame
(interaction.interaction_channels, masked_blocks). The kernel is the
derivative of that mean field, so the two are consistent by construction.
Without the rotated channels the acoustic magnon of a J1=3 Neel honeycomb
sits at 1.89; with them the Goldstone residual is 1e-10, and the whole
spectrum at every momentum matches a brute-force TDHF of the same model
written from the Pauli matrices (tests/magnon/test_exchange_rung.py). At
finite Q it differs from the site-basis RPA's, which has no place for the
Fock rung of the exchange bonds (1.3687 against 1.3296 at q=0.1).

The test that this is right is the Goldstone theorem. A mean-field state
that orders magnetically without spin-orbit coupling breaks SU(2)
spontaneously, so the exact response must have a zero-frequency mode at
Q=0 -- the uniform spin rotation costs nothing. Time-dependent
Hartree-Fock inherits that exactly, but only when the SAME interaction
generates the mean field and the kernel; goldstone_residual below measures
it. Measured on a Neel honeycomb Hubbard mean field, the residual is
proportional to the SCF tolerance and to nothing else -- 1.8e-6, 1.8e-8,
1.8e-10 at maxerror 1e-6, 1e-8, 1e-10 -- so it is zero up to how well the
mean field was converged. See tests/magnon/.

The corresponding EIGENVALUE is a much weaker statement, and worth not
confusing with it: the zero mode of a Casida matrix is defective (the
generator and its conjugate span a Jordan block), so a perturbation eps in
the mean field moves the eigenvalue by sqrt(eps). At maxerror 1e-10 the
acoustic branch at Q=0 therefore sits at 4e-5 -- as a residual imaginary
part, its real part being zero to 1e-13 -- which is the expected
signature rather than a problem.

Two consequences of that Ward identity are worth stating before use:

  - the mean field and the kernel must use the SAME k-mesh. A mean field
    converged at nk=20 and a magnon solved at nk=6 is not self-consistent
    on the nk=6 mesh, and the Goldstone mode acquires a real gap (measured
    0.38 on the honeycomb Hubbard case, against ~1e-5 for matched meshes).
    Nothing here can check that for you -- the mesh the mean field was
    converged on is not recorded on the Hamiltonian -- so it is on the
    caller, and goldstone_residual is how to find out.
  - the interaction must not be screened. Screening the kernel while the
    mean field keeps its own interaction breaks the identity at first
    order in the mismatch; future_development/magnons_screening.md records
    the measurements. That is also why the ab initio BSE magnons of
    arXiv:2502.06598 miss the Goldstone mode by 1.25 eV and are shifted by
    hand, while the construction here does not need to be.

Metals work too, with metal=True: the occupied and empty sets are then
decided per k-point instead of once for the whole mesh, so the number of
pairs varies across it, which everything here already tolerated. That is
what covers an itinerant magnet -- and in particular a ferromagnet ordered
by a neighbour-shell V1 alone, which the site-basis RPA cannot touch (its
vertex for one is identically zero) and which was gapless so this route
used to refuse it. The saturated case has a closed-form answer to check
against, and it is reproduced to five decimals; see tests/magnon/test_metal.py.

One thing to know before reading a metallic dispersion: E(q) is even in q
only if the OCCUPIED SET is symmetric under k -> -k, which on a finite mesh
it need not be. With an even number of occupied points around k=0 the +q
and -q magnons genuinely differ (measured 0.02413 against 0.00559 at
q=0.05 on one such mesh), and no method can paper over that.
"""

import numpy as np

from .interaction import (interaction_channels,
                          spin_block_parts)
from .pairbasis import PairBasis
from .solve import solve_pseudo_hermitian
from ..check import require_spin



def band_sz(ck):
    """Return <Sz> of every Bloch band, in units where a spin eigenstate
    gives +-1/2. ck has the (nk,norb,norb) shape PairBasis stores, with
    ck[ik][n] the coefficient vector of band n, and the spin-orbital
    convention of the rest of the library: index 2*i is the up component
    of site i and 2*i+1 the down one."""
    w = np.abs(ck)**2 # (nk,nband,norb)
    return 0.5*(np.sum(w[:, :, 0::2], axis=2) - np.sum(w[:, :, 1::2], axis=2))


def is_collinear(sz, tol=1e-6):
    """True if every band is a Sz eigenstate, i.e. |<Sz>| = 1/2 throughout.

    This is what lets the pair basis be split into a spin-flip block and a
    spin-conserving one. It fails for a non-collinear (canted, spiral) mean
    field and for anything with spin-orbit coupling, where no such split
    exists -- there the full pair basis has to be kept, which costs more
    but is not less correct."""
    return bool(np.max(np.abs(np.abs(sz) - 0.5)) < tol)


def spin_diagonalize(pb, tol=1e-8):
    """Rotate every degenerate multiplet of the pair basis into Sz
    eigenstates, in place, and rebuild the pair arrays.

    algebra.eigh returns an arbitrary basis inside a degenerate subspace,
    and a magnetic band structure has plenty of accidental degeneracies
    between an up band and a down band (whether the mesh hits one depends
    on nk -- the honeycomb Neel state is clean at nk=4 and degenerate at
    nk=6). The states it returns there are arbitrary up/down mixtures,
    which have no Sz character at all, so the spin-flip block cannot be
    identified without fixing them first.

    Doing so is free: a unitary rotation inside a degenerate subspace
    changes no energy and no observable of the two-particle problem, for
    exactly the reason bsetk/gauge.py's rotations do not. Here the
    rotation is fixed by diagonalizing Sz within each multiplet, which is
    legitimate because Sz commutes with a collinear mean-field
    Hamiltonian, so the two are simultaneously diagonalizable and this
    picks the common eigenbasis."""
    from .. import algebra
    norb = pb.ck.shape[1]
    Sz = np.zeros((norb, norb), dtype=np.complex128)
    for i in range(norb//2):
        Sz[2*i, 2*i] = 0.5
        Sz[2*i+1, 2*i+1] = -0.5
    def fix(es, cs): # cs[ik][n] = coefficients of band n
        out = np.array(cs, dtype=np.complex128, copy=True)
        for ik in range(out.shape[0]):
            for group in _degenerate_groups(es[ik], tol=tol):
                if len(group) == 1: continue
                sub = out[ik][group] # (ng,norb)
                m = np.conj(sub)@Sz@sub.T # Sz inside the multiplet
                _, u = algebra.eigh((m+m.conj().T)/2.)
                out[ik][group] = (np.array(u).T)@sub
        return out
    pb.ck = fix(pb.ek, pb.ck)
    pb.ckq = fix(pb.ekq, pb.ckq)
    pb.build() # rebuild el/ho/elA/hoA from the rotated coefficients
    return pb


def _degenerate_groups(es, tol=1e-8):
    """Group the indices of a sorted energy list into degenerate blocks"""
    groups, cur = [], [0]
    for i in range(1, len(es)):
        if abs(es[i]-es[i-1]) < tol: cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def occupancy_masks(pb):
    """Return (o1,o2), the masks of the pairs that actually exist as
    excitations of this mean field.

    A resonant pair (v,k) -> (c,k+Q) needs band v occupied at k and band c
    empty at k+Q. Its antiresonant partner runs the other way -- hole at
    (v,k+Q), electron at (c,k) -- so it needs v occupied at k+Q and c empty
    at k. The two conditions differ, which is why they are returned
    separately and why PairBasis does not apply them itself.

    For a gapped reference this is a no-op: the band window already put
    every v below the gap and every c above it, so both masks are all True
    and nothing changes. It is only for a METAL that they bite -- there a
    band can be occupied at one k-point and empty at another, and the
    number of pairs varies across the mesh. That is what lets the magnons
    of an itinerant magnet be computed here at all; see PairBasis's
    metal=True."""
    ik = np.array([l[0] for l in pb.labels])
    iv = np.array([l[1] for l in pb.labels])
    ic = np.array([l[2] for l in pb.labels])
    o1 = pb.occk[ik, iv] & (~pb.occkq[ik, ic]) # v occupied at k, c empty at k+Q
    o2 = pb.occkq[ik, iv] & (~pb.occk[ik, ic]) # v occupied at k+Q, c empty at k
    return o1, o2


def spinflip_masks(pb, tol=1e-6):
    """Return (m1,m2), the masks of the pairs that make up the magnon
    block of the Casida matrix, or None if the mean field is not collinear
    and no such block exists.

    The two halves of the matrix need DIFFERENT subsets, which is the one
    thing about this restriction that is easy to get wrong:

      m1  resonant pairs that lower the total spin, hole up and electron
          down, so that S^- creates them
      m2  pairs whose ANTIRESONANT partner lowers the total spin. The
          antiresonant state of the pair labelled (v,c) is the
          de-excitation c^dag_v c_c, which raises Sz by exactly what the
          resonant one lowers -- so m2 is the mask of the pairs with the
          OPPOSITE resonant spin change, hole down and electron up.

    Restricting both halves to m1 (the obvious thing to do) throws away
    the de-excitation half of the spin-lowering generator, and with it the
    Goldstone mode: measured on the honeycomb Neel Hubbard state, the
    acoustic branch moves from 0 to 0.77 while nothing else about the
    calculation changes.

    m2 is empty for a saturated ferromagnet -- there are no minority
    electrons left to promote -- which is correct rather than degenerate:
    the magnon problem is then purely resonant, and the Tamm-Dancoff
    approximation is exact for it.

    Which of the two spin channels is the resonant one depends on which
    way the state polarized, and that is not something a caller should
    have to arrange: a mean field that came out polarized DOWN has no
    S^--lowerable pair at all, and its magnons are the S^+ excitations
    instead. The returned third element says which generator the block
    belongs to, -1 for S^- and +1 for S^+, so goldstone_vector can build
    the matching one. Both are equally physical -- they are each other's
    image under a spin flip -- and an antiferromagnet has both."""
    o1, o2 = occupancy_masks(pb) # which pairs exist at all, see above
    szk = band_sz(pb.ck) # <Sz> at k
    szkq = band_sz(pb.ckq) # <Sz> at k+Q
    if not (is_collinear(szk, tol=tol) and is_collinear(szkq, tol=tol)):
        return None # no spin quantum number, no block structure
    ik = np.array([l[0] for l in pb.labels])
    iv = np.array([l[1] for l in pb.labels])
    ic = np.array([l[2] for l in pb.labels])
    dsz = szkq[ik, ic] - szk[ik, iv] # spin change of the resonant pair
    lower = (np.abs(dsz + 1.) < tol) # excitations S^- creates
    raise_ = (np.abs(dsz - 1.) < tol) # excitations S^+ creates
    if np.any(lower & o1): return lower & o1, raise_ & o2, -1 # S^- channel
    if np.any(raise_ & o1): return raise_ & o1, lower & o2, +1 # S^+ channel
    return None # no spin-flip pair at all, nothing to restrict to


def masked_blocks(pb, W, m1, m2):
    """Return (A,Abar,B) of the Casida matrix with the resonant half
    restricted to the pairs m1 and the antiresonant half to the pairs m2.

    This is kernel.build_blocks with two independent pair subsets instead
    of one. It is written here rather than folded into build_blocks
    because the exciton path has no use for it -- there the two halves are
    always the same set -- and because the only thing that changes is
    which arrays are handed to the same three block builders, so the
    physics stays in kernel.py where it belongs.

    W is either a single laboratory-frame density-density interaction or
    the list of (W,R) spin-frame channels interaction.interaction_channels
    returns. The kernel is linear in the interaction, so the channels are
    simply summed, each built from the state coefficients rotated into
    its own frame. For an exchange interaction the rotated channels are
    its transverse rung J/2 (S+_i S-_j + h.c.), and they are what gives it
    a Goldstone mode.

    Note the per-pair interaction index of the coupling block:
    iq[m,n] has to be built from the k-points of m1 on the rows and of m2
    on the columns, since that block connects the two different sets."""
    if isinstance(W, list): channels = W
    else: channels = [(W, None)]
    A = np.diag(pb.dE[m1]).astype(np.complex128)
    Abar = np.diag(pb.dEA[m2]).astype(np.complex128)
    B = np.zeros((int(np.sum(m1)), int(np.sum(m2))), dtype=np.complex128)
    for Wc, R in channels:
        dA, dAbar, dB = _channel_blocks(pb, Wc, R, m1, m2)
        A, Abar, B = A + dA, Abar + dAbar, B + dB
    return A, Abar, B


def _channel_blocks(pb, W, R, m1, m2):
    """The interaction part of the three masked Casida blocks for one
    density-density channel W acting in the spin frame R (None for the
    laboratory frame), see masked_blocks"""
    from .kernel import (direct_block, exchange_block, interaction_tensor,
                         nonzero_pattern, qdifference_map)
    from .interaction import interaction_at_q, rotate_spinors
    g = pb.geometry
    norm = 1.0/len(pb.kpoints) # 1/N, N the number of unit cells
    el, ho = rotate_spinors(pb.el, R), rotate_spinors(pb.ho, R)
    elA, hoA = rotate_spinors(pb.elA, R), rotate_spinors(pb.hoA, R)
    # exchange (Hartree) term. In the laboratory frame it vanishes
    # identically on the spin-flip block -- its form factors are pair
    # densities conj(electron)*hole, and an up electron and a down hole
    # have no orbital in common -- but not in a rotated one: there it is
    # the J(Q) S+ S- bubble of the transverse rung, the piece that puts a
    # collinear exchange magnet's Goldstone mode back at zero. It is
    # computed rather than assumed away in every frame, so that a mask
    # which turned out not to be exactly spin-flip would show up as a
    # wrong energy instead of a silently dropped term
    WQ = interaction_at_q(W, g, pb.Q)
    WmQ = np.conj(WQ) # W(-Q)
    Fr = (np.conj(el)*ho)[m1] # resonant density form factors
    Fa = (np.conj(elA)*hoA)[m2] # antiresonant ones
    A = exchange_block(Fr, Fr, WQ, norm)
    Abar = exchange_block(Fa, Fa, WmQ, norm)
    B = exchange_block(Fr, Fa, WQ, norm, conjugate=False)
    # direct (ladder) term, W(k-k')
    qs, iqk = qdifference_map(pb.kpoints)
    Wqs = interaction_tensor(W, g, qs)
    rows, cols = nonzero_pattern(Wqs)
    k1, k2 = pb.kindex[m1], pb.kindex[m2]
    iq11 = iqk[np.ix_(k1, k1)]
    iq22 = iqk[np.ix_(k2, k2)]
    iq12 = iqk[np.ix_(k1, k2)]
    A = A - direct_block(el[m1], el[m1], ho[m1], ho[m1],
                         Wqs, iq11, rows, cols, norm)
    Abar = Abar - direct_block(elA[m2], elA[m2], hoA[m2], hoA[m2],
                               Wqs, iq22, rows, cols, norm)
    B = B - direct_block(el[m1], hoA[m2], elA[m2], ho[m1],
                         Wqs, iq12, rows, cols, norm)
    return A, Abar, B


def _as_channels(W):
    """A single interaction dictionary, a MultiHopping, a plain onsite
    matrix, or already a list of (W,R) channels, as a list of channels"""
    from ..multihopping import MultiHopping
    if isinstance(W, list): return W
    if isinstance(W, MultiHopping): W = W.get_dict()
    if not isinstance(W, dict): W = {(0, 0, 0): W}
    return [(W, None)]


def _channel_couplings(channels):
    """Return ({bond: ising coupling per frame}, worst non-Ising deviation
    and where it is) for a list of channels.

    The frames are labelled by what the Ising matrix of the channel is in
    the laboratory frame: Sz_i Sz_j for R=None, and Sx_i Sx_j or Sy_i Sy_j
    for a rotated one, read off R itself (see _frame_axis). Every bond is
    also checked for the one thing no frame can make spin-rotation
    invariant, an Sz_i n_j-like field term."""
    ising = dict() # (d,i,j) -> [z, x, y]
    worst, where = 0., None
    for W, R in channels:
        slot = _frame_axis(R)
        for key, (dens, isg, field) in spin_block_parts(W).items():
            d, i, j = key
            if d == (0, 0, 0) and i == j: # onsite: Hubbard-like is fine
                continue # checked separately in check_su2_interaction
            if field > worst: worst, where = field, key
            ising.setdefault(key, [0., 0., 0.])[slot] += isg
    return ising, worst, where


def _frame_axis(R):
    """Which laboratory spin component the z axis of the frame R is: 0 for
    z (R=None), 1 for x, 2 for y. A state psi is R psi in that frame, so a
    frame operator sigma_z is R^dag sigma_z R in the laboratory one, which
    is plus or minus one Pauli matrix for the rotations the exchange SCF
    uses; the sign does not matter for an Ising coupling."""
    if R is None: return 0
    sz = np.array([[1., 0.], [0., -1.]])
    s = R.conj().T@sz@R
    comps = [abs(s[0, 0]), abs(s[0, 1].real), abs(s[0, 1].imag)] # z, x, y
    return int(np.argmax(comps))


def conserves_sz(W, tol=1e-8):
    """True if the interaction commutes with the total Sz, which is what
    makes the spin-flip block of a z-collinear state an exact block of
    the kernel. An isotropic or uniaxial (x = y) exchange does; an
    in-plane anisotropy (x != y) adds (Jx-Jy)/4 (S+S+ + h.c.), which
    changes Sz by two and couples the S^- block to the S^+ one."""
    ising, _, _ = _channel_couplings(_as_channels(W))
    return all(abs(c[1]-c[2]) < tol for c in ising.values())


def check_su2_interaction(W, tol=1e-8, recorded=False):
    """Raise unless the interaction is invariant under a global spin
    rotation, which is what the Goldstone theorem this whole module rests
    on actually requires.

    W is a real-space density-density interaction in the spin-orbital
    basis (the {(n1,n2,n3): matrix} dictionary of interaction.py), or the
    list of (W,R) spin-frame channels interaction.interaction_channels
    builds from a Hamiltonian that recorded them. Written as
    H = 1/2 sum W_(i s)(j s') n_(i s) n_(j s') in each frame, spin-rotation
    invariance means:

      - on a SINGLE site, up-up = down-down and up-down = down-up. The
        Hubbard term U n_up n_dn is spin-rotation invariant despite
        looking spin dependent in this basis (n_up n_dn = n^2/4 - Sz^2,
        and n^2 = n for one orbital), which is why the onsite block is
        checked differently from the bond ones.
      - between DIFFERENT sites, no Sz_i n_j-like field term, and an Ising
        coupling Sz_i Sz_j only if the same coupling is present as
        Sx_i Sx_j and Sy_i Sy_j, i.e. in the two rotated channels. That is
        the isotropic exchange J S_i.S_j, whose transverse part
        J/2 (S+_i S-_j + h.c.) is carried by those channels.

    The two ways this fails are different statements, and the message
    says which one it is:

      - an Ising bond coupling with no rotated channels at all. That is
        what h.V looks like after SzSz, after SxSx/SySy (whose h.V is in
        a rotated frame), for a hand-built exchange matrix, or with the
        transverse channels switched off, and nothing on the Hamiltonian
        says which. Only the exchange SCF (VJinteraction, and
        get_mean_field_hamiltonian with J1/J2/J3/Jr) records the channels.
      - rotated channels that do not match the Ising one: an anisotropic
        exchange. The kernel carries it correctly, but the symmetry is
        broken explicitly, the magnon gap is real, and there is no
        Goldstone mode to measure.

    recorded=True says the channels were read off h.Vchannels, so an
    Ising coupling with no rotated channel beside it is a recorded
    anisotropy (J1z alone, say) rather than a missing one, and the second
    message is the right one.

    Rejecting rather than proceeding matters because the failure is
    silent: a kernel missing its transverse rung returns an
    ordinary-looking magnon dispersion with a gap of order J at Q=0."""
    channels = _as_channels(W)
    worst, where = 0., None
    for Wc, R in channels: # onsite blocks, frame by frame
        for (d, i, j), (_, _, field) in spin_block_parts(Wc).items():
            if d == (0, 0, 0) and i == j and field > worst:
                worst, where = field, (d, i, j)
    if worst > tol:
        d, i, j = where
        raise ValueError("the interaction is not invariant under a global "
            "spin rotation: its onsite block on site %d at lattice "
            "vector %s is spin dependent beyond a Hubbard term (largest "
            "deviation %g), which is a one-body exchange field in "
            "disguise, so a kernel built from it has no Goldstone mode "
            "whatever symmetry the state has"%(i, d, worst))
    ising, worst, where = _channel_couplings(channels)
    if worst > tol:
        d, i, j = where
        raise ValueError("the interaction is not invariant under a global "
            "spin rotation: its block between sites %d and %d at "
            "lattice vector %s has an Sz_i n_j-like term (largest deviation "
            "%g), which no spin frame can make isotropic, so a kernel "
            "built from it has no Goldstone mode"%(i, j, d, worst))
    nrot = sum(1 for _, R in channels if R is not None)
    for key, (cz, cx, cy) in ising.items():
        if max(abs(cz-cx), abs(cz-cy)) < tol: continue
        d, i, j = key
        if nrot == 0 and not recorded:
            raise ValueError(
                "the interaction has an Ising coupling Sz_i Sz_j between "
                "orbitals %d and %d at lattice vector %s (%g) and nothing "
                "else, so it is not invariant under a global spin rotation "
                "and a kernel built from it has no Goldstone mode. This is "
                "what h.V holds after SzSz, after SxSx/SySy (whose h.V is "
                "written in a rotated spin frame), or for a hand-built "
                "exchange matrix, and nothing on the Hamiltonian says "
                "which: a genuine Ising interaction has a real magnon gap "
                "of order J, while the Ising half of an isotropic exchange "
                "is missing its transverse part J/2 (S+_i S-_j + h.c.) and "
                "gives the same-looking gap where zero was required. For an "
                "isotropic or anisotropic exchange, converge with "
                "VJinteraction or get_mean_field_hamiltonian(J1=...), which "
                "record the spin channels in h.Vchannels, and the "
                "transverse part is then included. For a genuine SzSz "
                "state, check_su2=False solves the Ising kernel of h.V as "
                "it stands, which is the time-dependent Hartree-Fock of "
                "that interaction (not of SxSx/SySy, whose h.V is in the "
                "wrong frame)"%(i, j, d, 4*cz.real))
        raise ValueError(
            "the exchange channels recorded in h.Vchannels are not equal "
            "(between sites %d and %d at lattice vector %s: Sx_i Sx_j "
            "%g, Sy_i Sy_j %g, Sz_i Sz_j %g), i.e. the exchange is "
            "anisotropic and breaks spin-rotation symmetry explicitly. The "
            "kernel carries every channel, so its spectrum is right, but "
            "the magnon gap it gives is a real one and there is no "
            "Goldstone mode to measure. Pass check_su2=False to compute "
            "it"%(i, j, d, 4*cx.real, 4*cy.real, 4*cz.real))


def _check_memory(dim, max_memory):
    """Raise before allocating a magnon matrix that would not fit.

    solve.py's check_memory is written in terms of a pair count and
    advises narrowing the band window or turning the Tamm-Dancoff
    approximation on, neither of which applies here -- the magnon problem
    has no free tda switch, and its dimension is already the restricted
    one. Same arithmetic (a complex128 matrix plus the eigensolver's
    working copies, measured at about four times the matrix itself),
    different advice."""
    gb = dim*dim*16/1e9*4.
    if gb > max_memory:
        raise MemoryError("this magnon calculation needs about %.1f GB "
            "(a %d x %d dense matrix plus eigensolver workspace), above "
            "the max_memory = %.1f GB limit. Reduce nk, narrow the band "
            "window with nv/nc, or raise max_memory"%(gb, dim, dim,
                                                      max_memory))


class MagnonProblem():
    """The magnon eigenproblem at one momentum: the pair basis pb, the
    Casida matrix M, the two pair masks its halves were built from, and
    the sign op of the spin generator its block belongs to (-1 for S^-,
    +1 for S^+, 0 when no restriction was made and the generator is picked
    by weight instead). Held together because none of the five means
    anything without the others -- M's rows cannot even be counted without
    the masks."""
    def __init__(self, pb, M, m1, m2, op):
        self.pb, self.M, self.m1, self.m2, self.op = pb, M, m1, m2, op
        self.n1 = int(np.sum(m1)) # size of the excitation half
        self.n2 = int(np.sum(m2)) # size of the de-excitation half
    def __iter__(self): # so (pb,M,m1,m2,op) = magnon_matrix(...) works
        return iter((self.pb, self.M, self.m1, self.m2, self.op))


def magnon_matrix(h, Q=None, nk=10, V=None, channel="auto", nv=None, nc=None,
                  max_memory=2.0, check_su2=True, metal=False,
                  transverse=True):
    """Return (pb,M,m1,m2): the pair basis, the time-dependent
    Hartree-Fock matrix whose eigenvalues are the magnon energies at Q,
    and the resonant/antiresonant pair masks it was built from.

    M is the Casida matrix [[A,B],[-B^dag,-conj(Abar)]] that
    bsetk/solve.py builds for excitons, with two differences: it is
    restricted to the magnon block of the pair basis (channel, see
    spinflip_masks), and the interaction is checked for spin-rotation
    invariance first (check_su2), since without that there is no Goldstone
    mode to expect.

    The interaction is the one the mean field was converged with: h.V,
    plus, when the exchange SCF recorded them in h.Vchannels, the Sx_i Sx_j
    and Sy_i Sy_j exchange channels, each as a density-density kernel in
    its own rotated spin frame (interaction.interaction_channels). Those
    two channels are the transverse rung J/2 (S+_i S-_j + h.c.) of an
    exchange interaction; without them an isotropic J1=3 Neel honeycomb
    gets a Q=0 magnon at 1.89 instead of zero. transverse=False leaves them
    out, which is only useful to see that. V= replaces the whole
    interaction by one laboratory-frame density-density matrix.

    An exchange interaction whose Sx_i Sx_j and Sy_i Sy_j channels differ
    does not conserve Sz, so the spin-flip block is not a block of its
    kernel: channel='auto' then keeps the whole pair basis, and
    channel='spinflip' raises.

    channel:
      "auto" (default)  restrict to the magnon block when the mean field
                        is collinear, keep the whole pair basis when it is
                        not. The restriction is exact -- a spin-conserving
                        interaction does not connect the blocks -- and
                        cuts the matrix dimension by about two, so the
                        dense solve by about eight.
      "spinflip"        restrict, and raise if the state is not collinear
      "all"             never restrict. The magnons are still in the
                        spectrum, mixed in with the charge excitons, and
                        every energy returned is a real excitation of the
                        system -- just not necessarily a magnon.

    metal=True lifts the requirement that the mean field be gapped: the
    band window becomes every band and which pairs exist is decided per
    k-point from the occupations (see PairBasis and occupancy_masks). That
    is what lets an itinerant magnet -- a doped ferromagnetic chain, say --
    be treated here at all. It changes nothing for a gapped reference,
    where the occupancy filter is a no-op (checked: identical pair counts
    and identical Goldstone residual either way), so it is safe to turn on
    when unsure; it is off by default only because the gapped path fails
    loudly and informatively when the reference is not what the caller
    thought it was.

    Two things are different about a metal once it runs. The Casida matrix
    picks up Fermi-surface pairs of nearly zero energy, so the Cholesky in
    solve_pseudo_hermitian can fail and fall back to the general solver.
    And the magnon is no longer the lowest mode -- it sits inside the
    Stoner continuum -- so it has to be found by spectral weight, which is
    what magnon_spectrum is for.

    Interpreting the returned M needs the masks: its first sum(m1) rows
    are the excitation half and the remaining sum(m2) the de-excitation
    one, and those two counts differ (m2 is empty for a saturated
    ferromagnet)."""
    require_spin(h,"magnons (there is no spin to flip in a spinless "
            "Hamiltonian)")
    if channel not in ("auto", "spinflip", "all"):
        raise ValueError("channel must be 'auto', 'spinflip' or 'all', "
                "got %r"%(channel,))
    # bare interaction, i.e. TDHF, one density-density matrix per spin frame
    # the mean field was decoupled in (see interaction_channels)
    W = interaction_channels(h, V=V, transverse=transverse)
    if check_su2: # recorded: the channels came from h.Vchannels
        recorded = (V is None and transverse
                    and getattr(h, "Vchannels", None) is not None)
        check_su2_interaction(W, recorded=recorded)
    if not conserves_sz(W): # an in-plane exchange anisotropy, x != y
        if channel == "spinflip":
            raise ValueError("this interaction does not conserve the total "
                "Sz (its Sx_i Sx_j and Sy_i Sy_j exchange channels differ), "
                "so the kernel couples the spin-lowering and spin-raising "
                "pairs and there is no spin-flip block to restrict to. Use "
                "channel='all'")
        channel = "all"
    pb = PairBasis(h, Q=Q, nk=nk, nv=nv, nc=nc, metal=metal)
    masks = None
    if channel != "all":
        spin_diagonalize(pb) # make the degenerate multiplets Sz eigenstates
        masks = spinflip_masks(pb)
        if masks is None and channel == "spinflip":
            raise ValueError("this mean field is not collinear (its bands "
                "are not Sz eigenstates), so its pair basis has no "
                "spin-flip block to restrict to. Use channel='all', which "
                "keeps every pair and still contains the magnons")
    if masks is None: # whole pair basis, both halves, both spin channels
        m1, m2 = occupancy_masks(pb) # still only the pairs that exist
        op = 0 # generator picked by weight, see goldstone_vector
    else: m1, m2, op = masks
    _check_memory(int(np.sum(m1)+np.sum(m2)), max_memory)
    A, Abar, B = masked_blocks(pb, W, m1, m2)
    if Abar.shape[0] == 0: M = A # saturated: no de-excitation half at all
    else: M = np.block([[A, B], [-B.conj().T, -np.conj(Abar)]])
    return MagnonProblem(pb, M, m1, m2, op)


def magnon_energies(h, n=None, **kwargs):
    """Return the magnon energies at a single center-of-mass momentum Q.

    The Casida spectrum splits into an excitation branch, one energy per
    resonant pair kept, and a de-excitation branch which is the mirror of
    the opposite spin channel at -Q. The first is what comes back, and
    which is which is decided by the conserved norm of the linear-response
    problem, ||X||^2 - ||Y||^2: it is positive on excitations and negative
    on de-excitations, and the n1 largest are kept.

    Neither of the two obvious alternatives works. Filtering on E>0 fails
    on the Goldstone mode, which sits at zero with a residual imaginary
    part of order sqrt(SCF tolerance, see the module docstring), so its
    sign decides nothing. Taking the n1 highest energies fails on a
    saturated ferromagnet, where the two halves are not mirror images of
    each other -- the de-excitation half describes the empty opposite spin
    channel -- and the two branches interleave rather than separating at
    zero: measured on the fully polarized Hubbard chain at Q=0.1, the
    energy comes out +0.076 that way against the correct -0.076 (a
    negative energy, i.e. that ferromagnet is an unstable saddle of the
    mean-field problem, which at half filling in one dimension it is).

    A sizable imaginary part on any energy means the mean-field reference
    is unstable against that excitation, and is returned rather than
    dropped, exactly as in a BSE solve."""
    p = magnon_matrix(h, **kwargs)
    if p.n2 == 0: # no de-excitation half at all, ordinary Hermitian problem
        es, _ = _solve_resonant(p.M)
        es = np.sort(es)
    else:
        es, ws = solve_pseudo_hermitian(p.M)
        ws = ws/np.linalg.norm(ws, axis=1)[:, None]
        norm = (np.sum(np.abs(ws[:, 0:p.n1])**2, axis=1)
                - np.sum(np.abs(ws[:, p.n1:])**2, axis=1))
        keep = np.argsort(-norm)[0:p.n1] # the excitation branch
        es = np.sort_complex(es[keep])
    if np.max(np.abs(es.imag)) < 1e-10*max(1., np.max(np.abs(es.real))):
        es = es.real + 0.0j
    if n is not None: es = es[0:n]
    return es


def _solve_resonant(A):
    """Diagonalize a saturated-ferromagnet magnon problem, which has no
    de-excitation half at all (see spinflip_masks) and is therefore an
    ordinary Hermitian eigenproblem rather than a Casida one. Returned in
    the same (eigenvalues,eigenvectors-as-rows) shape as
    solve_pseudo_hermitian so the caller does not have to branch twice."""
    import scipy.linalg as lg
    es, ws = lg.eigh((A+A.conj().T)/2.)
    return es.astype(np.complex128), np.array(ws.T, dtype=np.complex128)


def spin_generator(pb, op):
    """Return the one-body matrix of the total spin generator: S^- for
    op=-1 and S^+ for op=+1, both as sum over the sites of the unit cell,
    in the spin-orbital convention where index 2*i is the up component of
    site i."""
    norb = pb.ck.shape[1]
    O = np.zeros((norb, norb), dtype=np.complex128)
    for i in range(norb//2):
        if op < 0: O[2*i+1, 2*i] = 1.0 # S^-, down <- up
        else: O[2*i, 2*i+1] = 1.0 # S^+, up <- down
    return O


def goldstone_vector(p):
    """Return the total spin generator of the broken symmetry, written in
    the electron-hole pair basis as the concatenated (X,-Y) vector the
    Casida matrix of magnon_matrix acts on.

    This is the mode the Goldstone theorem is about: rotating every spin
    by the same angle costs no energy, so this vector -- and no
    approximation to it -- must be annihilated by the time-dependent
    Hartree-Fock matrix at Q=0. Its two halves are the matrix elements of
    the generator between the occupied and empty states of the mean field,

      X_m = <c,k+Q| S |v,k>  over m1     Y_m = <v,k+Q| S |c,k>  over m2

    Note the minus sign on the de-excitation half: the Casida matrix built
    here is [[A,B],[-B^dag,-conj(Abar)]], whose lower block row already
    carries a sign, so a one-body operator enters it as (X,-Y) rather than
    (X,Y). With the wrong sign the residual comes out at 2.4 instead of
    1e-9 on the honeycomb Hubbard case, i.e. this is not a convention that
    can be left ambiguous.

    Which generator (S^- or S^+) is set by the block the problem was
    restricted to. With no restriction (channel='all', a non-collinear
    state) both are legitimate -- for a state polarized along an arbitrary
    axis, S^- and S^+ are both combinations of broken generators -- and
    the one with the larger weight in this pair basis is used, since a
    state polarized exactly along -z leaves S^- with none."""
    pb, m1, m2, op = p.pb, p.m1, p.m2, p.op
    def build(o):
        O = spin_generator(pb, o)
        X = np.einsum("ma,ab,mb->m", np.conj(pb.el[m1]), O, pb.ho[m1])
        Y = np.einsum("ma,ab,mb->m", np.conj(pb.hoA[m2]), O, pb.elA[m2])
        return np.concatenate([X, -Y]) if p.n2 > 0 else X
    if op != 0: return build(op)
    down, up = build(-1), build(+1) # unrestricted: pick the one with weight
    return down if np.linalg.norm(down) > np.linalg.norm(up) else up


def magnon_spectrum(h, **kwargs):
    """Return (energies,weights): the magnon energies at one momentum Q and
    how much of the spin generator each mode carries.

    In an insulator the magnon is the lowest mode and nothing else is
    nearby, so magnon_energies is enough. In a METAL it is not: the
    spin-flip particle-hole continuum (the Stoner continuum) reaches down
    to zero, so the spectrum is dense at low energy and "the lowest
    eigenvalue" is a continuum state rather than the collective mode. What
    separates them is spectral weight -- the magnon is the mode the uniform
    spin rotation actually couples to, and the continuum states carry
    almost none of it.

    weights[i] is |<generator|mode_i>|^2 in the metric of the
    linear-response problem, normalized to sum to one over the branch, so
    it is directly the fraction of the transverse spectral weight in each
    mode. Sorting by it, rather than by energy, is how to read a magnon
    dispersion out of a metal."""
    p = magnon_matrix(h, **kwargs)
    if p.n2 == 0:
        es, ws = _solve_resonant(p.M)
        keep = np.arange(len(es))
        metric = np.ones(len(es))
    else:
        es, ws = solve_pseudo_hermitian(p.M)
        ws = ws/np.linalg.norm(ws, axis=1)[:, None]
        metric = (np.sum(np.abs(ws[:, 0:p.n1])**2, axis=1)
                  - np.sum(np.abs(ws[:, p.n1:])**2, axis=1))
        keep = np.argsort(-metric)[0:p.n1] # the excitation branch
        es, ws, metric = es[keep], ws[keep], metric[keep]
    v = goldstone_vector(p)
    S = np.ones(p.M.shape[0])
    if p.n2 > 0: S[p.n1:] = -1. # the Casida metric diag(1,-1)
    ov = np.abs(ws.conj()@(S*v))**2
    denom = np.sum(ov)
    if denom > 0: ov = ov/denom
    order = np.argsort(es.real)
    return es[order], ov[order]


def goldstone_residual(h, nk=10, relative=True, **kwargs):
    """Return how far the mean field is from satisfying the Goldstone
    theorem: the norm of the time-dependent Hartree-Fock matrix at Q=0
    applied to the uniform spin-rotation generator, ||M v|| / ||v||.

    Zero means the magnon spectrum starts at exactly zero energy, which
    every magnetic state without spin-orbit coupling must do. In practice
    it comes out proportional to the SCF tolerance the mean field was
    converged to (measured 1.8e-10 at maxerror 1e-10 on the honeycomb Neel
    Hubbard state, 2.0e-10 with a V1 neighbor-shell interaction on top),
    so it measures the convergence of the mean field rather than an error
    of the method -- which is what makes it a sharp test: nothing else in
    the calculation is allowed to contribute to it.

    relative=True divides by the scale of the matrix (its largest
    transition energy), so the number is comparable between models with
    different bandwidths.

    This is deliberately not an eigenvalue search. The zero eigenvalue is
    defective -- the generator and its conjugate span a Jordan block -- so
    it only converges as the square root of the same error, and picking
    "the eigenvalue closest to zero" out of a spectrum that also contains
    the Stoner continuum is a much weaker statement than showing that the
    generator itself is annihilated."""
    p = magnon_matrix(h, Q=[0., 0., 0.], nk=nk, **kwargs)
    v = goldstone_vector(p)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        raise ValueError("the spin generator has no weight in this pair "
            "basis, so there is no Goldstone mode to measure. The usual "
            "reason is a non-magnetic mean field: check h.get_vev('sz') "
            "before reading anything into a magnon calculation")
    res = np.linalg.norm(p.M@v)/nv
    if relative: res = res/max(1., float(np.max(np.abs(p.pb.dE))))
    return res


def magnon_bands_tdhf(h, qpath=None, nq=20, n=None, by="energy", **kwargs):
    """Return the magnon bands E(Q) of a magnetic mean-field state, from
    the spin-flip channel of the Bethe-Salpeter equation.

    One eigenproblem is solved per q-point, so the cost is nq times that
    of a single magnon_energies call. qpath/nq select the path in the same
    convention as get_bands/chitk.spinchi.magnon_bands (high-symmetry
    labels or explicit q-vectors), and n keeps only the n lowest branches
    at each q.

    by= decides which n branches are kept at each q-point, and for a metal
    it is the whole difference between a dispersion and noise. "energy"
    (the default) keeps the n lowest, which is what an insulator wants --
    there the magnon IS the lowest mode. "weight" keeps the n modes
    carrying the most of the spin generator, which is what a metal wants:
    its magnon sits inside the Stoner continuum, so the lowest mode at any
    q is a continuum state and the acoustic branch read off by energy is
    not the magnon at all. See magnon_spectrum.

    Returns (qs,es): qs is the integer index of the q-point along the path
    and es the magnon energy, both flat 1D arrays ready for a scatter
    plot, matching what chitk.spinchi.magnon_bands returns. es is complex
    if the mean-field state is unstable against some excitation.

    The acoustic branch starts at zero at Q=0 by the Goldstone theorem,
    and it is worth checking that it does (goldstone_residual) before
    reading a dispersion: everything else in the calculation -- the
    interaction, the mesh matching, the convergence of the mean field --
    shows up there first."""
    from .. import parallel
    if h.dimensionality < 1:
        raise ValueError("magnon bands need a periodic Hamiltonian; a 0d "
                "system has no momentum to disperse in. Use "
                "magnon_energies for its discrete spin excitations")
    if by not in ("energy", "weight"):
        raise ValueError("by must be 'energy' or 'weight', got %r"%(by,))
    qpath = h.geometry.get_kpath(qpath, nk=nq) # generate the q-path
    def f(q):
        if by == "energy": return magnon_energies(h, Q=q, n=n, **kwargs)
        es, w = magnon_spectrum(h, Q=q, **kwargs)
        keep = np.argsort(-w)[0:(len(es) if n is None else n)]
        return np.sort_complex(es[keep])
    outs = parallel.pcall(f, qpath) # one eigenproblem per q-point
    qs = np.concatenate([np.full(len(es), iq) for iq, es in enumerate(outs)])
    es = np.concatenate(outs)
    return qs, es
