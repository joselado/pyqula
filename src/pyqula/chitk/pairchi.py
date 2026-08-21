"""The transverse spin response with the interaction's pair index kept.

chitk/rpa.py dresses an N x N site-resolved response with a vertex that is
one number per site. That is exact for an onsite Hubbard U and empty for
anything longer ranged, and the reason is visible as soon as the ladder is
written down. For a density-density interaction

    H_int = 1/2 sum_{ij,ss'} V_ij n_{is} n_{js'}

the rung of the transverse (spin-flip) ladder is

    K_{(ij),(kl)} = -V_ij delta_ik delta_jl ,

i.e. DIAGONAL IN THE PAIR INDEX (i,j) -- not in the site index. For an
onsite U it collapses onto the pairs with i=j, which is precisely the
site-basis of rpa.py, and the collapse is what makes that basis exact
there. An extended V_ij lives entirely in the pairs with i != j, which
that basis does not have, so its contribution is not approximated there
but absent (chitk.spinchi.V2K_matrix maps it to exactly zero).

This module keeps the pair index. The response is built in the basis of
pair operators

    A_P = sum_r c^dag_{i up, r} c_{j dn, r+R} ,   P = (i,j,R)

one per non-zero entry of the real-space interaction, and the Dyson
equation solved there:

    chi = chi0 (1 + V chi0)^-1 ,   V = diag(V_P) .

The physical transverse susceptibility is the diagonal-pair block,
chi_{+-}(i,k) = chi_{(i,i,0),(k,k,0)}.

The cost is set by how many pairs the interaction actually has, not by
N^2: the kernel is diagonal, so only pairs in the support of V enter the
inversion, and for a short-ranged V that is N*(z+1) -- linear in N, eight
pairs for a honeycomb cell with a nearest-neighbour V. What it buys over
bsetk/spinflip.py's pair basis is the frequency-resolved response itself
(a chi(omega), not an eigenvalue list), which is what get_spinchi_full and
the IETS maps consume, and no gap requirement anywhere.

Conventions follow the rest of the library: the Fermi energy is at zero,
the spin-orbital index of site i is 2*i for up and 2*i+1 for down, and the
real-space interaction is the {(n1,n2,n3): matrix} dictionary of
bsetk/interaction.py -- so bare_interaction(h) is the right way to get one
from a mean-field Hamiltonian, factor of two included.
"""

import numpy as np
from numba import jit

from .. import algebra


def site_interaction(W, tol=1e-10, channel="+-"):
    """Return (pairs,values): the list of (i,j,R) the interaction couples
    and the transverse ladder coupling of each, from a spin-orbital
    density-density dictionary W.

    The coupling is the UP-DOWN element W[2i, 2j+1] (down-up for the "-+"
    channel), and that is not a choice. Writing the interaction as a
    general two-body term, the transverse rung comes out as the exchange
    integral

        K_{(ij),(kl)} = -W_{(i up),(l dn)} delta_ik delta_jl ,

    so only the opposite-spin element enters. It is worth noticing that
    this covers the two cases that LOOK different in this basis with one
    formula: an extended spin-independent V_ij has W[2i,2j+1] = V_ij, and
    an onsite Hubbard U has W[2i,2i+1] = U while its same-spin entries are
    zero (n^2 = n for one orbital, so there is no self-interaction to
    write down). Reading the same element in both is what makes the ladder
    reduce to the familiar chi0/(1-U chi0) for a Hubbard model.

    The interaction still has to be spin-rotation invariant for any of
    this to describe a Goldstone-carrying magnet, but that check lives in
    bsetk.spinflip.check_su2_interaction and is not duplicated here."""
    from ..multihopping import MultiHopping
    if isinstance(W, MultiHopping): W = W.get_dict()
    if not isinstance(W, dict): W = {(0, 0, 0): W}
    si, sj = (0, 1) if channel == "+-" else (1, 0)
    pairs, values = [], []
    for d, m in W.items():
        m = np.array(m)
        n = m.shape[0]//2
        for i in range(n):
            for j in range(n):
                v = m[2*i + si, 2*j + sj]
                if abs(v) > tol:
                    pairs.append((i, j, tuple(int(x) for x in d)))
                    values.append(v)
    return pairs, np.array(values, dtype=np.complex128)


def diagonal_pairs(pairs, nsite):
    """Return the index of the pair (i,i,0) for every site i, or -1 where
    the interaction has no onsite entry for it.

    The physical transverse susceptibility is the block of the pair
    response on these, so an interaction with no onsite term at all (a
    bare V1 with no Hubbard U) has no diagonal pairs to read it off. They
    are added to the pair list by build_pairs for exactly that reason --
    a pair can be needed as an OUTPUT index without appearing in the
    kernel."""
    index = {p: n for n, p in enumerate(pairs)}
    return np.array([index.get((i, i, (0, 0, 0)), -1) for i in range(nsite)],
                    dtype=np.int64)


def build_pairs(W, nsite, channel="+-"):
    """Return (pairs,values,diag): the pair basis of the ladder.

    Every pair in the support of the interaction, plus the diagonal pairs
    (i,i,0) whether or not the interaction has an entry there. The latter
    carry a zero coupling and so do not change the kernel; they are in the
    basis because the physical response is read off them."""
    pairs, values = site_interaction(W, channel=channel)
    have = set(pairs)
    extra = [(i, i, (0, 0, 0)) for i in range(nsite)
             if (i, i, (0, 0, 0)) not in have]
    pairs = pairs + extra
    values = np.concatenate([values, np.zeros(len(extra), dtype=np.complex128)])
    return pairs, values, diagonal_pairs(pairs, nsite)


@jit(nopython=True, cache=True)
def _accumulate(chi0, e1, u1, e2, u2, iP, jP, phase, omegas, delta, flip):
    """Add one k-point's contribution to the pair response.

    chi0[P,P',w] += sum_{nm} (f_n - f_m) M_P conj(M_P') / (e1_n-e2_m-w+i d)

    with M_P = conj(u1[n,iP]) u2[m,jP] phase_P, iP/jP the spin-orbital
    indices the pair operator connects and phase_P its Bloch phase. flip
    selects the spin channel: the up index of a site is 2*i and the down
    one 2*i+1, so a (+,-) operator takes jP odd and iP even and a (-,+)
    one the other way round -- the caller has already resolved that into
    iP/jP, and flip is carried only so the two channels can share this
    kernel without rebuilding the index arrays."""
    npair = iP.shape[0]
    nb = e1.shape[0]
    nw = omegas.shape[0]
    M = np.zeros(npair, dtype=np.complex128)
    for n in range(nb):
        fn = 1.0 if e1[n] < 0. else 0.0
        for m in range(nb):
            fm = 1.0 if e2[m] < 0. else 0.0
            df = fn - fm
            if df == 0.: continue
            for P in range(npair):
                M[P] = np.conj(u1[n, iP[P]])*u2[m, jP[P]]*phase[P]
            for w in range(nw):
                den = e1[n] - e2[m] - omegas[w] + 1j*delta
                pref = df/den
                for P in range(npair):
                    if M[P] == 0.: continue
                    for Pp in range(npair):
                        chi0[P, Pp, w] += pref*M[P]*np.conj(M[Pp])
    return chi0


def pair_chi0(h, pairs, q=None, energies=None, delta=1e-2, nk=20,
              channel="+-"):
    """Return the non-interacting response in the pair basis, shape
    (npair,npair,nomega).

    channel picks which transverse operator the pairs stand for: "+-"
    means A_P = c^dag_{i up} c_{j dn} and "-+" the other way round. The
    two are the two spin-flip sectors; a collinear state generally has
    weight in one or both, and they do not mix under a spin-conserving
    interaction, which is why they are solved separately."""
    if energies is None: energies = np.linspace(-1., 1., 100)
    energies = np.array(energies, dtype=np.float64)
    if q is None: q = [0., 0., 0.]
    q = np.array(q, dtype=np.float64)
    h = h.get_multicell().get_dense()
    hk = h.get_hk_gen()
    g = h.geometry
    if channel == "+-": si, sj = 0, 1 # up index of i, down index of j
    elif channel == "-+": si, sj = 1, 0
    else: raise ValueError("channel must be '+-' or '-+', got %r"%(channel,))
    iP = np.array([2*p[0] + si for p in pairs], dtype=np.int64)
    jP = np.array([2*p[1] + sj for p in pairs], dtype=np.int64)
    ds = [p[2] for p in pairs]
    npair, nw = len(pairs), len(energies)
    chi0 = np.zeros((npair, npair, nw), dtype=np.complex128)
    ks = g.get_kmesh(nk=nk)
    for k in ks:
        k = np.array(k, dtype=np.float64)
        e1, w1 = algebra.eigh(hk(k))
        e2, w2 = algebra.eigh(hk(k + q))
        u1 = np.array(w1.T, dtype=np.complex128)
        u2 = np.array(w2.T, dtype=np.complex128)
        # the pair operator carries exp(2 pi i (k+q).R), the same Bloch
        # convention the Hamiltonian's own hoppings use
        phase = np.array([g.bloch_phase(d, k + q) for d in ds],
                         dtype=np.complex128)
        _accumulate(chi0, e1, u1, e2, u2, iP, jP, phase, energies, delta,
                    0 if channel == "+-" else 1)
    return chi0/len(ks)


def pair_chi_rpa(h, W=None, q=None, energies=None, delta=1e-2, nk=20,
                 channel="+-"):
    """Return (energies,chi): the RPA-dressed transverse response in the
    SITE basis, one N x N matrix per frequency.

    The ladder is summed in the pair basis, where the interaction is
    diagonal, and then read off the diagonal pairs. W defaults to the
    interaction the mean field was converged with (bsetk.interaction's
    bare_interaction, factor of two included), so this is
    time-dependent Hartree-Fock on top of Hartree-Fock, the same
    self-consistent choice bsetk/spinflip.py makes."""
    from ..bsetk.interaction import bare_interaction
    if energies is None: energies = np.linspace(-1., 1., 100)
    energies = np.array(energies, dtype=np.float64)
    W = bare_interaction(h, V=W)
    nsite = len(h.geometry.r)
    pairs, values, diag = build_pairs(W, nsite, channel=channel)
    chi0 = pair_chi0(h, pairs, q=q, energies=energies, delta=delta, nk=nk,
                     channel=channel)
    npair = len(pairs)
    iden = np.identity(npair, dtype=np.complex128)
    out = np.zeros((len(energies), nsite, nsite), dtype=np.complex128)
    if np.any(diag < 0):
        raise ValueError("the interaction has no onsite entry for every "
            "site, so the physical response cannot be read off the "
            "diagonal pairs. build_pairs adds them; this should not "
            "happen and means the pair basis was built by hand")
    for w in range(len(energies)):
        c0 = chi0[:, :, w]
        chi = c0@algebra.inv(iden + (values[:, None]*c0))
        out[w] = chi[np.ix_(diag, diag)]
    return energies, out


def pair_rpa_kernel(h, W=None, q=None, energies=None, delta=1e-2, nk=20,
                    channel="+-"):
    """Return (energies,kernels): the ladder kernel 1 + V chi0 in the pair
    basis, one matrix per frequency. Its zeros are the collective modes --
    the magnons -- exactly as rpa.py's 1 - V chi is for the site basis."""
    from ..bsetk.interaction import bare_interaction
    if energies is None: energies = np.linspace(-1., 1., 100)
    energies = np.array(energies, dtype=np.float64)
    W = bare_interaction(h, V=W)
    pairs, values, _ = build_pairs(W, len(h.geometry.r), channel=channel)
    chi0 = pair_chi0(h, pairs, q=q, energies=energies, delta=delta, nk=nk,
                     channel=channel)
    iden = np.identity(len(pairs), dtype=np.complex128)
    return energies, [iden + values[:, None]*chi0[:, :, w]
                      for w in range(len(energies))]


def pair_rpa_poles(h, **kwargs):
    """Return the poles of the pair-basis ladder: the frequencies where an
    eigenvalue of 1 + V chi0 crosses zero, i.e. the magnons. Same
    (npoles,2) [frequency, residual imaginary part] convention as
    chitk.rpa.rpa_kernel_poles, and the same warning about judging a mode
    by the MAGNITUDE of that imaginary part rather than its sign."""
    from .rpa import _poles_from_chi_matrix, _track_eigenvalue_branches
    es, kernels = pair_rpa_kernel(h, **kwargs)
    raw = np.array([np.linalg.eigvals(k) for k in kernels])
    eigs = _track_eigenvalue_branches(raw)
    poles = []
    for ib in range(eigs.shape[1]):
        re, im = eigs[:, ib].real, eigs[:, ib].imag
        for k in range(len(es)):
            if re[k] == 0.0: poles.append((es[k], im[k]))
            elif k+1 < len(es) and re[k]*re[k+1] < 0.:
                t = -re[k]/(re[k+1]-re[k])
                poles.append((es[k] + t*(es[k+1]-es[k]),
                              im[k] + t*(im[k+1]-im[k])))
    if len(poles) == 0: return np.zeros((0, 2))
    poles.sort(key=lambda p: p[0])
    return np.array(poles)


def magnon_bands_pair(h, qpath=None, nq=20, channel="+-", **kwargs):
    """Return the magnon bands from the pair-basis ladder, scanned along a
    q-path: the third route to the same physics.

    Compared with the other two -- chitk.spinchi.magnon_bands (site basis)
    and bsetk.spinflip.magnon_bands_tdhf (electron-hole pair basis) -- this
    one keeps the frequency scan of the first while carrying the interaction
    rung of the second, so it handles a neighbour-shell density-density
    interaction, needs no gap, and returns a chi(omega) rather than an
    eigenvalue list. It costs a frequency grid, and its poles are limited
    by the broadening delta the way any frequency scan is.

    Returns (qs,ws,gammas), the same flat-array convention as
    chitk.spinchi.magnon_bands: judge how well defined a mode is by
    abs(gammas), not by its sign."""
    from .. import parallel
    qpath = h.geometry.get_kpath(qpath, nk=nq)
    def f(q): return pair_rpa_poles(h, q=q, channel=channel, **kwargs)
    outs = parallel.pcall(f, qpath)
    qs, ws, gammas = [], [], []
    for iq, poles in enumerate(outs):
        for (w, gm) in poles:
            qs.append(iq) ; ws.append(w) ; gammas.append(gm)
    return np.array(qs), np.array(ws), np.array(gammas)
