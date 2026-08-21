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

    A_P = sum_r c^dag_{a,r} c_{b,r+R} ,   P = (a,b,R)

with a,b SPIN-ORBITAL indices, one per non-zero entry of the real-space
interaction plus the diagonal ones, and the Dyson equation solved there:

    chi = chi0 (1 + K chi0)^-1 .

The kernel K has both terms of the time-dependent Hartree-Fock kernel: the
exchange rung W_ab, diagonal in the pair index, and the Hartree rung
-W_ac(q) between diagonal pairs (see ladder_kernel). Restricting to a
single spin-flip sector would let the second be dropped, but only for a
state with a global spin quantization axis; keeping both, in a basis with
no spin structure assumed, is what makes this work for a NON-COLLINEAR
mean field too (measured on a 120-degree triangular spiral: the Goldstone
mode is at zero to the broadening, delta^2, where a transverse-only ladder
left it gapped by 0.41).

The physical spin response is contracted out of the diagonal pairs with
the Pauli matrices, and comes back in the same (Sx,Sy,Sz) x site layout
chitk.spinchi.spinchi_full uses.

The cost is set by how many pairs the interaction actually has, not by
N^2: only pairs in its support enter, plus the diagonal ones, so a
short-ranged V gives O(N*(z+1)) -- linear in N. What it buys over
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


def spinorbital_pairs(W, norb, tol=1e-10):
    """Return (pairs,xvals,diag): the pair basis of the ladder and the
    exchange coupling of each pair.

    A pair is P = (a,b,R) with a,b SPIN-ORBITAL indices, standing for the
    operator A_P = sum_r c^dag_{a,r} c_{b,r+R}. The basis is every pair
    the interaction couples, plus every diagonal pair (a,a,0) -- the
    latter because the Hartree rung acts between those and because the
    physical spin response is read off them, whether or not the
    interaction has an entry there.

    Using spin-orbital rather than site indices is what makes this work
    for a state with no global spin quantization axis: the basis is then
    complete for any spin structure, and no rotation into an ordering
    axis -- which a non-collinear state does not have -- is needed
    anywhere."""
    from ..multihopping import MultiHopping
    if isinstance(W, MultiHopping): W = W.get_dict()
    if not isinstance(W, dict): W = {(0, 0, 0): W}
    pairs, xvals = [], []
    for d, m in W.items():
        m = np.array(m)
        for a in range(norb):
            for b in range(norb):
                if abs(m[a, b]) > tol:
                    pairs.append((a, b, tuple(int(x) for x in d)))
                    xvals.append(m[a, b])
    have = set(pairs)
    extra = [(a, a, (0, 0, 0)) for a in range(norb)
             if (a, a, (0, 0, 0)) not in have]
    pairs = pairs + extra
    xvals = np.concatenate([np.array(xvals, dtype=np.complex128),
                            np.zeros(len(extra), dtype=np.complex128)])
    index = {p: n for n, p in enumerate(pairs)}
    diag = np.array([index[(a, a, (0, 0, 0))] for a in range(norb)],
                    dtype=np.int64)
    return pairs, xvals, diag


def ladder_kernel(pairs, xvals, diag, Wq):
    """Return the time-dependent Hartree-Fock kernel in the pair basis.

    Differentiating the Hartree-Fock Hamiltonian
    h_ab = t_ab + delta_ab sum_c W_ac rho_cc - W_ab rho_ab with respect to
    the density matrix gives two terms, and both are needed as soon as the
    basis is not restricted to a single spin-flip sector:

      exchange   W_ab delta_ac delta_bd  -- DIAGONAL in the pair index,
                 with the real-space interaction. This is the ladder rung
                 that binds the magnon, and the whole reason the pair
                 index has to be kept at all: for an extended V_ij it
                 lives on the pairs with a != b, which a site-basis vertex
                 does not have.
      Hartree    -delta_ab delta_cd W_ac(q) -- between DIAGONAL pairs
                 only, with the interaction at the transferred momentum q.

    For a transverse sector quantized along a global axis every pair has
    a != b, so the Hartree term drops out entirely -- which is why a
    transverse-only ladder can get away without it, and why it cannot once
    the state has no such axis and the sectors mix."""
    n = len(pairs)
    K = np.diag(xvals).astype(np.complex128)
    for a in range(len(diag)):
        for c in range(len(diag)):
            K[diag[a], diag[c]] -= Wq[a, c]
    return K


@jit(nopython=True, cache=True)
def _accumulate(chi0, e1, u1, e2, u2, iP, jP, phase, omegas, delta, flip):
    """Add one k-point's contribution to the pair response.

    chi0[P,P',w] += sum_{nm} (f_n - f_m) M_P conj(M_P') / (e1_n-e2_m-w+i d)

    with M_P = conj(u1[n,iP]) u2[m,jP] phase_P, iP/jP the spin-orbital
    indices the pair operator connects and phase_P its Bloch phase. The
    indices are spin-orbital ones and carry no assumption about a spin
    quantization axis, which is what lets a non-collinear state through
    (flip is unused and kept only for signature stability)."""
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


def pair_chi0(h, pairs, q=None, energies=None, delta=1e-2, nk=20):
    """Return the non-interacting response in the pair basis, shape
    (npair,npair,nomega), chi0_{P,P'} = <<A_P ; A_P'^dag>>."""
    if energies is None: energies = np.linspace(-1., 1., 100)
    energies = np.array(energies, dtype=np.float64)
    if q is None: q = [0., 0., 0.]
    q = np.array(q, dtype=np.float64)
    h = h.get_multicell().get_dense()
    hk = h.get_hk_gen()
    g = h.geometry
    iP = np.array([p[0] for p in pairs], dtype=np.int64)
    jP = np.array([p[1] for p in pairs], dtype=np.int64)
    ds = [p[2] for p in pairs]
    chi0 = np.zeros((len(pairs), len(pairs), len(energies)),
                    dtype=np.complex128)
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
        _accumulate(chi0, e1, u1, e2, u2, iP, jP, phase, energies, delta, 0)
    return chi0/len(ks)


def _setup(h, W=None, q=None):
    """Shared preamble: the interaction, the pair basis and the kernel's
    momentum-dependent Hartree block."""
    from ..bsetk.interaction import bare_interaction, interaction_at_q
    W = bare_interaction(h, V=W)
    norb = h.get_multicell().get_dense().intra.shape[0]
    pairs, xvals, diag = spinorbital_pairs(W, norb)
    if q is None: q = [0., 0., 0.]
    Wq = interaction_at_q(W, h.geometry, q)
    return pairs, xvals, diag, ladder_kernel(pairs, xvals, diag, Wq)


def pair_rpa_kernel(h, W=None, q=None, energies=None, delta=1e-2, nk=20):
    """Return (energies,kernels): the ladder kernel 1 + K chi0 in the pair
    basis, one matrix per frequency. Its zeros are the collective modes --
    the magnons -- exactly as chitk.rpa's 1 - V chi is for the site
    basis."""
    if energies is None: energies = np.linspace(-1., 1., 100)
    energies = np.array(energies, dtype=np.float64)
    pairs, xvals, diag, K = _setup(h, W=W, q=q)
    chi0 = pair_chi0(h, pairs, q=q, energies=energies, delta=delta, nk=nk)
    iden = np.identity(len(pairs), dtype=np.complex128)
    return energies, [iden + K@chi0[:, :, w] for w in range(len(energies))]


def pair_chi_rpa(h, W=None, q=None, energies=None, delta=1e-2, nk=20,
                 component=None):
    """Return (energies,chi): the RPA-dressed SPIN response, one 3N x 3N
    tensor per frequency in the (Sx,Sy,Sz) x site convention
    chitk.spinchi.spinchi_full uses, so the two are directly comparable.

    The ladder is summed in the pair basis, where the interaction is
    simple, and the spin response contracted out of it afterwards:

        <<S_a(i);S_b(j)>> = 1/4 sum (sigma_a)_{ss'} conj((sigma_b)_{t't})
                                 chi_{(is,is'),(jt',jt)}

    component, if given as a pair of indices (a,b), returns only that spin
    block instead of the full tensor.

    W defaults to the interaction the mean field was converged with
    (bsetk.interaction's bare_interaction, factor of two included), so
    this is time-dependent Hartree-Fock on top of Hartree-Fock -- the same
    self-consistent choice bsetk/spinflip.py makes."""
    if energies is None: energies = np.linspace(-1., 1., 100)
    energies = np.array(energies, dtype=np.float64)
    pairs, xvals, diag, K = _setup(h, W=W, q=q)
    chi0 = pair_chi0(h, pairs, q=q, energies=energies, delta=delta, nk=nk)
    iden = np.identity(len(pairs), dtype=np.complex128)
    norb = len(diag)
    nsite = norb//2
    index = {p: n for n, p in enumerate(pairs)}
    sigma = [np.array([[0, 1], [1, 0]], dtype=np.complex128),
             np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
             np.array([[1, 0], [0, -1]], dtype=np.complex128)]
    # every (i s, i s') pair is needed as an index; they are all diagonal
    # in the site, so they are in the basis already only when s == s'.
    # The rest are added here rather than in spinorbital_pairs, which
    # builds the KERNEL's basis -- these carry no coupling
    need = [(2*i + s, 2*i + sp, (0, 0, 0))
            for i in range(nsite) for s in range(2) for sp in range(2)]
    missing = [p for p in need if p not in index]
    if len(missing) > 0:
        pairs = pairs + missing
        xvals = np.concatenate([xvals, np.zeros(len(missing),
                                                dtype=np.complex128)])
        K2 = np.zeros((len(pairs), len(pairs)), dtype=np.complex128)
        K2[0:K.shape[0], 0:K.shape[1]] = K
        K = K2
        index = {p: n for n, p in enumerate(pairs)}
        chi0 = pair_chi0(h, pairs, q=q, energies=energies, delta=delta,
                         nk=nk)
        iden = np.identity(len(pairs), dtype=np.complex128)
    out = np.zeros((len(energies), 3*nsite, 3*nsite), dtype=np.complex128)
    for w in range(len(energies)):
        c0 = chi0[:, :, w]
        chi = c0@algebra.inv(iden + K@c0)
        for a in range(3):
            for b in range(3):
                for i in range(nsite):
                    for j in range(nsite):
                        acc = 0.0+0.0j
                        for ss in range(2):
                            for sp in range(2):
                                if sigma[a][ss, sp] == 0.: continue
                                for tp in range(2):
                                    for t in range(2):
                                        if sigma[b][tp, t] == 0.: continue
                                        P = index[(2*i+ss, 2*i+sp, (0, 0, 0))]
                                        Q = index[(2*j+tp, 2*j+t, (0, 0, 0))]
                                        acc += (sigma[a][ss, sp]
                                                *np.conj(sigma[b][tp, t])
                                                *chi[P, Q])
                        out[w, a*nsite+i, b*nsite+j] = acc/4.
    if component is not None:
        a, b = component
        out = out[:, a*nsite:(a+1)*nsite, b*nsite:(b+1)*nsite]
    return energies, out


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


def magnon_bands_pair(h, qpath=None, nq=20, **kwargs):
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
    def f(q): return pair_rpa_poles(h, q=q, **kwargs)
    outs = parallel.pcall(f, qpath)
    qs, ws, gammas = [], [], []
    for iq, poles in enumerate(outs):
        for (w, gm) in poles:
            qs.append(iq) ; ws.append(w) ; gammas.append(gm)
    return np.array(qs), np.array(ws), np.array(gammas)
