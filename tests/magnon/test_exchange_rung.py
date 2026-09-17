"""The transverse rung of an exchange interaction in the magnon kernels.

An isotropic exchange J S_i.S_j is not a density-density interaction. Its
Ising part J Sz_i Sz_j is, and that is what h.V stores; its transverse part
J/2 (S+_i S-_j + h.c.) is a spin-flip two-body term. The exchange SCF
decouples it by writing Sx_i Sx_j and Sy_i Sy_j as the same Ising matrix in
two rotated spin frames, and records the three channels in h.Vchannels.
The time-dependent Hartree-Fock kernel is the derivative of that mean field
with respect to the density matrix, so it is the sum of the three
density-density kernels, each built from the states rotated into its frame
(bsetk.interaction.interaction_channels). Without the two rotated ones the
acoustic magnon of a J1=3 Neel honeycomb sits at 1.89 instead of zero.

Three things are checked here:

  - the whole TDHF spectrum, at every momentum of the mesh, against a
    brute-force reference that shares nothing with the kernel: the Casida
    matrix of a Gamma-only ring of N cells, with the interaction written
    as a four-index tensor straight from the Pauli matrices. Measured
    agreement 4e-10 for isotropic exchange, 9e-14 for XXZ, 2e-11 for an
    in-plane anisotropy, 6e-14 for SzSz;
  - the Goldstone mode, as ||M v|| on the rotation generator, on states
    whose moments point along a generic axis;
  - that what cannot be done honestly is refused with a reason: an Ising
    h.V with no recorded channels (a hand-built matrix, or one that lost
    them; SzSz, SxSx and SySy record theirs),
    and an anisotropic exchange when a Goldstone mode is asked for.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk import spinflip
from pyqula.meanfield import SzSz, VJinteraction

N = 6  # cells of the ring = k-points of the mesh, the SCF shares it

_SIGMA = [np.array([[0, 1], [1, 0]], dtype=complex),
          np.array([[0, -1j], [1j, 0]], dtype=complex),
          np.array([[1, 0], [0, -1]], dtype=complex)]


def _chain_cell():
    g = geometry.chain().get_supercell(2)
    g.get_sublattice()
    return g


def _neel_chain(**kw):
    """A Neel state of the two-site chain, gapped, converged on the N mesh"""
    return VJinteraction(_chain_cell().get_hamiltonian(), filling=0.5,
                         mf="antiferro", nk=N, maxerror=1e-12, mix=0.3,
                         maxite=3000, **kw).hamiltonian


def _ring_hamiltonian(h):
    """The mean-field Hamiltonian of h on a ring of N cells, real space"""
    md = h.get_multicell().get_dense().get_multihopping().get_dict()
    no = md[(0, 0, 0)].shape[0]
    H = np.zeros((N*no, N*no), dtype=complex)
    for d, m in md.items():
        for c in range(N):
            c2 = (c + int(d[0])) % N
            H[c*no:(c+1)*no, c2*no:(c2+1)*no] += np.array(m)
    return H


def _ring_interaction(U=0., Jxyz=(0., 0., 0.)):
    """v[p,q,r,s] of H = 1/2 sum v c^dag_p c^dag_q c_s c_r on the ring of
    2N sites: U n_up n_dn on every site, sum_a J_a S^a_i S^a_j on every
    first-neighbour bond, written from the Pauli matrices"""
    ns = 2*N
    v = np.zeros((2*ns,)*4, dtype=complex)
    for i in range(ns):
        v[2*i, 2*i+1, 2*i, 2*i+1] += U
        v[2*i+1, 2*i, 2*i+1, 2*i] += U
        for j in ((i+1) % ns, (i-1) % ns): # both orderings of each bond
            for s in range(2):
                for sp in range(2):
                    for t in range(2):
                        for tp in range(2):
                            v[2*i+s, 2*j+t, 2*i+sp, 2*j+tp] += sum(
                                Jxyz[a]/4.*_SIGMA[a][s, sp]*_SIGMA[a][t, tp]
                                for a in range(3))
    return v


def _ring_tdhf(H, v):
    """Excitation energies of the textbook Casida problem,
    A_ia,jb = (e_a-e_i) d + <aj|ib> - <aj|bi>, B_ia,jb = <ab|ij> - <ab|ji>"""
    e, C = np.linalg.eigh(H)
    occ, vir = np.where(e < 0)[0], np.where(e >= 0)[0]
    V = np.einsum("pP,qQ,pqrs,rR,sS->PQRS", C.conj(), C.conj(), v, C, C,
                  optimize=True)
    no, nv = len(occ), len(vir)
    A = (np.einsum("ajib->iajb", V[np.ix_(vir, occ, occ, vir)])
         - np.einsum("ajbi->iajb", V[np.ix_(vir, occ, vir, occ)]))
    A = A.reshape(no*nv, no*nv) + np.diag(
        (e[vir][None, :] - e[occ][:, None]).reshape(-1))
    Bt = V[np.ix_(vir, vir, occ, occ)]
    B = (np.einsum("abij->iajb", Bt)
         - np.einsum("abji->iajb", Bt)).reshape(no*nv, no*nv)
    w, X = np.linalg.eig(np.block([[A, B], [-B.conj(), -A.conj()]]))
    X = X/np.linalg.norm(X, axis=0)[None, :]
    n = no*nv
    norm = np.sum(np.abs(X[:n])**2, axis=0) - np.sum(np.abs(X[n:])**2, axis=0)
    return np.sort(w[np.argsort(-norm)[:n]].real)


def _pyqula_tdhf(h, **kw):
    """The same spectrum from the pair-basis kernel, one momentum at a
    time over the whole mesh"""
    es = [spinflip.magnon_energies(h, nk=N, Q=[iq/N, 0., 0.], channel="all",
                                   **kw) for iq in range(N)]
    return np.sort(np.concatenate(es).real)


def _tilted(h):
    t = h.copy()
    t.global_spin_rotation(vector=[1., 0.3, 0.], angle=0.37)
    return t


def test_the_isotropic_exchange_spectrum_matches_a_brute_force_reference():
    """Every TDHF energy at every momentum, on a state along z and on the
    same state rotated off every axis, with U, J1 and V1 together. The
    reference is exact for this model on this ring and knows nothing about
    spin frames, so agreement pins both halves of the rung: the S+S-
    bubble, which acts inside the spin-flip block, and the bond Fock term,
    which only a non-collinear pair basis sees."""
    h = _neel_chain(U=2.0, J1=1.5)
    ref = _ring_tdhf(_ring_hamiltonian(h), _ring_interaction(U=2.0,
                                                    Jxyz=(1.5, 1.5, 1.5)))
    assert np.max(np.abs(_pyqula_tdhf(h) - ref)) < 1e-7
    t = _tilted(h)
    assert abs(t.get_vev("sz")[0]) < 0.5*abs(h.get_vev("sz")[0])
    ref = _ring_tdhf(_ring_hamiltonian(t), _ring_interaction(U=2.0,
                                                    Jxyz=(1.5, 1.5, 1.5)))
    assert np.max(np.abs(_pyqula_tdhf(t) - ref)) < 1e-7
    # and without the rotated channels it is a different spectrum entirely
    ising = _pyqula_tdhf(h, transverse=False, check_su2=False)
    assert np.max(np.abs(ising - _ring_tdhf(_ring_hamiltonian(h),
                  _ring_interaction(U=2.0, Jxyz=(1.5, 1.5, 1.5))))) > 0.1


@pytest.mark.parametrize("kw,Jxyz", [
    (dict(U=4.0, J1=1.0, J1z=0.5), (1.0, 1.0, 1.5)),  # uniaxial, XXZ
    (dict(U=4.0, J1=1.0, J1x=0.4), (1.4, 1.0, 1.0)),  # in-plane, x != y
])
def test_an_anisotropic_exchange_spectrum_matches_the_reference(kw, Jxyz):
    """Anisotropic exchange breaks spin rotation explicitly, so there is
    no Goldstone mode and check_su2 refuses it by default. The kernel still
    carries every channel, and with the check off its spectrum is the
    exact TDHF one. The in-plane case does not conserve Sz either, so the
    spin-flip restriction is not available for it."""
    h = _neel_chain(**kw)
    with pytest.raises(ValueError, match="anisotropic"):
        h.get_goldstone_residual(nk=N)
    ref = _ring_tdhf(_ring_hamiltonian(h), _ring_interaction(U=4.0,
                                                             Jxyz=Jxyz))
    assert np.max(np.abs(_pyqula_tdhf(h, check_su2=False) - ref)) < 1e-7
    if Jxyz[0] != Jxyz[1]:
        with pytest.raises(ValueError, match="Sz"):
            spinflip.magnon_matrix(h, nk=N, check_su2=False,
                                   channel="spinflip")


def test_szsz_sxsx_sysy_record_their_channel_and_match_the_reference():
    """SzSz records its coupling as the z channel with x and y at zero,
    SxSx/SySy as their own axis, the layout an anisotropic exchange
    (J1z alone) has. So a Goldstone mode is refused with the anisotropy
    message, and check_su2=False solves the kernel: the whole spectrum
    matches the reference with only Sz_i Sz_j on the bonds, and the three
    axes give the same spectrum, measured 3e-13 apart.

    Without the recorded channel, what SzSz used to leave, the Ising h.V
    is refused, since it is also what an isotropic exchange that lost its
    transverse part looks like."""
    from pyqula.meanfield import SxSx, SySy
    kw = dict(J1=3.0, filling=0.5, nk=N, maxerror=1e-12, mix=0.3,
              maxite=3000)
    h = SzSz(_chain_cell().get_hamiltonian(), mf="antiferro", **kw).hamiltonian
    assert abs(h.get_vev("sz")[0]) > 0.1
    with pytest.raises(ValueError, match="anisotropic"):
        h.get_goldstone_residual(nk=N)
    ref = _ring_tdhf(_ring_hamiltonian(h), _ring_interaction(
                         Jxyz=(0., 0., 3.0)))
    spec = _pyqula_tdhf(h, check_su2=False)
    assert np.max(np.abs(spec - ref)) < 1e-7
    g = _chain_cell()
    for axis, solver in ((0, SxSx), (1, SySy)):
        v = np.zeros(3)
        v[axis] = 1.0
        mf = g.get_hamiltonian()
        mf.add_exchange([v*g.sublattice[i] for i in range(len(g.r))])
        t = solver(g.get_hamiltonian(), mf=mf, **kw).hamiltonian
        m = [t.get_vev(op)[0] for op in ("sx", "sy", "sz")]
        assert abs(m[axis]) > 0.1
        assert np.max(np.abs(_pyqula_tdhf(t, check_su2=False) - spec)) < 1e-8
    lost = h.copy()
    lost.Vchannels = None
    with pytest.raises(ValueError, match="SzSz"):
        lost.get_magnon_energies(nk=N)


def test_goldstone_of_a_ferromagnet_seeded_along_a_generic_axis():
    """A saturated ferromagnet converged from a guess along an arbitrary
    direction, so the x and y channels of the SCF did real work and the
    bands are not Sz eigenstates. Measured 6e-16."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    n = np.array([0.3, -0.5, 0.8])
    mf = h.copy()
    mf.add_exchange(3.0*n/np.linalg.norm(n)) # seed the guess, never h
    hm = VJinteraction(h, U=10.0, J1=-1.0, filling=0.5, mf=mf, nk=N,
                       maxerror=1e-12, mix=0.3, maxite=3000).hamiltonian
    m = np.array([hm.get_vev(op)[0] for op in ("sx", "sy", "sz")])
    assert np.min(np.abs(m)) > 0.2*np.linalg.norm(m)  # no component small
    assert hm.get_goldstone_residual(nk=N) < 1e-10
    assert hm.get_goldstone_residual(nk=N, transverse=False,
                                     check_su2=False) > 1e-2


def _neel_honeycomb(**kw):
    g = geometry.honeycomb_lattice()
    return VJinteraction(g.get_hamiltonian(), filling=0.5, mf="antiferro",
                         nk=N, maxerror=1e-10, mix=0.3, maxite=3000,
                         **kw).hamiltonian


@pytest.mark.slow
def test_goldstone_of_an_exchange_antiferromagnet():
    """The case the roadmap was written about. J1=3 alone on the honeycomb
    and U=3 with J1=1: the acoustic magnon was at 1.89 without the rung,
    and the residual is now 9.7e-11 and 6.6e-11, the SCF tolerance, along
    z (spin-flip block) and tilted (whole pair basis) alike."""
    for kw in (dict(J1=3.0), dict(U=3.0, J1=1.0)):
        h = _neel_honeycomb(**kw)
        assert abs(h.get_vev("sz")[0]) > 0.1
        assert h.get_goldstone_residual(nk=N) < 1e-8
        assert _tilted(h).get_goldstone_residual(nk=N) < 1e-8
    ising = h.get_magnon_energies(nk=N, n=1, transverse=False,
                                  check_su2=False)
    assert ising[0].real > 1.0  # J1=1, U=3: the gap the rung removes


@pytest.mark.slow
def test_goldstone_of_an_itinerant_exchange_ferromagnet():
    """A metal, with metal=True, polarized along a generic axis. The
    Stoner continuum reaches zero here, so only the residual can say the
    collective mode is exactly at zero. Measured 2e-16."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    mf = h.copy()
    mf.add_exchange([0.3, 0.4, 0.5])
    nk = 200
    hm = VJinteraction(h, U=3.0, J1=-1.0, filling=41/400., mf=mf, nk=nk,
                       maxerror=1e-10, mix=0.2, maxite=3000).hamiltonian
    assert abs(hm.get_gap()) < 1e-6
    assert abs(hm.get_vev("sx")[0]) > 0.05
    assert hm.get_goldstone_residual(nk=nk, metal=True) < 1e-8


@pytest.mark.slow
def test_goldstone_of_a_non_collinear_exchange_magnet():
    """The 120-degree spiral of the triangular lattice with U=6 and J1=1,
    which has no spin-flip block in any frame. Measured 2.0e-10 with the
    rung and 2.3e-3 without."""
    g = geometry.triangular_lattice().get_supercell([3, 3])
    h = g.get_hamiltonian()

    def spiral(r):
        ph = 2*np.pi*(r[0] + 2*r[1])/3.
        return [np.cos(ph), np.sin(ph), 0.]

    mf = h.copy()
    mf.add_exchange(spiral)
    nk = 3
    hm = VJinteraction(h, U=6.0, J1=1.0, filling=0.5, mf=mf, nk=nk,
                       maxerror=1e-10, mix=0.2, maxite=3000).hamiltonian
    m = np.array([hm.get_vev("sx"), hm.get_vev("sy"), hm.get_vev("sz")]).T
    n = m/np.linalg.norm(m, axis=1)[:, None]
    assert 1 - np.min(np.abs(n@n[0])) > 0.4  # genuinely non-collinear
    assert hm.get_goldstone_residual(nk=nk) < 1e-8
    assert hm.get_goldstone_residual(nk=nk, transverse=False,
                                     check_su2=False) > 1e-4
