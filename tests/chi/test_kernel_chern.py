"""The Chern number of a collective mode from the null vectors of the kernel
K(q,w) = chi0(q,w)^-1 - V(q) of a response function (chitk/kernelchern.py).

The engine is checked on two kernels whose zeros and null vectors are known
exactly: w - H(q), whose null vectors are the Bloch eigenvectors, and
w - H0(q) - V (w - E_b)^-1 V, a Haldane model with one flat bath level per
site, whose zeros are the bands of the enlarged Hamiltonian
[[H0, V], [V, E_b]] and whose null vectors are the system part of its
eigenvectors. The bath there changes the Chern number of the lowest band
from 0 to 1, so the integer comes from the frequency-dependent
self-energy and not from H0.

The adaptors are checked against the routes they replace or reproduce: the
poles of the site-basis spin RPA of an onsite U against the magnon energies
of time-dependent Hartree-Fock (bsetk.spinflip.magnon_matrix), with a
sublattice imbalance that makes E(q) differ from E(-q), so that the sign
of the momentum is tested too; the null vector against the site-diagonal
spin-flip part of the TDHF eigenvector, which fixes the conjugation and so
the sign of the integer; and the responses against kondotk.momentchi and
the spinon mean field. An antiferromagnet in a staggered field is the case
with a de-excitation half, where dK/dw is indefinite: there the zero and
the null vector are checked against the eigenvector (X,Y) of the Casida
matrix, and the Y part is large enough for a wrong sign of it to fail."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk.spinflip import magnon_matrix
from pyqula.chitk import kernelchern as kc
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.spinon import SpinonHamiltonian
from pyqula.topologytk.plaquette import chern_from_states

Q = np.array([0.13, 0.29, 0.]) # a generic momentum, off every mesh used
SHIFT = (0.03, 0.07)


def _overlap(a, b):
    return abs(np.vdot(a, b))/np.linalg.norm(a)/np.linalg.norm(b)


# ---------------------------------------------------------------- engine


def _haldane(t2, mass=0.):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    if mass != 0.: h.add_sublattice_imbalance(mass)
    return h


@pytest.mark.parametrize("t2", [0.1, -0.1])
def test_linear_kernel_gives_the_bands_and_get_chern(t2):
    h = _haldane(t2)
    hk = h.get_hk_gen()
    def K(q, w): return w*np.identity(2) - hk(q)
    es, vs = kc.kernel_modes(K, Q, (-10., 10.))
    e0, u0 = np.linalg.eigh(hk(Q))
    assert np.allclose(es, e0, atol=1e-10)
    for n in range(2): assert _overlap(vs[n], u0[:, n]) > 1. - 1e-10
    c0 = kc.kernel_chern(K, (-10., 0.), nq=12)
    c1 = kc.kernel_chern(K, (0., 10.), nq=12)
    ref = h.get_chern(nk=20, write=False) # the lower band, occupied
    assert abs(round(ref)) == 1
    assert abs(c0 - round(ref)) < 1e-8
    assert abs(c1 + round(ref)) < 1e-8


# a Haldane model in its trivial phase (mass above 4.5 t2) with a bath
# level on each sublattice, whose self-energy pushes the lowest band
# into the topological one
BATH_E = np.array([1.0, 4.0])
BATH_V = np.array([1.0, 0.3])
BATH_WINDOW = (-5., -0.34) # the lowest band, below both bath levels


def _bath():
    h0 = _haldane(0.1, mass=0.6)
    hk0 = h0.get_hk_gen()
    def K(q, w):
        return w*np.identity(2) - hk0(q) - np.diag(BATH_V**2/(w - BATH_E))
    def enlarged(q):
        m = np.zeros((4, 4), dtype=np.complex128)
        m[:2, :2] = hk0(q)
        m[2:, 2:] = np.diag(BATH_E)
        m[:2, 2:] = m[2:, :2] = np.diag(BATH_V)
        return m
    def continuum(q): return (-np.inf, np.min(BATH_E)) # K is smooth there
    return h0, K, enlarged, continuum


def test_bath_kernel_is_the_enlarged_hamiltonian():
    h0, K, enlarged, continuum = _bath()
    es, vs = kc.kernel_modes(K, Q, BATH_WINDOW, continuum=continuum)
    ee, ue = np.linalg.eigh(enlarged(Q))
    assert len(es) == 1 and abs(es[0] - ee[0]) < 1e-10
    assert _overlap(vs[0], ue[:2, 0]) > 1. - 1e-10 # the system part
    c = kc.kernel_chern(K, BATH_WINDOW, nq=10, continuum=continuum)
    ref = chern_from_states(lambda q: np.linalg.eigh(enlarged(q))[1].T[0:1],
                            nq=10)
    assert round(ref) == 1 and abs(c - ref) < 1e-8
    # without the bath the lowest band is trivial, so the self-energy at the
    # frequency of the zero is what makes the integer
    assert abs(h0.get_chern(nk=16, write=False)) < 1e-6


def test_degenerate_zeros_are_one_group():
    """The two spin copies of a Haldane band are degenerate at every q:
    they come out as one group with twice the Chern number, and a group
    that takes one copy only is refused"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_haldane(0.1)
    hk = h.get_hk_gen()
    def K(q, w): return w*np.identity(4) - hk(q)
    es, vs = kc.kernel_modes(K, Q, (-10., 0.))
    assert len(es) == 2 and abs(es[1] - es[0]) < 1e-9
    assert abs(kc.kernel_chern(K, (-10., 0.), nq=10) - 2.) < 1e-8
    with pytest.raises(ValueError, match="degenerate"):
        kc.kernel_chern(K, (-10., 0.), bands=[0], nq=10)


def test_engine_raises_on_a_window_it_cannot_count():
    h0, K, enlarged, continuum = _bath()
    with pytest.raises(ValueError, match="continuum"): # contains a bath level
        kc.kernel_modes(K, Q, (-5., 1.5), continuum=continuum)
    with pytest.raises(ValueError, match="increases|infinity|not finite"):
        kc.kernel_modes(K, Q, (-5., 1.5)) # the same, found by the count
    with pytest.raises(ValueError, match="changes over the zone"):
        kc.kernel_chern(K, (-5., -1.5), nq=10, continuum=continuum)
    with pytest.raises(ValueError, match="not Hermitian"):
        kc.kernel_modes(lambda q, w: np.array([[w, 1.], [0., -w]]), Q,
                        (-1., 1.))


@pytest.mark.parametrize("w0", [0.3137, -0.2718, 0.1, 0.55])
def test_a_sign_change_through_infinity_is_not_a_mode(w0):
    """An eigenvalue 1/(w - w0) goes from minus to plus infinity at w0, so
    the count of negative eigenvalues decreases there by one as it does at
    a mode, and the count alone cannot refuse it: only the size of the
    eigenvalue at the root tells a pole from a zero, on whichever side of
    the pole the root search stops"""
    def K(q, w):
        with np.errstate(divide="ignore"):
            pole = np.float64(1.)/np.float64(w - w0)
        return np.array([[pole, 0.3], [0.3, w - 5.]])
    with pytest.raises(ValueError, match="infinity|not finite"):
        kc.kernel_modes(K, Q, (-1., 1.))


# ------------------------------------------------- the magnet of an onsite U

_magnets = {}


def magnet(kane_mele=0.15, imbalance=0.):
    """A ferromagnet saturated along -z by its Zeeman field, with an
    onsite U only, so that the site-basis RPA is time-dependent
    Hartree-Fock and there is no Goldstone mode"""
    key = (kane_mele, imbalance)
    if key not in _magnets:
        h = geometry.honeycomb_lattice().get_hamiltonian(tij=[1.0, 0.3])
        h.add_kane_mele(kane_mele)
        h.add_zeeman([0., 0., 3.0])
        if imbalance != 0.: h.add_sublattice_imbalance(imbalance)
        hmf = h.get_mean_field_hamiltonian(U=8.0, filling=0.5, mf="ferro",
                                           nk=8, maxerror=1e-10)
        assert np.allclose(hmf.get_vev("sz"), -1., atol=1e-8) # saturated
        _magnets[key] = hmf
    return _magnets[key]


# the two magnon bands are between 4.54 and 4.88 and between 5.05 and
# 5.95, and the spin-flip continuum starts at 7.08 or above
LOWER, UPPER, BOTH = (3.0, 4.96), (4.96, 7.0), (3.0, 7.0)


def _tdhf(hmf, q, nk):
    p = magnon_matrix(hmf, Q=q, nk=nk)
    assert p.n2 == 0 # no de-excitation half
    es, ws = np.linalg.eigh((p.M + p.M.conj().T)/2.)
    return p, es, ws


def _site_spin_flip(p, x):
    """The amplitude of the TDHF eigenvector x on S^+_i, the element
    (i up, i down) of the two-body wavefunction summed over k"""
    el, ho = p.pb.el[p.m1], p.pb.ho[p.m1]
    n = el.shape[1]//2
    return np.array([np.sum(x*el[:, 2*i]*np.conj(ho[:, 2*i+1]))
                     for i in range(n)])


def test_poles_are_the_tdhf_magnons_at_the_same_momentum():
    hmf = magnet(imbalance=0.5) # no inversion, so E(q) differs from E(-q)
    nk = 6
    K = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3)
    es, vs = kc.kernel_modes(K, Q, BOTH)
    p, ref, ws = _tdhf(hmf, Q, nk)
    assert len(es) == 2
    assert np.allclose(es, ref[:2], atol=1e-9)
    em, vm = kc.kernel_modes(K, -Q, BOTH)
    assert abs(em[0] - es[0]) > 1e-2 # so the sign of q is tested
    lo, hi = K.continuum(Q)
    assert abs(hi - np.min(p.pb.dE[p.m1])) < 1e-12 # the same pair set


def test_null_vector_is_the_site_part_of_the_tdhf_eigenvector():
    hmf = magnet()
    nk = 6
    K = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3)
    Kf = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3, channel="full")
    es, vs = kc.kernel_modes(K, Q, BOTH)
    ef, vf = kc.kernel_modes(Kf, Q, BOTH)
    assert np.allclose(ef, es, atol=1e-9)
    assert Kf.basis(Q).shape == (6, 4) # Sz of the two sites left out
    p, ref, ws = _tdhf(hmf, Q, nk)
    for n in range(2):
        phi = _site_spin_flip(p, ws[:, n])
        assert _overlap(vs[n], phi) > 1. - 1e-9
        assert _overlap(np.conj(vs[n]), phi) < 1. - 1e-3 # not its conjugate
        # the full channel: (Sx,Sy,Sz) on the two sites, Sz is out of the
        # range of chi0 and the mode is in the S^+ = Sx + i Sy component
        x, y, z = vf[n][0:2], vf[n][2:4], vf[n][4:6]
        assert np.linalg.norm(z) < 1e-10
        assert np.linalg.norm(x + 1j*y) < 1e-10
        assert _overlap(x - 1j*y, phi) > 1. - 1e-9


@pytest.mark.parametrize("nk,nq", [(6, 6), (8, 8)])
def test_magnon_chern_numbers(nk, nq):
    hmf = magnet()
    K = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3)
    c0 = kc.kernel_chern(K, LOWER, nq=nq, shift=SHIFT)
    c1 = kc.kernel_chern(K, UPPER, nq=nq, shift=SHIFT)
    # c0 is the measurement, fixed against the TDHF eigenvectors below. The
    # two null vectors are independent vectors of the two-dimensional space
    # of S^+ on the two sites, so c1 = -c0 for any such pair, and the two
    # bands together give 0 whatever the vectors are (not asserted, since
    # it cannot fail); c1 checks that the upper window holds the upper band
    assert abs(c0 - 1.) < 1e-8 and abs(c1 + 1.) < 1e-8
    # the selection of one band inside a window that holds both
    assert abs(kc.kernel_chern(K, BOTH, bands=[0], nq=nq, shift=SHIFT)
               - c0) < 1e-8
    assert abs(kc.kernel_chern(K, BOTH, bands=[1], nq=nq, shift=SHIFT)
               - c1) < 1e-8
    Kf = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3, channel="full")
    assert abs(kc.kernel_chern(Kf, LOWER, nq=nq, shift=SHIFT) - c0) < 1e-8


def test_magnon_chern_numbers_against_the_tdhf_eigenvectors():
    """The same integers from the site part of the TDHF eigenvectors, and
    the opposite sign under the opposite spin-orbit coupling"""
    nk, nq = 6, 6
    for km, sign in ((0.15, 1.), (-0.15, -1.)):
        hmf = magnet(kane_mele=km)
        K = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3)
        for band, window in ((0, LOWER), (1, UPPER)):
            def phi(q, band=band):
                p, es, ws = _tdhf(hmf, q, nk)
                return _site_spin_flip(p, ws[:, band])[None, :]
            ref = chern_from_states(phi, nq=nq, shift=SHIFT)
            c = kc.kernel_chern(K, window, nq=nq, shift=SHIFT)
            assert abs(c - ref) < 1e-8
            assert abs(c - sign*(1. if band == 0 else -1.)) < 1e-8


def test_magnet_along_plus_z_has_its_magnons_in_the_lowering_channel():
    """The time-reversed magnet: the same energies, now created by S^-, so
    the raising channel has no mode at positive frequency, and the
    opposite Chern numbers"""
    h = geometry.honeycomb_lattice().get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(0.15)
    h.add_zeeman([0., 0., -3.0])
    hmf = h.get_mean_field_hamiltonian(U=8.0, filling=0.5, mf="ferro", nk=8,
                                       maxerror=1e-10)
    assert np.allclose(hmf.get_vev("sz"), 1., atol=1e-8)
    nk, nq = 6, 6
    Kl = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3, channel="lowering")
    Kr = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3, channel="raising")
    ref = kc.kernel_modes(kc.spin_rpa_kernel(magnet(), nk=nk, T=1e-3),
                          -Q, BOTH)[0]
    assert np.allclose(kc.kernel_modes(Kl, Q, BOTH)[0], ref, atol=1e-9)
    assert len(kc.kernel_modes(Kr, Q, BOTH)[0]) == 0
    assert abs(kc.kernel_chern(Kl, LOWER, nq=nq, shift=SHIFT) + 1.) < 1e-8
    assert abs(kc.kernel_chern(Kl, UPPER, nq=nq, shift=SHIFT) - 1.) < 1e-8


def test_spin_kernel_refuses_what_it_does_not_take():
    hmf = magnet()
    with pytest.raises(ValueError, match="raising"):
        kc.spin_rpa_kernel(hmf, nk=4, T=1e-3, channel="transverse")
    with pytest.raises(ValueError, match="continuum"): # starts near 7.4
        kc.kernel_modes(kc.spin_rpa_kernel(hmf, nk=4, T=1e-3), Q, (3., 8.))
    for T in (-1e-3, None): # T=0, the default, is sharp occupations
        with pytest.raises(ValueError, match="temperature"):
            kc.spin_rpa_kernel(hmf, nk=4, T=T)
    h = hmf.copy()
    h.add_rashba(0.1) # mixes the two spins
    with pytest.raises(ValueError, match="conserves Sz"):
        kc.spin_rpa_kernel(h, nk=4, T=1e-3, channel="raising")
    # a vertex that couples Sx to Sz is not a block in the raising channel
    V = np.kron(np.array([[1., 0., 0.3], [0., 1., 0.], [0.3, 0., 1.]]),
                -16.*np.identity(2))
    K = kc.LocalSpinKernel(hmf, "raising", [0, 1], lambda q: V, 4, 1e-3)
    with pytest.raises(ValueError, match="couples the channel"):
        K(Q, 4.0)


def test_a_fermi_surface_is_a_continuum_down_to_zero():
    """A metal at q=0: the spin flips inside a band are degenerate pairs,
    which have no pole at finite frequency but stand for the continuum
    that starts at zero, so no window is accepted. At a temperature far
    below the level spacing of the mesh no state is partially filled, and
    what is left is the interband edge of the finite mesh"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_onsite(0.8) # the Fermi energy inside the upper band
    K = kc.LocalSpinKernel(h, "raising", [0, 1], None, 6, 0.05)
    assert K.continuum(np.zeros(3)) == (0., 0.)
    with pytest.raises(ValueError, match="continuum"):
        kc.kernel_modes(K, np.zeros(3), (0.5, 0.9))
    K = kc.LocalSpinKernel(h, "raising", [0, 1], None, 6, 1e-3)
    assert K.continuum(np.zeros(3))[1] > 1.


def test_sharp_occupations_give_the_zeros_of_a_small_temperature():
    """T=0, the default, takes f=1 and f=0 on the two sides of the Fermi
    energy. On the gapped magnet the occupations at T=1e-3 are already 0
    and 1 to machine precision, so the zeros are the same, and they are
    the TDHF energies, whose pair basis takes sharp occupations too"""
    hmf = magnet()
    nk = 6
    K0 = kc.spin_rpa_kernel(hmf, nk=nk)
    K3 = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3)
    assert K0.T == 0.
    for q in (Q, np.array([0.41, 0.07, 0.]), np.zeros(3)):
        e0, v0 = kc.kernel_modes(K0, q, BOTH)
        e3, v3 = kc.kernel_modes(K3, q, BOTH)
        p, ref, ws = _tdhf(hmf, q, nk)
        assert len(e0) == 2 and np.max(np.abs(e0 - e3)) < 1e-10
        assert np.max(np.abs(e0 - ref[:2])) < 1e-9
        for n in range(2): assert _overlap(v0[n], v3[n]) > 1. - 1e-10


def test_sharp_occupations_at_the_fermi_energy():
    """T=0 gives a state at the Fermi energy the occupation 1/2, the limit
    of a small T. First the half-filled honeycomb lattice on a mesh that
    holds K, where four states sit at the Fermi energy: their pairs with
    each other are a Fermi surface, so the continuum starts at zero at
    q=0. Then a gapped honeycomb lattice shifted so that one state, the top
    of one spin band at K, sits at the Fermi energy: away from q=0 its
    pairs with the states at K - q enter with the weight 1/2, which a
    sharp 0 or 1 would change"""
    K_ = np.array([1./3., 1./3., 0.])
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    e = np.linalg.eigvalsh(h.get_hk_gen()(K_))
    assert np.sum(np.abs(e) < 1e-10) == 4 # the Dirac point is on the mesh
    K0 = kc.LocalSpinKernel(h, "raising", [0, 1], None, 6, 0.)
    Kt = kc.LocalSpinKernel(h, "raising", [0, 1], None, 6, 1e-6)
    assert K0.continuum(np.zeros(3)) == (0., 0.)
    assert Kt.continuum(np.zeros(3)) == (0., 0.)
    h.add_sublattice_imbalance(0.5)
    h.add_zeeman([0., 0., 0.2])
    e = np.linalg.eigvalsh(h.get_hk_gen()(K_))
    h.add_onsite(-np.max(e[e < 0.])) # the top of one spin band to zero
    e = np.linalg.eigvalsh(h.get_hk_gen()(K_))
    assert np.sum(np.abs(e) < 1e-10) == 1 and np.min(np.abs(e[e != e[np.argmin(
        np.abs(e))]])) > 0.3 # one state at the Fermi energy, alone
    K0 = kc.LocalSpinKernel(h, "raising", [0, 1], None, 6, 0.)
    Kt = kc.LocalSpinKernel(h, "raising", [0, 1], None, 6, 1e-6)
    q = np.array([1./6., 0., 0.]) # pairs the state at K with those at K - q
    c0, ct = K0.bare(q, -0.25), Kt.bare(q, -0.25)
    assert np.max(np.abs(c0)) > 0.1
    assert np.max(np.abs(c0 - ct)) < 1e-9




def test_bands_must_be_a_contiguous_group_of_zeros():
    hmf = magnet()
    K = kc.spin_rpa_kernel(hmf, nk=4)
    with pytest.raises(ValueError, match="contiguous"):
        kc.kernel_chern(K, BOTH, bands=[0, 2], nq=4, shift=SHIFT)
    with pytest.raises(ValueError, match="indices of the 2 zeros"):
        kc.kernel_chern(K, BOTH, bands=[2], nq=4, shift=SHIFT)


def test_the_record_holds_the_zeros_the_gap_and_the_bound():
    """full_output=True: the energies are the zeros of the group at the
    points of the mesh, gap is the distance of the group to the other zero
    of the window or to an end of it, bound the distance of the top of the
    group below the continuum, and a kernel with no continuum has none"""
    hmf = magnet()
    nk, nq = 4, 4
    K = kc.spin_rpa_kernel(hmf, nk=nk)
    for bands in ([0], [1], None):
        r = kc.kernel_chern(K, BOTH, bands=bands, nq=nq, shift=SHIFT,
                            full_output=True)
        assert abs(r.chern - kc.kernel_chern(K, BOTH, bands=bands, nq=nq,
                                             shift=SHIFT)) < 1e-14
        assert r.qs.shape == (nq, nq, 3)
        assert np.allclose(r.qs[2, 1], [2./nq + SHIFT[0], 1./nq + SHIFT[1],
                                        0.])
        e = np.array([[kc.kernel_modes(K, q, BOTH)[0] for q in row]
                      for row in r.qs])
        edge = np.array([[K.continuum(q)[1] for q in row] for row in r.qs])
        sel = [0, 1] if bands is None else bands
        assert np.max(np.abs(r.energies - e[:, :, sel])) < 1e-12
        ends = np.concatenate([np.full((nq, nq, 1), BOTH[0]), e,
                               np.full((nq, nq, 1), BOTH[1])], axis=2)
        gap = min(np.min(ends[:, :, sel[0]+1] - ends[:, :, sel[0]]),
                  np.min(ends[:, :, sel[-1]+2] - ends[:, :, sel[-1]+1]))
        assert abs(r.gap - gap) < 1e-12
        assert abs(r.bound - np.min(edge - e[:, :, sel[-1]])) < 1e-12
        assert abs(r.max_flux - np.max(np.abs(r.fluxes))) < 1e-15
        # tol is the smallest link overlap accepted: above the one of the
        # record the same mesh is refused (the two bands together span the
        # whole space of S^+ on the two sites, so their overlap is 1)
        if len(sel) == 1:
            assert r.min_overlap < 1. - 1e-6
            with pytest.raises(ValueError, match="almost orthogonal"):
                kc.kernel_chern(K, BOTH, bands=bands, nq=nq, shift=SHIFT,
                                tol=0.5*(1. + r.min_overlap))
    h = _haldane(0.1)
    hk = h.get_hk_gen()
    r = kc.kernel_chern(lambda q, w: w*np.identity(2) - hk(q), (-10., 0.),
                        nq=6, full_output=True)
    assert r.bound is None and abs(abs(r.chern) - 1.) < 1e-8


# ------------------------------- an antiferromagnet: a de-excitation half


def test_antiferromagnet_zero_is_the_casida_mode():
    """With a staggered field the Neel state has its Goldstone mode gapped,
    and the Casida matrix of magnon_matrix has a de-excitation half, so
    the transitions of the kernel have both signs of f_a - f_b and dK/dw
    is indefinite. The TDHF block is the one in which S^- creates the
    mode, the lowering channel. The zero is its lowest positive energy,
    and the null vector is the site part of the lifted eigenvector: the
    resonant part X on the pairs (el, ho) plus the antiresonant part Y on
    the pairs (hoA, elA), an occupied state at k+Q and an empty one at k"""
    h = geometry.honeycomb_lattice().get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(0.15)
    h.add_antiferromagnetism(0.3)
    h.add_sublattice_imbalance(0.4)
    hmf = h.get_mean_field_hamiltonian(U=6.0, filling=0.5, mf="antiferro",
                                       nk=8, maxerror=1e-10)
    sz = hmf.get_vev("sz")
    assert abs(sz[0] + sz[1]) < 1e-8 and abs(sz[0]) > 0.5
    nk = 6
    K = kc.spin_rpa_kernel(hmf, nk=nk, T=1e-3, channel="lowering")
    for q in (Q, np.array([0.03, 0.07, 0.])):
        p = magnon_matrix(hmf, Q=q, nk=nk)
        assert p.n2 > 0
        w, c = np.linalg.eig(p.M)
        assert np.max(np.abs(w.imag)) < 1e-9 # a stable reference
        i = np.argmin(np.where(w.real > 0., w.real, np.inf))
        x, y = c[:p.n1, i], c[p.n1:, i]
        assert np.linalg.norm(y) > 0.1*np.linalg.norm(x) # Y matters
        el, ho = p.pb.el[p.m1], p.pb.ho[p.m1]
        elA, hoA = p.pb.elA[p.m2], p.pb.hoA[p.m2]
        phi = np.array([np.sum(x*el[:, 2*s+1]*np.conj(ho[:, 2*s]))
                        + np.sum(y*hoA[:, 2*s+1]*np.conj(elA[:, 2*s]))
                        for s in range(2)]) # the element (s down, s up)
        lo, hi = K.continuum(q)
        assert abs(hi - np.min(p.pb.dE[p.m1])) < 1e-12
        es, vs = kc.kernel_modes(K, q, (1e-3, 0.8*hi))
        assert len(es) == 1 and abs(es[0] - w[i].real) < 1e-9
        assert _overlap(vs[0], phi) > 1. - 1e-9


# ------------------------------------------------------ the Kondo lattice

_kondos = {}


def kondo(t2):
    """A honeycomb Kondo insulator, two moments per cell, with a Haldane
    term on the conduction electrons that breaks time reversal"""
    if t2 not in _kondos:
        hc = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
        hc.add_haldane(t2)
        h = KondoLatticeHamiltonian(hc)
        n = len(h._kondo_pairs)
        _kondos[t2] = h.get_mean_field_hamiltonian(J=4.0, filling=0.5, nk=12,
                mf=([1.0+0j]*n, [0.0]*n), maxerror=1e-8, maxite=4000)
    return _kondos[t2]


def test_moment_kernel_is_moment_spinchi():
    h = kondo(0.2)
    U = np.array([[-2.0, 0.3], [0.3, -2.0]])
    nk, w = 10, 1.2 # below the continuum, which starts at 1.69
    q = np.array([0.3, 0.2, 0.]) # on the mesh, where the two sums coincide
    K = kc.moment_rpa_kernel(h, U=U, nk=nk, T=1e-3)
    e, chi = h.get_moment_spinchi(q=-q, energies=[w], U=U, nk=nk,
                                  delta=1e-8, T=1e-3)
    e, chi0 = h.get_moment_spinchi(q=-q, energies=[w], nk=nk, delta=1e-8,
                                   T=1e-3)
    assert np.max(np.abs(chi[0])) > 0.1
    assert np.allclose(K.bare(q, w), chi0[0], atol=1e-6)
    assert np.allclose(K.response(q, w), chi[0], atol=1e-6)
    # a residual interaction that is not even in q, so that the kernel at
    # the momentum q of the mode has to evaluate it at -q
    def Uq(k):
        c = 0.3*np.exp(2j*np.pi*k[0])
        return np.array([[-2.0, c], [np.conj(c), -2.0]])
    assert np.max(np.abs(Uq(q) - Uq(-q))) > 0.1
    K = kc.moment_rpa_kernel(h, U=Uq, nk=nk, T=1e-3)
    e, chi = h.get_moment_spinchi(q=-q, energies=[w], U=Uq, nk=nk,
                                  delta=1e-8, T=1e-3)
    assert np.allclose(K.response(q, w), chi[0], atol=1e-6)


# with U' = -4 one spin exciton is bound below the continuum, between 0.60
# and 1.36, and the second one starts at 1.51 (at K)
KONDO_WINDOW = (0.01, 1.43)


@pytest.mark.parametrize("t2,nk,nq,sign", [(0.2, 8, 6, 1.), (0.2, 10, 8, 1.),
                                           (-0.2, 8, 6, -1.)])
def test_kondo_spin_exciton_chern_number(t2, nk, nq, sign):
    K = kc.moment_rpa_kernel(kondo(t2), U=-4.0, nk=nk, T=1e-3)
    c = kc.kernel_chern(K, KONDO_WINDOW, nq=nq, shift=SHIFT)
    assert abs(c - sign) < 1e-8


def test_kondo_spin_exciton_is_a_triplet():
    """Spin rotation symmetry: the exciton has the same energy and the
    same null vector in the raising channel, in the lowering one and in
    each of the three components of the full one. So the lowering channel
    gives the same Chern number and the triplet of the full channel three
    times it. The triplet needs the finer mesh, since its plaquette flux
    is three times the one of a single mode: it reads 0, 2 and 1 at nq=6,
    8 and 10, and 3 at nq=12 and 16. The largest flux of the single mode
    is 0.30 pi at nq=12 and 0.19 pi at nq=16, so nq=16 keeps the triplet
    away from the branch cut at pi"""
    h = kondo(0.2)
    nk = 6
    K = {c: kc.moment_rpa_kernel(h, U=-4.0, nk=nk, T=1e-3, channel=c)
         for c in kc.get_channel_names()}
    er, vr = kc.kernel_modes(K["raising"], Q, KONDO_WINDOW)
    el, vl = kc.kernel_modes(K["lowering"], Q, KONDO_WINDOW)
    ef, vf = kc.kernel_modes(K["full"], Q, KONDO_WINDOW)
    assert len(er) == 1 and len(el) == 1 and len(ef) == 3
    assert abs(el[0] - er[0]) < 1e-8 and np.allclose(ef, er[0], atol=1e-8)
    assert _overlap(vl[0], vr[0]) > 1. - 1e-9
    # the null space of the full channel is psi (x) C^3, with the component
    # c of the site i at the index c*2 + i
    P = vf.T@np.linalg.inv(vf.conj()@vf.T)@vf.conj()
    for comp in np.identity(3):
        t = np.kron(comp, vr[0])/np.linalg.norm(vr[0])
        assert abs(np.vdot(t, P@t) - 1.) < 1e-8
    c = kc.kernel_chern(K["raising"], KONDO_WINDOW, nq=6, shift=SHIFT)
    assert abs(c - 1.) < 1e-8
    assert abs(kc.kernel_chern(K["lowering"], KONDO_WINDOW, nq=6,
                               shift=SHIFT) - c) < 1e-8
    assert abs(kc.kernel_chern(K["full"], KONDO_WINDOW, nq=16, shift=SHIFT)
               - 3.*c) < 1e-8


def test_a_touching_on_the_default_mesh_is_refused():
    """With no Haldane term the two spin excitons touch at K and K'
    (1.53057), and the second one goes into the continuum at Gamma, so no
    window holds a fixed number of zeros. An unshifted mesh that holds K,
    nq=6 or the default nq=12, refuses the lowest band as degenerate with
    the second one; nq=10 misses K and returns an integer with no error,
    which is why the default is 12. The gap of its record, taken on the
    points of the mesh, is 0.14 there, as large as the one of the gapped
    case with t2=0.2, so only a second nq tells"""
    K = kc.moment_rpa_kernel(kondo(0.0), U=-4.0, nk=8)
    window = (0.01, 1.5661) # above the touching, below the continuum
    es = kc.kernel_modes(K, [1./3., 1./3., 0.], window)[0]
    assert len(es) == 2 and abs(es[1] - es[0]) < 1e-9
    for kw in (dict(nq=6), dict()):
        with pytest.raises(ValueError, match="degenerate"):
            kc.kernel_chern(K, window, bands=[0], **kw)
    r = kc.kernel_chern(K, window, bands=[0], nq=10, full_output=True)
    assert abs(r.chern - round(r.chern)) < 1e-8
    assert r.gap > 0.1 # the gap on the mesh does not see it either


def test_moment_magnon_chern_is_the_kernel_chern_of_the_moments():
    """KondoLatticeHamiltonian.get_moment_magnon_chern delegates to
    moment_rpa_chern, which is kernel_chern of moment_rpa_kernel, and both
    refuse a lattice that is not two-dimensional"""
    h = kondo(0.2)
    kw = dict(nq=6, shift=SHIFT, full_output=True)
    ref = kc.kernel_chern(kc.moment_rpa_kernel(h, U=-4.0, nk=6),
                          KONDO_WINDOW, **kw)
    out = [kc.moment_rpa_chern(h, KONDO_WINDOW, U=-4.0, nk=6, **kw),
           h.get_moment_magnon_chern(KONDO_WINDOW, U=-4.0, nk=6, **kw)]
    assert abs(ref.chern - 1.) < 1e-8 and ref.bound > 0.1
    for r in out: # the same zeros, so every keyword reaches the kernel
        assert r.chern == ref.chern and r.bound == ref.bound
        assert np.max(np.abs(r.energies - ref.energies)) < 1e-14
    # and the defaults of the method are those of the kernel
    a = h.get_moment_magnon_chern(KONDO_WINDOW, U=-4.0, nq=6, shift=SHIFT,
                                  full_output=True)
    b = kc.kernel_chern(kc.moment_rpa_kernel(h, U=-4.0), KONDO_WINDOW, **kw)
    assert kc.moment_rpa_kernel(h, U=-4.0).T == 0. # sharp occupations
    assert np.max(np.abs(a.energies - b.energies)) < 1e-14
    hc = geometry.chain().get_hamiltonian(has_spin=True)
    chain = KondoLatticeHamiltonian(hc)
    for call in (lambda: kc.moment_rpa_chern(chain, KONDO_WINDOW, U=-4.0),
                 lambda: chain.get_moment_magnon_chern(KONDO_WINDOW)):
        with pytest.raises(ValueError, match="two-dimensional"):
            call()


# ------------------------------------------------------- the spinon RPA


def test_spinon_bond_kernel_is_get_spinchi_full():
    """The kernel of a gapped chiral spinon state, whose vertex is the
    exchange J(q), against the dressed response of the spinon mean field;
    and the uniform S^+ of the spin singlet, which couples to no
    transition, is left out of the kernel at q=0"""
    g = geometry.honeycomb_lattice()
    ansatz = g.get_hamiltonian(has_spin=True)
    ansatz.add_haldane(0.2) # holds imaginary second-neighbor hoppings
    h = SpinonHamiltonian(g).get_mean_field_hamiltonian(J1=1.0, J2=0.5,
            decoupling="bond", ansatz=ansatz, mix=0.5, maxerror=1e-7,
            T=0.01, nk=12)
    nk = 12
    K = kc.spin_rpa_kernel(h, nk=nk, T=1e-3)
    q = np.array([1./6., 1./12., 0.]) # on the mesh
    lo, hi = K.continuum(q)
    assert hi > 0.3 # gapped spinons
    w = 0.5*hi
    e, chi = h.get_spinchi_full(q=-q, energies=[w], nk=nk, delta=1e-8, T=1e-3)
    c = chi[0]
    x, y = slice(0, 2), slice(2, 4)
    ref = c[x, x] + c[y, y] - 1j*c[x, y] + 1j*c[y, x] # S^+ S^-
    assert np.max(np.abs(ref)) > 0.1
    assert np.allclose(K.response(q, w), ref, atol=1e-6)
    B = K.basis(np.zeros(3))
    assert B is not None and B.shape == (2, 1)
    assert abs(np.vdot(np.ones(2)/np.sqrt(2.), B[:, 0])) < 1e-8
