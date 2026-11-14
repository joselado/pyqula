"""The Chern number of the magnons of a saturated ferromagnet from the
winding of the site-basis spin RPA kernel (chitk/responsewinding.py),
against two routes that do not use the contour: the null vectors of the
Hermitian kernel at the real mode energies, and the TDHF eigenvectors of
bsetk/spinflip.py on the local spin flips. The magnet has an onsite U only,
so the site-basis RPA and the TDHF describe the same modes, and a Zeeman
field, so there is no Goldstone mode."""
import numpy as np
import pytest
from scipy.optimize import brentq

from pyqula import geometry
from pyqula.chitk import localspinchi
from pyqula.chitk import responsewinding as rw
from pyqula.topologytk import greenwinding as gw
from pyqula.topologytk.plaquette import chern_from_states

NK = 8
LOWER = [4.0, 4.964] # bands 4.547 to 4.832 and 5.096 to 5.946 at NK=8
BOTH = [4.0, 6.687] # the continuum starts at 7.43


@pytest.fixture(scope="module")
def hmf():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(0.15)
    h.add_zeeman([0., 0., 3.0])
    return h.get_mean_field_hamiltonian(U=8.0, filling=0.5, mf="ferro",
                                        nk=NK, maxerror=1e-10)


def _direct_chi0(h, q, z, T=1e-3):
    """chi0 of (Sx,Sy,Sz)/2 at a complex z from its Lehmann sum"""
    n = len(h.geometry.r)
    sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
           np.array([[1, 0], [0, -1]])]
    ops = []
    for a in range(3):
        for i in range(n):
            m = np.zeros((2*n, 2*n), dtype=np.complex128)
            m[2*i:2*i+2, 2*i:2*i+2] = sig[a]/2.
            ops.append(m)
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=NK)
    out = 0.
    for k in ks:
        e1, u1 = np.linalg.eigh(hk(k))
        e2, u2 = np.linalg.eigh(hk(np.array(k) + q))
        fac = np.subtract.outer(0.5*(1. - np.tanh(e1/(2.*T))),
                                0.5*(1. - np.tanh(e2/(2.*T))))
        X = np.array([np.conj(u1.T)@o@u2 for o in ops])
        out = out + np.einsum("amn,bmn,mn->ab", X, np.conj(X),
                              fac/(np.subtract.outer(e1, e2) - z))
    return out/len(ks)


def test_complex_bubble_is_the_analytic_continuation(hmf):
    q = np.array([0.13, 0.27, 0.])
    x, y = 4.7, 0.3
    # spin_chi at energies=x, delta=y is chi0 in the lower half plane
    _, c = localspinchi.bare_spin_chi(hmf, q=q, energies=[x], nk=NK,
                                      delta=y, T=1e-3)
    assert np.max(np.abs(c[0] - _direct_chi0(hmf, q, x - 1j*y))) < 1e-12
    assert np.max(np.abs(c[0] - _direct_chi0(hmf, q, x + 1j*y))) > 1e-4
    zs = np.array([x + 1j*y, x - 1j*y, 5.1 + 1j*y])
    calls = []
    def counting(h, energies=None, **kwargs):
        calls.append(list(energies))
        return bare(h, energies=energies, **kwargs)
    bare = localspinchi.bare_spin_chi
    localspinchi.bare_spin_chi = counting
    try: out = rw.bare_spin_chi_complex(hmf, q, zs, nk=NK)
    finally: localspinchi.bare_spin_chi = bare
    for (z, m) in zip(zs, out):
        assert np.max(np.abs(m - _direct_chi0(hmf, q, z))) < 1e-12
    # a frequency and its mirror image share one evaluation of the bubble
    assert calls == [[4.7, 5.1]]


def test_full_spin_basis_is_singular_in_a_collinear_magnet(hmf):
    """The zz block of chi0 vanishes in a magnet collinear along z, so the
    response kernel of the full channel raises and names the two remedies,
    while the default Dyson kernel takes it"""
    with pytest.raises(ValueError, match="kernel='dyson'.*'raising' or "
                       "'lowering'"):
        rw.spin_response_chern(hmf, LOWER, nq=4, nz=2, nk=NK,
                               kernel="response")
    assert np.isfinite(rw.spin_response_chern(hmf, LOWER, nq=4, nz=2, nk=NK))
    with pytest.raises(ValueError, match="dyson, response"):
        rw.spin_kernel(hmf, kernel="inverse")
    with pytest.raises(ValueError, match="'raising', 'lowering', 'full'"):
        rw.spin_kernel(hmf, channel="xy")
    spinless = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError, match="spinful"):
        rw.spin_kernel(spinless)
    chain = geometry.chain().get_hamiltonian()
    chain = chain.get_mean_field_hamiltonian(U=4.0, filling=0.5, mf="ferro",
                                             nk=8, maxerror=1e-8)
    with pytest.raises(ValueError, match="two-dimensional"):
        rw.spin_response_chern(chain, LOWER, nq=4, nz=2, nk=8)


def test_a_channel_that_is_not_a_block_raises(hmf):
    """A term that mixes the spins couples S^+ to the other components in
    chi0, and a vertex that couples Sx to Sz does in V, so the raising
    channel is not a block of the response there; the full one takes both"""
    zs = np.array([4.5 + 0.2j])
    q = np.array([0.13, 0.27, 0.])
    h = hmf.copy()
    h.add_rashba(0.3)
    ginv, _ = rw.spin_kernel(h, nk=4, channel="raising")
    with pytest.raises(ValueError, match="chi0 couples the channel"):
        ginv(zs, q)
    assert np.all(np.isfinite(rw.spin_kernel(h, nk=4)[0](zs, q)))
    from pyqula.chitk.spinchi import _full_spin_U
    V = _full_spin_U(hmf)
    M = np.kron(np.array([[1., 0., 0.3], [0., 1., 0.], [0.3, 0., 1.]]),
                np.identity(2))
    ginv, _ = rw.spin_kernel(hmf, V={d: M@m@M for (d, m) in V.items()},
                             nk=4, channel="raising")
    with pytest.raises(ValueError, match="the vertex couples the channel"):
        ginv(zs, q)


def test_the_channels_are_those_of_route_2(hmf):
    """On the magnet polarized along -z the raising channel holds the
    magnons and gives the integer of route 2 in the same channel, the
    lowering one has no mode at positive frequency and gives zero, and the
    default, the Dyson kernel in the full channel, gives the raising one"""
    from pyqula.chitk import kernelchern as kc
    c2 = kc.kernel_chern(kc.spin_rpa_kernel(hmf, nk=NK), LOWER, nq=6,
                         shift=(0.03, 0.07))
    assert abs(c2 - 1.) < 1e-8
    kw = dict(nk=NK, shift=(0.03, 0.07))
    cr = rw.spin_response_chern(hmf, LOWER, channel="raising", nq=8, nz=4,
                                **kw)
    assert abs(cr - c2) < 2e-2 # 0.9951 measured
    # the same contour for the three, a coarse one, since they are compared
    # with each other and not with the integer
    kw.update(nq=6, nz=2)
    cr = rw.spin_response_chern(hmf, LOWER, channel="raising", **kw)
    cl = rw.spin_response_chern(hmf, LOWER, channel="lowering", **kw)
    cf = rw.spin_response_chern(hmf, LOWER, **kw)
    assert abs(cl) < 1e-6 and abs(cr) > 0.5
    assert abs(cf - cr) < 1e-6


def test_bare_response_is_negative_definite_below_the_continuum(hmf):
    """Every transition is above the two windows, so the transverse chi0 is
    Hermitian and negative definite there and has no zero inside them
    (-0.024 is its largest eigenvalue on an 8 x 8 mesh of q)"""
    ws = np.linspace(BOTH[0], BOTH[1], 6) + 0j
    for q in ([0., 0., 0.], [1./3., 1./3., 0.], [0.5, 0., 0.],
              [0.13, 0.27, 0.]):
        chi = rw.bare_spin_chi_complex(hmf, q, ws, nk=NK)[:, :4, :4]
        assert np.max(np.abs(chi - np.conj(np.transpose(chi, (0, 2, 1))))) < 1e-12
        assert np.max(np.linalg.eigvalsh(chi)) < -1e-3


def _null_vector(ginv, q, wa, wb):
    """The energy of the mode in [wa, wb] and the null vector of the
    Hermitian kernel there"""
    K = lambda w: ginv(np.array([w + 0j]), q)[0]
    w = brentq(lambda w: np.linalg.det(K(w)).real, wa, wb, xtol=1e-12)
    e, u = np.linalg.eigh(K(w))
    return w, u[:, np.argmin(np.abs(e))]


def _tdhf_amplitude(h, Q):
    """The energy of the lowest TDHF magnon at Q and its amplitude on the
    local spin flip S^+ = Sx + i Sy of every site, <0|A_Q|mode> with
    A_Q = sum_k c^dag_k A^dag c_(k+Q)"""
    from pyqula.bsetk.spinflip import magnon_matrix
    p = magnon_matrix(h, Q=Q, nk=NK)
    assert p.n2 == 0 # saturated: no de-excitation half
    e, X = np.linalg.eigh((p.M + np.conj(p.M.T))/2.)
    el, ho = p.pb.el[p.m1], p.pb.ho[p.m1]
    n = len(h.geometry.r)
    out = []
    for i in range(n):
        A = np.zeros((2*n, 2*n), dtype=np.complex128)
        A[2*i, 2*i+1] = 1. # S^+ of the site i
        me = np.einsum("pa,ab,pb->p", np.conj(el), A, ho)
        out.append(np.sum(X[:, 0]*np.conj(me)))
    return e[0], np.array(out)


@pytest.fixture(scope="module")
def integers(hmf):
    """(c_null, c_tdhf): the plaquette Chern numbers of the lower magnon
    band from two routes that use no contour, the null vectors of the
    response kernel of the raising channel and the TDHF amplitudes on the
    local spin flips. Both are the same at nq=6, 8, 10 and 12, with and
    without a shift of the mesh"""
    ginv, hermitian = rw.spin_kernel(hmf, nk=NK, channel="raising",
                                     kernel="response")
    assert hermitian
    c_null = chern_from_states(
            lambda q: _null_vector(ginv, q, *LOWER)[1][None], nq=6)
    c_tdhf = chern_from_states(lambda q: _tdhf_amplitude(hmf, q)[1][None],
                               nq=6)
    assert abs(round(c_null)) == 1 # not a trivial band
    assert round(c_tdhf) == -round(c_null) # the conjugate bundle
    return round(c_null), round(c_tdhf)


@pytest.mark.slow
def test_magnon_band_winding_is_the_tdhf_integer(hmf, integers):
    """spin_response_chern against the plaquette sum of the TDHF
    amplitudes, which fixes its sign with no recorded number"""
    c_tdhf = integers[1]
    # the response kernel takes the upper half of the contour only, so it
    # is the cheaper one for the meshes, and the default is compared with
    # it at the end
    kw = dict(nz=4, nk=NK, channel="raising", kernel="response")
    c8 = rw.spin_response_chern(hmf, LOWER, nq=8, **kw)
    c6 = rw.spin_response_chern(hmf, LOWER, nq=6, **kw)
    # 1.0024 measured, 0.99994 at nq=20. The approach is not monotonic,
    # 1.023 at nq=9 and 0.986 at nq=10, and the tolerance is the largest
    # error from nq=8 on, not the one of this mesh
    assert abs(c8 - c_tdhf) < 3e-2
    # not quantized on a mesh: 1.166 at nq=6
    assert abs(c8 - c_tdhf) < abs(c6 - c_tdhf)/3.
    # 6.5e-4 measured at this nz, -3e-7 at nz=8
    assert abs(rw.spin_response_chern(hmf, BOTH, nq=8, **kw)) < 5e-3
    # the zeros of chi0, if any, carry no net Chern number in the window
    kw["kernel"] = "bare"
    assert abs(rw.spin_response_chern(hmf, LOWER, nq=4, **kw)) < 1e-8
    # and 1 - chi0 V, the default, wound on both halves of the contour,
    # agrees with chi0^-1 - V on the same mesh (1e-5 apart at nq=8, nz=8)
    kw["kernel"] = "dyson"
    cd = rw.spin_response_chern(hmf, LOWER, nq=6, **kw)
    assert abs(cd - c6) < 1e-2


def test_winding_is_the_chern_number_of_the_null_vectors(hmf, integers):
    """The null vector of K at -Q is the conjugate of the TDHF amplitude at
    Q, and the winding of K is the Chern number of the null vectors, so
    minus the winding is the one of the TDHF amplitude"""
    c_null = integers[0]
    ginv, hermitian = rw.spin_kernel(hmf, nk=NK, channel="raising",
                                     kernel="response")
    # a Q of the k-mesh: the bubble at -Q and the TDHF at Q are the same sum
    Q = np.array([1./NK, 2./NK, 0.])
    wk, v = _null_vector(ginv, -Q, *LOWER)
    et, w = _tdhf_amplitude(hmf, Q)
    assert abs(wk - et) < 1e-9 # 1e-14 measured
    assert abs(np.vdot(v, np.conj(w)))/np.linalg.norm(w) > 1. - 1e-8
    assert abs(np.vdot(v, w))/np.linalg.norm(w) < 1. - 1e-3 # not w itself
    # off the mesh the two sample the occupied states on meshes shifted by
    # Q, so they agree only as well as nk is converged (7e-6 measured)
    Q = np.array([0.13, 0.27, 0.])
    wk, v = _null_vector(ginv, -Q, *LOWER)
    et, w = _tdhf_amplitude(hmf, Q)
    assert 1e-9 < abs(wk - et) < 1e-4
    assert abs(np.vdot(v, np.conj(w)))/np.linalg.norm(w) > 1. - 1e-8
    winding = gw.window_chern(ginv, LOWER, nq=8, nz=4, hermitian=True,
                              batched=True)
    assert abs(winding - c_null) < 3e-2 # 2.4e-3 measured, see above


def test_a_bubble_that_is_not_finite_raises(hmf, monkeypatch):
    """On the real axis the bubble is 0/0 at the energy difference of a
    pair of states with equal occupations (seen on a Lieb ferrimagnet at
    Q=0), which bare_spin_chi_complex reports instead of returning it"""
    def broken(h, q=None, energies=None, **kwargs):
        chi = np.zeros((len(energies), 6, 6), dtype=np.complex128)
        chi[-1, 0, 0] = np.nan
        return energies, chi
    monkeypatch.setattr(localspinchi, "bare_spin_chi", broken)
    with pytest.raises(ValueError, match="not finite at z=\\(5.1"):
        rw.bare_spin_chi_complex(hmf, [0., 0., 0.], [4.7, 5.1], nk=NK)



def test_arguments_reach_the_engine_and_the_vertex(hmf):
    """spin_response_chern is minus the winding of the kernel with the same
    contour, mesh and derivative, the vertex V= enters as chi0^-1 - V, in
    the raising channel as the vertex of route 2 there, and the flag of
    each kernel says whether it has the conjugation symmetry"""
    from pyqula.chitk import kernelchern as kc
    from pyqula.chitk.spinchi import _full_spin_U
    ginv, hermitian = rw.spin_kernel(hmf, nk=NK) # the default, Dyson, full
    assert not hermitian
    kw = dict(nq=4, nz=2, height=0.2, shift=(0.03, 0.07),
              derivative="central")
    c = rw.spin_response_chern(hmf, LOWER, nk=NK, **kw)
    w = gw.window_chern(ginv, LOWER, hermitian=False, batched=True, **kw)
    assert abs(c + w) < 1e-12
    for key in ("height", "shift", "derivative"): # each one is used
        other = {k: v for (k, v) in kw.items() if k != key}
        c1 = rw.spin_response_chern(hmf, LOWER, nk=NK, **other)
        assert abs(c1 - c) > 1e-3
    V = _full_spin_U(hmf)
    q = np.array([0.13, 0.27, 0.])
    zs = np.array([4.7 + 0.3j, 4.7 - 0.3j])
    g1, _ = rw.spin_kernel(hmf, nk=NK, channel="raising", kernel="response")
    g2, _ = rw.spin_kernel(hmf, V={d: 2.*m for (d, m) in V.items()}, nk=NK,
                           channel="raising", kernel="response")
    d = g1(zs, q) - g2(zs, q) # the vertex of the channel, at every z
    assert np.max(np.abs(d[0] - d[1])) < 1e-10 and np.max(np.abs(d)) > 1.
    # route 2 builds its kernel at minus the momentum of route 3
    Vr = kc.spin_rpa_kernel(hmf, nk=4)._vertex(-q)
    assert np.max(np.abs(d[0] - Vr)) < 1e-10
    # a vertex that differs between the two sites does not commute with
    # chi0, so 1 - chi0 V loses the symmetry that chi0^-1 - V keeps
    D = np.diag(np.tile([1., np.sqrt(2.)], 3))
    V3 = {d: D@m@D for (d, m) in V.items()}
    for (kernel, flag) in (("response", True), ("dyson", False),
                           ("bare", True)):
        g, herm = rw.spin_kernel(hmf, V=V3, nk=NK, channel="raising",
                                 kernel=kernel)
        a = g(zs, q)
        symmetric = np.max(np.abs(a[1] - np.conj(a[0].T))) < 1e-10
        assert herm == flag and symmetric == flag
