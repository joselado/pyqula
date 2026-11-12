import numpy as np
import pytest

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.chitk import localspinchi
from pyqula.multihopping import MultiHopping

# Spin susceptibility of the local moments of the Kondo lattice mean field,
# the convolution of two f propagators, and its random phase approximation
# with a residual interaction U'(q) (Chen, Stoudenmire, Komijani and
# Coleman, arXiv:2302.09701). The paper writes the Kondo term as
# J'(c^dag sigma c).S, and the loop of this library takes J = 2J'. Every
# test checks an invariant of the construction against an independent
# route, not a recorded number.

DELTA = 0.05
T = 1e-3
JP = 2.0 # the J' of the paper, in units of the hopping


def _chain_kondo(natoms=1):
    g = geometry.chain()
    if natoms > 1: g = g.get_supercell(natoms)
    return KondoLatticeHamiltonian(g.get_hamiltonian(has_spin=True))


def _converge(h, jp, nk=200):
    n = len(h._kondo_pairs)
    return h.get_mean_field_hamiltonian(J=2*jp, filling=0.5, nk=nk,
            mf=([jp+0j]*n, [0.0]*n), maxerror=1e-8, maxite=3000)


@pytest.fixture(scope="module")
def hchain():
    """Converged half-filled chain, the Kondo insulator of the paper"""
    return _converge(_chain_kondo(), JP)


def _bands(k, V):
    """Two bands of the half-filled chain and the weight of the f orbital
    in the upper one"""
    e = 2*np.cos(2*np.pi*k)
    r = np.sqrt((e/2)**2+V**2)
    ep, em = e/2+r, e/2-r
    wp = (ep-e)**2/((ep-e)**2+V**2)
    return em, ep, wp


def _closed_form(V, q, ws, nk):
    """chi^{-+}(q,w) of the appendix of the paper: a hole in the lower
    band and a particle in the upper one, each with its f weight"""
    ks = np.arange(nk)/nk
    em, ep, wp = _bands(ks, V)
    emq, epq, wpq = _bands(ks+q, V)
    z = ws[:, None]-1j*DELTA # this library has the pole at w - i delta
    return np.mean((1-wp)*wpq/(z-(epq-em))-wp*(1-wpq)/(z+(ep-emq)), axis=1)


def _gaps(V):
    e = 2*np.cos(2*np.pi*np.arange(4000)/4000)
    r = np.sqrt((e/2)**2+V**2)
    return 2*V, np.min(e/2+r)-np.max(e/2-r) # direct and indirect gap


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_matches_the_two_propagator_formula(hchain, scale):
    V = hchain.hybridization[0].real*scale
    ws = np.linspace(0.0, 8.0, 81)
    nk = 100
    for q in (0.0, 0.17, 0.5):
        e, chi = hchain.get_moment_spinchi(q=[q, 0., 0.], energies=ws, nk=nk,
                delta=DELTA, T=T, hybridization_scale=scale)
        assert chi.shape == (len(ws), 1, 1)
        ref = _closed_form(V, q, ws, nk)
        assert np.max(chi[:, 0, 0].imag) > 0.3 # there is a signal
        assert np.allclose(chi[:, 0, 0], ref, atol=1e-10)


def test_continuum_starts_at_the_direct_and_indirect_gaps(hchain):
    """At q=0 the particle and the hole are at the same k and the
    continuum starts at 2V. At q=pi it starts at the indirect gap, which
    is smaller"""
    V = hchain.hybridization[0].real
    direct, indirect = _gaps(V)
    assert indirect < direct-1.0
    ws = np.linspace(0.0, 8.0, 801)
    kw = dict(energies=ws, nk=400, delta=0.01, T=T)
    for q, gap in [(0.0, direct), (0.5, indirect)]:
        e, chi = hchain.get_moment_spinchi(q=[q, 0., 0.], **kw)
        im = chi[:, 0, 0].imag
        assert np.all(im[ws > 0.05] > 0) # absorption at positive frequency
        onset = ws[np.argmax(im > 0.2*im.max())]
        assert abs(onset-gap) < 0.05
        assert np.max(im[ws < gap-0.3]) < 0.02*im.max()


def test_local_sum_rule(hchain):
    """The integral of the local spectral weight over positive frequency
    is <S^- S^+> of the site. For free f fermions it is
    n_down (1 - n_up) = 1/4, half of the 1/2 of a spin 1/2: the mean
    field keeps the constraint on average only. It does not depend on V"""
    ws = np.linspace(0.0, 60.0, 3001)
    for scale in (1.0, 1.5):
        e, chi = hchain.get_moment_spinchi(energies=ws, nk=12, delta=DELTA,
                T=T, hybridization_scale=scale)
        weight = np.sum(chi[:, 0, 0].imag)*(ws[1]-ws[0])/np.pi
        assert abs(weight-0.25) < 5e-3


def test_rpa_is_the_formula_of_the_paper(hchain):
    """chi_RPA = 1/(1/chi - U'(q)), for a number and for a function of q"""
    ws = np.linspace(0.0, 6.0, 61)
    q = [0.5, 0., 0.]
    kw = dict(q=q, energies=ws, nk=100, delta=DELTA, T=T)
    e, chi0 = hchain.get_moment_spinchi(**kw)
    e, chi = hchain.get_moment_spinchi(U=-2.0, **kw)
    assert np.allclose(chi[:, 0, 0], 1/(1/chi0[:, 0, 0]+2.0), atol=1e-10)
    # the U'(q) of the weak-coupling panel of the paper
    Uq = lambda q: -0.75*np.sin(np.pi*q[0])
    for qx in (0.0, 0.2, 0.5):
        kw["q"] = [qx, 0., 0.]
        e, chi0 = hchain.get_moment_spinchi(**kw)
        e, chi = hchain.get_moment_spinchi(U=Uq, **kw)
        ref = 1/(1/chi0[:, 0, 0]+0.75*np.sin(np.pi*qx))
        assert np.allclose(chi[:, 0, 0], ref, atol=1e-10)
    assert not np.allclose(chi0[:, 0, 0], chi[:, 0, 0], atol=1e-3)


def test_attractive_residual_interaction_binds_a_magnon(hchain):
    """U' < 0 gives a pole below the continuum, the magnon of the paper
    at strong coupling; U' > 0 pushes the weight up"""
    V = hchain.hybridization[0].real
    indirect = _gaps(V)[1]
    ws = np.linspace(0.0, 6.0, 601)
    kw = dict(q=[0.5, 0., 0.], energies=ws, nk=200, delta=0.02, T=T)
    e, chi0 = hchain.get_moment_spinchi(**kw)
    e, chim = hchain.get_moment_spinchi(U=-2.0, **kw)
    e, chip = hchain.get_moment_spinchi(U=2.0, **kw)
    peak = lambda c: ws[np.argmax(c[:, 0, 0].imag)]
    assert peak(chi0) > indirect-0.05
    assert peak(chim) < indirect-0.1 # a bound state inside the gap
    assert peak(chip) > peak(chi0)
    # the pole is where 1/chi = U', with chi real below the continuum
    ibound = np.argmax(chim[:, 0, 0].imag)
    assert abs(1/chi0[ibound, 0, 0].real+2.0) < 0.05


def test_free_moment_in_a_field_is_excited_by_s_plus():
    """A decoupled moment in a Zeeman field along z has its f electron
    in the spin-down level. S^+ takes it to the spin-up one, so the
    response has one pole at the Zeeman splitting, at positive frequency,
    with weight |<up|S^+|down>|^2 = 1. This goes through the route for a
    spin-split Hamiltonian"""
    b = 0.3
    h = _chain_kondo()
    h.add_zeeman([0., 0., b])
    h2 = h.get_mean_field_hamiltonian(J=0.0, filling=0.5, nk=20,
            maxerror=1e-7, maxite=500)
    # the levels of the f site, to know the splitting and its sign
    fup, fdn = h2.intra[2, 2].real, h2.intra[3, 3].real
    assert fup > fdn
    ws = np.linspace(-2.0, 2.0, 801)
    e, chi = h2.get_moment_spinchi(q=[0.1, 0., 0.], energies=ws, nk=8,
            delta=0.01, T=T)
    im = chi[:, 0, 0].imag
    assert abs(ws[np.argmax(np.abs(im))]-(fup-fdn)) < 0.01
    assert abs(np.sum(im)*(ws[1]-ws[0])/np.pi-1.0) < 0.02
    assert np.max(np.abs(im[ws < 0])) < 0.02*np.max(im)


def test_supercell_gives_the_primitive_response(hchain):
    """The matrix between the two moments of a doubled cell at q=0
    contains the primitive responses at q=0 (its sum) and at q=pi (the
    staggered combination), bare and with the residual interaction"""
    h2 = _converge(_chain_kondo(2), JP, nk=100)
    ws = np.linspace(0.0, 6.0, 31)
    for U in (None, -1.0):
        kw = dict(energies=ws, delta=DELTA, T=T, U=U)
        e, cs = h2.get_moment_spinchi(q=[0., 0., 0.], nk=50, **kw)
        assert cs.shape == (len(ws), 2, 2)
        e, c0 = hchain.get_moment_spinchi(q=[0., 0., 0.], nk=100, **kw)
        e, cpi = hchain.get_moment_spinchi(q=[0.5, 0., 0.], nk=100, **kw)
        uniform = (cs[:, 0, 0]+cs[:, 1, 1]+cs[:, 0, 1]+cs[:, 1, 0])/2
        stagger = (cs[:, 0, 0]+cs[:, 1, 1]-cs[:, 0, 1]-cs[:, 1, 0])/2
        scale = np.max(np.abs(cpi))
        assert np.max(np.abs(uniform-c0[:, 0, 0])) < 1e-4*scale
        assert np.max(np.abs(stagger-cpi[:, 0, 0])) < 1e-4*scale


def test_first_index_is_the_moment_that_s_plus_excites():
    """chi[i,j] has S^+ on the moment i and S^- on the moment j. The sum
    over the pairs of states is written here from the eigenvectors, in a
    doubled cell with two different moments and at a generic q, where the
    matrix is not symmetric"""
    h2 = _converge(_chain_kondo(2), JP, nk=100)
    # two different moments: the hybridization of the first pair is made
    # larger by hand, which keeps the gap and the half filling
    hop = dict(h2.get_multihopping().get_dict())
    m = np.array(hop[(0, 0, 0)], dtype=np.complex128)
    (ci, fi) = h2._kondo_pairs[0]
    m[2*ci:2*ci+2, 2*fi:2*fi+2] *= 1.4
    m[2*fi:2*fi+2, 2*ci:2*ci+2] *= 1.4
    hop[(0, 0, 0)] = m
    h2.set_multihopping(MultiHopping(hop))
    q = np.array([0.23, 0., 0.])
    ws = np.linspace(0.5, 4.0, 5)
    nk, temp = 12, 0.05
    e, chi = h2.get_moment_spinchi(q=q, energies=ws, nk=nk, delta=DELTA,
            T=temp)
    fs = [fi for (ci, fi) in h2._kondo_pairs]
    hk = h2.get_hk_gen()
    ref = np.zeros((len(ws), 2, 2), dtype=np.complex128)
    for k in h2.geometry.get_kmesh(nk=nk):
        e1, u1 = np.linalg.eigh(hk(np.array(k)))
        e2, u2 = np.linalg.eigh(hk(np.array(k)+q))
        occ1, occ2 = 1/(np.exp(e1/temp)+1), 1/(np.exp(e2/temp)+1)
        # <m,k|S^+_i|n,k+q> = conj(u_m(i,up)) u_n(i,down)
        sp = np.array([np.conj(u1[2*f, :])[:, None]*u2[2*f+1, :][None, :]
                for f in fs])
        fac = occ1[:, None]-occ2[None, :]
        de = e1[:, None]-e2[None, :]
        for iw, w in enumerate(ws):
            g = fac/(de-w+1j*DELTA)
            ref[iw] += np.einsum("imn,mn,jmn->ij", sp, g, np.conj(sp))/nk
    assert np.max(np.abs(ref[:, 0, 1]-ref[:, 1, 0])) > 1e-3 # not symmetric
    assert np.allclose(chi, ref, atol=1e-8)


def test_site_restriction_of_the_bubble_is_a_block(hchain):
    """The bubble restricted to a list of sites is the block of the full
    one, for a spin-degenerate and for a spin-split Hamiltonian"""
    hz = hchain.copy()
    hz.add_zeeman([0.05, 0.02, 0.1])
    ws = np.linspace(0.0, 5.0, 7)
    for h in (hchain, hz):
        kw = dict(q=[0.3, 0., 0.], energies=ws, nk=20, delta=DELTA, T=T)
        full = localspinchi.bare_spin_chi(h, **kw)[1]
        n = len(h.geometry.r)
        for sites in ([1], [1, 0]):
            part = localspinchi.bare_spin_chi(h, sites=sites, **kw)[1]
            idx = [a*n+i for a in range(3) for i in sites]
            assert np.max(np.abs(full)) > 1e-2
            assert np.allclose(part, full[:, idx][:, :, idx], atol=1e-12)
    with pytest.raises(ValueError, match="sites"):
        localspinchi.bare_spin_chi(hchain, sites=[5], q=[0., 0., 0.], nk=4)


def test_requires_a_kondo_lattice_and_valid_options(hchain):
    from pyqula.kondotk import momentchi
    h = geometry.chain().get_hamiltonian(has_spin=True)
    with pytest.raises(TypeError, match="KondoLatticeHamiltonian"):
        momentchi.moment_spinchi(h, q=[0., 0., 0.])
    with pytest.raises(ValueError, match="delta"):
        hchain.get_moment_spinchi(q=[0., 0., 0.], delta=0.0, nk=4)
    with pytest.raises(ValueError, match="residual interaction"):
        hchain.get_moment_spinchi(q=[0., 0., 0.], U=np.zeros((2, 2)), nk=4)
    with pytest.raises(ValueError, match="hybridization_scale"):
        hchain.get_moment_spinchi(q=[0., 0., 0.], hybridization_scale=-1.,
                nk=4)
