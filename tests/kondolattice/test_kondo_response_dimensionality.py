import numpy as np
import pytest

from pyqula import geometry
from pyqula import islands
from pyqula.kondolattice import KondoLatticeHamiltonian

# The matrix spectral function of (c, F) and the spin susceptibility of
# the moments of the Kondo lattice mean field (arXiv:2302.09701) are not
# specific to the chain of the paper. Here they are checked in 0D, 1D, 2D
# and 3D, with one and with several Kondo pairs per unit cell, against
# routes written in the test: the inverse of z - H(k) for the Green
# functions, and the sum over pairs of eigenstates for the susceptibility.

DELTA = 0.05
JP = 2.0 # the J' of the paper; the loop takes J = 2J'


# name: (geometry, k-mesh of the loop, filling), built when asked for
_GEOMETRIES = {
    "0d": (lambda: islands.get_geometry(name="honeycomb", n=2, nedges=6),
            1, 0.5),
    "1d": (lambda: geometry.chain().get_supercell(2), 60, 0.5),
    "2d_square": (geometry.square_lattice, 16, 0.5),
    "2d_honeycomb": (geometry.honeycomb_lattice, 16, 0.5),
    # no particle-hole symmetry: the f level is not at zero
    "2d_triangular": (geometry.triangular_lattice, 16, 0.2),
    "3d": (geometry.cubic_lattice, 8, 0.5)}


@pytest.fixture(scope="module", params=list(_GEOMETRIES))
def kondo(request):
    """Converged Kondo lattice mean field in each dimensionality"""
    gen, nk, filling = _GEOMETRIES[request.param]
    h = KondoLatticeHamiltonian(gen().get_hamiltonian(has_spin=True))
    n = len(h._kondo_pairs)
    h2 = h.get_mean_field_hamiltonian(J=2*JP, filling=filling, nk=nk,
            mf=([1.0+0j]*n, [0.0]*n), maxerror=1e-7, maxite=4000)
    assert np.min(np.abs(h2.hybridization)) > 0.3 # a hybridized state
    assert np.allclose(h2.local_occupation, 1.0, atol=1e-5)
    return request.param, h2


def _generic(dim, seed):
    """A generic momentum of a system of that dimensionality"""
    k = np.zeros(3)
    k[:dim] = np.random.default_rng(seed).random(dim)
    return k


def _scaled_hk(h, k, scale):
    """H(k) with the hybridization of every pair multiplied by scale,
    built here from the Bloch Hamiltonian"""
    m = np.array(h.get_hk_gen()(k), dtype=np.complex128)
    for (ci, fi) in h._kondo_pairs:
        m[2*ci:2*ci+2, 2*fi:2*fi+2] *= scale
        m[2*fi:2*fi+2, 2*ci:2*ci+2] *= scale
    return m


def test_dimensionality_and_cell(kondo):
    name, h = kondo
    assert h.dimensionality == int(name[0])
    if name == "0d": assert len(h._kondo_pairs) == 24 # a honeycomb flake
    if name == "2d_triangular": # not at the particle-hole symmetric point
        assert abs(h.constraint_lambda[0]) > 0.1


@pytest.mark.parametrize("J,scale", [(None, 1.0), (2*JP, 1.5)])
def test_green_functions_are_the_inverse_of_z_minus_h(kondo, J, scale):
    """Every element that the routine returns, against the inverse of
    z - H(k) at generic momenta, and g_F against the Schur complement
    G_FF - G_Fc G_cc^-1 G_cF, the inverse of the FF block of G^-1"""
    name, h = kondo
    dim = h.dimensionality
    pairs = h._kondo_pairs
    n = len(pairs)
    ks = np.array([_generic(dim, s) for s in (1, 2)])
    es = np.linspace(-5.0, 5.0, 11)
    kpath, e, G = h.get_composite_spectral_function(kpath=ks, energies=es,
            delta=DELTA, J=J, hybridization_scale=scale, green=True,
            site=list(range(n)))
    kpath, e, A = h.get_composite_spectral_function(kpath=ks, energies=es,
            delta=DELTA, J=J, hybridization_scale=scale,
            site=list(range(n)))
    kpath, e, Aav = h.get_composite_spectral_function(kpath=ks, energies=es,
            delta=DELTA, J=J, hybridization_scale=scale)
    assert kpath.shape == ((1, 3) if dim == 0 else (2, 3))
    assert G["cc"].shape == (n, len(kpath), len(es))
    assert Aav["cc"].shape == (len(kpath), len(es))
    assert np.allclose(Aav["FF"], np.mean(A["FF"], axis=0), atol=1e-12)
    c = np.array([2*ci+s for (ci, fi) in pairs for s in (0, 1)])
    f = np.array([2*fi+s for (ci, fi) in pairs for s in (0, 1)])
    for ik, k in enumerate(kpath):
        m = _scaled_hk(h, k, scale)
        V = np.array([(m[2*fi, 2*ci]+m[2*fi+1, 2*ci+1])/2
                for (ci, fi) in pairs])
        z = np.ones(n) if J is None else 4*np.conjugate(V)/(3*J)
        D = np.diag(np.repeat(z, 2)) # F = z f, for the two spins
        for iw, w in enumerate(es):
            g = np.linalg.inv((w+1j*DELTA)*np.identity(len(m))-m)
            gcc, gcf = g[np.ix_(c, c)], g[np.ix_(c, f)]@np.conjugate(D)
            gfc, gff = D@g[np.ix_(f, c)], D@g[np.ix_(f, f)]@np.conjugate(D)
            schur = gff-gfc@np.linalg.inv(gcc)@gcf
            for name_, ref in [("cc", gcc), ("cF", gcf), ("Fc", gfc),
                    ("FF", gff), ("gF", schur)]:
                d = np.diag(ref)
                per_pair = (d[0::2]+d[1::2])/2 # average over the spin
                assert np.allclose(G[name_][:, ik, iw], per_pair, atol=1e-9)
            assert np.allclose(A["cc"][:, ik, iw],
                    -G["cc"][:, ik, iw].imag/np.pi, atol=1e-12)
            assert np.allclose(A["FF"][:, ik, iw],
                    -G["FF"][:, ik, iw].imag/np.pi, atol=1e-12)


def _pair_sum(h, q, ws, nk, temp):
    """chi_ij = 1/Nk sum_k sum_mn (f_m - f_n) <m,k|S^+_i|n,k+q>
    <n,k+q|S^-_j|m,k> / (e_m - e_n - w + i delta), from the eigenvectors"""
    fs = [fi for (ci, fi) in h._kondo_pairs]
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=nk)
    out = np.zeros((len(ws), len(fs), len(fs)), dtype=np.complex128)
    for k in ks:
        e1, u1 = np.linalg.eigh(hk(np.array(k)))
        e2, u2 = np.linalg.eigh(hk(np.array(k)+q))
        o1, o2 = 1/(np.exp(e1/temp)+1), 1/(np.exp(e2/temp)+1)
        sp = np.array([np.conj(u1[2*i, :])[:, None]*u2[2*i+1, :][None, :]
                for i in fs])
        fac = o1[:, None]-o2[None, :]
        de = e1[:, None]-e2[None, :]
        for iw, w in enumerate(ws):
            out[iw] += np.einsum("imn,mn,jmn->ij", sp, fac/(de-w+1j*DELTA),
                    np.conj(sp))/len(ks)
    return out


def test_moment_response_is_the_sum_over_pairs_of_states(kondo):
    """The bare response at a generic q, and the residual interaction as
    the matrix form of the formula of the paper, chi (1 - U chi)^-1"""
    name, h = kondo
    dim = h.dimensionality
    n = len(h._kondo_pairs)
    nk = {0: 1, 1: 12, 2: 5, 3: 3}[dim]
    q = _generic(dim, 3)
    ws = np.linspace(0.4, 5.0, 4)
    kw = dict(q=q, energies=ws, nk=nk, delta=DELTA, T=0.05)
    e, chi = h.get_moment_spinchi(**kw)
    ref = _pair_sum(h, q, ws, nk, 0.05)
    assert chi.shape == (len(ws), n, n)
    assert np.max(np.abs(ref.imag)) > 1e-3 # there is a signal
    assert np.allclose(chi, ref, atol=1e-8)
    # a number, a matrix between the moments, and a function of q
    rng = np.random.default_rng(4)
    Um = rng.random((n, n))-0.5
    Um = (Um+Um.T)/2
    for U, Uref in [(-0.8, -0.8*np.identity(n)), (Um, Um),
            (lambda qq: -0.5*np.cos(2*np.pi*qq[0])*np.identity(n),
             -0.5*np.cos(2*np.pi*q[0])*np.identity(n))]:
        e, rpa = h.get_moment_spinchi(U=U, **kw)
        for iw in range(len(ws)):
            r = ref[iw]@np.linalg.inv(np.identity(n)-Uref@ref[iw])
            assert np.allclose(rpa[iw], r, atol=1e-8)


def test_finite_system_has_no_momentum(kondo):
    """In 0D the response does not depend on q, and the local response
    is the same one; the spectral function has one k-point"""
    name, h = kondo
    if name != "0d": pytest.skip("only for the finite system")
    ws = np.linspace(0.0, 5.0, 11)
    kw = dict(energies=ws, delta=DELTA, T=0.05, U=-0.5)
    e, c0 = h.get_moment_spinchi(q=[0., 0., 0.], **kw)
    e, c1 = h.get_moment_spinchi(q=[0.3, 0.1, 0.7], **kw)
    e, cl = h.get_moment_spinchi(**kw)
    assert np.allclose(c0, c1, atol=1e-12)
    assert np.allclose(c0, cl, atol=1e-12)
    ks, e, A = h.get_composite_spectral_function(energies=ws, delta=DELTA)
    assert ks.shape == (1, 3) and A["cc"].shape == (1, len(ws))
    ks, e, A2 = h.get_composite_spectral_function(energies=ws, delta=DELTA,
            kpath=[[0.2, 0.4, 0.]], nk=7)
    assert np.allclose(A["FF"], A2["FF"], atol=1e-12)


def test_local_sum_rule_in_every_dimensionality(kondo):
    """The local spectral weight of each moment integrates to
    n_down (1 - n_up) = 1/4, in any dimensionality and for any lattice"""
    name, h = kondo
    dim = h.dimensionality
    nk = {0: 1, 1: 8, 2: 4, 3: 2}[dim]
    ws = np.linspace(0.0, 80.0, 1601)
    e, chi = h.get_moment_spinchi(energies=ws, nk=nk, delta=DELTA, T=1e-3)
    weight = np.sum(chi.imag, axis=0)*(ws[1]-ws[0])/np.pi
    # occupation of each spin of each moment on the same k-mesh
    fs = [fi for (ci, fi) in h._kondo_pairs]
    occ = np.zeros(len(fs))
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=nk)
    for k in ks:
        es, vs = np.linalg.eigh(hk(np.array(k)))
        occ += np.array([np.sum(np.abs(vs[2*i, es < 0])**2) for i in fs])
    occ /= len(ks)
    assert np.allclose(np.diag(weight), occ*(1-occ), atol=5e-3)
    if name != "2d_triangular":
        assert np.allclose(np.diag(weight), 0.25, atol=5e-3)


def test_square_lattice_loop_is_the_gap_equation(kondo):
    """In 2D too the loop gives the V of
    1/J' = 1/N sum_k 1/sqrt((eps_k/2)^2 + V^2), on the same k-mesh"""
    name, h = kondo
    if name != "2d_square": pytest.skip("only for the square lattice")
    from scipy.optimize import brentq
    k = 2*np.pi*np.arange(16)/16
    eps = 2*(np.cos(k)[:, None]+np.cos(k)[None, :])
    # the loop runs at T=0.02, far below the gap
    V0 = brentq(lambda V: np.mean(1/np.sqrt((eps/2)**2+V**2))-1/JP, 1e-6, 50)
    assert abs(h.hybridization[0]-V0) < 1e-4
