"""A non-reciprocal hopping, t_ij != t_ji^*, built from a hopping function
on a periodic 1d or 2d geometry came back reciprocal: the single-cell
storage keeps one direction of each bond (inter, tx, ty...) and takes the
other as its adjoint, so the Hatano-Nelson chain had a real spectrum
instead of the ellipse 2t cos k + 2i gamma sin k. A non-Hermitian hopping
function is now built as multicell, which evaluates both directions."""

import numpy as np

from pyqula import geometry


t, gamma = 1.0, 0.3


def _tij(r1, r2):
    """Hopping from r2 to r1 along x, larger to the right than to the left"""
    dr = r2 - r1
    if abs(np.sqrt(dr.dot(dr)) - 1.) > 1e-3: return 0.
    if abs(dr[0]) < 1e-3: return t  # a bond along y is reciprocal
    return t + gamma*np.sign(dr[0])


def _spectrum(h, k):
    """Sorted complex eigenvalues of the Bloch Hamiltonian at k"""
    return np.sort_complex(np.linalg.eigvals(h.get_hk_gen()(k)))


def test_hatano_nelson_chain_is_non_reciprocal():
    """The periodic Hatano-Nelson chain has the complex band
    2t cos(2 pi k) + 2i gamma sin(2 pi k), for a one-site cell and for a
    two-site cell, whose two bands are that band at k/2 and (k+1)/2"""
    for (g, n) in [(geometry.chain(), 1), (geometry.bichain(), 2)]:
        h = g.get_hamiltonian(has_spin=False, non_hermitian=True, tij=_tij)
        for k in np.linspace(0., 1., 7, endpoint=False):
            ks = [(k + j)/n for j in range(n)]  # the folded momenta
            exact = [2*t*np.cos(2*np.pi*q) + 2j*gamma*np.sin(2*np.pi*q)
                     for q in ks]
            e = _spectrum(h, [k, 0., 0.])
            assert np.allclose(e, np.sort_complex(np.array(exact)))


def test_non_reciprocal_square_lattice():
    """In two dimensions the same holds for the x bonds, with the y bonds
    left reciprocal: 2t cos kx + 2i gamma sin kx + 2t cos ky"""
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=False, non_hermitian=True, tij=_tij)
    for k in np.random.default_rng(0).random((5, 2)):
        kx, ky = 2*np.pi*k
        exact = 2*t*np.cos(kx) + 2j*gamma*np.sin(kx) + 2*t*np.cos(ky)
        (e,) = _spectrum(h, [k[0], k[1], 0.])
        assert np.isclose(e, exact)


def test_reciprocal_hopping_is_unchanged():
    """A Hermitian hopping function gives the same Bloch Hamiltonian with
    and without the non-Hermitian flag, so the switch to multicell only
    changes how the hoppings are stored"""
    def tij(r1, r2):
        dr = r2 - r1
        if abs(np.sqrt(dr.dot(dr)) - 1.) > 1e-3: return 0.
        return 1. + 0.2j*np.sign(dr[0])  # a Peierls-like phase, Hermitian
    for g in [geometry.chain(), geometry.bichain(),
              geometry.honeycomb_lattice()]:
        h0 = g.get_hamiltonian(has_spin=False, tij=tij)
        h1 = g.get_hamiltonian(has_spin=False, tij=tij, non_hermitian=True)
        for k in np.random.default_rng(1).random((4, 3)):
            m0, m1 = h0.get_hk_gen()(k), h1.get_hk_gen()(k)
            assert np.allclose(m0, m1)
            assert np.allclose(m1, np.conjugate(m1.T))


def _spinful_tij(r1, r2):
    """A spin-diagonal Hatano-Nelson hopping, with half the hopping and
    half the non-reciprocity for spin down"""
    return np.diag([_tij(r1, r2), 0.5*_tij(r1, r2)])


def test_spinful_generator_hatano_nelson_chain():
    """A non-Hermitian spinful generator in one dimension is built with
    both directions of each bond too: the two spins give the ellipses of
    (t, gamma) and (t/2, gamma/2)"""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True, non_hermitian=True,
                          spinful_generator=True, tij=_spinful_tij)
    for k in np.linspace(0., 1., 7, endpoint=False):
        q = 2*np.pi*k
        exact = [s*(2*t*np.cos(q) + 2j*gamma*np.sin(q)) for s in (1., 0.5)]
        assert np.allclose(_spectrum(h, [k, 0., 0.]),
                           np.sort_complex(np.array(exact)))


def test_spinful_generator_builds_the_same_as_multicell():
    """A Hermitian spinful generator gives the same Bloch Hamiltonian from
    the single-cell storage and from the multicell builder, which did not
    accept a spinful generator before"""
    def tij(r1, r2):  # a Rashba-like hopping, t_ji = t_ij^dagger
        dr = r2 - r1
        if abs(np.sqrt(dr.dot(dr)) - 1.) > 1e-3: return np.zeros((2, 2))
        sx = np.array([[0., 1.], [1., 0.]])
        sy = np.array([[0., -1j], [1j, 0.]])
        return np.identity(2) + 0.3j*(dr[0]*sy - dr[1]*sx)
    for g in [geometry.chain(), geometry.honeycomb_lattice()]:
        h0 = g.get_hamiltonian(spinful_generator=True, tij=tij)
        h1 = g.get_hamiltonian(spinful_generator=True, tij=tij,
                               is_multicell=True)
        for k in np.random.default_rng(2).random((4, 3)):
            m0, m1 = h0.get_hk_gen()(k), h1.get_hk_gen()(k)
            assert np.allclose(m0, m1)
            assert np.allclose(m1, np.conjugate(m1.T))


def test_non_reciprocal_function_without_the_flag_warns():
    """Without non_hermitian=True a non-reciprocal hopping function cannot
    be represented, since the Hamiltonian is built as Hermitian, and that
    is said rather than done in silence, whichever way it is stored"""
    import pytest
    for g in [geometry.chain(), geometry.square_lattice()]:
        for kw in [dict(has_spin=False), dict(has_spin=True),
                   dict(has_spin=False, is_multicell=True),
                   dict(has_spin=True, is_sparse=True)]:
            with pytest.warns(UserWarning, match="not reciprocal"):
                g.get_hamiltonian(tij=_tij, **kw)
        with pytest.warns(UserWarning, match="not reciprocal"):
            g.get_hamiltonian(spinful_generator=True, tij=_spinful_tij)


def test_reciprocal_functions_build_without_a_warning():
    """A reciprocal hopping, real or with a Peierls-like phase, scalar or
    as a spinful generator, builds with no warning in every storage"""
    import warnings

    def peierls(r1, r2):
        dr = r2 - r1
        if abs(np.sqrt(dr.dot(dr)) - 1.) > 1e-3: return 0.
        return np.exp(0.4j*dr[0])  # t_ji = t_ij^*

    def spinful(r1, r2):
        return np.diag([peierls(r1, r2), np.conjugate(peierls(r1, r2))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for g in [geometry.chain(), geometry.bichain(),
                  geometry.honeycomb_lattice(), geometry.square_lattice()]:
            for kw in [dict(has_spin=False), dict(has_spin=True),
                       dict(is_multicell=True), dict(is_sparse=True)]:
                g.get_hamiltonian(tij=peierls, **kw)
            g.get_hamiltonian(spinful_generator=True, tij=spinful)
