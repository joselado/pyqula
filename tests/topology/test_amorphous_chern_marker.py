"""The amorphous Chern insulator of Agarwala and Shenoy, and its local
Chern marker.

A. Agarwala and V. B. Shenoy, Phys. Rev. Lett. 118, 236402 (2017),
arXiv:1701.00374: sites placed at random in a square, two orbitals per
site, and a 2x2 hopping that depends on the length and the direction of
each bond (class A row of their Table I). The model is topological with
Bott index -1 in a window of the mass M (-2.5 < M < 3.7 at density 1,
their Fig. 3(b)), and the bulk local Chern marker must be quantized to the
same value there and vanish outside.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import topology
from pyqula.geometrytk import amorphous
from pyqula.topologytk.realspace import site_areas


def test_amorphous_lattice_places_the_sites_in_the_square():
    g = geometry.amorphous_lattice(L=10., density=0.6, seed=4)
    assert g.dimensionality == 0
    assert len(g.r) == 60 # round(density*L*L)
    assert np.all(np.abs(g.r[:, 0:2]) <= 5.) and np.all(g.r[:, 2] == 0.)
    same = geometry.amorphous_lattice(L=10., density=0.6, seed=4)
    assert np.allclose(g.r, same.r) # the seed fixes the sample


def _hopping(M, t2, lam, R):
    """The hopping of Table I of arXiv:1701.00374 for one pair of sites,
    written independently of the vectorized generator of the library"""
    onsite = np.array([[2+M, (1-1j)*lam], [(1+1j)*lam, -(2+M)]])
    def f(r1, r2):
        dr = r2 - r1 # bond from the first site to the second
        r = np.sqrt(dr.dot(dr))
        if r < 1e-9: return onsite
        if r > R: return np.zeros((2, 2), dtype=complex)
        theta = np.arctan2(dr[1], dr[0])
        s2 = np.sin(theta)**2
        T = 0.5*np.array([[-1+t2, -1j*np.exp(-1j*theta)+lam*(s2*(1+1j)-1)],
                          [-1j*np.exp(1j*theta)+lam*(s2*(1-1j)-1), 1+t2]])
        return np.e*np.exp(-r)*T
    return f


def test_the_generator_matches_the_hopping_of_the_paper():
    """The vectorized generator against the pair-by-pair hopping, built
    through the spinful hopping-function route of get_hamiltonian"""
    g = geometry.amorphous_lattice(L=5., density=1., seed=2)
    pars = dict(M=0.3, t2=0.25, lam=0.5, R=4.)
    h = g.get_hamiltonian(tij=_hopping(**pars), spinful_generator=True)
    fm = amorphous.amorphous_chern_generator(**pars)
    h2 = g.get_hamiltonian(mgenerator=fm, spinful_generator=True)
    assert np.allclose(h.intra, h2.intra, atol=1e-12)
    assert np.allclose(h2.intra, np.conjugate(h2.intra.T), atol=1e-12)


def _reference_marker(h):
    """Local Chern marker times the area of the site, from the eigenvectors,
    in the form 4*pi*Im<i|Q x P y Q|i> that Marsal, Varjas and Grushin use
    (arXiv:2003.13701, code at zenodo.3741829), with the imaginary part of
    the diagonal taken in the site basis"""
    (e, v) = np.linalg.eigh(np.array(h.intra)) # h.intra may be a np.matrix
    occupied = v[:, e < 0.]
    P = occupied@np.conjugate(occupied.T)
    Q = np.identity(len(e)) - P
    x = np.repeat(h.geometry.r[:, 0], 2) # two orbitals per site
    y = np.repeat(h.geometry.r[:, 1], 2)
    QxPyQ = (Q*x[None, :])@P@(y[:, None]*Q)
    C = 4*np.pi*np.diagonal(QxPyQ).imag
    return C[0::2] + C[1::2] # sum over the two orbitals of each site


def test_the_marker_matches_an_independent_implementation(tmp_path):
    import os
    g = geometry.amorphous_lattice(L=8., density=1., seed=3)
    h = amorphous.amorphous_chern_hamiltonian(g, M=-0.5)
    old = os.getcwd()
    os.chdir(tmp_path)
    try: (r, c) = topology.real_space_chern(h)
    finally: os.chdir(old)
    assert np.allclose(c*site_areas(g), _reference_marker(h), atol=1e-8)


def _bulk_marker(M, tmp_path):
    """Area-weighted marker of the sites within 4 of the center of a 20x20
    sample at density 1, which is 6 away from the edges"""
    import os
    g = geometry.amorphous_lattice(L=20., density=1., seed=0)
    h = amorphous.amorphous_chern_hamiltonian(g, M=M)
    es = np.linalg.eigvalsh(h.intra)
    n = len(g.r) # half of the states
    assert es[n-1] < 0. < es[n] # half filling, the Fermi energy at zero
    old = os.getcwd()
    os.chdir(tmp_path)
    try: (r, c) = topology.real_space_chern(h)
    finally: os.chdir(old)
    A = site_areas(g)
    bulk = np.max(np.abs(g.r[:, 0:2]), axis=1) < 4.
    return np.sum(c[bulk]*A[bulk])/np.sum(A[bulk])


@pytest.mark.slow
def test_bulk_marker_is_quantized_in_the_topological_phase(tmp_path):
    marker = _bulk_marker(-0.5, tmp_path)
    assert np.isclose(marker, -1., atol=0.03), marker


@pytest.mark.slow
def test_bulk_marker_vanishes_in_the_trivial_phase(tmp_path):
    marker = _bulk_marker(-3.0, tmp_path)
    assert np.abs(marker) < 0.02, marker
