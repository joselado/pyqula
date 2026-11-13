"""Density matrices of embedded systems with a given selfenergy.

get_dm integrates the Green's function on two semicircles, the upper one
with delta>0 and the lower one with delta<0. The lower one needs the
advanced selfenergy, the Hermitian conjugate of the retarded one, or the
poles of the lower function sit inside the lower contour. The oracle here
is the real-axis integral of i(G - G^dagger)/2pi built from the object's
own retarded Green's function, on the same energy window, which does not
go through the lower branch at all."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import embedding
from pyqula.embeddingtk.embedded import embed_hamiltonian

DELTA = 5e-2
EMIN = -10.


def _realaxis_rho(gr, emin=EMIN, npanel=200, ng=16):
    """rho = int_emin^0 dE i(G(E) - G(E)^dagger)/2pi, composite
    Gauss-Legendre, from the retarded Green's function gr(E)"""
    x, w = np.polynomial.legendre.leggauss(ng)
    edges = np.linspace(emin, 0., npanel+1)
    rho = 0.
    for a, b in zip(edges[:-1], edges[1:]):
        for xi, wi in zip(0.5*(x+1.)*(b-a)+a, 0.5*w*(b-a)):
            g = np.array(gr(xi))
            rho = rho + wi*1j*(g - g.conj().T)/(2*np.pi)
    return rho


def _finite_chain(n):
    g = geometry.chain().get_supercell(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 0.3*np.cos(r[0])) # breaks particle-hole symmetry
    return h


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


@pytest.mark.parametrize("selfenergy", [1.0,
        lambda e: 1j*0.8/(e - 0.5 + 0.6j)], # a retarded, energy-dependent one
        ids=["constant", "callable"])
def test_boundary_selfenergy_density_matrix_matches_the_real_axis(selfenergy):
    """Embedding(h, selfenergy=...) used the retarded boundary selfenergy
    -i Gamma on the lower contour too, which made the diagonal of a 6-site
    chain 0.603 instead of 0.508 at its ends. The contour result must agree
    with the real-axis integral and be Hermitian."""
    h = _finite_chain(6)
    eb = embedding.Embedding(h, selfenergy=selfenergy)
    dm = np.array(eb.get_density_matrix(delta=DELTA, emin=EMIN))
    rho = _realaxis_rho(lambda e: eb.get_gf(energy=e, delta=DELTA))
    assert np.max(np.abs(dm - dm.conj().T)) < 1e-6
    assert np.max(np.abs(dm.T - rho)) < 1e-3


def test_boundary_selfenergy_agrees_with_the_same_matrix_selfenergy():
    """The boundary selfenergy of Embedding and the same matrix handed to
    embed_hamiltonian are two routes to one density matrix."""
    h = _finite_chain(6)
    eb = embedding.Embedding(h, selfenergy=0.7)
    dm1 = np.array(eb.get_density_matrix(delta=DELTA, emin=EMIN))
    sigma = 0.7*_dense(eb.boundary_embedding_generator) # -0.7i on the edges
    dm2 = np.array(embed_hamiltonian(h, selfenergy=sigma).get_density_matrix(
            delta=DELTA, emin=EMIN))
    assert np.max(np.abs(dm1 - dm2)) < 1e-3


def test_complex_hermitian_selfenergy_gives_a_hermitian_density_matrix():
    """A matrix selfenergy Lam - i Gam with Lam complex Hermitian (the shape
    of a lead selfenergy with spin-orbit coupling or a flux) was conjugated
    element-wise on the lower contour instead of Hermitian conjugated, which
    left the density matrix non-Hermitian by 0.1. It must be Hermitian and
    agree with the real-axis integral."""
    h = _finite_chain(4)
    n = _dense(h.intra).shape[0]
    rng = np.random.default_rng(1)
    p = rng.random((n, n))
    lam = 0.4j*(p - p.T) # Hermitian and purely imaginary, so conj(lam) = -lam
    sigma = lam - 1j*np.diag([0.5, 0., 0., 0.5])
    eh = embed_hamiltonian(h, selfenergy=sigma)
    dm = np.array(eh.get_density_matrix(delta=DELTA, emin=EMIN))
    rho = _realaxis_rho(lambda e: eh.get_gf(energy=e, delta=DELTA))
    assert np.max(np.abs(dm - dm.conj().T)) < 1e-6
    assert np.max(np.abs(dm.T - rho)) < 1e-3


def test_a_defect_with_a_boundary_selfenergy_is_refused():
    """with selfenergy= the boundary embedding is built from h alone, so a
    defect m used to be dropped silently (bug_audit_6.md #12)"""
    h = _finite_chain(6)
    hd = h.copy()
    hd.add_onsite(0.5)
    with pytest.raises(ValueError, match="not both"):
        embedding.Embedding(h, m=hd, selfenergy=1.)


def _flake_without_time_reversal():
    """honeycomb flake with a Haldane coupling, an exchange field and
    Rashba coupling, and the edge sites at the largest |x| as the lead"""
    g = geometry.honeycomb_lattice().get_supercell(2)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_haldane(0.1)
    h.add_exchange([0.1, 0.2, 0.3])
    h.add_rashba(0.2)
    xm = np.max(np.abs(g.r[:, 0]))
    edge = lambda r: 1.0 if abs(abs(r[0]) - xm) < 1e-6 else 0.
    return h, edge


def test_semimetal_spectral_function_with_broken_time_reversal():
    """the semimetal selfenergy returned the retarded one for both signs
    of delta, so i(G^R-G^A)/2pi was zero; the advanced side is the
    Hermitian conjugate, whatever the symmetries of the Hamiltonian
    (bug_audit_6.md #13)"""
    from pyqula.embeddingtk.embedded import get_A
    h, edge = _flake_without_time_reversal()
    se = embedding.get_selfenergy_from_potential(h, edge, mode="semimetal")
    eh = embed_hamiltonian(h, selfenergy=se)
    for e in [0.3, -0.7]:
        A = get_A(eh, energy=e, delta=1e-3)
        gr = _dense(eh.get_gf(energy=e, delta=1e-3))
        ref = -np.trace(gr).imag/np.pi
        assert ref > 0.1
        assert abs(np.trace(A).real - ref) < 1e-10
        assert np.max(np.abs(A - np.conjugate(A.T))) < 1e-10


def test_density_matrix_refuses_the_non_analytic_semimetal_selfenergy():
    """the contour of the density matrix needs a selfenergy analytic in
    the upper half plane; -i Gamma |E| is not, and gave entries off by 0.3"""
    h, edge = _flake_without_time_reversal()
    se = embedding.get_selfenergy_from_potential(h, edge, mode="semimetal")
    with pytest.raises(NotImplementedError, match="analytic"):
        embed_hamiltonian(h, selfenergy=se).get_density_matrix()


def test_analytic_callable_selfenergy_density_matrix_matches_the_real_axis():
    """a callable is the retarded selfenergy, continued below the real
    axis as Sigma(z^*)^dagger, the same contract as the boundary
    embedding; before, the retarded one was used on the lower contour.
    Here Sigma(z) = Gamma 0.8/(z - 0.5 + 0.6i), with its pole in the lower
    half plane, so analytic above it"""
    h = _finite_chain(6)
    n = _dense(h.intra).shape[0]
    gam = np.diag([1.] + [0.]*(n-2) + [1.])
    se = lambda energy=0., **kwargs: gam*0.8/(energy - 0.5 + 0.6j)
    eh = embed_hamiltonian(h, selfenergy=se)
    dm = np.array(eh.get_density_matrix(delta=DELTA, emin=EMIN))
    rho = _realaxis_rho(lambda e: eh.get_gf(energy=e, delta=DELTA))
    assert np.max(np.abs(dm - dm.conj().T)) < 1e-6
    assert np.max(np.abs(dm.T - rho)) < 1e-3


def test_boundary_embedding_refuses_a_non_analytic_callable():
    """one contract for both embeddings: a callable marked analytic=False
    is refused by the density matrix of the boundary embedding too"""
    se = lambda e: abs(e) # |E| has no analytic continuation
    se.analytic = False
    eb = embedding.Embedding(_finite_chain(6), selfenergy=se)
    with pytest.raises(NotImplementedError, match="analytic"):
        eb.get_density_matrix(delta=DELTA, emin=EMIN)
