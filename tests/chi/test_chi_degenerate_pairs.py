import numpy as np
from scipy.integrate import quad

from pyqula import geometry


def _chain():
    return geometry.chain().get_hamiltonian(has_spin=True)


def _band_integral(T):
    """Static response of the half-filled chain at q=pi, both spins, in
    the limit of zero broadening: -int d(eps) rho(eps) tanh(eps/2T)/eps
    with rho the density of states per spin, written with eps=2 sin(th)"""
    f = lambda th: np.tanh(np.sin(th)/T)/(2.*np.sin(th)) if th > 1e-12 \
        else 1./(2.*T)
    return -(2./np.pi)*quad(f, 0., np.pi/2., limit=400)[0]


def test_static_response_does_not_depend_on_where_the_mesh_lands():
    """On a mesh with nk a multiple of 4, k=pi/2 and k+pi=3pi/2 are both
    mesh points at the Fermi level with the same energy. That pair used to
    be dropped, since f_a-f_b=0, although it stands for the intraband
    continuum around it, where (f_a-f_b)/(e_a-e_b) tends to f': nk=200
    gave -1.1635 against the -1.2135 of odd nk and of the band integral.
    Outside the particle-hole continuum, at omega=5, the two meshes agree
    as well."""
    h = _chain()
    kw = dict(q=[0.5, 0., 0.], energies=[0., 5.], delta=1e-3, T=0.1)
    even = np.array(h.get_chi(nk=200, **kw)[1])
    odd = np.array(h.get_chi(nk=199, **kw)[1])
    assert abs(even[0].real - odd[0].real) < 1e-3
    assert abs(even[0].real/_band_integral(0.1) - 1.) < 1e-2
    assert abs(even[1] - odd[1]) < 1e-3


def test_degenerate_pairs_give_the_static_limit_only():
    """At q=0 every pair of the chain is degenerate: the response is the
    compressibility, the average of f' over the band, in the static limit,
    a peak of width delta around zero frequency, and it vanishes for
    |omega| >> delta, as the conservation of the charge requires. It used
    to vanish at every frequency, static limit included."""
    h = _chain()
    T, delta, nk = 0.1, 0.02, 400
    chi = np.array(h.get_chi(q=[0., 0., 0.], energies=[0., 1.], nk=nk,
                             delta=delta, T=T)[1])
    ks = 2.*np.pi*np.arange(nk)/nk
    f = 1./(1. + np.exp(2.*np.cos(ks)/T))  # the band is 2t cos k, t=1
    fprime = -2.*np.mean(f*(1. - f))/T  # both spins
    assert abs(chi[0]/fprime - 1.) < 1e-2
    assert abs(chi[1]) < 2.*delta*abs(fprime)
    assert chi[1].imag > 0.  # positive spectral weight at positive frequency
