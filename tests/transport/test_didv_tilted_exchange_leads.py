import numpy as np
import pytest
import warnings

from pyqula import geometry, heterostructures, algebra
from pyqula.greentk import rg

# A chain with an exchange field of 0.5 has its two spin bands centred at
# +-0.5. With the field tilted away from z the two spin channels mix in
# the lead's own basis, and at exactly E=+-0.5 and delta=1e-12 (the
# broadening every S-matrix is evaluated at) the Sancho-Rubio decimation
# converged to the advanced solution for one of the two channels. That
# solution satisfies the Dyson equation just as well as the retarded one
# (residual ~1e-14), so the residual check accepted it, and the junction
# lost that channel's transmission: a conductance of 0.96 instead of 1.90
# at theta=0.6pi, and the same at 0.8pi and 0.9pi, only at those two
# energies and only for some angles. The rg module now also requires the
# answer to be retarded, see rg.causality_violation.

THETAS = [0.6*np.pi, 0.8*np.pi, 0.9*np.pi] # angles that used to fail


def _lead(theta):
    h = geometry.chain().get_hamiltonian()
    h.add_exchange([0.5*np.sin(theta), 0., 0.5*np.cos(theta)])
    return h


def _junction(theta):
    h1 = geometry.chain().get_hamiltonian()
    h1.add_exchange([0., 0., 0.5])
    return heterostructures.create_leads_and_central(h1, _lead(theta), h1)


def _didv(ht, e):
    with warnings.catch_warnings(): # the broadening is raised there, and says so
        warnings.simplefilter("ignore")
        return ht.didv(energy=e)


@pytest.mark.parametrize("numba", [False, True])
@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("energy", [0.5, -0.5])
def test_surface_green_is_retarded_or_refused(theta, energy, numba):
    """At delta=1e-12 the decimation either returns a retarded surface
    Green's function or raises; it never returns the advanced one"""
    h = _lead(theta)
    intra = algebra.todense(h.intra)
    inter = algebra.todense(algebra.dagger(h.inter))
    try:
        gb, gs = rg.green_renormalization(intra, inter, energy=energy,
                                          delta=1e-12, numba=numba)
    except ValueError:
        return # refusing is allowed, a wrong answer is not
    assert rg.causality_violation(gs) < rg.dyson_tolerance
    assert rg.causality_violation(gb) < rg.dyson_tolerance


@pytest.mark.parametrize("theta", THETAS)
def test_batched_surface_green_is_retarded_or_refused(theta):
    h = _lead(theta)
    intra = algebra.todense(h.intra)
    inter = algebra.todense(algebra.dagger(h.inter))
    try:
        gb, gs = rg.green_renormalization_jit_batch(intra, inter,
                            np.array([0.4, 0.5, -0.5]), delta=1e-12)
    except ValueError:
        return
    assert np.all(rg.causality_violation(gs) < rg.dyson_tolerance)


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("energy", [0.5, -0.5])
def test_conductance_is_continuous_at_the_band_centre(theta, energy):
    """The conductance at exactly E=+-0.5 matches its value a hair away,
    instead of losing one of the two channels"""
    ht = _junction(theta)
    g0 = _didv(ht, energy)
    g1 = _didv(ht, energy*(1. - 1e-6))
    assert abs(g0 - g1) < 1e-3
    assert g0 > 1.5 # both channels conduct


def test_conductance_does_not_depend_on_what_was_computed_before():
    ht = _junction(THETAS[0])
    fresh = _didv(ht, 0.5)
    ht2 = _junction(THETAS[0])
    for e in np.linspace(-0.5, 0.5, 11): _didv(_junction(THETAS[1]), e)
    for e in np.linspace(-0.5, 0.5, 11): _didv(ht2, e)
    assert abs(_didv(ht2, 0.5) - fresh) < 1e-10


@pytest.mark.parametrize("numba", [False, True])
def test_a_negative_delta_gives_the_advanced_green_function(numba):
    """A negative delta asks for the advanced Green's function, which the
    causality check must accept rather than refuse as non-retarded"""
    h = _lead(THETAS[0])
    intra = algebra.todense(h.intra)
    inter = algebra.todense(algebra.dagger(h.inter))
    for delta in [-1e-3, -5.001]:
        gb, gs = rg.green_renormalization(intra, inter, energy=-5.,
                                          delta=delta, numba=numba)
        assert rg.causality_violation(gs, sign=-1.) < rg.dyson_tolerance
        gb2, gs2 = rg.green_renormalization(intra, inter, energy=-5.,
                                            delta=-delta, numba=numba)
        assert np.allclose(gs, np.conjugate(gs2.T)) # advanced = retarded^dag
