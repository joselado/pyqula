"""Nambu/BdG entanglement. In the Nambu basis every orbital appears twice
(as a particle and as a hole), so the correlation matrix restricted to a
region has eigenvalues in (zeta,1-zeta) pairs and the entropy sum counts
every physical mode twice; entanglement.py divides it by two. These tests
pin that factor against the normal-state result of the very same
Hamiltonian at zero pairing, which is the only way to catch it: a wrong
factor would still produce a smooth, plausible-looking number."""

import numpy as np
import pytest

from pyqula import geometry


def _normal_chain(has_spin=True):
    """Chain away from half filling, so no level sits at E=0"""
    h = geometry.chain().get_hamiltonian(has_spin=has_spin)
    h.add_onsite(0.3)
    return h


@pytest.mark.parametrize("has_spin,mode",
                         [(True, "spinful_nambu"), (False, "spinless_nambu")])
def test_zero_pairing_bdg_reproduces_the_normal_state_entropy(has_spin, mode):
    """Adding a Nambu doubling with zero pairing does not change the
    physical state, so the entanglement entropy must be unchanged. This
    checks both the factor 1/2 and the assumption that the Nambu
    components of a site are contiguous (a wrong orbital layout would
    scramble particle and hole blocks across the cut and break this).
    Both Nambu flavors are checked: 4 components per site for a spinful
    Hamiltonian and 2 for a spinless one."""
    h = _normal_chain(has_spin=has_spin)
    h_bdg = h.copy()
    h_bdg.add_swave(0.0)
    assert h_bdg.has_eh
    assert h_bdg.check_mode(mode)
    for nsuper in [10, 14, 18]:
        s_normal = h.get_entanglement_entropy(nsuper=nsuper)
        s_bdg = h_bdg.get_entanglement_entropy(nsuper=nsuper)
        assert abs(s_bdg - s_normal) < 1e-9


def test_bdg_entanglement_spectrum_is_particle_hole_symmetric():
    """The Nambu doubling makes the single-particle entanglement spectrum
    symmetric under xi -> -xi, for any pairing amplitude."""
    h = _normal_chain()
    h.add_swave(0.3)
    xi = h.get_entanglement_spectrum(nsuper=14)
    finite = xi[np.abs(xi) < 20.0]  # drop the numerically empty/full levels
    assert len(finite) > 0
    assert np.allclose(np.sort(finite), np.sort(-finite), atol=1e-8)


def test_pairing_changes_the_entropy_but_keeps_it_finite():
    """A superconducting gap is a real change of the ground state, so the
    entropy must move; it must also stay finite and positive."""
    h = _normal_chain()
    h_bdg = h.copy()
    h_bdg.add_swave(0.0)
    h_sc = h.copy()
    h_sc.add_swave(0.4)
    s_normal = h_bdg.get_entanglement_entropy(nsuper=14)
    s_sc = h_sc.get_entanglement_entropy(nsuper=14)
    assert 0.0 < s_sc < s_normal
    assert abs(s_sc - s_normal) > 1e-3


def test_shifting_the_fermi_energy_of_a_bdg_hamiltonian_raises():
    """The chemical potential of a BdG Hamiltonian is already inside the
    Hamiltonian; shifting the Bogoliubov spectrum instead would silently
    return a wrong number, so it is rejected."""
    h = _normal_chain()
    h.add_swave(0.2)
    with pytest.raises(ValueError):
        h.get_entanglement_entropy(nsuper=14, fermi=0.1)
