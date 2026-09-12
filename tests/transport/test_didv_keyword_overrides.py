"""The keyword and the attribute must be the same knob.

`delta` (the broadening) and, for a LocalProbe, `T` (the probe
transparency) are attributes of the junction/probe object that several
routines below `didv` read independently. Both were also accepted as
keywords of `didv` itself and then dropped: the answer came out
byte-identical to the call that passed nothing, so a caller who asked for
a different broadening (or a different transparency) got the object's own
value back without being told.

These tests assert the invariant rather than a recorded number: passing
`delta=X` to the call has to return exactly what building the junction
with that `delta` returns, and has to differ from the unmodified call --
the second half is what makes the first half non-vacuous.
"""
import numpy as np
import pytest

from pyqula import geometry, heterostructures
from pyqula.transporttk.localprobe import LocalProbe


def _normal_junction(delta):
    """Spinless chain junction, weakly coupled, with a given broadening"""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    ht = heterostructures.build(h, h)
    ht.set_coupling(0.3)
    ht.delta = delta
    return ht


def _bdg_junction(delta):
    """Normal lead against a superconducting one, in the Nambu basis"""
    g = geometry.chain()
    h1 = g.get_hamiltonian(); h1.add_swave(0.0)
    h2 = g.get_hamiltonian(); h2.add_swave(0.1)
    ht = heterostructures.build(h1, h2)
    ht.set_coupling(0.3)
    ht.delta = delta
    return ht


def _local_probe(delta):
    h = geometry.chain().get_hamiltonian()
    h.shift_fermi(1.)
    h.add_swave(0.1)
    lp = LocalProbe(h, delta=delta)
    lp.T = 0.2
    return lp


def test_delta_keyword_matches_the_attribute_on_a_normal_junction():
    """E=1.98 is right at the band edge, where the broadening genuinely
    changes the answer."""
    E, big = 1.98, 3e-1
    bare = _normal_junction(1e-6).didv(energy=E)
    passed = _normal_junction(1e-6).didv(energy=E, delta=big)
    attr = _normal_junction(big).didv(energy=E)
    assert abs(passed - attr) < 1e-12*abs(attr)
    assert abs(passed - bare) > 1e-3*abs(bare)


def test_delta_keyword_matches_the_attribute_on_a_BdG_junction():
    """The BdG branch (didv_BdG) never forwarded delta to get_smatrix at
    all, unlike the normal branch next to it."""
    E, big = 0.02, 2e-1
    bare = _bdg_junction(1e-6).didv(energy=E)
    passed = _bdg_junction(1e-6).didv(energy=E, delta=big)
    attr = _bdg_junction(big).didv(energy=E)
    assert abs(passed - attr) < 1e-12*abs(attr)
    assert abs(passed - bare) > 1e-3*abs(bare)


def test_delta_keyword_matches_the_construction_delta_on_a_local_probe():
    """A LocalProbe spends its delta in two places -- the probe
    selfenergy and the bulk Green's function of the sample -- so the
    keyword has to reach both to mean what LocalProbe(delta=...) means."""
    E, big = 0.05, 1e-1
    bare = _local_probe(1e-4).didv(energy=E)
    passed = _local_probe(1e-4).didv(energy=E, delta=big)
    attr = _local_probe(big).didv(energy=E)
    assert abs(passed - attr) < 1e-12*abs(attr)
    assert abs(passed - bare) > 1e-3*abs(bare)


def _full_junction(delta):
    """get_tmatrix needs a junction whose central part is not block
    diagonal, which is what a single central cell produces"""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    ht = heterostructures.create_leads_and_central(h, h, h, num_central=1)
    ht.delta = delta
    return ht


def test_tmatrix_delta_keyword_matches_the_attribute():
    """heterostructures.get_tmatrix declared a delta and never forwarded
    it to get_smatrix either."""
    E, big = 1.98, 3e-1
    bare = heterostructures.get_tmatrix(_full_junction(1e-6), energy=E)
    passed = heterostructures.get_tmatrix(_full_junction(1e-6), energy=E,
                                          delta=big)
    attr = heterostructures.get_tmatrix(_full_junction(big), energy=E)
    assert np.allclose(passed, attr, atol=1e-12, rtol=0.)
    assert not np.allclose(passed, bare, atol=1e-6, rtol=0.)


def test_didv_BdG_refuses_an_unknown_component():
    """component is selected by a string, so a typo must name itself
    instead of falling off the end of the if-chain with an unbound G."""
    from pyqula.transporttk.didv import didv_BdG
    ht = _bdg_junction(1e-4)
    for good in [None, "electron", "hole", "Andreev"]:
        didv_BdG(ht, energy=0.02, component=good)
    with pytest.raises(ValueError):
        didv_BdG(ht, energy=0.02, component="andreev")
