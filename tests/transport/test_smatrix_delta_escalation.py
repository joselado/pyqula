import numpy as np
import pytest
import warnings

from pyqula import geometry, heterostructures, multihopping


def _lead():
    """A two-orbital 1d lead whose inter-cell hopping is genuinely
    non-Hermitian and non-symmetric, with a vanishing onsite block.

    Both properties matter. A one-orbital chain cannot exercise any of
    this: its decimation is exact (residual 0.0 even at delta=1e-12), so
    the surface Green's function always resolves. 62 of the 63 lead
    geometries in tests/keldysh and tests/transport were
    geometry.chain(), which is why this went unnoticed."""
    g = geometry.chain().supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    d = h.get_multihopping().get_dict()
    t = np.array([[0.3, 0.9], [0.1, 0.4]], dtype=complex)
    d[(1, 0, 0)] = np.matrix(t)
    d[(-1, 0, 0)] = np.matrix(t.conj().T)
    d[(0, 0, 0)] = np.matrix(np.zeros((2, 2), dtype=complex))
    h.set_multihopping(multihopping.MultiHopping(d))
    return h


def test_smatrix_raises_its_broadening_instead_of_failing():
    """get_smatrix clamped the lead broadening to delta_smatrix=1e-12
    unconditionally. On a lead with a state essentially at the evaluated
    energy the surface Green's function grows like 1/delta, and at 1e-12
    on a multi-orbital lead the cancellation needed is ~1e-12 out of
    numbers of size 1e11 -- which double precision cannot carry, so
    greentk.rg refuses it rather than return a wrong Green's function.
    didv() therefore could not produce a conductance at that energy at
    all, while landauer() on the identical junction could, because it
    uses the junction's own delta.

    The broadening is now raised only as far as it must be and never past
    the junction's own delta, so the call returns."""
    ht = heterostructures.build(_lead(), _lead())
    ht.set_coupling(0.3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = ht.didv(energy=0.0)
        assert any("delta" in str(w.message) for w in caught), \
            "a raised broadening must be reported, not applied silently"
    assert np.isfinite(g) and g >= 0.


def test_escalation_agrees_with_the_landauer_route():
    """The independent check: landauer() evaluates the same transmission
    without going through the Fisher-Lee S-matrix, and never had the
    clamp. The two must agree at the energy that used to fail, and away
    from it."""
    central = [_lead() for _ in range(3)]
    ht = heterostructures.build(_lead(), _lead(), central=central)
    ht.set_coupling(0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for energy in [0.0, 0.2, 0.5]:
            d = ht.didv(energy=energy)
            ll = ht.landauer(energy=energy)
            assert abs(d - ll) < 1e-3 + 1e-2*abs(ll)


def test_a_resolvable_energy_is_untouched_and_silent():
    """The escalation must not perturb the ordinary case: where the
    clamped broadening resolves, the answer and the broadening are what
    they always were, and nothing is warned about. 0.14489064219000647 is
    the value this junction returned before the change."""
    ht = heterostructures.build(_lead(), _lead())
    ht.set_coupling(0.3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = ht.didv(energy=0.5)
        # specifically no broadening warning; unrelated warnings (numpy's
        # np.matrix deprecation, say) must not make this test fail
        assert not [w for w in caught if "delta" in str(w.message)]
    assert abs(g - 0.14489064219000647) < 1e-12


def test_a_one_orbital_chain_never_needs_the_escalation():
    """Control: the case the whole test suite used to be built out of.
    The decimation is exact on a single-orbital chain, so even at E=0 the
    clamped broadening resolves and no warning is issued."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    ht = heterostructures.build(h, h)
    ht.set_coupling(0.3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = ht.didv(energy=0.0)
        assert not [w for w in caught if "delta" in str(w.message)]
    assert np.isfinite(g)
