import numpy as np
import pytest

from pyqula import geometry, meanfield


def _hamiltonian_for(mode):
    """A Hamiltonian carrying whatever degree of freedom the mode needs"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    if mode in ("swave", "s-wave superconductivity", "pwave"):
        h.setup_nambu_spinor()
    return h


def test_every_advertised_guess_mode_builds_a_mean_field():
    """Every name the guess dispatch advertises must actually build a guess.

    known_guesses used to be a second list maintained by hand next to the
    if/elif chain, so a mode could be advertised without being dispatched
    (or dispatched without being advertised) and nothing would notice. It
    is now derived from the registry, and this pins the other half: each
    advertised name produces a Hermitian mean field of the right size.
    """
    assert len(meanfield.known_guesses) > 0
    for mode in meanfield.known_guesses:
        h = _hamiltonian_for(mode)
        n = h.intra.shape[0]
        out = meanfield.guess(h, mode=mode, fun=0.17)
        assert out is not None, mode
        # the guess is either the intracell matrix or a hopping dictionary
        ms = list(out.values()) if isinstance(out, dict) else [out]
        assert len(ms) > 0, mode
        for m in ms:
            m = np.array(m.todense() if hasattr(m, "todense") else m)
            assert m.shape == (n, n), (mode, m.shape)
        if isinstance(out, dict): # a hopping dict must be Hermitian as a whole
            for d in out:
                dm = tuple(-i for i in d)
                assert dm in out, (mode, d)
                m1 = np.array(out[d].todense() if hasattr(out[d], "todense")
                              else out[d])
                m2 = np.array(out[dm].todense() if hasattr(out[dm], "todense")
                              else out[dm])
                assert np.max(np.abs(m1 - np.conjugate(m2).T)) < 1e-8, (mode, d)
        else:
            m = np.array(out.todense() if hasattr(out, "todense") else out)
            assert np.max(np.abs(m - np.conjugate(m).T)) < 1e-8, mode


def test_known_guesses_is_derived_from_the_dispatch():
    """The advertised names and the dispatch keys are the same object, so
    the two can no longer drift apart"""
    assert list(meanfield.known_guesses) == meanfield.get_guess_names()
    for mode in meanfield.known_guesses:
        assert mode in meanfield._guesses


@pytest.mark.parametrize("mode", ["feroo", "s_wave", "not_a_mode"])
def test_an_unknown_guess_mode_lists_the_accepted_ones(mode):
    """A mode selected by a string must be self-diagnosing: the error names
    the offending value and enumerates what is accepted"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError) as e:
        meanfield.guess(h, mode=mode)
    msg = str(e.value)
    assert mode in msg
    for name in meanfield.known_guesses:
        assert name in msg, (name, msg)


def test_a_spinful_guess_on_a_spinless_hamiltonian_names_the_requirement():
    """The guard must say what is missing and how to get it, not fail later
    inside add_zeeman with a shape error"""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError) as e:
        meanfield.guess(h, mode="ferro")
    msg = str(e.value)
    assert "ferro" in msg and "spinful" in msg and "turn_spinful" in msg


def test_a_superconducting_guess_without_nambu_names_the_requirement():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError) as e:
        meanfield.guess(h, mode="swave")
    msg = str(e.value)
    assert "swave" in msg and "Nambu" in msg and "setup_nambu_spinor" in msg


def test_a_sublattice_guess_without_a_sublattice_names_the_requirement():
    """A charge-density-wave guess needs a two-colored cell; without one it
    must say so rather than seed an array of zeros"""
    g = geometry.square_lattice()
    assert not g.has_sublattice
    with pytest.raises(ValueError) as e:
        meanfield.guess(g.get_hamiltonian(), mode="CDW")
    msg = str(e.value)
    assert "CDW" in msg and "sublattice" in msg
