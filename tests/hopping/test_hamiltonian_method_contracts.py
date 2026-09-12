"""Four public Hamiltonian methods that broke their own contract.

`remove_sites` replaced the geometry before checking whether it could do the
job at all, `get_no_multicell` handed back the receiver instead of a new
Hamiltonian, `print_hamiltonian` read an attribute only a non-multicell 1d
Hamiltonian carries, and `enforce_eh` died on a Python-2 absolute import two
lines above the NotImplementedError it meant to raise.

Each test states the invariant rather than a recorded number: a method that
raises must leave the object as it found it, a get_* must return something
the caller can mutate, and a printer must work on every dimensionality the
library builds."""
import numpy as np
import pytest

from pyqula import geometry


def bandwidth(h):
    (k, e) = h.get_bands(write=False)  # no BANDS.OUT in the working directory
    return np.max(e) - np.min(e)


# ---------------------------------------------------------------- remove_sites

def test_remove_sites_refuses_a_spinful_hamiltonian_without_touching_it():
    """The class invariant every other method maintains is
    intra.shape[0] == (spin factor)*len(geometry.r). A call that raises must
    not be the one that breaks it."""
    g = geometry.honeycomb_lattice().get_supercell(2)
    h = g.get_hamiltonian()  # spinful by default
    n0, s0 = len(h.geometry.r), h.intra.shape[0]
    assert s0 == 2*n0
    store = np.ones(n0, dtype=int)
    store[0] = 0  # drop one site
    with pytest.raises(NotImplementedError):
        h.remove_sites(store)
    assert len(h.geometry.r) == n0
    assert h.intra.shape[0] == s0
    assert h.intra.shape[0] == 2*len(h.geometry.r)


def test_remove_sites_keeps_the_supported_case_consistent():
    """The spinless branch is the one that is implemented; check it leaves
    the geometry and the matrices the same size."""
    g = geometry.honeycomb_lattice().get_supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    n0 = len(h.geometry.r)
    store = np.ones(n0, dtype=int)
    store[0] = 0  # drop one site
    h.remove_sites(store)
    assert len(h.geometry.r) == n0 - 1
    assert h.intra.shape[0] == len(h.geometry.r)


# ------------------------------------------------------------ get_no_multicell

def test_get_no_multicell_returns_a_hamiltonian_the_caller_may_mutate():
    """ccdee4a's acceptance test, applied to the sibling it missed: for any
    get_* that promises a new object, mutating the result must leave the
    receiver's spectrum alone."""
    h = geometry.chain().get_hamiltonian()  # already non-multicell
    b0 = bandwidth(h)
    h2 = h.get_no_multicell()
    assert h2 is not h
    h2.add_onsite(1.0)
    assert abs(bandwidth(h) - b0) < 1e-8, (bandwidth(h), b0)


def test_get_no_multicell_does_not_alias_in_three_dimensions():
    """A 3d Hamiltonian has no non-multicell form, so the routine passes it
    through -- but passing it through must still not be an alias."""
    h = geometry.cubic_lattice().get_hamiltonian()
    assert h.is_multicell and h.dimensionality == 3
    b0 = bandwidth(h)
    h2 = h.get_no_multicell()
    assert h2 is not h
    h2.add_onsite(1.0)
    assert abs(bandwidth(h) - b0) < 1e-8, (bandwidth(h), b0)


def test_a_multicell_hamiltonian_still_converts_and_round_trips():
    """The conversion itself has to keep working: the non-multicell form of
    a 2d Hamiltonian must reproduce the same Bloch matrices."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.turn_multicell()
    h2 = h.get_no_multicell()
    assert h2 is not h
    assert not h2.is_multicell
    assert h.same_hamiltonian(h2)


# --------------------------------------------------------- print_hamiltonian

@pytest.mark.parametrize("name", ["chain", "honeycomb_lattice",
                                  "square_lattice", "cubic_lattice"])
def test_print_hamiltonian_works_in_every_dimensionality(name, capsys):
    """It used to read h.inter, an attribute only a non-multicell 1d
    Hamiltonian carries, so it died on every 2d and 3d lattice."""
    h = getattr(geometry, name)().get_hamiltonian()
    dd = h.get_multihopping().get_dict()  # the blocks that actually exist
    h.print_hamiltonian()
    out = capsys.readouterr().out
    assert "Intracell" in out
    # every cell the Hamiltonian couples to has to be named: the old code
    # could only ever show one of them
    for key in dd:
        if np.max(np.abs(key)) == 0: continue
        assert str(tuple([int(i) for i in key])) in out, (key, out)


def test_print_hamiltonian_shows_the_entries_of_a_chain(capsys):
    """The 1d chain is the case that used to work. Its intercell block has
    one entry per spin, and they must still be printed."""
    h = geometry.chain().get_hamiltonian()
    h.print_hamiltonian()
    out = capsys.readouterr().out
    assert "2 stored elements" in out, out


# ------------------------------------------------------------------ enforce_eh

def test_enforce_eh_raises_its_own_guard_and_leaves_the_hamiltonian_alone():
    """The guard the author wrote is a NotImplementedError; the exception
    the caller used to get was a ModuleNotFoundError naming a top-level
    package that does not exist -- raised after the method had already
    converted the Hamiltonian to multicell form."""
    h = geometry.chain().get_hamiltonian()
    assert not h.is_multicell
    with pytest.raises(NotImplementedError):
        h.enforce_eh()
    assert not h.is_multicell


# -------------------------------------------------------------------- rotate90

def test_rotating_a_hamiltonian_leaves_the_original_unrotated():
    """multicell.rotate90 built its working copy with turn_multicell, which
    hands back the receiver itself when it is already multicell -- so
    rotating an already-multicell Hamiltonian swapped the hopping directions
    and the lattice vectors of the input, in place."""
    from pyqula import multicell
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.turn_multicell()
    a1, a2 = np.array(h.geometry.a1).copy(), np.array(h.geometry.a2).copy()
    dirs = [tuple(t.dir) for t in h.hopping]
    hr = multicell.rotate90(h)
    assert hr is not h
    assert np.allclose(h.geometry.a1, a1) and np.allclose(h.geometry.a2, a2)
    assert [tuple(t.dir) for t in h.hopping] == dirs
    # and the rotation really happened on the object that was returned
    assert np.allclose(hr.geometry.a1, a2) and np.allclose(hr.geometry.a2, a1)
