"""multicell.unit_cell_hoppings must describe exactly the cells
multicell.turn_multicell builds.

It was factored out of turn_multicell so that a read-only caller --
htk.kchain.detect_longest_hopping, called once per energy in a decimation
-- can look at a non-multicell Hamiltonian's hoppings without paying
turn_multicell's deepcopy of the whole Hamiltonian, geometry included.
Two places now describe the same cells, so the risk this file guards
against is that they drift apart, which would make detect_longest_hopping
quietly answer a different question from the one it used to.
"""
import numpy as np
import pytest

from pyqula import geometry, multicell
from pyqula.htk.kchain import detect_longest_hopping


def _cases():
    yield "chain", geometry.chain().get_hamiltonian()
    yield "square", geometry.square_lattice().get_hamiltonian()
    yield "honeycomb", geometry.honeycomb_lattice().get_hamiltonian()
    yield "kagome", geometry.kagome_lattice().get_hamiltonian()
    yield "honeycomb spinful", geometry.honeycomb_lattice().get_hamiltonian(
            has_spin=True)


@pytest.mark.parametrize("name,h", list(_cases()), ids=lambda x: str(x)[:20])
def test_unit_cell_hoppings_reproduces_turn_multicell(name, h):
    """Every cell turn_multicell attaches, and no other, with the same
    matrix -- including the daggered direction, which unit_cell_hoppings
    leaves implicit."""
    ref = {tuple(t.dir): np.array(t.m) for t in multicell.turn_multicell(h.copy())
                                                 .hopping}
    got = {}
    for (d, m) in multicell.unit_cell_hoppings(h):
        m = np.array(m)
        got[d] = m
        got[tuple(-np.array(d))] = np.conjugate(m).T
    assert set(got) == set(ref)
    for d in ref:
        assert np.max(np.abs(got[d]-ref[d])) == 0.


@pytest.mark.parametrize("name,h", list(_cases()), ids=lambda x: str(x)[:20])
def test_detect_longest_hopping_agrees_with_the_multicell_route(name, h):
    """The non-copying branch of detect_longest_hopping must answer what
    the turn_multicell one does, for a non-multicell Hamiltonian and for
    the multicell form of the same system."""
    multi = h.copy()
    multi.turn_multicell()
    assert detect_longest_hopping(h) == detect_longest_hopping(multi)


def test_detect_longest_hopping_sees_longer_range_hoppings():
    """A second-neighbour chain is the case the fast branch must NOT be
    taken for: get_hamiltonian(tij=...) returns a multicell Hamiltonian,
    and the answer has to be 2, not 1."""
    h = geometry.chain().get_hamiltonian(tij=[1., 0.4])
    assert detect_longest_hopping(h) == 2


def test_unit_cell_hoppings_refuses_a_multicell_hamiltonian():
    """It only describes the dedicated attributes (inter, tx/ty/txy/txmy);
    a multicell Hamiltonian stores h.hopping instead, so asking is a
    caller mistake rather than an empty answer."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.turn_multicell()
    with pytest.raises(ValueError):
        multicell.unit_cell_hoppings(h)


def test_detect_longest_hopping_does_not_copy_the_hamiltonian():
    """The whole point of the split: a read-only question must not
    deepcopy the Hamiltonian it is asked about."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    calls = []
    h.copy = lambda *a, **k: calls.append(1)  # would return None if called
    detect_longest_hopping(h)
    assert calls == []
