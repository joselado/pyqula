"""The shared Hilbert-space guards, and the routines that were converted to
use them.

The same requirement -- "this needs the spin degree of freedom", "this needs
the electron-hole (Nambu) degree of freedom" -- used to be hand-written at
each call site, so it was worded a dozen different ways and a new routine had
to reinvent both the wording and the remedy. check.require_spin/require_nambu/
require_sublattice are the single home for it. What these tests pin is the
part a user depends on: the error names what was being computed, what is
missing, and how to get it.
"""

import numpy as np
import pytest

from pyqula import check, geometry


def spinless():
    return geometry.chain().get_hamiltonian(has_spin=False)


def spinful():
    return geometry.honeycomb_lattice().get_hamiltonian()


def test_require_spin_names_the_caller_the_requirement_and_the_remedy():
    with pytest.raises(ValueError) as e:
        check.require_spin(spinless(), "the widget")
    msg = str(e.value)
    assert "the widget" in msg      # what was being computed
    assert "spinful" in msg          # what is missing
    assert "turn_spinful" in msg     # how to get it
    check.require_spin(spinful(), "the widget") # passes, returns None


def test_require_nambu_names_the_caller_the_requirement_and_the_remedy():
    with pytest.raises(ValueError) as e:
        check.require_nambu(spinful(), "the widget")
    msg = str(e.value)
    assert "the widget" in msg
    assert "Nambu" in msg
    assert "setup_nambu_spinor" in msg
    h = spinful()
    h.setup_nambu_spinor()
    check.require_nambu(h, "the widget")


def test_require_sublattice_accepts_a_hamiltonian_or_a_geometry():
    g = geometry.square_lattice()
    assert not g.has_sublattice
    for obj in [g, g.get_hamiltonian()]:
        with pytest.raises(ValueError) as e:
            check.require_sublattice(obj, "the widget")
        msg = str(e.value)
        assert "the widget" in msg and "sublattice" in msg
    check.require_sublattice(geometry.honeycomb_lattice(), "the widget")


# Entry points that really reach a guard. Note that most Hamiltonian
# *methods* that add a spin term (add_exchange, add_zeeman, add_kane_mele)
# call turn_spinful() first and so promote a spinless Hamiltonian rather
# than rejecting it -- that is deliberate, and the guards below sit on the
# routines that cannot do that because they only *read* the spin.
SPINFUL_CALLS = [
    ("a spin operator", lambda h: h.get_operator("sx")),
    ("the magnetization", lambda h: h.get_magnetization()),
    ("a spin rotation", lambda h: h.global_spin_rotation(angle=0.3)),
    ("a spin spiral", lambda h: h.generate_spin_spiral(qspiral=[0.3, 0., 0.])),
    ("the spin mixing", lambda h: h.extract("spin_mixing")),
    ("the magnetization by name", lambda h: h.extract("mz")),
    ("the magnetic correlator",
     lambda h: _susceptibility().dominant_correlation(h)),
    ("the spin splitting", lambda h: h.get_average_spin_splitting()),
]


def _susceptibility():
    from pyqula import susceptibility
    return susceptibility


@pytest.mark.parametrize("what,call", SPINFUL_CALLS,
                         ids=[c[0] for c in SPINFUL_CALLS])
def test_a_routine_needing_spin_says_so_on_a_spinless_hamiltonian(what, call):
    """Each of these used to raise its own hand-written message, in its own
    wording, sometimes without naming the remedy at all (and the
    magnetization extracted by name was reported as an *unknown quantity*).
    They must all name the missing degree of freedom and how to get it."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError) as e:
        call(h)
    msg = str(e.value)
    assert "spinful" in msg, (what, msg)
    assert "turn_spinful" in msg, (what, msg)


NAMBU_CALLS = [
    ("a pairing operator", lambda h: h.get_operator("spair")),
    ("the superfluidity", lambda h: h.extract("superfluidity")),
]


@pytest.mark.parametrize("what,call", NAMBU_CALLS,
                         ids=[c[0] for c in NAMBU_CALLS])
def test_a_routine_needing_nambu_says_so_without_it(what, call):
    h = geometry.honeycomb_lattice().get_hamiltonian()
    assert not h.has_eh
    with pytest.raises(ValueError) as e:
        call(h)
    msg = str(e.value)
    assert "Nambu" in msg, (what, msg)
    assert "setup_nambu_spinor" in msg, (what, msg)


def test_the_guards_do_not_fire_on_a_hamiltonian_that_has_what_is_needed():
    """The converted guards must be no-ops in the supported case -- the
    point of the sweep was to unify the message, not to add restrictions"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_exchange([0., 0., 0.3])
    assert np.max(np.abs(h.get_magnetization())) > 1e-6
    h.add_kane_mele(0.05)
    hn = h.copy()
    hn.setup_nambu_spinor()
    hn.add_swave(0.2)
    assert hn.get_operator("spair") is not None
    assert np.max(np.abs(hn.extract("superfluidity"))) > 1e-6


UNUSED_KWARG_CALLS = [
    (lambda: geometry.chain().get_hamiltonian().generate_spin_spiral(
        qspiral=[0.3, 0., 0.], fractionl=True), "fractional"),
    (lambda: __import__("pyqula.strain", fromlist=["x"]).uniaxial_strain(
        geometry.honeycomb_lattice().get_hamiltonian(), d=[1., 0., 0.],
        ss=0.1), "s"),
    (lambda: __import__("pyqula.specialhamiltonian",
                        fromlist=["x"]).excitonic_bilayer(gapp=1.), "gap"),
]


@pytest.mark.parametrize("call,suggestion", UNUSED_KWARG_CALLS,
                         ids=["spin_spiral", "uniaxial_strain",
                              "excitonic_bilayer"])
def test_a_mistyped_keyword_is_rejected_rather_than_swallowed(call, suggestion):
    """These three took a **kwargs bag they never read, so a mistyped
    keyword was silently ignored and the routine ran with its default --
    the worst failure mode, because the result looks fine. The bag is gone,
    so Python itself rejects the typo and suggests the real name."""
    with pytest.raises(TypeError) as e:
        call()
    assert suggestion in str(e.value)
