import numpy as np
import pytest

from pyqula import algebra
from pyqula import geometry
from pyqula import operatorlist

# get_operator resolves named operators through a registry (name -> builder)
# in operatorlist.py rather than an if/elif chain. These tests pin the two
# properties that made the registry worth having: every advertised name is
# actually resolvable, and an unknown name fails loudly with the valid names
# in the message instead of the bare `raise` the chain used to end with
# (which surfaced as "RuntimeError: No active exception to reraise", saying
# nothing about the name that was wrong).


def get_hamiltonians():
    """A spinful and a Nambu Hamiltonian, both on a Kekule-commensurate
    honeycomb cell with a sublattice index, so that between the two of them
    every named operator has the degrees of freedom it needs (the pairing
    operators require Nambu, and are undefined without it)"""
    # store_primal is what the "unfold" operator needs to be definable
    g = geometry.honeycomb_lattice().get_supercell(3,store_primal=True)
    h = g.get_hamiltonian(has_spin=True)
    hsc = g.get_hamiltonian(has_spin=True)
    hsc.setup_nambu_spinor() ; hsc.add_swave(0.1)
    return [h,hsc]


def test_every_registered_name_is_resolvable():
    """Every name get_operator_names advertises builds in at least one of the
    Hilbert spaces, and none of them is reported as an unknown name"""
    hs = get_hamiltonians()
    names = operatorlist.get_operator_names()
    assert len(names)>40 # the registry is not accidentally empty
    for name in names:
        errors = []
        for h in hs:
            try:
                assert h.get_operator(name) is not None,name
                break # built somewhere, that is enough
            except Exception as e:
                assert "unknown operator" not in str(e),name # not a lookup miss
                errors.append(repr(e))
        else: raise AssertionError(name+" built nowhere: "+str(errors))


def test_unknown_name_raises_valueerror_listing_the_valid_ones():
    h = get_hamiltonians()[0]
    with pytest.raises(ValueError) as info:
        h.get_operator("not_an_operator")
    message = str(info.value)
    assert "not_an_operator" in message # says what was wrong
    assert "sz" in message # and what would have been right


def test_aliases_agree_with_their_canonical_name():
    """The alternative spellings return the same operator, not a similar one.
    Compared by their action on a vector rather than by their matrix, because
    some named operators (the bulk/edge projectors, for instance) are only
    defined as a function and have no matrix representation"""
    h = get_hamiltonians()[0]
    v = np.random.random(h.intra.shape[0]) + 1j*np.random.random(h.intra.shape[0])
    for (alias,name) in [("Sz","sz"),("Sx","sx"),("Sy","sy"),("Berry","berry"),
            ("Bulk","bulk"),("edge","surface"),("Edge","surface"),
            ("Surface","surface"),("electrons","electron"),("down","dn"),
            ("IPR","ipr"),
            ("valley_top","valley_upper"),("valley_bottom","valley_lower")]:
        oa = h.get_operator(alias).m(v,k=[0.,0.,0.])
        on = h.get_operator(name).m(v,k=[0.,0.,0.])
        assert np.max(np.abs(np.array(oa)-np.array(on)))<1e-10,alias


def test_objects_are_passed_through_not_looked_up():
    """Non-string inputs keep working: Operator, matrix, callable, None"""
    h = get_hamiltonians()[0]
    sz = h.get_operator("sz")
    assert h.get_operator(sz) is not None
    assert h.get_operator(None) is None
    assert h.get_operator(sz.get_matrix()) is not None
    assert h.get_operator(lambda r: 1.*(r[0]>0.)) is not None
