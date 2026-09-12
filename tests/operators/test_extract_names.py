import numpy as np
import pytest

from pyqula import extract, geometry


def _hamiltonian_for(name):
    """A Hamiltonian carrying whatever degree of freedom the quantity needs"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_exchange([0., 0., 0.3])
    if name in ("swave", "SC", "superfluidity", "deltak", "absolute_delta",
                "absolute_spatial_delta"):
        h.setup_nambu_spinor()
        h.add_swave(0.2)
    return h


def test_every_advertised_extractable_name_dispatches():
    """The accepted names used to be re-listed by hand inside the error
    message of the if/elif chain. They are now derived from the registry;
    this pins that each one really dispatches and returns something."""
    assert len(extract.extractable_names) > 0
    for name in extract.extractable_names:
        h = _hamiltonian_for(name)
        out = h.extract(name)
        assert out is not None, name


def test_extractable_names_is_derived_from_the_dispatch():
    assert list(extract.extractable_names) == extract.get_extractable_names()


@pytest.mark.parametrize("name", ["densty", "Mz", "not_a_quantity"])
def test_an_unknown_extractable_lists_the_accepted_ones(name):
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError) as e:
        h.extract(name)
    msg = str(e.value)
    assert name in msg
    for known in extract.extractable_names:
        assert known in msg, (known, msg)


def test_a_magnetization_on_a_spinless_hamiltonian_names_the_requirement():
    """mx/my/mz on a spinless Hamiltonian used to fall through to the
    else-branch and be reported as an *unknown quantity*, which named
    neither the real requirement nor its remedy"""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    for name in ["mx", "my", "mz"]:
        with pytest.raises(ValueError) as e:
            h.extract(name)
        msg = str(e.value)
        assert "spinful" in msg and "turn_spinful" in msg, (name, msg)
        assert "unknown quantity" not in msg, (name, msg)


def test_the_CDW_order_parameter_without_a_sublattice_raises():
    """It used to return None silently, so the caller got a TypeError
    somewhere downstream instead of being told the cell has no sublattice"""
    g = geometry.square_lattice()
    assert not g.has_sublattice
    with pytest.raises(ValueError) as e:
        g.get_hamiltonian().extract("CDW")
    assert "sublattice" in str(e.value)


def test_the_superfluidity_without_nambu_raises():
    """It used to return None silently for a Hamiltonian with no
    electron-hole degree of freedom"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    assert not h.has_eh
    with pytest.raises(ValueError) as e:
        h.extract("superfluidity")
    msg = str(e.value)
    assert "Nambu" in msg and "setup_nambu_spinor" in msg


def test_the_CDW_order_parameter_is_the_sublattice_staggered_density():
    """The invariant behind the fixed branch: on a sublattice-imbalanced
    honeycomb cell, extract('CDW') is the density measured against the
    two-coloring, so it is nonzero and opposite on the two sublattices"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_sublattice_imbalance(0.4)
    h = h.get_mean_field_hamiltonian(U=0.0, filling=0.5, nk=6, mf="CDW")
    cdw = np.array(h.extract("CDW"))
    assert np.max(np.abs(cdw)) > 1e-6
