import pytest

from pyqula import geometry


def test_valley_exchange_refuses_a_function_of_position():
    """most add_* methods take a callable of position, so a callable here
    is an easy mistake; it used to die unpacking it into (vx,vy,vz)"""
    h = geometry.honeycomb_lattice().get_supercell(3).get_hamiltonian()
    with pytest.raises(TypeError, match="add_valley_exchange"):
        h.add_valley_exchange(lambda r: [0.,0.,0.1])
    with pytest.raises(ValueError, match="three numbers"):
        h.add_valley_exchange([0.1,0.])


def test_crystal_field_refuses_a_function_and_points_at_add_onsite():
    """it used to die multiplying the function by a float, inside the
    crystal-field potential"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(TypeError, match="add_onsite"):
        h.add_crystal_field(lambda r: 0.1)


def test_incommensurate_inplane_valley_exchange_names_the_public_routine():
    """the message used to name kekule_registries, an internal function,
    and not the routine the user called or what to do about it"""
    g = geometry.honeycomb_lattice().get_supercell([2,2,1])
    h = g.get_hamiltonian()
    with pytest.raises(ValueError) as e:
        h.add_valley_exchange([0.1,0.,0.])
    msg = str(e.value)
    assert "add_valley_exchange" in msg
    assert "multiple of 3" in msg
    assert "kekule_registries" not in msg
