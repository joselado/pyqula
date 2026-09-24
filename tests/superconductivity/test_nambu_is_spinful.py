"""A Nambu Hamiltonian is always spinful in pyqula. There used to be a
spinless Nambu mode, spinor (c, c^dag) per site, built by add_swave on a
spinless Hamiltonian, while turn_nambu on the same Hamiltonian made it
spinful first; half of the package then had to refuse the spinless one."""
import numpy as np
import pytest

from pyqula import geometry


def test_add_swave_on_a_spinless_chain_gives_a_spinful_bdg():
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h.add_swave(0.3)
    assert h.has_spin and h.has_eh
    assert h.intra.shape == (4, 4)  # 2 spin x 2 electron-hole, one site
    assert abs(h.get_gap() - 0.6) < 1e-8


def test_setup_nambu_spinor_and_turn_nambu_agree_on_a_spinless_chain():
    a = geometry.chain().get_hamiltonian(has_spin=False)
    a.add_onsite(0.3)
    a.setup_nambu_spinor()
    b = geometry.chain().get_hamiltonian(has_spin=False)
    b.add_onsite(0.3)
    b.turn_nambu()
    assert a.check_mode("spinful_nambu") and b.check_mode("spinful_nambu")
    assert (a - b).is_zero()


def test_there_is_no_spinless_nambu_mode():
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError) as e:
        h.check_mode("spinless_nambu")
    for name in ["spinless", "spinful", "spinful_nambu"]:
        assert name in str(e.value)


def test_a_hand_built_spinless_nambu_hamiltonian_fails_the_check():
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h.has_eh = True
    with pytest.raises(ValueError, match="always spinful"):
        h.check()


def test_attractive_hubbard_from_a_spinless_chain(tmp_path, monkeypatch):
    """The spinless attractive-Hubbard SCF is gone; the spinful one takes a
    Hamiltonian built as spinless, since setup_nambu_spinor makes it
    spinful"""
    monkeypatch.chdir(tmp_path)  # the SCF caches MF.pkl in the cwd
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h.setup_nambu_spinor()
    h2 = h.get_mean_field_hamiltonian(U=-2.0, filling=0.4, mf="swave", nk=20)
    assert h2 is not None and h2.get_gap() > 1e-2
