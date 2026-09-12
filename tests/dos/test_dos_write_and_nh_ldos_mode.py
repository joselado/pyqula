import numpy as np
import pytest

from pyqula import geometry


def test_green_dos_honours_write(tmp_path, monkeypatch):
    """Every DOS mode takes write=; the Green/RG branch was the one that
    both refused it and ignored it.

    It forwarded **kwargs straight into green.green_operator, which has no
    write argument, so h.get_dos(mode="Green", write=False) raised
    TypeError -- while the branch wrote DOS.OUT unconditionally, which is
    why the missing argument had never been noticed. Both halves are
    asserted here: the call succeeds, and the file appears only when asked
    for."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    energies = np.linspace(-1.0, 1.0, 5)
    (e, d) = h.get_dos(mode="Green", energies=energies, delta=0.1,
                       write=False)
    assert not (tmp_path/"DOS.OUT").exists()
    assert np.all(np.isfinite(d)) and np.all(d >= 0.)
    (e2, d2) = h.get_dos(mode="Green", energies=energies, delta=0.1,
                         write=True)
    assert (tmp_path/"DOS.OUT").exists()
    # write= must not change the numbers, only whether they are saved
    assert np.allclose(d, d2)


def test_non_hermitian_ldos_refuses_an_unimplemented_mode():
    """nonhermitiantk.ldos.get_ldos supports only the diagonalization
    mode. It used to print "<mode> not implemented" and then forward the
    unsupported mode anyway, so the caller silently received the
    diagonalization answer under another name. It raises now, and the
    supported mode still works."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False, non_hermitian=True)
    h.add_onsite(0.2j)
    with pytest.raises(NotImplementedError, match="diagonalization"):
        h.get_ldos(mode="KPM", write=False)
    d = h.get_ldos(mode="diagonalization", write=False)
    assert d is not None
