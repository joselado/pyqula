"""integration="qtci" of the mean-field density matrix in 3D
(qtcitk/densitymatrix_qtci.py): the tetrahedron weights of dmtk/simplexdm.py
on a 2^R x 2^R x 2^R mesh whose band energies and band projectors come from
quantics cross interpolations with three quantics coordinates. On the same
mesh it must agree with integration="triangle" (the tetrahedra in 3D) to the
interpolation tolerance, entry by entry, and the self-consistent state it
reaches must be the triangle one."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.dmtk import simplexdm
from pyqula.qtcitk.densitymatrix_qtci import (get_fermi4filling_qtci,
        dm_entries_qtci, get_dm_qtci)
from pyqula.scftk.densitydensity import Vinteraction


def _low_symmetry_metal():
    """Cubic metal with Rashba coupling and a tilted exchange field:
    complex projectors that vary along every band, no spin symmetry left"""
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=True)
    h.add_rashba(0.5)
    h.add_exchange([0.4, 0., 0.1])
    return h.get_multicell()


def test_qtci_3d_matches_the_tetrahedra_on_the_same_mesh_in_a_metal():
    """Every entry, with Bloch phases along the three axes, agrees with
    integration="triangle" at the same nk to the interpolation tolerance,
    with the Fermi level of the qtci search"""
    h = _low_symmetry_metal()
    nk = 8
    h.shift_fermi(-get_fermi4filling_qtci(h, 0.3, nk=nk, T=1e-3))
    ds = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)]
    needed = {(d, i, j) for d in ds for i in range(2) for j in range(2)}
    dq = dm_entries_qtci(h, needed, nk=nk, T=1e-3)
    ref = simplexdm.get_dm(h, ds, nk=nk, T=1e-3)
    err = max(abs(dq[d][i, j]-ref[d][i, j]) for (d, i, j) in needed)
    assert err < 1e-6
    assert abs(ref[(0, 0, 1)][0, 0]) > 1e-2 # a nontrivial bond along z
    assert abs(np.trace(dq[(0, 0, 0)]).real-0.6) < 1e-6 # the filling


def test_get_dm_qtci_still_rejects_1d():
    h = geometry.chain().get_hamiltonian()
    v = {(0, 0, 0): np.zeros((1, 1), dtype=np.complex128)}
    with pytest.raises(NotImplementedError, match="2D and 3D"):
        get_dm_qtci(h, v, nk=4)


def _magnetization(dm):
    n = dm.shape[0]
    return sum((dm[2*i, 2*i]-dm[2*i+1, 2*i+1]).real for i in range(n//2))


def test_qtci_scf_reaches_the_tetrahedron_state_of_the_cubic_ferromagnet():
    """The cubic Hubbard ferromagnet (U=10, filling 0.3, T=1e-3) is a
    metal: the majority band is partly filled, and the minority band holds
    about 0.005 electrons. The qtci mean field must reach the tetrahedron
    state of the same 8^3 mesh: the same magnetization, Fermi level and
    total energy, with the filling exact"""
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=True)
    out = []
    for integration in ("qtci", "triangle"):
        np.random.seed(1)
        out.append(Vinteraction(h.copy(), nk=8, integration=integration,
            U=10.0, filling=0.3, mf="ferro", T=1e-3,
            solver="broyden_mixing", maxerror=1e-8, load_mf=False,
            write=False, verbose=0))
    sq, st = out
    assert sq.converged and st.converged
    mq = _magnetization(sq.dm[(0, 0, 0)])
    mt = _magnetization(st.dm[(0, 0, 0)])
    assert 0.5 < abs(mt) < 0.599 # polarized, both spins occupied
    assert abs(mq-mt) < 1e-6
    assert abs(sq.hamiltonian.fermi-st.hamiltonian.fermi) < 1e-6
    assert abs(sq.total_energy-st.total_energy) < 1e-6
    assert abs(np.trace(sq.dm[(0, 0, 0)]).real-0.6) < 1e-6
