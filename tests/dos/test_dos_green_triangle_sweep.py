"""h.get_dos(mode="Green", gmode="triangle") is a sweep in energy of the
trace of the bulk Green's function of a 3d crystal, from the tetrahedra of
the nk^3 mesh. The mesh does not depend on the energy, so the sweep
diagonalizes it once (green.green_operator_batch); it used to call
green.green_operator at every energy, one diagonalization of the mesh per
energy. The numbers are the ones of the single calls, and without an
operator they are the tetrahedron DOS of mode="ED", which sums the same
simplices without the projectors."""
import numpy as np

from pyqula import geometry
from pyqula import green


ES = np.array([-1.1, -0.3, 0.05, 0.4, 1.3])
DELTA, NK = 0.05, 6


def _crystal():
    """Spinful diamond with a sublattice imbalance and an exchange field,
    so that the sz trace is not zero"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.3)
    h.add_exchange([0., 0., 0.2])
    return h


def _dos(h, **kwargs):
    return np.array(h.get_dos(mode="Green", gmode="triangle", energies=ES,
                              delta=DELTA, nk=NK, write=False, **kwargs)[1])


def test_the_sweep_diagonalizes_the_mesh_once(monkeypatch):
    """one call of the tetrahedron Green's function, with all the
    energies, where a loop over the energies makes one call per energy"""
    from pyqula.dostk import simplexdos
    calls = []
    green_triangle = simplexdos.green_triangle

    def counted(h, energies, **kwargs):
        calls.append(len(np.atleast_1d(energies)))
        return green_triangle(h, energies, **kwargs)

    monkeypatch.setattr(simplexdos, "green_triangle", counted)
    h = _crystal()
    green.green_operator(h, e=ES[0], delta=DELTA, nk=NK, gmode="triangle")
    assert calls == [1] # the counter sees the single-energy route
    calls.clear()
    _dos(h)
    assert calls == [len(ES)]


def test_the_sweep_gives_the_single_energy_traces():
    h = _crystal()
    for op in [None, "sz"]:
        ds = _dos(h, operator=op)
        ref = np.array([green.green_operator(h, operator=op, e=e,
                delta=DELTA, nk=NK, gmode="triangle") for e in ES])/np.pi
        assert np.max(np.abs(ref)) > 1e-2 # not a vacuous trace
        assert np.max(np.abs(ds-ref)) < 1e-10*np.max(np.abs(ref))


def test_the_sweep_is_the_tetrahedron_dos():
    """-Im Tr G/pi from the band projectors is the simplex sum of the DOS
    on the same mesh at the same delta"""
    h = _crystal()
    ds = _dos(h)
    ed = np.array(h.get_dos(mode="ED", integration="triangle", energies=ES,
                            delta=DELTA, nk=NK, write=False)[1])
    assert np.max(np.abs(ed)) > 1e-2
    assert np.max(np.abs(ds-ed)) < 1e-10*np.max(np.abs(ed))
