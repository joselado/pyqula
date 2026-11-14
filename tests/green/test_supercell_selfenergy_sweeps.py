"""green.supercell_selfenergy_batch and green.green_operator_batch answer, for
a whole set of energies, what supercell_selfenergy and green_operator answer
one energy at a time. With the tetrahedra of a 3d crystal (mode "triangle")
the sweep comes from one diagonalization of the nk^3 mesh, so the two agree
to rounding; every other mode still goes energy by energy, so the numbers
are identical."""
import numpy as np

from pyqula import geometry
from pyqula import green


ES = np.array([-1.1, -0.3, 0.05, 0.4, 1.3])
DELTA, NK = 0.05, 6


def _crystal():
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.3)
    h.add_exchange([0., 0., 0.2])
    return h


def _close(a, b, rtol=1e-10):
    a, b = np.array(a), np.array(b)
    return np.max(np.abs(a-b)) <= rtol*np.max(np.abs(b))


def test_the_triangle_selfenergy_sweep_matches_one_energy_at_a_time():
    h = _crystal()
    gs, ss = green.supercell_selfenergy_batch(h, ES, delta=DELTA, nk=NK,
                                              nsuper=1, gf_mode="triangle")
    for (e, g, s) in zip(ES, gs, ss):
        g1, s1 = green.supercell_selfenergy(h, e=e, delta=DELTA, nk=NK,
                                            nsuper=1, gf_mode="triangle")
        assert np.max(np.abs(g1)) > 1e-2
        assert _close(g, g1) and _close(s, s1)


def test_the_triangle_operator_trace_sweep_matches_one_energy_at_a_time():
    h = _crystal()
    for op in [None, "sz"]:
        got = green.green_operator_batch(h, ES, operator=op, delta=DELTA,
                                         nk=NK, gmode="triangle")
        ref = [green.green_operator(h, operator=op, e=e, delta=DELTA, nk=NK,
                                    gmode="triangle") for e in ES]
        assert np.max(np.abs(ref)) > 1e-2 # not a vacuous trace
        assert _close(got, ref)


def test_the_other_modes_are_unchanged():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    gs, ss = green.supercell_selfenergy_batch(h, ES[:3], delta=0.1, nk=6,
                                              nsuper=2)
    for (e, g, s) in zip(ES[:3], gs, ss):
        g1, s1 = green.supercell_selfenergy(h, e=e, delta=0.1, nk=6,
                                            nsuper=2)
        assert np.max(np.abs(g-g1)) == 0. and np.max(np.abs(s-s1)) == 0.
    got = green.green_operator_batch(h, ES[:3], delta=0.1, nk=6, gmode="full")
    ref = [green.green_operator(h, e=e, delta=0.1, nk=6, gmode="full")
           for e in ES[:3]]
    assert np.max(np.abs(ref)) > 1e-2
    assert np.max(np.abs(got-np.array(ref))) == 0.
