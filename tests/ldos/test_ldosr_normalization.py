import numpy as np
import pytest

from pyqula import islands, ldos


def test_real_space_ldos_integrates_to_one_state():
    """The continuum-space LDOS from ldos.ldosr_generator is a density of
    states, so its integral over all energies is the total weight the
    eigenstates put on the sampled neighbourhood -- and since the sampling
    weights are normalized to one and the site densities of a complete set
    of eigenstates sum to one per site, that integral is exactly 1.

    It used to be pi, because ldostk/ldosr.py accumulated calculate_dos's
    output raw. calculate_dos returns pi times a sum of unit-normalized
    Lorentzians (see dostk/eigtodos.py), and every routine that reports a
    density of states divides by pi -- dos.dos_kmesh, dos's two
    energy-window routines, dostk/adaptivedos, ldos.multi_ldos_tb and
    ldostk/atomicmultildos all do. This one did not, so a real-space LDOS
    map was pi times a real-space LDOS map.

    The window has to cover the whole spectrum and delta has to be small
    against it, so the residual here is the Lorentzian tail outside the
    grid, not the normalization.
    """
    g = islands.get_geometry(name="honeycomb", n=3, nedges=6, rot=0.0)
    h = g.get_hamiltonian(has_spin=False)
    es = np.linspace(-8.0, 8.0, 6001)
    f = ldos.ldosr_generator(h, es=es, delta=2e-2, nn=6, rs=0.2)
    for r in [g.r[0], g.r[len(g.r)//2]]:
        (e, y) = f(r)
        assert np.all(y >= 0.)  # a density of states is non-negative
        assert abs(np.trapezoid(y, e) - 1.0) < 5e-3


def test_real_space_ldos_refuses_spinless_nambu():
    """The routine has no spinless-Nambu branch and says so, rather than
    falling through to a wrong Hilbert-space slicing."""
    g = islands.get_geometry(name="honeycomb", n=2, nedges=3, rot=0.0)
    h = g.get_hamiltonian(has_spin=False)
    h.add_swave(0.2)  # spinless Nambu
    f = ldos.ldosr_generator(h, es=np.linspace(-2., 2., 50), nn=4)
    with pytest.raises(NotImplementedError, match="spinless Nambu"):
        f(g.r[0])
