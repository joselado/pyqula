import numpy as np

from pyqula import geometry
from pyqula.strain import graphene_buckling


def _buckled_honeycomb(dt):
    """Honeycomb supercell(3) with a non-uniform (buckling) strain whose
    period matches the supercell. dt sets how strongly the strain modulates
    the hoppings; dt=0 leaves the pristine lattice."""
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(3)
    h = g.get_hamiltonian(has_spin=False, is_sparse=True)
    if dt != 0.:
        omega = np.pi * 2. / np.sqrt(g.a1.dot(g.a1))
        pot = graphene_buckling(omega=omega, dt=dt, geometry=g)
        h.add_strain(pot, mode="non_uniform")
    return g, h


def test_buckling_strain_modulates_the_hoppings_but_keeps_the_lattice_bipartite(
        tmp_path, monkeypatch):
    """Strain rescales the hoppings but does not add onsite terms, so the
    honeycomb lattice stays bipartite and its spectrum stays symmetric under
    E -> -E. That symmetry is the entire content of the old sum(eb)
    reference (sum(eb) = sum_k Tr H(k) = 0), which held for every strain
    amplitude; it is asserted here pointwise instead.

    What does depend on the amplitude is how far the strain pushes the band
    edge past the pristine 3t and how strongly it modulates the zero-energy
    LDOS across the supercell. At dt=0.2 the band edge moves by about 1% and
    the LDOS varies by a factor of about three from site to site; with no
    strain the LDOS is exactly uniform and the edge sits at 3t, and at
    dt=2.0 the edge reaches 5.4 and the LDOS varies by two orders of
    magnitude."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    dt = 0.2
    (g, h) = _buckled_honeycomb(dt)
    (kb, eb) = h.get_bands(num_bands=20)
    eb = np.sort(np.array(eb))
    assert np.allclose(eb, -eb[::-1], atol=1e-8)
    assert 3.005 < np.max(np.abs(eb)) < 3.2  # pristine band edge is 3t

    (x, y, ld) = h.get_ldos(e=0., nrep=2)
    ld = np.array(ld)
    assert 2. < np.max(ld) / np.min(ld) < 10.

    # sum rule: the density of states integrates to the number of orbitals
    # (the 5% slack is the 0.01 broadening leaking past the band edge and
    # the 60-point quadrature)
    energies = np.linspace(-3.5, 3.5, 60)
    h.turn_dense()
    (e, d) = h.get_dos(energies=energies, nk=8, delta=1e-2)
    norb = len(g.r)
    assert abs(np.sum(d) * (energies[1] - energies[0]) - norb) < 0.05 * norb
