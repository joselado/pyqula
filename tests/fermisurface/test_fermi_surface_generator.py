import numpy as np

from pyqula import geometry, algebra
from pyqula.fermisurface import fermi_surface_generator, fermi_weight


def _serial_reference(h, nk, energies, delta):
    hk_gen = h.get_hk_gen()
    kxs = np.linspace(-1, 1, nk, endpoint=True)
    kys = np.linspace(-1, 1, nk, endpoint=True)
    fR = h.geometry.get_k2K_generator()
    from pyqula.klist import kgrid2d
    rs = np.array(kgrid2d(kxs, kys))
    out = []
    for r in rs:
        hk = hk_gen(fR(r))
        es = algebra.eigvalsh(hk)
        out.append(fermi_weight(es, np.array(energies), delta=delta))
    return rs, np.array(out)


def test_fermi_surface_generator_matches_serial_reference():
    """fermi_surface_generator's default (dense, no operator) case is now
    batched through numba prange; check it still agrees with the plain
    per-kpoint loop it replaced. This is the multi-energy sibling of
    fermisurfacetk.singlefs.fermi_surface (already converted separately)."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    energies = [0.0, 0.2]
    delta = 0.1
    _, rs_new, kdos_new = fermi_surface_generator(h, energies=energies, nk=5, delta=delta)
    rs_old, kdos_old = _serial_reference(h, 5, energies, delta)
    assert np.allclose(rs_new, rs_old)
    assert np.allclose(kdos_new, kdos_old)


def test_fermi_surface_generator_operator_mode_still_works():
    """Operator-given case falls back to the original pcall path -- smoke
    test it still runs and returns the right shape."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    _, rs, kdos = fermi_surface_generator(h, energies=[0.0], nk=4, delta=0.1,
                                            operator="sz")
    assert kdos.shape == (len(rs), 1)
