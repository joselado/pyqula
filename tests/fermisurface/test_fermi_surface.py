import numpy as np

from pyqula import geometry
from pyqula import algebra
from pyqula import parallel
from pyqula.fermisurfacetk.singlefs import fermi_surface


def _serial_reference(h, nk, e, delta):
    """Verbatim pre-refactor per-kpoint loop for mode='eigen', operator=None."""
    hk_gen = h.get_hk_gen()
    R = h.geometry.get_k2K_generator()
    kxs = np.linspace(-1, 1, nk)
    kys = np.linspace(-1, 1, nk)
    out = []
    for x in kxs:
        for y in kys:
            k = R(np.array([x, y, 0.]))
            hk = hk_gen(k)
            es = algebra.eigvalsh(hk)
            out.append(np.sum(delta/((e-es)**2+delta**2)))
    return np.array(out)


def test_fermi_surface_matches_serial_reference(tmp_path, monkeypatch):
    """h.get_fermi_surface()'s default mode ('eigen', no operator) is now
    batched through numba prange; check it still agrees with the plain
    per-kpoint loop it replaced."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    nk = 6
    e, delta = 0.2, 0.1
    kx, ky, new = fermi_surface(h, write=False, nk=nk, e=e, delta=delta)
    old = _serial_reference(h, nk, e, delta)
    assert new.shape == old.shape
    assert np.allclose(new, old)


def test_fermi_surface_independent_of_core_count(tmp_path, monkeypatch):
    """The batched path doesn't use parallel.pcall at all; result must not
    depend on the (now-irrelevant, for this mode) process-pool setting."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    try:
        outs = []
        for cores in (1, 2, 4):
            parallel.set_cores(cores)
            _, _, kdos = fermi_surface(h, write=False, nk=6, e=0.1, delta=0.1)
            outs.append(kdos)
        for o in outs[1:]:
            assert np.allclose(o, outs[0])
    finally:
        parallel.set_cores(1)


def test_fermi_surface_operator_mode_still_uses_pcall_path(tmp_path, monkeypatch):
    """mode='eigen' with an operator falls back to the original per-kpoint
    dispatch (h.get_dos per kpoint) -- just a smoke test that this path
    still runs and returns the right shape."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    kx, ky, kdos = fermi_surface(h, write=False, nk=4, e=0.1, delta=0.1,
                                  operator="sz")
    assert kdos.shape == kx.shape
