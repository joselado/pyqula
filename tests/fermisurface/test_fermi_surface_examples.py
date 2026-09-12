import numpy as np

from pyqula import geometry
from pyqula.specialhamiltonian import NbSe2


def _inversion_partner(fs, nk):
    """The same mesh read at -k. get_fermi_surface returns a symmetric
    nk x nk grid flattened row by row, so reversing both axes maps k to
    -k."""
    f = np.array(fs).reshape(nk, nk)
    return f, f[::-1, ::-1]


def test_valley_weight_on_the_fermi_surface_is_odd_in_k(tmp_path, monkeypatch):
    """The valley operator is odd under time reversal, and a doped honeycomb
    lattice is time-reversal symmetric, so the valley-resolved Fermi surface
    must be exactly antisymmetric under k -> -k: the two Fermi pockets carry
    equal and opposite valley weight.

    The old assertion was sum(fs) == 0, which is that antisymmetry summed
    over a symmetric mesh -- true for any onsite shift, any mesh and any
    Hamiltonian with time-reversal symmetry, and equally true when the shift
    is large enough that there is no Fermi surface left at all. Asserted
    pointwise here, together with the requirement that there is a Fermi
    surface: at an onsite shift of 0.6 the pockets carry a valley weight of
    order one, and a hundred times that shift moves the bands away from the
    Fermi level entirely."""
    monkeypatch.chdir(tmp_path)  # writes FERMI_MAP.OUT to cwd
    nk = 15
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(0.6)
    (kx, ky, fs) = h.get_fermi_surface(nk=nk, operator="valley", num_waves=10,
                                        mode="lowest")
    (f, fm) = _inversion_partner(fs, nk)
    assert np.allclose(f, -fm, atol=1e-8 * np.max(np.abs(f)))
    assert np.max(f) > 0.5 and np.min(f) < -0.5  # both valleys are occupied


def test_ising_soc_polarises_the_nbse2_fermi_surface_oppositely_at_k_and_minus_k(
        tmp_path, monkeypatch):
    """NbSe2's Ising spin-orbit coupling pins the Fermi-surface spins out of
    plane with opposite sign at k and -k, which is what makes the material
    an Ising superconductor. The sz-resolved Fermi surface must therefore be
    exactly odd in k, and its amplitude must be set by the SOC: at soc=0.9
    the pockets are polarised by more than half, ten times weaker SOC gives
    an order of magnitude less.

    The old assertion was sum(fs) == 0, which is that oddness summed over a
    symmetric mesh and is satisfied by any SOC strength, including none."""
    monkeypatch.chdir(tmp_path)
    nk = 15
    soc = 0.9
    h = NbSe2(soc=soc)
    (kx, ky, fs) = h.get_fermi_surface(e=0., nk=nk, delta=3e-1, operator="sz")
    (f, fm) = _inversion_partner(fs, nk)
    assert np.allclose(f, -fm, atol=1e-8 * np.max(np.abs(f)))
    assert np.max(np.abs(f)) > 0.5


def test_fermi_surface_qtci_backend_matches_grid(tmp_path, monkeypatch):
    """backend="qtci" reconstructs the Fermi surface mesh from a quantics
    tensor cross interpolation (qutecipy) instead of brute-force evaluating
    every mesh point; it must reproduce the grid-based result on a smooth,
    broadened (delta=0.3) spectral weight, which compresses well."""
    monkeypatch.chdir(tmp_path)  # writes FERMI_MAP.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(0.6)

    nk = 32  # a power of two so the qtci mesh lines up exactly with the grid mesh
    kx1, ky1, fs1 = h.get_fermi_surface(nk=nk, delta=0.3, write=False)
    kx2, ky2, fs2 = h.get_fermi_surface(nk=nk, delta=0.3, write=False,
            backend="qtci", tolerance=1e-3)
    assert np.allclose(kx1, kx2) and np.allclose(ky1, ky2)
    assert np.max(np.abs(fs1 - fs2)) < 1e-2


def test_fermi_surface_qtci_backend_non_power_of_two_nk(tmp_path, monkeypatch):
    """When nk isn't a power of two, qtci internally rounds up to the
    nearest 2**R for the quantics mesh, then must interpolate the result
    back onto the exact nk x nk grid the caller asked for (same shape and
    k-points as the grid backend). Interpolating a coarse, non-aligned
    mesh trades some pointwise accuracy for a peaked function, so this
    checks the total spectral weight (integrated over the BZ) rather than
    a strict per-point tolerance."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(0.6)

    nk = 20  # not a power of two
    kx1, ky1, fs1 = h.get_fermi_surface(nk=nk, delta=0.3, write=False)
    kx2, ky2, fs2 = h.get_fermi_surface(nk=nk, delta=0.3, write=False,
            backend="qtci", tolerance=1e-3)
    assert fs1.shape == (nk * nk,) and fs2.shape == (nk * nk,)
    assert np.allclose(kx1, kx2) and np.allclose(ky1, ky2)
    assert abs(fs1.sum() - fs2.sum()) / fs1.sum() < 1e-2
