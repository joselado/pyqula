import numpy as np

from pyqula import geometry


def _brute_force_band_edges(h, nk=14):
    """Band edges from a genuine 3D mesh, as a reference for the k-path."""
    hk = h.get_hk_gen()
    ks = np.linspace(0., 1., nk, endpoint=False)
    es = [np.linalg.eigvalsh(np.array(hk([kx, ky, kz])))
          for kx in ks for ky in ks for kz in ks]
    es = np.array(es)
    return es.min(), es.max()


def test_default_path_of_a_cubic_lattice_reaches_the_band_edges():
    """A simple cubic lattice with t=1 has E(k) = -2(cos kx + cos ky + cos kz)
    and hence band edges at exactly -6 and +6. The default band path used to
    be built from the first two reciprocal vectors only, so it stayed in the
    k3=0 plane and reported a minimum of -2 -- a planar cut sold as the band
    structure of a 3D crystal."""
    h = geometry.cubic_lattice().get_hamiltonian()
    (k, e) = h.get_bands()
    assert abs(e.min() - (-6.)) < 1e-6, e.min()
    assert abs(e.max() - 6.) < 1e-6, e.max()


def test_default_path_of_a_diamond_lattice_reaches_the_band_edges():
    """Same check on a two-site 3D cell, where the edges are +-4."""
    h = geometry.diamond_lattice_minimal().get_hamiltonian()
    (k, e) = h.get_bands()
    emin, emax = _brute_force_band_edges(h)
    assert abs(e.min() - emin) < 1e-6, (e.min(), emin)
    assert abs(e.max() - emax) < 1e-6, (e.max(), emax)


def test_labelled_three_dimensional_path_uses_the_third_direction():
    """Every high-symmetry label used to sit in the k3=0 plane and the
    closest-replica search had no 3D branch, so a labelled 3D path was
    impossible to write down. G-X-M-G-R must now reach both band edges."""
    h = geometry.cubic_lattice().get_hamiltonian()
    (k, e) = h.get_bands(kpath=["G", "X", "M", "G", "R"])
    assert abs(e.min() - (-6.)) < 1e-6, e.min()
    assert abs(e.max() - 6.) < 1e-6, e.max()


def test_lower_dimensional_default_paths_are_unchanged():
    """The 1D and 2D branches must be untouched by the 3D one."""
    (k, e) = geometry.chain().get_hamiltonian().get_bands()
    assert abs(e.min() - (-2.)) < 1e-3 and abs(e.max() - 2.) < 1e-3
    (k, e) = geometry.honeycomb_lattice().get_hamiltonian().get_bands()
    assert abs(e.min() - (-3.)) < 1e-3 and abs(e.max() - 3.) < 1e-3
