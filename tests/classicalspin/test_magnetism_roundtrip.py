"""`SpinModel.write()` / `load_magnetism()` are documented as a checkpoint
round trip. They could not be: write_magnetization writes MAGNETISM.OUT with
six columns x,y,z,mx,my,mz, while load_magnetism defaulted to a filename
nothing writes and read columns 1,2,3 -- y, z and mx -- as the moment."""
import numpy as np

from pyqula import classicalspin, geometry


def test_write_then_load_restores_the_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # write() drops its files in the cwd
    g = geometry.chain().get_supercell(4)
    g.dimensionality = 0
    sm = classicalspin.SpinModel(g)
    sm.add_heisenberg()
    sm.minimize_energy()
    before = np.array(sm.get_magnetization())
    assert np.max(np.abs(before)) > 1e-6  # a nontrivial configuration
    sm.write()
    other = classicalspin.SpinModel(g)
    other.add_heisenberg()
    other.load_magnetism()
    after = np.array(other.get_magnetization())
    assert np.allclose(before, after, atol=1e-10), (before, after)


def test_xyz_generating_function_takes_a_plain_anisotropy_vector():
    """The XYZ branch passed the difference *vector* to the cutoff function
    fc, which takes a distance, so any non-callable v raised "truth value of
    an array with more than one element is ambiguous"."""
    g = geometry.chain().get_supercell(4)
    g.dimensionality = 0
    sm = classicalspin.SpinModel(g)
    sm.add_tensor(classicalspin.generating_functions(name="XYZ",
                                                     v=[1., 1., 0.5]))
    np.random.seed(0)
    sm.minimize_energy()
    m = np.array(sm.get_magnetization())
    # easy plane: the anisotropy is weakest along z, so the moments lie in xy
    # (to the minimizer's own tolerance, not to machine precision)
    assert np.max(np.abs(m[2])) < 1e-3, m
    assert np.max(np.abs(m[:2])) > 0.1, m
