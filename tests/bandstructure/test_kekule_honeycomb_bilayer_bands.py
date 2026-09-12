import numpy as np
import pytest

from pyqula import specialhamiltonian


def _kekule_bilayer(ktop, kbot, ti=0.2):
    """Bilayer graphene (layers at z=+1.5 and z=-1.5) with an independent
    Kekule distortion on each layer. The supercell(3) multiplier is what the
    Kekule periodicity requires; it folds K and K' onto Gamma, where the
    distortion opens the gap."""
    h = specialhamiltonian.multilayer_graphene(l=[0, 1], ti=ti)
    h = h.get_supercell(3)
    h.add_kekule(lambda r: (r[2] > 0) * ktop)
    h.add_kekule(lambda r: (r[2] < 0) * kbot)
    return h


@pytest.mark.slow
def test_kekule_distortion_gaps_each_layer_by_its_own_amplitude(tmp_path,
                                                                monkeypatch):
    """A Kekule distortion of amplitude t opens a gap 2t at the folded Dirac
    point, so with the layers decoupled (ti=0) the band edge of each layer
    sits at |E| = t for that layer's own amplitude -- and <zposition> says
    which layer each band edge belongs to.

    This is the assertion the old sum(e)/sum(c) pair could not make: sum(e)
    is sum_k Tr H(k) = 0 for any Kekule amplitude on either layer, and
    sum(c) over a full band structure is nk*Tr(zposition) = 0 because the
    two layers sit at +-1.5. Both are satisfied by swapping the two
    amplitudes, by setting them to zero, or by making them ten times
    larger."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    ktop, kbot = 0.1, 0.2
    h = _kekule_bilayer(ktop, kbot, ti=0.)
    (k, e, z) = h.get_bands(operator="zposition", nk=20)
    e, z = np.array(e), np.array(z)
    # band edge of the weakly distorted (top) layer
    assert np.isclose(np.min(np.abs(e)), ktop, atol=1e-6)
    assert np.all(z[np.abs(np.abs(e) - ktop) < 1e-6] > 0.)
    # and of the strongly distorted (bottom) one
    assert np.all(z[np.abs(np.abs(e) - kbot) < 1e-6] < 0.)


@pytest.mark.slow
def test_kekule_bilayer_band_edge_lives_on_the_weakly_distorted_layer(tmp_path,
                                                                      monkeypatch):
    """With the layers coupled (ti=0.2) the gap is no longer a single
    layer's, but the states at the gap edge still polarize onto the layer
    with the *smaller* Kekule amplitude -- <zposition> follows whichever
    layer got the 0.1, and reverses when the two amplitudes are swapped.
    Marked slow: the supercell(3) multiplier is required by the Kekule
    distortion's periodicity and cannot be shrunk further."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    h = _kekule_bilayer(0.1, 0.2)
    (k, e, z) = h.get_bands(operator="zposition", nk=20)
    e, z = np.array(e), np.array(z)
    assert np.min(np.abs(e)) > 0.05  # the distortion gaps the bilayer
    edge = np.argsort(np.abs(e))[:4]
    assert np.all(z[edge] > 0.3)  # top layer, the one with 0.1

    hs = _kekule_bilayer(0.2, 0.1)  # swap the two layers' amplitudes
    (k, es, zs) = hs.get_bands(operator="zposition", nk=20)
    es, zs = np.array(es), np.array(zs)
    assert np.allclose(np.sort(es), np.sort(e))  # same spectrum, mirrored
    assert np.all(zs[np.argsort(np.abs(es))[:4]] < -0.3)
