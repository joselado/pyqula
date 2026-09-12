import numpy as np

from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix


def _tbg_inplane(b, phi=0.828, nk=201, num_bands=4):
    """Twisted bilayer graphene at the smallest commensurate moire index
    with the interlayer hopping switched off (ti=0), so the only thing the
    in-plane field can do is act on each layer separately.

    Returns, for each k, the num_bands band energies closest to zero and
    their <zposition> (the layers sit at z = +-1.5). The 28-orbital cell is
    diagonalised densely and the bands are picked here rather than through
    get_bands(num_bands=...): the sparse solver starts from a random vector
    and, where the bands are degenerate, does not always return the same
    four of them."""
    g = specialgeometry.twisted_bilayer(1)
    h = g.get_hamiltonian(is_sparse=True, has_spin=False,
                          mgenerator=twisted_matrix(ti=0.0))
    h.turn_dense()
    if b != 0.:
        h.add_inplane_bfield(b=b, phi=phi)
    (k, e, c) = h.get_bands(operator="zposition", nk=nk)
    k, e, c = np.array(k), np.array(e), np.array(c)
    nb = len(e) // nk
    sel = np.argsort(np.abs(e.reshape(nk, nb)), axis=1)[:, :num_bands]
    return (np.take_along_axis(e.reshape(nk, nb), sel, 1),
            np.take_along_axis(c.reshape(nk, nb), sel, 1))


def _dirac_node_index(e, c):
    """k-index at which the top layer's Dirac node sits."""
    top = c > 0.5
    ik = np.indices(e.shape)[0]
    return ik[top][np.argmin(np.abs(e[top]))]


def test_inplane_field_polarises_the_layers_and_shifts_them_in_momentum(
        tmp_path, monkeypatch):
    """An in-plane field enters through a Peierls phase proportional to the
    height of the bond above the mid-plane, so with the interlayer hopping
    off it is exactly a rigid momentum shift, opposite for the two layers.
    Three consequences, none of which the old references could see:

    * with no field the two decoupled layers are identical, so every band is
      exactly twofold degenerate;
    * with the field on, the degeneracy is lifted and every state becomes
      fully layer-polarised (|<zposition>| = 1.5), because the field cannot
      hybridise layers that do not hop into each other;
    * the momentum shift is linear in the field, so the Dirac node of the
      top layer travels twice as far along the k-path at b=0.02 as at
      b=0.01.

    The old sum(c) reference was nk*Tr(zposition) restricted to the selected
    bands, which the layer symmetry sends to zero at b=0.02 just as it does
    at b=0 and at b=1.98."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    b = 0.02

    (e0, c0) = _tbg_inplane(0.)
    p = np.sort(e0, axis=1)
    assert np.allclose(p[:, 1] - p[:, 0], 0., atol=1e-9)
    assert np.allclose(p[:, 3] - p[:, 2], 0., atol=1e-9)

    (e1, c1) = _tbg_inplane(b)
    p = np.sort(e1, axis=1)
    assert np.max(p[:, 1] - p[:, 0]) > 0.05  # degeneracy lifted
    assert np.allclose(np.abs(c1), 1.5, atol=1e-6)  # and fully layer-polarised

    # the shift of the Dirac node is linear in the field
    n0 = _dirac_node_index(e0, c0)
    (eh, ch) = _tbg_inplane(b / 2.)
    assert abs((_dirac_node_index(e1, c1) - n0)
               - 2 * (_dirac_node_index(eh, ch) - n0)) <= 1
