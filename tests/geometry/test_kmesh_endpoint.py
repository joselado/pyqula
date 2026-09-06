import numpy as np

from pyqula.kpointstk.kmesh import kmesh

# kmesh forwarded `endpoint` to np.linspace in the 1D and 3D branches but
# called kmesh2d(nk,nsuper), whose two linspaces were pinned to
# endpoint=False -- so a 2D mesh came back the same either way.


def test_endpoint_is_honoured_in_every_dimensionality():
    nk = 3
    for dim in [1, 2, 3]:
        closed = np.array(kmesh(dim, nk=nk, endpoint=True))
        half_open = np.array(kmesh(dim, nk=nk, endpoint=False))
        assert closed.shape == half_open.shape
        assert np.max(np.abs(closed - half_open)) > 1e-8
        # a closed mesh reaches the zone boundary, a half-open one does not
        assert abs(np.max(closed[:, 0:dim]) - 1.0) < 1e-12
        assert np.max(half_open[:, 0:dim]) < 1.0


def test_the_default_mesh_is_unchanged():
    for dim in [1, 2, 3]:
        assert np.max(np.abs(np.array(kmesh(dim, nk=4))
                             - np.array(kmesh(dim, nk=4, endpoint=False)))) == 0
