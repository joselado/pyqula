import numpy as np

from pyqula import geometry, spectrum


def _zeeman_honeycomb():
    """A gapped, spin-split honeycomb model: the occupied and the empty
    state at a given k carry OPPOSITE <sz>, so an operator expectation
    taken from the wrong one of the two is a sign flip, not a small
    numerical difference."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.25])
    h.add_rashba(0.2)
    return h


def test_negative_band_reports_the_occupied_state(tmp_path, monkeypatch):
    """selected_bands2d's i<0 branch writes the VALENCE energy but took
    the operator expectation from wfpos, the CONDUCTION eigenvector. The
    oracle is an explicit <psi|O|psi> over the highest occupied state of
    the same H(k), from numpy.linalg.eigh -- not a recorded number."""
    monkeypatch.chdir(tmp_path)
    h = _zeeman_honeycomb()
    op = h.get_operator("sz")
    nk = 4
    spectrum.selected_bands2d(h, nindex=[-1, 1], operator=[op], nk=nk,
                              reciprocal=False)
    m = np.genfromtxt("BANDS2D__-1.OUT")
    # one row per k-point, four columns (kx, ky, energy, <sz>): the i<0
    # branch used to break the line right after the energy
    assert m.shape == (nk*nk, 4)
    M = op.get_matrix()
    hk = h.get_hk_gen()
    for (kx, ky, e, sz) in m:
        es, ws = np.linalg.eigh(hk(np.array([kx, ky, 0.])))
        ws = ws.T
        occ = [(ei, w) for (ei, w) in zip(es, ws) if ei < 0.]
        (eref, wref) = max(occ, key=lambda p: p[0]) # highest occupied
        assert abs(e-eref) < 1e-8
        szref = (np.conjugate(wref)@(M@wref)).real
        assert abs(sz-szref) < 1e-8


def test_ky_offset_is_honoured(tmp_path, monkeypatch):
    """The 2d k-maps build `kys = linspace(...)+k0[1]` and then iterated
    kxs twice, so the y offset of k0 was dropped. The invariant is the
    line's own statement: the second column must be that kys grid."""
    monkeypatch.chdir(tmp_path)
    h = _zeeman_honeycomb()
    nk, nsuper, k0 = 3, 1, [0., 0.5]
    expect = np.linspace(-nsuper, nsuper, nk) + k0[1]
    spectrum.selected_bands2d(h, nindex=[1], nk=nk, nsuper=nsuper,
                              reciprocal=False, k0=k0)
    m = np.genfromtxt("BANDS2D__1.OUT")
    assert np.max(np.abs(np.unique(m[:, 1]) - expect)) < 1e-10
    spectrum.ev2d(h, nk=nk, nsuper=nsuper, k0=k0)
    m = np.genfromtxt("EV2D.OUT")
    assert np.max(np.abs(np.unique(m[:, 1]) - expect)) < 1e-10


def test_ev2d_agrees_between_sparse_and_dense(tmp_path, monkeypatch):
    """ev2d sums an operator over EVERY occupied state, so its answer
    cannot depend on how the Hamiltonian happens to be stored. The sparse
    branch referred to an undefined `nindex` and had never run."""
    monkeypatch.chdir(tmp_path)
    h = _zeeman_honeycomb()
    hs = h.copy()
    hs.turn_sparse()
    op = h.get_operator("sz")
    spectrum.ev2d(h, nk=3, operator=[op])
    dense = np.genfromtxt("EV2D.OUT")
    spectrum.ev2d(hs, nk=3, operator=[op])
    sparse = np.genfromtxt("EV2D.OUT")
    assert np.max(np.abs(dense-sparse)) < 1e-8
