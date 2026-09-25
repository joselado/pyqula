"""The simultaneous density matrix accepts the same k-meshes as the
accumulate one.

full_dm_simultaneous used to normalize by 1/nk**dimensionality from the raw
argument, so a list or tuple nk, which klist.kmesh expands per direction,
stopped it with a TypeError while every other route took it."""
import itertools

import numpy as np
import pytest

from pyqula import geometry, multicell

FERMI = -0.4


def _haldane():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(0.1) # complex hoppings, so a transposed dm would show
    return h


def _supercell_dm(h, n1, n2):
    """<c_i^dag c_j> in one cell of a periodic n1 x n2 supercell, from a
    real-space diagonalization that shares no k-mesh or Bloch code with
    pyqula"""
    hd = {tuple(int(x) for x in d): np.array(m, dtype=complex)
          for d, m in multicell.get_hopping_dict(h).items()}
    n = h.intra.shape[0]
    cells = list(itertools.product(range(n1), range(n2)))
    idx = {c: i for i, c in enumerate(cells)}
    H = np.zeros((len(cells)*n,)*2, dtype=complex)
    for c in cells:
        for d, m in hd.items():
            c2 = ((c[0]+d[0]) % n1, (c[1]+d[1]) % n2)
            H[idx[c]*n:(idx[c]+1)*n, idx[c2]*n:(idx[c2]+1)*n] += m
    es, vs = np.linalg.eigh(H)
    return ((vs.conj()*(es < FERMI)) @ vs.T)[0:n, 0:n]


@pytest.mark.parametrize("nk", [[3, 5], (4, 4)])
def test_list_nk_matches_a_real_space_supercell(nk):
    h = _haldane()
    ref = _supercell_dm(h, *nk)
    dm = h.get_density_matrix(nk=nk, fermi=FERMI, dm_mode="simultaneous")
    assert np.max(np.abs(dm - ref)) < 1e-10


@pytest.mark.parametrize("nk", [[3, 5], (2, 3, 4)])
def test_simultaneous_matches_accumulate_with_directions(nk):
    if len(nk) == 3:
        h = geometry.cubic_lattice().get_hamiltonian(has_spin=True)
        h.add_exchange([0.3, 0.5, 0.2])
    else:
        h = _haldane()
    ds = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, -1, 0]]
    kw = dict(nk=nk, fermi=FERMI)
    a = h.get_density_matrix(dm_mode="accumulate", **kw)
    s = h.get_density_matrix(dm_mode="simultaneous", **kw)
    assert np.max(np.abs(a - s)) < 1e-12
    a = h.get_density_matrix(dm_mode="accumulate", ds=ds, **kw)
    s = h.get_density_matrix(dm_mode="simultaneous", ds=ds, **kw)
    for d in a:
        assert np.max(np.abs(a[d] - s[d])) < 1e-12
