"""The 2d k-maps of spectrum: ev2d on a Nambu Hamiltonian, operators given
by name, and operators that depend on the k-point.

ev2d sums an operator over every negative-energy state at each k-point. On
a Bogoliubov-de Gennes Hamiltonian at zero pairing these are the occupied
electrons at k and the holes of the empty states at -k, so a normal
observable came out as S(k)+S(-k) instead of S(k), unless the hole-hole
block of the operator is dropped as get_vev does.
"""

import numpy as np

from pyqula import geometry, spectrum
from pyqula.operators import Operator


def _spin_textured_honeycomb():
    """Rashba with a generic exchange: every spin component is textured in
    k"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.3, 0.5, 0.2])
    h.add_onsite(0.2)
    return h


def test_ev2d_bdg_copy_at_zero_pairing_matches_the_normal_state():
    h = _spin_textured_honeycomb()
    hb = h.copy()
    hb.setup_nambu_spinor() # the same state, described with BdG
    kw = dict(nk=2, nsuper=1, k0=[0.13, 0.29], write=False)
    for name in ["sx", "sy", "sz"]:
        rn = spectrum.ev2d(h, operator=[h.get_operator(name)], **kw)
        rb = spectrum.ev2d(hb, operator=[hb.get_operator(name)], **kw)
        assert np.max(np.abs(rn[:, :2]-rb[:, :2])) < 1e-12 # same k-points
        assert np.max(np.abs(rn[:, 2])) > 1e-3 # a nonzero texture
        assert np.max(np.abs(rn[:, 2]-rb[:, 2])) < 1e-8


def test_kmaps_resolve_an_operator_given_by_name():
    h = _spin_textured_honeycomb()
    kw = dict(nk=2, nsuper=1, k0=[0.13, 0.29], write=False)
    a = spectrum.ev2d(h, operator="sz", **kw)
    b = spectrum.ev2d(h, operator=[h.get_operator("sz")], **kw)
    assert np.max(np.abs(a-b)) < 1e-12
    a = spectrum.selected_bands2d(h, operator="sz", nindex=[-1, 1], **kw)
    b = spectrum.selected_bands2d(h, operator=[h.get_operator("sz")],
                                  nindex=[-1, 1], **kw)
    for (ai, bi) in zip(a, b): assert np.max(np.abs(ai-bi)) < 1e-12


def test_kmaps_apply_a_k_dependent_operator_at_its_k_point():
    """The Bloch Hamiltonian itself, as an operator of the k-point: summed
    over the occupied states it is the sum of the negative energies at k,
    and for a single band it is the energy of that band"""
    h = _spin_textured_honeycomb()
    hk = h.get_hk_gen()
    op = Operator(lambda v, k=None: hk(k) @ v)
    kw = dict(nk=2, nsuper=1, k0=[0.13, 0.29], write=False)
    r = spectrum.ev2d(h, operator=[op], **kw)
    for (x, y, s) in r:
        m = hk(np.array([x, y, 0.]))
        m = m.toarray() if hasattr(m, "toarray") else np.array(m)
        es = np.linalg.eigvalsh(m)
        assert abs(s-np.sum(es[es < 0.])) < 1e-8
    for band in spectrum.selected_bands2d(h, operator=[op], nindex=[-1, 1],
                                          reciprocal=False, **kw):
        assert np.max(np.abs(band[:, 2]-band[:, 3])) < 1e-8
