import numpy as np
import pytest

from pyqula import geometry
from pyqula.scftk import mfconstrains


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


def _mean_field(sparse, nambu):
    """A mean field with charge, both magnetizations and a bond term"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True, is_sparse=sparse)
    h.add_rashba(0.3)
    h.add_exchange([0.1, 0.2, 0.3])
    h.add_onsite(0.2)
    if nambu: h.add_swave(0.2)
    return h, h.get_dict()


@pytest.mark.parametrize("nambu", [False, True])
def test_constraints_on_sparse_and_dense_mean_fields_agree(nambu):
    """The constraints now set the onsite spin entries of every site in one
    step, linear in the number of sites, where setting them one at a time
    rebuilt a sparse matrix whenever its pattern changed; a sparse and a
    dense mean field give the same constrained mean field"""
    for c in mfconstrains.known_constrains:
        out = []
        for sparse in [False, True]:
            h, mf = _mean_field(sparse, nambu)
            if c == "no_normal_term" and not nambu: continue
            out.append(mfconstrains.enforce_constrains(
                {k: v.copy() for k, v in mf.items()}, h, [c]))
        if len(out) < 2: continue
        for d in out[0]:
            assert np.max(np.abs(_dense(out[0][d]) - _dense(out[1][d]))) < 1e-14, c


def test_constraints_remove_what_they_name():
    """On the onsite spin block of every site: no_magnetism leaves the
    charge times the identity, no_inplane_magnetism removes the spin-flip
    entries, no_offplane_magnetism equalizes the two spins, and no_charge
    removes the trace while keeping the magnetization"""
    for sparse in [False, True]:
        h, mf = _mean_field(sparse, False)
        m0 = _dense(mf[(0, 0, 0)])
        def onsite(c):
            out = mfconstrains.enforce_constrains(
                {k: v.copy() for k, v in mf.items()}, h, [c])
            return _dense(out[(0, 0, 0)])
        n = m0.shape[0]//2
        up, dn = 2*np.arange(n), 2*np.arange(n) + 1
        charge = (m0[up, up] + m0[dn, dn])/2.
        m = onsite("no_magnetism")
        assert np.allclose(m[up, up], charge) and np.allclose(m[dn, dn], charge)
        assert np.allclose(m[up, dn], 0.) and np.allclose(m[dn, up], 0.)
        m = onsite("no_inplane_magnetism")
        assert np.allclose(m[up, dn], 0.) and np.allclose(m[up, up], m0[up, up])
        m = onsite("no_offplane_magnetism")
        assert np.allclose(m[up, up], m[dn, dn]) and np.allclose(m[up, dn], m0[up, dn])
        m = onsite("no_charge")
        assert np.allclose(m[up, up] + m[dn, dn], 0.)
        assert np.allclose(m[up, up] - m[dn, dn], m0[up, up] - m0[dn, dn])
