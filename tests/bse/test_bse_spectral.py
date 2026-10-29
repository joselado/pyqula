"""Exciton spectral functions along a q-path (bsetk/spectral.py).

The Chebyshev route never forms the BSE block, the dense route takes every
exciton from its eigenvectors, so the two are independent and must agree
once the trace is exact. The projections on the atoms must add up to the
unprojected density of states, the electron and the hole must sit on the
sublattices the mass term puts them on, the lowest peak must be the
lowest exciton of get_exciton_bands, and the structure factor must not
depend on the unit cell chosen, which only holds with the phase on the
orbital positions.
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain, gapped_honeycomb
from pyqula.bsetk.interaction import density_interaction


def _honeycomb():
    h = gapped_honeycomb(spinful=False)
    return h, density_interaction(h, V1=0.6, V2=0.2)


QPATH = [[0.13, 0.27, 0.], [0., 0., 0.]] # off the mesh, and Gamma
ES = np.linspace(-1., 8., 300)


@pytest.mark.parametrize("operator,particle", [(None, "electron"),
    ("sublatticeA", "electron"), ("sublatticeA", "hole")])
def test_chebyshev_density_of_states_equals_the_dense_one(operator, particle):
    """With nrand reaching the number of pairs the trace is exact, so the
    Chebyshev resolvent and the dense eigenvectors give the same map"""
    h, V = _honeycomb()
    kw = dict(qpath=QPATH, energies=ES, nk=5, V=V, delta=0.1,
            operator=operator, particle=particle)
    qs, es, a = h.get_exciton_kdos_bands(nrand=25, **kw)
    qs2, es2, b = h.get_exciton_kdos_bands(method="dense", **kw)
    assert np.array_equal(qs, qs2) and np.array_equal(es, es2)
    assert np.max(np.abs(a - b)) < 1e-8*np.max(b)


def test_projections_on_the_atoms_add_up_to_the_total():
    """The projectors on the two sublattices add up to one, so with the
    same random vectors the projected maps add up to the unprojected one;
    the mass term puts the conduction band, and so the electron, on
    sublattice A and the hole on B"""
    h, V = _honeycomb()
    kw = dict(qpath=QPATH, energies=ES, nk=6, V=V, delta=0.1, nrand=8)
    t = h.get_exciton_kdos_bands(**kw)[2]
    a = h.get_exciton_kdos_bands(operator="sublatticeA", **kw)[2]
    b = h.get_exciton_kdos_bands(operator="sublatticeB", **kw)[2]
    assert np.max(np.abs(a + b - t)) < 1e-12*np.max(t)
    ah = h.get_exciton_kdos_bands(operator="sublatticeA", particle="hole",
            **kw)[2]
    assert np.sum(a) > 2.*np.sum(ah)


def test_lowest_peak_is_the_lowest_exciton():
    """A random-vector trace only changes the weights of the poles, not
    where they are, so the onset of the map at each Q is the lowest
    Tamm-Dancoff exciton of get_exciton_bands"""
    h, V = _honeycomb()
    qpath = [[0.1, 0.05, 0.], [0.2, 0.1, 0.]]
    qs, es = h.get_exciton_bands(qpath=qpath, nk=6, V=V, tda=True, n=1)
    e = np.linspace(1., 3., 2001)
    q2, e2, d = h.get_exciton_kdos_bands(qpath=qpath, nk=6, V=V, nrand=4,
            energies=e, delta=0.01)
    for iq in range(len(qpath)):
        di = d[q2 == iq]
        # the first local maximum of the map
        i = np.nonzero((di[1:-1] > di[:-2]) & (di[1:-1] > di[2:]))[0][0] + 1
        assert abs(e[i] - np.real(es[qs == iq][0])) < 2e-3


def test_structure_factor_chebyshev_equals_dense():
    h, V = _honeycomb()
    kw = dict(qpath=QPATH, energies=ES, nk=6, V=V, delta=0.1)
    a = h.get_exciton_structure_factor(**kw)[2]
    b = h.get_exciton_structure_factor(method="dense", **kw)[2]
    assert np.max(np.abs(a - b)) < 1e-8*np.max(b)


def test_structure_factor_does_not_depend_on_the_unit_cell():
    """The same chain with a doubled cell: the density fluctuation at the
    same physical Q selects the same excitons out of the folded ones, and
    the supercell holds two primitive cells. Without the phase on the
    orbital positions, or with its sign flipped, the two differ by more
    than half"""
    out = []
    for n in [1, 2]:
        h = gapped_ionic_chain(nsuper=n)
        V = density_interaction(h, U=1.0, V1=0.5)
        out.append(h.get_exciton_structure_factor(
            qpath=[[0.1*n, 0., 0.], [0.37*n, 0., 0.]], nq=2,
            energies=np.linspace(0., 6., 200), nk=12//n, V=V, delta=0.1)[2])
    assert np.max(np.abs(out[1]/2. - out[0])) < 1e-8*np.max(out[0])


def test_wrong_options_are_refused():
    h, V = _honeycomb()
    with pytest.raises(ValueError, match="'electron', 'hole'"):
        h.get_exciton_kdos_bands(particle="both", V=V, nk=4)
    with pytest.raises(ValueError, match="'dense', 'kpm'"):
        h.get_exciton_structure_factor(method="lanczos", V=V, nk=4)
    with pytest.raises(TypeError, match="operator"):
        h.get_exciton_structure_factor(operator="sublatticeA", V=V, nk=4)
    with pytest.raises(ValueError, match="delta"):
        h.get_exciton_kdos_bands(delta=0., V=V, nk=4)


@pytest.mark.parametrize("particle", ["electron", "hole"])
def test_projection_is_the_density_of_the_exciton(particle):
    """The projected weight of one exciton x on an orbital a is the
    density of its electron, sum_kv |sum_c x_kvc C_c(k+Q)_a|^2, or of its
    hole, sum_kc |sum_v x_kvc conj(C_v(k)_a)|^2, written here from the
    eigenvectors. The Rashba coupling and the exchange field mix the spins
    and split the bands, so both valence bands of the window live on the
    same orbital and interfere there, and the hole side fails if the
    transpose of its operator is wrong. With the spin conserved (the MoS2
    model) each orbital sees one valence band, and it would not"""
    from pyqula.bsetk.solve import BSE
    from pyqula.bsetk.spectral import _PairOperator
    h = gapped_honeycomb(spinful=True)
    h.add_rashba(0.3)
    h.add_exchange([0.1, 0.2, 0.3])
    V = density_interaction(h, V1=0.6, V2=0.2)
    b = BSE(h, V=V, Q=[0.13, 0.27, 0.], nk=4, nv=2, nc=2, tda=True)
    pb = b.pairs
    nk, nv, nc = len(pb.kpoints), len(pb.vbands), len(pb.cbands)
    Cc = pb.ckq[:, pb.cbands, :] # (nk,nc,norb), at k+Q
    Cv = pb.ck[:, pb.vbands, :] # (nk,nv,norb), at k
    norb = Cc.shape[2]
    for a in [0, 3]: # spin up on A, spin down on B
        P = np.zeros((norb, norb)) ; P[a, a] = 1.
        op = _PairOperator(h, pb, P, particle)
        X = np.array(b.amplitudes[:3].T) # three excitons, (npair,3)
        w = np.real(np.sum(np.conj(X)*op.apply(X), axis=0))
        Xk = X.reshape(nk, nv, nc, -1)
        if particle == "electron":
            ref = np.sum(np.abs(np.einsum("kvcn,kc->kvn", Xk, Cc[:, :, a]))**2,
                    axis=(0, 1))
        else:
            ref = np.sum(np.abs(np.einsum("kvcn,kv->kcn", Xk,
                np.conj(Cv[:, :, a])))**2, axis=(0, 1))
        assert np.max(np.abs(w - ref)) < 1e-12
        assert np.max(ref) > 1e-2 # a weight that can fail
