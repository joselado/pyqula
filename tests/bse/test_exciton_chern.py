"""The Chern number of an exciton band, from the eigenvectors of the BSE
lifted to the orbital pair basis (bsetk/topology.py).

Four independent things are pinned here:

1. The lift itself, exactly: a one-body operator at momentum transfer Q,
   written in the pair basis, lifts to the operator projected between the
   empty states at k+Q and the occupied ones at k (and the other way round
   for the antiresonant block), whatever phases the Bloch states carry.
2. A nonzero integer against an independent route: in the Frenkel limit,
   decoupled molecules with flat bands, the BSE restricted to the pairs on
   one molecule is exactly a tight-binding model of the exciton, written
   down here from H_int and a Frenkel exciton without the BSE code. Its
   spectrum at Q and its Chern number are what the BSE must give.
3. A zero by construction: an electron in a Chern band bound to a hole on
   an orbital with no hopping and no hybridization. The exciton band is
   flat and spanned by localized states, so C = 0 on any mesh of Q. On a
   mesh of Q equal to the k-mesh every plaquette flux vanishes on top of
   that, since the BSE at two neighbouring Q is the same problem
   translated by one k-point, which checks the lift and not the physics.
4. The Bloch gauge: gauge=None, gauge="phase" and random phases on the
   Bloch states give the same lifted state and the same integer, on a
   saturated ferromagnet whose electron-hole pairs are all spin flips, so
   its excitons are magnons with Chern numbers +1 and -1; the raw
   amplitudes, with no lift, are not the same state.

Three more things are pinned, none of which the integer can see. The
record of full_output (energies, gap, distance below the continuum) is
compared with the oracle at the same mesh points, which also fixes the
momentum each flux belongs to, since solving at -Q leaves the integer.
The Casida metric is compared with the norm of the eigenproblem and with
the engine. And the three refusals (a group that touches another mode, a
mesh that does not resolve the group, an unstable reference) are each
raised on a case built for it.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.bsetk import pairbasis
from pyqula.bsetk.pairbasis import PairBasis
from pyqula.bsetk.solve import BSE
from pyqula.bsetk.interaction import density_interaction
from pyqula.bsetk.topology import (lift_amplitudes, exciton_chern,
        mode_states, mode_berry_fluxes, get_metric_names, get_mode_names,
        casida_metric)
from pyqula.topologytk.plaquette import chern_from_states, berry_fluxes

SHIFT = (0.013, 0.029) # keeps the special points out of the mesh of Q


def _scrambler(seed):
    """A replacement of pairbasis.mesh_eigenstates that puts a random phase
    on every Bloch eigenvector, a gauge transformation"""
    orig = pairbasis.mesh_eigenstates
    rng = np.random.default_rng(seed)
    def scrambled(hk, ks):
        ek, ck = orig(hk, ks)
        return ek, ck*np.exp(2j*np.pi*rng.random(ck.shape[0:2]))[:, :, None]
    return scrambled


def _phase_free_overlap(v0, v1):
    return abs(np.vdot(v0, v1))/np.linalg.norm(v0)/np.linalg.norm(v1)


# the Frenkel model: kagome lattice whose up triangles are decoupled
# molecules threaded by a flux, spinless, one electron per molecule

def _molecule(phi, e0=0.5):
    H = np.zeros((3, 3), dtype=np.complex128)
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        H[j, i] = -np.exp(1j*phi/3.)
        H[i, j] = np.conj(H[j, i])
    return H + e0*np.eye(3)


def _frenkel(phi=0.3, V1=1.0):
    g = geometry.kagome_lattice()
    h = g.get_hamiltonian(has_spin=False).get_multicell()
    h.set_multihopping(MultiHopping({(0, 0, 0): _molecule(phi)}))
    return h, density_interaction(h, V1=V1)


def _frenkel_oracle(phi, W, Q):
    """H[c',c] = (e_c-e_v) delta + sum_ab rho_c'(a) W(Q)_ab conj(rho_c(b))
                 - sum_ab conj(u_c'(a)) u_c(a) W_ab(0) |u_v(b)|^2,
    rho_c(a) = conj(u_c(a)) u_v(a), W(Q) = sum_d W(d) exp(2 pi i Q.d), for
    the exciton sum_R exp(2 pi i Q.R) sum_c x_c c^dag_{Rc} c_{Rv}|GS> of
    H_int = 1/2 sum W_ij(R'-R) n_Ri n_R'j"""
    e, u = np.linalg.eigh(_molecule(phi))
    u = u.T
    cs = [1, 2]
    rho = np.array([np.conj(u[c])*u[0] for c in cs])
    WQ = sum(np.array(m)*np.exp(2j*np.pi*(Q[0]*d[0] + Q[1]*d[1]))
            for d, m in W.items())
    H = np.diag([e[c] - e[0] for c in cs]).astype(np.complex128)
    H = H + rho@WQ@np.conj(rho).T
    W0, dens = np.array(W[(0, 0, 0)]), np.abs(u[0])**2
    for i, ci in enumerate(cs):
        for j, cj in enumerate(cs):
            H[i, j] -= np.sum(np.conj(u[ci])[:, None]*u[cj][:, None]
                    *W0*dens[None, :])
    return H


def _oracle_chern(phi, W, nq):
    def states(q):
        _, v = np.linalg.eigh(_frenkel_oracle(phi, W, q))
        return v[:, 0:1].T
    return chern_from_states(states, nq=nq, shift=SHIFT)


# the zero model: a Haldane honeycomb plus a filled orbital at the center
# of the hexagon, with no hopping, bound to its neighbours by V1

def _haldane_and_flat_hole(V1=3.0, eflat=-3.0):
    g = geometry.honeycomb_lattice()
    h2 = g.get_hamiltonian(has_spin=False)
    h2.add_haldane(0.15)
    h2.add_onsite(3.5) # both Haldane bands empty
    center = g.r[0] - (g.r[1] - g.r[0]) # six honeycomb sites at distance 1
    g3 = g.copy()
    g3.r = np.vstack([g.r, center[None, :]])
    g3.r2xyz()
    g3.has_sublattice = False
    g3.get_fractional()
    d3 = dict()
    for d, m in h2.get_multicell().get_hopping_dict().items():
        m3 = np.zeros((3, 3), dtype=np.complex128)
        m3[0:2, 0:2] = np.array(m.todense() if hasattr(m, "todense") else m)
        d3[tuple(int(x) for x in d)] = m3
    d3[(0, 0, 0)][2, 2] = eflat
    h = g3.get_hamiltonian(has_spin=False).get_multicell()
    h.set_multihopping(MultiHopping(d3))
    return h, density_interaction(h, V1=V1)


def test_lift_reproduces_a_projected_one_body_operator():
    """sum_k c^dag_{a,k+Q} O_ab c_{b,k} has the pair amplitudes
    X = <c,k+Q|O|v,k> and Y = <v,k+Q|O|c,k>, and its lift must be
    P_emp(k+Q) O P_occ(k) and P_occ(k+Q) O P_emp(k), in any Bloch gauge"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(0.15)
    h.add_sublattice_imbalance(0.8)
    h.add_zeeman([0.1, 0.05, 0.3]) # no conserved spin
    rng = np.random.default_rng(1)
    orig = pairbasis.mesh_eigenstates
    for scramble in (False, True):
        if scramble: pairbasis.mesh_eigenstates = _scrambler(2)
        try: pb = PairBasis(h, Q=[0.13, 0.27, 0.], nk=4)
        finally: pairbasis.mesh_eigenstates = orig
        norb, nk = pb.ck.shape[1], len(pb.kpoints)
        O = rng.normal(size=(norb, norb)) + 1j*rng.normal(size=(norb, norb))
        X = np.einsum("ma,ab,mb->m", np.conj(pb.el), O, pb.ho)
        Y = np.einsum("ma,ab,mb->m", np.conj(pb.hoA), O, pb.elA)
        v = lift_amplitudes(pb, X, Y)[0]
        Phi = v[:nk*norb**2].reshape(nk, norb, norb)
        Psi = v[nk*norb**2:].reshape(nk, norb, norb)
        P = lambda C, bands: sum(np.outer(C[n], np.conj(C[n])) for n in bands)
        for ik in range(nk):
            Pv, Pc = P(pb.ck[ik], pb.vbands), P(pb.ck[ik], pb.cbands)
            Pvq, Pcq = P(pb.ckq[ik], pb.vbands), P(pb.ckq[ik], pb.cbands)
            assert np.max(np.abs(Phi[ik] - Pcq@O@Pv)) < 1e-12
            assert np.max(np.abs(Psi[ik] - Pvq@O@Pc)) < 1e-12


def test_frenkel_spectrum_is_the_exciton_tight_binding_model():
    """The two lowest excitons are the eigenvalues of the oracle at Q, and
    not at -Q, which pins the sign of the momentum; the next level is the
    first charge-transfer pair, flat"""
    phi = 0.3
    h, W = _frenkel(phi)
    for Q in ([0.13, 0.27, 0.], [0.41, 0.07, 0.]):
        es = np.sort(BSE(h, V=W, Q=Q, nk=6, tda=True, max_memory=1.0)
                .get_energies().real)
        eo = np.linalg.eigvalsh(_frenkel_oracle(phi, W, Q))
        emq = np.linalg.eigvalsh(_frenkel_oracle(phi, W, [-Q[0], -Q[1], 0.]))
        assert np.max(np.abs(es[0:2] - eo)) < 1e-10
        assert np.max(np.abs(es[0:2] - emq)) > 1e-2
        assert es[2] > eo[1] + 0.1 # the Frenkel bands are isolated


@pytest.mark.parametrize("nk,nq", [(4, 6), (6, 6), (6, 8)])
def test_frenkel_chern_number_matches_the_oracle(nk, nq):
    phi = 0.3
    h, W = _frenkel(phi)
    co = _oracle_chern(phi, W, nq)
    assert abs(round(co)) == 1 # a Chern band, so not a vacuous check
    r = exciton_chern(h, bands=[0], nq=nq, shift=SHIFT, V=W, nk=nk,
            tda=True, max_memory=1.0, full_output=True)
    assert abs(r.chern - round(r.chern)) < 1e-8
    assert round(r.chern) == round(co)
    assert r.gap > 0.1 and r.bound > 0.1 # isolated and bound
    # the record against the oracle at the same mesh points. The integer
    # does not see a solve at -Q, a rotation of the plane; the energies
    # do, since E(Q) != E(-Q) with a flux
    assert r.qs.shape == (nq, nq, 3)
    assert np.allclose(r.qs[2, 1], [2./nq + SHIFT[0], 1./nq + SHIFT[1], 0.])
    eo = np.array([[np.linalg.eigvalsh(_frenkel_oracle(phi, W, q))
            for q in row] for row in r.qs])
    emq = np.array([[np.linalg.eigvalsh(_frenkel_oracle(phi, W, -q))
            for q in row] for row in r.qs])
    assert np.max(np.abs(r.energies[:, :, 0] - eo[:, :, 0])) < 1e-10
    assert np.max(np.abs(r.energies[:, :, 0] - emq[:, :, 0])) > 0.1
    # the gap is the one to the upper Frenkel band, and the continuum of
    # flat molecular levels starts at the lowest transition of a molecule
    assert abs(r.gap - np.min(eo[:, :, 1] - eo[:, :, 0])) < 1e-10
    em = np.linalg.eigvalsh(_molecule(phi))
    assert abs(r.bound - (em[1] - em[0] - np.max(eo[:, :, 0]))) < 1e-10


def test_casida_metric_is_the_norm_of_the_eigenproblem():
    """The lift is an isometry of each block, so the lifted mode has the
    Euclidean norm X^dag X + Y^dag Y and, in the metric of casida_metric,
    the norm X^dag X - Y^dag Y of the linear-response problem. The fluxes
    of metric="casida" are those the engine gives with that metric, they
    differ from the Euclidean ones plaquette by plaquette (measured: by
    up to 0.018, for fluxes up to 0.54), and the two sums agree"""
    h, W = _frenkel(0.3)
    kw = dict(V=W, nk=6, tda=False, max_memory=1.0)
    Q = [0.13, 0.27, 0.]
    b = BSE(h, Q=Q, **kw)
    v = mode_states(h, "exciton", Q=Q, bands=[0], **kw)[0]
    S = casida_metric(b.pairs)
    x2 = np.sum(np.abs(b.amplitudes[0])**2)
    y2 = np.sum(np.abs(b.amplitudesY[0])**2)
    assert y2 > 1e-3 # the two norms are different numbers
    assert abs(np.vdot(v, v).real - (x2 + y2)) < 1e-10
    assert abs(np.vdot(v, S*v).real - (x2 - y2)) < 1e-10
    fkw = dict(bands=[0], nq=6, shift=SHIFT, **kw)
    fe, _, _ = mode_berry_fluxes(h, "exciton", **fkw)
    fc, _, _ = mode_berry_fluxes(h, "exciton", metric="casida", **fkw)
    states = lambda q: mode_states(h, "exciton", Q=q, bands=[0], **kw)
    f2, _ = berry_fluxes(states, nq=6, shift=SHIFT, metric=S)
    assert np.max(np.abs(fc - f2)) < 1e-10
    assert np.max(np.abs(fc - fe)) > 1e-3
    assert abs(np.sum(fc) - np.sum(fe)) < 1e-8


def test_full_bse_gives_the_same_integer_in_both_metrics():
    """With the coupling block on, Y != 0, and the Casida metric diag(1,-1)
    and the Euclidean one give the same integer, the one of the oracle,
    also with random phases on the Bloch states"""
    phi = 0.3
    h, W = _frenkel(phi)
    b = BSE(h, V=W, Q=[0.13, 0.27, 0.], nk=6, tda=False, max_memory=1.0)
    assert np.linalg.norm(b.amplitudesY[0]) > 0.05*np.linalg.norm(b.amplitudes[0])
    co = round(_oracle_chern(phi, W, 6))
    orig = pairbasis.mesh_eigenstates
    for scramble in (False, True):
        if scramble: pairbasis.mesh_eigenstates = _scrambler(8)
        try:
            for metric in (None, "casida"):
                c = exciton_chern(h, bands=[0], nq=6, shift=SHIFT, V=W, nk=6,
                        tda=False, max_memory=1.0, metric=metric)
                assert abs(c - co) < 1e-8, (metric, scramble)
        finally:
            pairbasis.mesh_eigenstates = orig


@pytest.mark.parametrize("nk,nq", [(6, 4), (8, 6)])
def test_zero_by_construction(nk, nq):
    """Electron confined to the C = +1 band (nc=1), hole flat and
    localized: C = 0 on meshes of Q that are not commensurate with the
    k-mesh, where the plaquette fluxes are not identically zero"""
    h, W = _haldane_and_flat_hole()
    hk = h.get_hk_gen()
    def band(q):
        _, v = np.linalg.eigh(hk(q))
        return v[:, 1:2].T # the lowest Haldane band, the electron's
    assert abs(round(chern_from_states(band, nq=12))) == 1
    r = exciton_chern(h, bands=[0], nq=nq, shift=SHIFT, V=W, nk=nk, nc=1,
            tda=False, max_memory=1.0, full_output=True)
    assert abs(r.chern) < 1e-8
    assert np.max(np.abs(r.fluxes)) > 1e-5 # not zero plaquette by plaquette
    assert r.gap > 0.1 and r.bound > 0.1 # isolated and bound


def test_zero_plaquette_by_plaquette_when_the_meshes_match():
    """With nq = nk the BSE at two neighbouring Q is the same problem
    translated by one k-point, so with the right lift every plaquette flux
    vanishes, here with random phases on the Bloch states. This checks the
    lift, and the raw amplitudes of the pair basis fail it"""
    h, W = _haldane_and_flat_hole()
    nk = 6
    orig = pairbasis.mesh_eigenstates
    pairbasis.mesh_eigenstates = _scrambler(3)
    try:
        r = exciton_chern(h, bands=[0], nq=nk, shift=SHIFT, V=W, nk=nk,
                tda=True, max_memory=1.0, full_output=True)
        raw = lambda q: BSE(h, V=W, Q=q, nk=nk, tda=True).amplitudes[0:1]
        fraw, _ = berry_fluxes(raw, nq=nk, shift=SHIFT)
    finally:
        pairbasis.mesh_eigenstates = orig
    assert abs(r.chern) < 1e-8
    assert np.max(np.abs(r.fluxes)) < 1e-8
    assert r.gap > 0.1 and r.bound > 0.1 # isolated and bound
    assert np.max(np.abs(fraw)) > 0.1 # the test can tell a missing lift


def _saturated_ferromagnet(nk):
    """Honeycomb with t', Kane-Mele and a Zeeman field, fully polarized at
    U=8: every electron-hole pair is a spin flip, so its BSE is the magnon
    problem, with Bloch states that depend on k and a nonzero integer"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(0.15)
    h.add_zeeman([0., 0., 3.0])
    return h.get_mean_field_hamiltonian(U=8.0, filling=0.5, mf="ferro",
            nk=nk, maxerror=1e-10)


def test_gauge_changes_neither_the_states_nor_the_integer():
    """gauge=None, gauge="phase" and random Bloch phases: the lifted
    exciton is the same state up to an overall phase and its band has the
    same integer, +1, and the raw amplitudes are not the same state"""
    nk = 6
    h = _saturated_ferromagnet(nk)
    kw = dict(nk=nk, tda=False, max_memory=1.0)
    ckw = dict(bands=[0], nq=6, shift=(0.03, 0.07), **kw)
    Q = [0.21, 0.37, 0.]
    v0 = mode_states(h, "exciton", Q=Q, **kw)[0]
    v1 = mode_states(h, "exciton", Q=Q, gauge="phase", **kw)[0]
    c0 = exciton_chern(h, **ckw)
    c1 = exciton_chern(h, gauge="phase", **ckw)
    orig = pairbasis.mesh_eigenstates
    pairbasis.mesh_eigenstates = _scrambler(4)
    try:
        v2 = mode_states(h, "exciton", Q=Q, **kw)[0]
        a2 = BSE(h, Q=Q, **kw).amplitudes[0]
        c2 = exciton_chern(h, **ckw)
    finally:
        pairbasis.mesh_eigenstates = orig
    a0 = BSE(h, Q=Q, **kw).amplitudes[0]
    assert _phase_free_overlap(v0, v1) > 1. - 1e-10
    assert _phase_free_overlap(v0, v2) > 1. - 1e-10
    assert _phase_free_overlap(a0, a2) < 0.9 # the raw amplitudes move
    for c in (c0, c1, c2): assert abs(c - 1.) < 1e-8


def test_berry_fluxes_sum_to_the_chern_number():
    h, W = _frenkel(0.3)
    kw = dict(V=W, nk=4, tda=True, max_memory=1.0, nq=6, shift=SHIFT)
    f, es, worst = mode_berry_fluxes(h, "exciton", bands=[0], **kw)
    assert f.shape == (6, 6) and es.shape == (6, 6, 1) and worst > 0.5
    assert abs(np.sum(f)/(2.*np.pi) - exciton_chern(h, bands=[0], **kw)) < 1e-10


def test_an_unstable_reference_is_refused():
    """A flat level too close to the Haldane bands lets the exciton bind
    below zero energy, and the full BSE then puts the mirror of that mode,
    with X = 0, in its positive branch; the lowest exciton is then not an
    excitation of the reference, and the Chern number is refused"""
    h, W = _haldane_and_flat_hole(eflat=-1.0)
    kw = dict(Q=[0.13, 0.27, 0.], V=W, nk=6)
    with pytest.raises(ValueError, match="negative norm.*unstable"):
        mode_states(h, "exciton", tda=False, **kw)
    # the same exciton in the resonant block alone, at its negative energy
    with pytest.raises(ValueError, match="negative energy.*unstable"):
        mode_states(h, "exciton", tda=True, **kw)


def _hubbard(U, nsites):
    """An onsite Hubbard U as the V= of the BSE"""
    W = np.zeros((2*nsites, 2*nsites))
    for i in range(nsites): W[2*i, 2*i+1] = W[2*i+1, 2*i] = U
    return {(0, 0, 0): W}


def test_an_imaginary_energy_is_refused_whatever_the_roundoff():
    """A Neel state converged at U=4, with the kernel of a larger U, is
    unstable against the uniform rotation of the spins, at an imaginary
    energy. The norm of such a mode vanishes, and BSE.solve keeps or drops
    it by the roundoff of its real part (measured: 142 modes for 144 pairs
    at U=4.5, and a mode at 1.24i with norm +1e-15 at U=6 and Q=0), so
    neither its norm nor its energy says that the reference is unstable.
    A group that does not contain the unstable mode is refused too"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    h = h.get_mean_field_hamiltonian(U=4.0, filling=0.5, mf="antiferro",
            nk=6, maxerror=1e-10)
    kw = dict(bands=[9], nk=6, tda=False, max_memory=1.0)
    # with its own interaction the reference is stable
    v = mode_states(h, "exciton", Q=[0.21, 0.37, 0.], V=_hubbard(4.0, 2), **kw)
    assert v.shape[0] == 1
    for U in (4.5, 6.0):
        for Q in ([0., 0., 0.], [0.03, 0.07, 0.]):
            with pytest.raises(ValueError, match="unstable"):
                mode_states(h, "exciton", Q=Q, V=_hubbard(U, 2), **kw)


def test_a_group_that_touches_another_mode_is_refused():
    """With no flux the molecule has C3 and time reversal, its two empty
    levels are degenerate, and so are the two Frenkel excitons at Q=0. On
    a mesh that holds Q=0 the lowest exciton alone has no Chern number,
    and the group of the two has the zero of time reversal"""
    h, W = _frenkel(0.0)
    kw = dict(V=W, nk=4, tda=True, max_memory=1.0, nq=4)
    with pytest.raises(ValueError, match="degenerate with another mode"):
        exciton_chern(h, bands=[0], **kw)
    r = exciton_chern(h, bands=[0, 1], full_output=True, **kw)
    assert abs(r.chern) < 1e-8
    assert r.energies.shape == (4, 4, 2)
    assert r.gap > 0.1 and r.bound > 0.1


def test_a_mesh_that_does_not_resolve_the_group_is_refused():
    """tol is the smallest link overlap accepted. The overlaps of the
    Frenkel band on a 4 x 4 mesh go down to 0.64 (measured), so tol=0.9
    refuses that mesh and the default accepts it"""
    h, W = _frenkel(0.3)
    kw = dict(bands=[0], nq=4, shift=SHIFT, V=W, nk=4, tda=True)
    r = exciton_chern(h, full_output=True, **kw)
    assert 0.3 < r.min_overlap < 0.9
    with pytest.raises(ValueError, match="almost orthogonal"):
        exciton_chern(h, tol=0.9, **kw)


def test_guards():
    h, W = _frenkel(0.3)
    kw = dict(V=W, nk=4, tda=True)
    assert get_metric_names() == ["casida", "euclidean"]
    with pytest.raises(ValueError, match="metric must be"):
        exciton_chern(h, metric="minkowski", nq=2, **kw)
    with pytest.raises(ValueError, match="contiguous"):
        mode_states(h, "exciton", bands=[0, 2], **kw)
    with pytest.raises(ValueError, match="has 2 modes|has \\d+ modes"):
        mode_states(h, "exciton", bands=[500], **kw)
    with pytest.raises(NotImplementedError, match="solver"):
        mode_states(h, "exciton", solver="iterative", **kw)
    h1 = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError, match="two-dimensional"):
        exciton_chern(h1, nq=4, nk=4)


def test_the_method_and_the_general_function():
    """h.get_exciton_chern() is bse.exciton_chern, which is mode_chern with
    mode="exciton"; mode= takes the names of its registry and nothing
    else, and lists them when it is wrong"""
    from pyqula import bse
    from pyqula.bsetk.topology import mode_chern
    h, W = _frenkel(0.3)
    kw = dict(bands=[0], nq=4, shift=SHIFT, V=W, nk=4, tda=True)
    c = h.get_exciton_chern(**kw)
    assert abs(c + 1.) < 1e-8 # the C = -1 of the oracle
    assert c == bse.exciton_chern(h, **kw) == exciton_chern(h, **kw)
    assert c == mode_chern(h, mode="exciton", **kw)
    r = h.get_exciton_chern(full_output=True, **kw)
    assert r.chern == c and r.energies.shape == (4, 4, 1)
    assert get_mode_names() == ["exciton", "magnon"]
    with pytest.raises(ValueError, match="'exciton', 'magnon'"):
        mode_chern(h, mode="phonon", **kw)
    with pytest.raises(ValueError, match="'exciton', 'magnon'"):
        mode_states(h, "phonon", Q=[0.1, 0.2, 0.], bands=[0], V=W, nk=4)
