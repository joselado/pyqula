"""The Chern number of a magnon band of time-dependent Hartree-Fock, from
the eigenvectors of the magnon problem lifted to the orbital pair basis
(bsetk/topology.py).

Two magnets are used. The first is a saturated ferromagnet on the
honeycomb lattice with t', Kane-Mele coupling and a Zeeman field, which
has an onsite U only and no Goldstone mode; its magnon problem has no
de-excitation half (Y = 0), and its two magnon bands are bound below the
Stoner continuum. The second is the Lieb ferrimagnet with the same terms,
whose magnon problem does have a de-excitation half, so the lift of Y and
the Casida metric both enter. In both the integer must not depend on the
mesh of Q, on the internal k-mesh, or on the phases of the Bloch states,
and it must flip with the sign of the Kane-Mele coupling, whose product
with t' is what plays the role of a Dzyaloshinskii-Moriya term.

The sign of the integers of the ferromagnet is not a recorded number. It
is compared with linear spin-wave theory at strong coupling, written here
from the hoppings alone: a spin flipped on site i hops to site j with the
amplitude (2/U) t_ij(flipped spin) conj(t_ij(majority spin)), the
superexchange of second order in the hopping. This is a tight-binding
model of the magnon, whose Chern numbers the plaquette product gives with
no two-particle code. It is checked as the limit of the magnon problem at
U=40 before the integers at U=8 are compared with it.

The record of full_output is compared with magnon_energies and with the
Stoner edge computed from the two spin blocks of the Bloch Hamiltonian,
on a ferromagnet with a sublattice imbalance, whose magnons have
E(Q) != E(-Q). This fixes the momentum each flux belongs to, which the
integer cannot do: solving every point at -Q rotates the plane and leaves
the integer.
"""
import functools

import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk import pairbasis
from pyqula.bsetk.spinflip import magnon_energies
from pyqula.bsetk.topology import (magnon_chern, mode_states,
        mode_berry_fluxes)
from pyqula.topologytk.plaquette import chern_from_states

SHIFT = (0.03, 0.07) # keeps Q=0 and the corners out of the mesh of Q


@functools.lru_cache(maxsize=None)
def _ferromagnet(nk, lam=0.15, B=3.0, U=8.0, imbalance=0.0):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(lam)
    h.add_zeeman([0., 0., B])
    if imbalance != 0.: h.add_sublattice_imbalance(imbalance)
    return h.get_mean_field_hamiltonian(U=U, filling=0.5, mf="ferro",
            nk=nk, maxerror=1e-10)


@functools.lru_cache(maxsize=None)
def _ferrimagnet(nk, lam=0.15):
    g = geometry.lieb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(lam)
    h.add_zeeman([0., 0., 0.3])
    return h.get_mean_field_hamiltonian(U=4.0, filling=0.5, mf="ferro",
            nk=nk, maxerror=1e-10, mix=0.3, maxite=3000)


@functools.lru_cache(maxsize=None)
def _antiferromagnet(nk):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    return h.get_mean_field_hamiltonian(U=4.0, filling=0.5, mf="antiferro",
            nk=nk, maxerror=1e-10)


def _scrambled(seed):
    orig = pairbasis.mesh_eigenstates
    rng = np.random.default_rng(seed)
    def f(hk, ks):
        ek, ck = orig(hk, ks)
        return ek, ck*np.exp(2j*np.pi*rng.random(ck.shape[0:2]))[:, :, None]
    return f


def _phase_free_overlap(v0, v1):
    return abs(np.vdot(v0, v1))/np.linalg.norm(v0)/np.linalg.norm(v1)


def _dense(m):
    return np.array(m.todense() if hasattr(m, "todense") else m)


def _up_is_flipped(h):
    """True when the magnet points along -z, so a magnon is a spin up"""
    return bool(np.sum(np.real(h.get_vev("sz"))) < 0.)


def _spin_wave_hamiltonian(h, U):
    """The Bloch Hamiltonian q -> T(q) of one flipped spin in a saturated
    ferromagnet at strong coupling. Second-order perturbation theory in
    the hopping t_ij(s) (the coefficient of c^dag_{i,s} c_{j,s}) of the
    Hubbard model gives the exchange term
    (2/U) t_ij(dn) conj(t_ij(up)) S^+_j S^-_i + h.c. on every bond, so the
    flipped spin, up on a background of spins down, has the hopping
    matrix T_ij = (2/U) t_ij(up) conj(t_ij(dn)), and its conjugate when
    the background is up. The diagonal is left out: it is the same on the
    two sites of the honeycomb lattice when they are equivalent, and then
    it changes no eigenvector. T(q) uses the Fourier convention of the
    Bloch Hamiltonian, sum_d T(d) exp(2 pi i q.d)."""
    up = _up_is_flipped(h)
    T = dict()
    for d, m in h.get_multicell().get_hopping_dict().items():
        m = _dense(m)
        tup, tdn = m[0::2, 0::2], m[1::2, 1::2]
        t = (2./U)*(tup*np.conj(tdn) if up else tdn*np.conj(tup))
        d = tuple(int(x) for x in d)
        if d == (0, 0, 0): t = t - np.diag(np.diag(t))
        T[d] = t
    return lambda q: sum(t*np.exp(2j*np.pi*(q[0]*d[0] + q[1]*d[1]))
            for d, t in T.items())


def _spin_wave_chern(h, U, band):
    T = _spin_wave_hamiltonian(h, U)
    def states(q):
        _, v = np.linalg.eigh(T(q))
        return v[:, band:band+1].T
    return chern_from_states(states, nq=12, shift=SHIFT)


def _stoner_edge(h, Q, nk):
    """The lowest spin-flip transition of the channel of the magnon at Q,
    from the two spin blocks of the Bloch Hamiltonian diagonalized one by
    one: the flipped electron above the Fermi energy at k+Q, the hole of
    the other spin below it at k"""
    hk = h.get_hk_gen()
    el, ho = (0, 1) if _up_is_flipped(h) else (1, 0)
    out = np.inf
    for k in h.geometry.get_kmesh(nk=nk):
        ee = np.linalg.eigvalsh(_dense(hk(np.array(k) + Q))[el::2, el::2])
        eh = np.linalg.eigvalsh(_dense(hk(np.array(k)))[ho::2, ho::2])
        ee, eh = ee[ee > 0.], eh[eh < 0.]
        if len(ee) > 0 and len(eh) > 0: out = min(out, ee[0] - eh[-1])
    return out


def test_spin_wave_theory_is_the_strong_coupling_limit():
    """At U=40 the splitting of the two magnon bands is the one of the
    spin-wave model within 3 % (measured: 0.4 to 1.4 %; at U=8 it is 7 to
    30 %), and the Chern numbers of the two agree"""
    U = 40.
    h = _ferromagnet(6, U=U)
    assert np.allclose(np.real(h.get_vev("sz")), -1., atol=1e-6) # saturated
    # the Fourier convention the spin-wave model takes from the electrons
    q = np.array([0.13, 0.27, 0.])
    hq = sum(_dense(m)*np.exp(2j*np.pi*(q[0]*d[0] + q[1]*d[1]))
            for d, m in h.get_multicell().get_hopping_dict().items())
    assert np.max(np.abs(hq - _dense(h.get_hk_gen()(q)))) < 1e-10
    T = _spin_wave_hamiltonian(h, U)
    for Q in ([0.21, 0.37, 0.], [0.5, 0., 0.], [1/3., 1/3., 0.]):
        e = np.sort(magnon_energies(h, Q=Q, nk=6).real)[0:2]
        eo = np.linalg.eigvalsh(T(Q))
        assert abs((e[1] - e[0]) - (eo[1] - eo[0])) < 0.03*(eo[1] - eo[0])
    for band in (0, 1):
        co = _spin_wave_chern(h, U, band)
        assert abs(round(co)) == 1 # a Chern band, so not a vacuous check
        c = magnon_chern(h, bands=[band], nq=6, shift=SHIFT, nk=6)
        assert abs(c - round(co)) < 1e-8


@pytest.mark.parametrize("lam,B", [(0.15, 3.0), (-0.15, 3.0), (0.15, -3.0),
        (-0.15, -3.0)])
def test_the_sign_is_the_one_of_spin_wave_theory(lam, B):
    """Both magnon bands against the spin-wave model, for the two signs of
    the Kane-Mele coupling and the two signs of the field. A field along
    -z polarizes the magnet along +z and moves the magnon to the other
    spin channel (S^- creates it). Time reversal maps (lam, B) to
    (lam, -B) and flips every Chern number; a rotation of the spins by pi
    about x maps it to (-lam, -B) and keeps them"""
    h = _ferromagnet(6, lam=lam, B=B)
    assert np.allclose(np.real(h.get_vev("sz")), -np.sign(B), atol=1e-6)
    out = []
    for band in (0, 1):
        co = _spin_wave_chern(h, 8., band)
        assert abs(round(co)) == 1
        r = magnon_chern(h, bands=[band], nq=6, shift=SHIFT, nk=6,
                full_output=True)
        assert abs(r.chern - round(co)) < 1e-8
        assert r.gap > 0.1 and r.bound > 1. # isolated and bound
        out.append(round(r.chern))
    assert out[0] == -out[1]
    # the two symmetries, on the magnon problem itself
    c0 = magnon_chern(_ferromagnet(6), bands=[0], nq=6, shift=SHIFT, nk=6)
    assert out[0] == round(c0)*int(np.sign(lam*B))


def test_the_two_bound_magnons_of_the_ferromagnet():
    """The two bands have opposite Chern numbers, those of the spin-wave
    model, and the pair together has 0; both bands are below the Stoner
    continuum and separated from each other"""
    h = _ferromagnet(6)
    assert np.allclose(np.real(h.get_vev("sz")), -1., atol=1e-6) # saturated
    out = dict()
    for bands in ([0], [1], [0, 1]):
        r = magnon_chern(h, bands=bands, nq=6, shift=SHIFT, nk=6,
                full_output=True)
        assert abs(r.chern - round(r.chern)) < 1e-8
        assert r.gap > 0.1 and r.bound > 1. # isolated and bound
        out[len(bands), bands[0]] = round(r.chern)
    assert out[1, 0] == round(_spin_wave_chern(h, 8., 0))
    assert out[1, 1] == round(_spin_wave_chern(h, 8., 1))
    assert abs(out[1, 0]) == 1 and out[1, 1] == -out[1, 0]
    assert out[2, 0] == 0


@pytest.mark.parametrize("nk,nq", [(6, 8), (6, 10), (8, 6)])
def test_the_integer_is_converged_in_both_meshes(nk, nq):
    h = _ferromagnet(nk)
    c = magnon_chern(h, bands=[0], nq=nq, shift=SHIFT, nk=nk)
    assert abs(c - round(_spin_wave_chern(h, 8., 0))) < 1e-8


def test_the_record_is_the_magnon_spectrum_at_the_same_momentum():
    """A sublattice imbalance breaks inversion, so E(Q) != E(-Q) (measured:
    by up to 0.10). The energies of the record are those of
    magnon_energies at the points of the record and not at minus them,
    the gap is the distance to the third mode, and bound is the distance
    of the top of the group below the Stoner edge, the lowest spin-flip
    transition of the bands"""
    h = _ferromagnet(6, imbalance=0.6)
    assert np.allclose(np.real(h.get_vev("sz")), -1., atol=1e-6) # saturated
    nq = 4
    kw = dict(bands=[0, 1], nq=nq, shift=SHIFT, nk=6)
    r = magnon_chern(h, full_output=True, **kw)
    fluxes, energies, worst = mode_berry_fluxes(h, "magnon", **kw)
    assert np.max(np.abs(fluxes - r.fluxes)) < 1e-12
    assert np.max(np.abs(energies - r.energies)) < 1e-12
    assert worst == r.min_overlap
    assert abs(r.chern - np.sum(fluxes)/(2.*np.pi)) < 1e-12
    assert r.qs.shape == (nq, nq, 3) and r.energies.shape == (nq, nq, 2)
    assert np.allclose(r.qs[2, 1], [2./nq + SHIFT[0], 1./nq + SHIFT[1], 0.])
    spectrum = lambda q: np.sort(magnon_energies(h, Q=q, nk=6).real)[0:3]
    e = np.array([[spectrum(q) for q in row] for row in r.qs])
    em = np.array([[spectrum(-q) for q in row] for row in r.qs])
    assert np.max(np.abs(r.energies - e[:, :, 0:2])) < 1e-10
    assert np.max(np.abs(r.energies - em[:, :, 0:2])) > 0.03
    assert abs(r.gap - np.min(e[:, :, 2] - e[:, :, 1])) < 1e-10
    edge = np.array([[_stoner_edge(h, q, 6) for q in row] for row in r.qs])
    assert abs(r.bound - np.min(edge - e[:, :, 1])) < 1e-10
    # one band of the pair: the gap is then the distance to the other one
    r0 = magnon_chern(h, bands=[0], nq=nq, shift=SHIFT, nk=6,
            full_output=True)
    assert abs(r0.gap - np.min(e[:, :, 1] - e[:, :, 0])) < 1e-10
    assert abs(r0.bound - np.min(edge - e[:, :, 0])) < 1e-10


def test_random_bloch_phases_change_neither_the_states_nor_the_integer():
    """magnon_matrix takes no gauge, so the phases are put on the Bloch
    states directly; a lift that is missing or wrong moves the state"""
    h = _ferromagnet(6)
    Q = [0.21, 0.37, 0.]
    v0 = mode_states(h, "magnon", Q=Q, bands=[0], nk=6)[0]
    c0 = magnon_chern(h, bands=[0], nq=6, shift=SHIFT, nk=6)
    orig = pairbasis.mesh_eigenstates
    pairbasis.mesh_eigenstates = _scrambled(5)
    try:
        v1 = mode_states(h, "magnon", Q=Q, bands=[0], nk=6)[0]
        c = magnon_chern(h, bands=[0], nq=6, shift=SHIFT, nk=6)
    finally:
        pairbasis.mesh_eigenstates = orig
    assert _phase_free_overlap(v0, v1) > 1. - 1e-10
    assert abs(round(c0)) == 1 and abs(c - c0) < 1e-8


def test_the_states_are_periodic_in_the_momentum():
    """The plaquettes wrap around the torus of Q, which is right only when
    the lifted state at Q + G is the one at Q, for the ferromagnet and
    for the ferrimagnet with its de-excitation half"""
    for h in (_ferromagnet(6), _ferrimagnet(6)):
        Q = np.array([0.21, 0.37, 0.])
        v0 = mode_states(h, "magnon", Q=Q, bands=[0], nk=6)[0]
        for G in ([1., 0., 0.], [0., 1., 0.]):
            v1 = mode_states(h, "magnon", Q=Q + np.array(G), bands=[0], nk=6)[0]
            assert _phase_free_overlap(v0, v1) > 1. - 1e-10


def test_a_ferrimagnet_with_a_de_excitation_half():
    """Y != 0: the two bound magnons of the Lieb ferrimagnet have opposite
    Chern numbers of modulus one, the same in the Euclidean metric and in
    the Casida metric diag(1,-1), whose fluxes differ plaquette by
    plaquette (measured: by up to 0.073). The lifted state of the lowest
    one does not move under random Bloch phases, the sign of the
    Kane-Mele coupling flips the integer, and the distance below the
    continuum is the one to the Stoner edge of the spin-flip channel"""
    h = _ferrimagnet(6)
    Q = [0.21, 0.37, 0.]
    v0 = mode_states(h, "magnon", Q=Q, bands=[0], nk=6)[0]
    n = len(v0)//2
    assert np.linalg.norm(v0[n:]) > 0.1*np.linalg.norm(v0[:n]) # Y != 0
    kw = dict(nq=6, shift=SHIFT, nk=6)
    r = magnon_chern(h, bands=[0], full_output=True, **kw)
    c0, c1 = round(r.chern), round(magnon_chern(h, bands=[1], **kw))
    assert abs(c0) == 1 and c1 == -c0
    assert abs(magnon_chern(h, bands=[0], metric="casida", **kw) - c0) < 1e-8
    assert abs(magnon_chern(h, bands=[1], metric="casida", **kw) - c1) < 1e-8
    fc, _, _ = mode_berry_fluxes(h, "magnon", bands=[0], metric="casida", **kw)
    assert np.max(np.abs(fc - r.fluxes)) > 1e-2
    assert abs(np.sum(fc) - np.sum(r.fluxes)) < 1e-8
    edge = np.array([[_stoner_edge(h, q, 6) for q in row] for row in r.qs])
    assert r.gap > 0.1 and r.bound > 0.5 # isolated and bound
    assert abs(r.bound - np.min(edge - r.energies[:, :, 0])) < 1e-10
    orig = pairbasis.mesh_eigenstates
    pairbasis.mesh_eigenstates = _scrambled(6)
    try: v1 = mode_states(h, "magnon", Q=Q, bands=[0], nk=6)[0]
    finally: pairbasis.mesh_eigenstates = orig
    assert _phase_free_overlap(v0, v1) > 1. - 1e-10
    hm = _ferrimagnet(6, lam=-0.15)
    assert round(magnon_chern(hm, bands=[0], **kw)) == -c0


def _hubbard(U, nsites):
    """An onsite Hubbard U as the V= of the magnon problem"""
    W = np.zeros((2*nsites, 2*nsites))
    for i in range(nsites): W[2*i, 2*i+1] = W[2*i+1, 2*i] = U
    return {(0, 0, 0): W}


def test_an_unstable_reference_is_refused_whatever_the_roundoff():
    """A Neel state converged at U=4, with the kernel of a larger U, is
    unstable against the uniform rotation of the spins, at an imaginary
    energy at Q=0 (0.84i at U=4.5, 1.24i at U=6, measured). The norm of
    that mode vanishes, +1.5e-15 as computed, so its sign cannot say that
    the reference is unstable; the complex energy does, also for a group
    that does not contain the unstable mode. At Q=(0.21,0.37) and U=6 the
    instability is a negative energy, -0.47"""
    h = _antiferromagnet(6)
    kw = dict(bands=[5], nk=6, check_su2=False)
    # with its own interaction the reference is stable away from Q=0
    v = mode_states(h, "magnon", Q=[0.21, 0.37, 0.], V=_hubbard(4.0, 2), **kw)
    assert v.shape[0] == 1
    for U in (4.5, 6.0):
        for Q in ([0., 0., 0.], [0.03, 0.07, 0.]):
            with pytest.raises(ValueError, match="unstable"):
                mode_states(h, "magnon", Q=Q, V=_hubbard(U, 2), **kw)
    with pytest.raises(ValueError, match="negative energy.*unstable"):
        mode_states(h, "magnon", Q=[0.21, 0.37, 0.], V=_hubbard(6.0, 2), **kw)


def test_guards():
    """A band beyond the excitation branch and a Hamiltonian that is not
    two-dimensional are refused, with what was asked in the message"""
    h = _ferromagnet(6)
    with pytest.raises(ValueError, match="modes"):
        mode_states(h, "magnon", Q=[0.21, 0.37, 0.], bands=[100000], nk=6)
    g = geometry.chain()
    h1 = g.get_hamiltonian()
    with pytest.raises(ValueError, match="two-dimensional"):
        magnon_chern(h1, nq=4, nk=4)


def test_the_method_is_the_module_function():
    """h.get_magnon_chern() is bse.magnon_chern, which is mode_chern with
    mode="magnon", with the same record"""
    from pyqula import bse
    from pyqula.bsetk.topology import mode_chern
    h = _ferromagnet(6)
    kw = dict(bands=[0], nq=4, shift=SHIFT, nk=6)
    c = h.get_magnon_chern(**kw)
    assert abs(abs(c) - 1.) < 1e-8
    assert c == bse.magnon_chern(h, **kw) == magnon_chern(h, **kw)
    assert c == mode_chern(h, mode="magnon", **kw)
    r = h.get_magnon_chern(full_output=True, **kw)
    assert r.chern == c and r.gap > 0.1 and r.bound > 1.
    assert 0. < r.max_flux < np.pi
