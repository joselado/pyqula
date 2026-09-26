"""Unfolding with the kernel polynomial method.

The unfolding operator is O(k) = U U^dagger with U = P^dagger, whose n0
columns are the Bloch states of the primal cell, so its weight in the
density of states is sum_a <u_a|delta(E-H)|u_a>: one Chebyshev expansion
per column and no random vectors, which is how KITE computes the
momentum-resolved spectral function (arXiv:1910.05194, Sec. 4.4.1). The
oracle is an exact diagonalization: the moments of the expansion are
sum_m w_m T_j(E_m/scale) with w_m = <m|O|m>, to rounding."""

import numpy as np
import pytest
from scipy.linalg import eigh

from pyqula import geometry, kpm
from pyqula.unfolding import bloch_projector


def defective_supercell(setup="spinless"):
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell([[2, 1, 0], [0, 1, 0], [0, 0, 1]], store_primal=True)
    h = g.get_hamiltonian(has_spin=(setup != "spinless"))
    h.add_onsite(lambda r: 0.7 * (np.sum((r - g.r[0])**2) < 1e-2))
    if setup == "nambu":
        h.setup_nambu_spinor()
    return h


def dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.array(m)


@pytest.mark.parametrize("setup", ["spinless", "spinful", "nambu"])
def test_moments_are_the_exact_ones(setup):
    h = defective_supercell(setup)
    k = np.array([0.37, 0.81, 0.])
    scale, n = 10., 40
    hk = dense(h.get_hk_gen()(k))
    U = bloch_projector(h).factor(k)
    mus, w = kpm.factored_moments(hk / scale, U, n=n)
    mus = np.sum(w[:, None] * mus, axis=0)  # the trace
    E, V = eigh(hk)
    Ud = dense(U)
    wm = np.sum(np.abs(np.conjugate(V.T) @ Ud)**2, axis=1)  # <m|O|m>
    j = np.arange(2 * n)
    exact = np.array([np.sum(wm * np.cos(i * np.arccos(E / scale)))
                      for i in j])
    assert np.isclose(np.sum(wm), h.intra.shape[0])  # Tr O, the sum rule
    assert np.allclose(mus.real, exact, atol=1e-10 * h.intra.shape[0])
    assert np.allclose(mus.imag, 0., atol=1e-10 * h.intra.shape[0])


@pytest.mark.parametrize("lattice,M", [
    ("triangular_lattice", [[3, 0, 0], [0, 3, 0], [0, 0, 1]]),
    ("triangular_lattice", np.sqrt(3)),
    ("honeycomb_lattice", [[2, 1, 0], [0, 1, 0], [0, 0, 1]])])
def test_clean_supercell_unfolds_onto_the_primal_cell(lattice, M, tmp_path,
                                                      monkeypatch):
    """At k_S = M@k0 the unfolded weight of a clean supercell is N_rep
    times the density of states of the primal cell at k0, which the same
    expansion gives exactly with the identity as the factor"""
    monkeypatch.chdir(tmp_path)
    g0 = getattr(geometry, lattice)()
    g = g0.get_supercell(M, store_primal=True)
    Ms = np.array(g.supercell_matrix, dtype=float)
    nrep = int(round(abs(np.linalg.det(Ms))))
    h = g.get_hamiltonian(has_spin=False)
    h0 = g.primal_geometry.get_hamiltonian(has_spin=False)
    energies, delta = np.linspace(-4., 4., 81), 0.1
    k0 = np.array([0.13, 0.29, 0.])
    out = h.get_kdos_bands(kpath=[Ms @ k0], operator="unfold", mode="KPM",
                           delta=delta, energies=energies)
    npol = 3 * int(10. / delta)  # what get_kdos_bands takes, with its
    hk0 = h0.get_hk_gen()(k0)  # default scale=10 and ewindow=4
    x, ref = kpm.factored_dos(hk0, np.identity(hk0.shape[0]), scale=10.,
                              npol=npol, ne=4 * npol, ewindow=4.,
                              x=energies)
    assert np.allclose(out[2], nrep * ref, atol=1e-9)
    assert np.max(ref) > 0.1  # the primal cell has states in the window


def test_kpm_and_eigenvector_unfolding_carry_the_same_weight(tmp_path,
                                                             monkeypatch):
    """The kernels broaden differently (Jackson against Lorentzian), so
    the profiles differ, but the weight each puts under the band is the
    trace of the unfolding operator, the number of orbitals"""
    monkeypatch.chdir(tmp_path)
    h = defective_supercell("spinful")
    kp = [[0.37, 0.81, 0.]]
    energies, delta = np.linspace(-6., 6., 1201), 0.02
    kw = dict(kpath=kp, operator="unfold", delta=delta, energies=energies)
    ed = h.get_kdos_bands(mode="ED", **kw)[2]
    kp_ = h.get_kdos_bands(mode="KPM", **kw)[2]
    n = h.intra.shape[0]
    assert np.isclose(np.trapezoid(kp_, energies), n, rtol=1e-3)
    assert np.isclose(np.trapezoid(ed, energies), n, rtol=1e-2)


def test_plain_kpm_kdos_is_the_trace_as_in_the_other_modes(tmp_path,
                                                           monkeypatch):
    """mode="KPM" used to average over unit random vectors and so came out
    N times smaller than mode="ED" and mode="green"; its integral is the
    zeroth moment, one per vector, exactly, whatever vectors are drawn"""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_supercell(2).get_hamiltonian()
    kp = [[0.13, 0.37, 0.]]
    energies, delta = np.linspace(-5., 5., 1001), 0.05
    n = h.intra.shape[0]
    for op in [None, np.identity(n)]:
        out = h.get_kdos_bands(kpath=kp, mode="KPM", delta=delta,
                               energies=energies, ntries=4, operator=op)
        assert np.isclose(np.trapezoid(out[2], energies), n, rtol=1e-3)


def test_kpm_refuses_what_it_cannot_take_exactly(tmp_path, monkeypatch):
    """A product with another operator has no factor, and random vectors
    have nothing to do in an exact trace"""
    monkeypatch.chdir(tmp_path)
    h = defective_supercell("spinful")
    kw = dict(kpath=[[0.1, 0.2, 0.]], mode="KPM",
              energies=np.linspace(-1., 1., 3))
    op = h.get_operator("unfold") * h.get_operator("sz")
    assert op.factor is None
    with pytest.raises(NotImplementedError):
        h.get_kdos_bands(operator=op, **kw)
    n = h.intra.shape[0]
    with pytest.raises(ValueError):
        h.get_kdos_bands(operator="unfold",
                         frand=lambda: np.random.random(n), **kw)


def test_a_copy_keeps_the_factor_and_a_new_operator_does_not():
    h = defective_supercell()
    op = h.get_operator("unfold")
    from pyqula.operators import Operator
    assert op.factor is not None
    assert Operator(op).factor is op.factor
    assert (2. * op).factor is None
    assert (op + op).factor is None
