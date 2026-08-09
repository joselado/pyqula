import numpy as np

from pyqula import geometry


def test_dos_kpm_electron_hole_projectors_sum_to_total(tmp_path, monkeypatch):
    """Regression check for dos.py's dos_kpm operator normalization: the
    stochastic trace estimator used for an operator/projector P (kpm.pdos)
    draws random vectors confined to and renormalized within P's own
    subspace, so it converges to Tr[P f(H)]/Tr[P] -- a *per-state* average
    over that subspace, not over the full N-dimensional Hilbert space. The
    old code rescaled every projected result by the full matrix dimension
    N=h.intra.shape[0] regardless, silently inflating any projector of rank
    < N by N/Tr[P] (e.g. 2x for the electron/hole split of a BdG chain,
    where each projector has rank N/2).

    This is checked here via an exact linear identity rather than an
    approximate physical symmetry: the electron and hole projectors of a
    Nambu Hamiltonian are complementary, P_electron + P_hole = I, so
    Tr[P_electron f(H)] + Tr[P_hole f(H)] = Tr[f(H)] exactly, for any f.
    The buggy normalization broke this (each side inflated by ~2x, so the
    projected sum came out ~2x the total instead of matching it)."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    g = g.get_supercell(60)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_swave(0.3)  # BdG Hamiltonian with an electron/hole (Nambu) structure

    energies = np.linspace(-2.0, 2.0, 80)
    kwargs = dict(mode="KPM", energies=energies, delta=5e-2, ntries=10)
    _, y_total = h.get_dos(**kwargs)
    _, y_electron = h.get_dos(operator="electron", **kwargs)
    _, y_hole = h.get_dos(operator="hole", **kwargs)

    total_sum = np.sum(y_total)
    split_sum = np.sum(y_electron) + np.sum(y_hole)
    assert np.isclose(split_sum, total_sum, rtol=0.2)

    # each projector has rank N/2 here, and the spectrum is particle-hole
    # symmetric, so the electron- and hole-projected DOS should each land
    # close to half the total, not on top of it (the pre-fix bug)
    assert np.isclose(np.sum(y_electron), total_sum / 2, rtol=0.3)
    assert np.isclose(np.sum(y_hole), total_sum / 2, rtol=0.3)
