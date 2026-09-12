import numpy as np

from pyqula import geometry, current


def _ribbon(bfield=0.0):
    g = geometry.honeycomb_zigzag_ribbon(6)
    h = g.get_hamiltonian(has_spin=False)
    if bfield != 0.0:
        h.add_peierls(bfield)  # orbital field, breaks time reversal
    return h


def test_ground_state_current_is_zero_without_a_field(tmp_path, monkeypatch):
    """current.gs_current and its two siblings were dead on arrival, with
    four independent defects stacked on one another: a call to ket_Aw,
    which is defined in bandstructure.py and was never imported here
    (NameError on every call); a bare float k reaching htk.bloch's
    generator, where it is indexed as an array (the class already repaired
    in dos1d_ewindow and current_bands); an elementwise product against an
    np.matrix, which silently becomes a matrix product and fails on any
    Hamiltonian with more than one orbital; and the computed current never
    being returned to the caller.

    With time-reversal symmetry every occupied state at +k is matched by
    one at -k with the opposite velocity, so the current density vanishes
    site by site, not merely in total."""
    monkeypatch.chdir(tmp_path)
    j = current.gs_current(_ribbon(), nk=400)
    assert j is not None  # it used to return None whatever it computed
    assert len(j) == 24
    assert np.max(np.abs(j)) < 1e-12


def test_orbital_field_gives_counterpropagating_edge_currents(tmp_path,
                                                              monkeypatch):
    """An orbital field drives equilibrium edge currents that run opposite
    ways on the two edges of the ribbon, so the *density* is large and
    edge-antisymmetric while the *total* still vanishes.

    The vanishing total is an exact identity rather than a numerical
    accident: summed over a full Brillouin zone, sum_k f(e(k)) de/dk is the
    integral of a derivative of a periodic function. It is therefore the
    sharpest available check that the velocity operator and the k-sum are
    consistent with each other -- and note it is why a *total* current must
    not be used as the observable here: on a plain chain with a flux
    threaded through it, gs_current's total looks nonzero at a coarse mesh
    (-4.4e-3 at nk=40) purely because the tanh occupation cutoff is
    unresolved, and converges to zero (-3.6e-11 at nk=2560)."""
    monkeypatch.chdir(tmp_path)
    h = _ribbon(bfield=0.05)
    j = current.gs_current(h, nk=400)
    y = np.array(h.geometry.y)
    assert abs(np.sum(j)) < 1e-10           # exact by periodicity
    assert np.max(np.abs(j)) > 1e-3         # but locally large
    assert abs(np.sum(j*np.sign(y))) > 1e-2  # and opposite on the two edges


def test_fermi_current_is_switched_on_by_breaking_particle_hole_symmetry(
        tmp_path, monkeypatch):
    """fermi_current reaches the same kernel through weighted_current, so
    it inherited all four defects; but its weight is a Lorentzian centred
    at zero, which is EVEN in energy, where gs_current's is an occupation
    step.

    On a bipartite ribbon the spectrum is particle-hole symmetric, and an
    even weight then cancels the two halves against each other exactly --
    so this current vanishes however large the orbital field is (7e-15
    even at delta=0.02). That cancellation is the invariant: it is not a
    null result, because adding an onsite term to break the symmetry
    switches the current back on at the same field and broadening."""
    monkeypatch.chdir(tmp_path)
    j0 = current.fermi_current(_ribbon(bfield=0.05), nk=400, delta=0.1)
    assert np.max(np.abs(j0)) < 1e-12  # even weight x p-h symmetry

    h = _ribbon(bfield=0.05)
    h.add_onsite(0.4)  # break particle-hole symmetry
    j = current.fermi_current(h, nk=400, delta=0.1)
    y = np.array(h.geometry.y)
    assert abs(np.sum(j)) < 1e-10            # still exact by periodicity
    assert np.max(np.abs(j)) > 1e-3          # now nonzero
    assert abs(np.sum(j*np.sign(y))) > 1e-2  # and edge-antisymmetric
