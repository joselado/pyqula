import numpy as np
import pytest

from pyqula import geometry, algebra


def _onsite_energies(h):
    """Diagonal of the intra-cell block, i.e. the onsite energy of each
    orbital."""
    return np.real(np.diag(algebra.todense(h.intra)))


def _scf(Vr, constrains=None, mass=0.5, nk=4):
    """Mean field of a honeycomb supercell whose two sublattices are made
    inequivalent by a staggered onsite term, so that a density-density
    interaction has a charge channel to fill in."""
    g = geometry.honeycomb_lattice()
    g = g.supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(mass)
    kwargs = dict(Vr=Vr, filling=0.5, nk=nk)
    if constrains is not None:
        kwargs["constrains"] = constrains
    return h, h.get_mean_field_hamiltonian(**kwargs)


@pytest.mark.slow
def test_no_charge_constraint_removes_the_hartree_shift(tmp_path, monkeypatch):
    """constrains=["no_charge"] is supposed to run the self-consistency with
    the charge (Hartree) channel switched off, so whatever else the mean
    field does it must not move any onsite energy. This checks exactly that,
    on a lattice whose sublattices are inequivalent so the unconstrained
    Hartree term has something to do: with the constraint every onsite
    energy comes back bit-for-bit unchanged, without it they all shift by
    about 0.58.

    It also checks the trivial limit the file used to exercise: with Vr = 0
    there is no interaction, so the mean field must vanish identically.

    The old assertion was sum(e) == 0 on the bands of the *unconstrained*
    input Hamiltonian (the SCF result was computed and then discarded), and
    sum(e) = sum_k Tr H(k) is zero for a bipartite lattice with a staggered
    onsite term whatever the interaction or the constraint does. Marked
    slow: the SCF convergence drives the runtime."""
    monkeypatch.chdir(tmp_path)  # writes MF.pkl to cwd

    def Vr(r1, r2):
        return 2.0 * np.exp(-np.linalg.norm(np.array(r1) - np.array(r2)))

    (h, hc) = _scf(Vr, constrains=["no_charge"])
    assert np.allclose(_onsite_energies(hc), _onsite_energies(h), atol=1e-8)

    (h, hu) = _scf(Vr)
    assert np.max(np.abs(_onsite_energies(hu) - _onsite_energies(h))) > 0.1

    # no interaction, no mean field
    (h, h0) = _scf(lambda r1, r2: 0., constrains=["no_charge"])
    assert np.allclose(_onsite_energies(h0), _onsite_energies(h), atol=1e-8)
