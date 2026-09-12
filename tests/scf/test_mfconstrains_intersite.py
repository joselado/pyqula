import numpy as np

from pyqula import geometry, meanfield
from pyqula.scftk.mfconstrains import remove_spinful_sector


def test_no_magnetism_is_enforced_with_an_intersite_interaction(tmp_path,
                                                                monkeypatch):
    """The oracle is the constraint's own claim: asking for
    constrains=["no_magnetism"] must return a state with no magnetization.

    With an intersite interaction the spin-dependent Fock term lives on the
    BONDS, not onsite, and the constraint only ever rewrote the (0,0,0)
    block of the mean-field dictionary -- so the constrained and the
    unconstrained run used to be bit-for-bit the same magnetic solution."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    out = dict()
    for con in ([], ["no_magnetism"]):
        h = g.get_hamiltonian()
        scf = meanfield.Vinteraction(h, V1=3.0, filling=0.3, nk=40,
                mf="ferroZ", constrains=con, mix=0.3, maxerror=1e-8,
                maxite=2000, load_mf=False, verbose=0)
        out[len(con)] = scf
    # the premise: the unconstrained solution really is magnetic, and the
    # magnetism really does sit on the bonds
    mag = out[0].hamiltonian.get_magnetization(nk=40)
    assert np.max(np.abs(mag)) > 1e-3
    bond = np.diag(np.array(out[0].mf[(1, 0, 0)])).real
    assert abs(bond[0]-bond[1]) > 1e-3
    # and the constraint has to remove it
    mag = out[1].hamiltonian.get_magnetization(nk=40)
    assert np.max(np.abs(mag)) < 1e-6


def test_bdg_mean_field_survives_the_constraint_on_every_direction():
    """Applying the constraint to every direction rewrites the electron
    block of a Nambu mean field at the BOND directions too, and the hole
    block is then rebuilt from it. With an identity removal that rebuild
    has to be the identity as well, at every direction -- otherwise the
    constraint would silently replace the hole hoppings of any BdG mean
    field it touched."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.turn_nambu()
    h.add_swave(0.1)
    h = h.get_multicell()
    mf = h.get_dict()
    out = remove_spinful_sector(h, lambda m: m)(mf) # identity removal
    assert set(out) == set(mf)
    for d in mf:
        assert np.max(np.abs(np.array(out[d])-np.array(mf[d]))) < 1e-12, d
