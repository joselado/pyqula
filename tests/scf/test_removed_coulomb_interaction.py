"""The Coulomb interactions of the old scfclass are removed.

meanfield.coulomb_interaction and fast_coulomb_interaction set
g.ncells on the caller's geometry, a box of cells set by |a1| only, and
then called g.replicas(d) positionally, while Geometry.replicas takes
keywords only. So both raised TypeError at the first cell, and their only
caller, scfclass.setup_interaction(mode="Coulomb"), raised it too. The
maintainer chose to delete them, with kanemele.add_kane_mele_old, which
added the coupling only to the cells that already had a hopping. An
interaction mode that is not there is now refused with the accepted ones.
"""

import pytest

from pyqula import geometry
from pyqula import kanemele
from pyqula import meanfield
from pyqula import scftypes


@pytest.mark.parametrize("name", ["coulomb_interaction",
    "fast_coulomb_interaction", "coulomb_interaction_spinless",
    "coulomb_interaction_spinful"])
def test_the_coulomb_builders_are_gone(name):
    assert not hasattr(meanfield, name)


def test_the_old_kane_mele_is_gone():
    assert not hasattr(kanemele, "add_kane_mele_old")


@pytest.mark.parametrize("mode", ["Coulomb", "fastCoulomb"])
def test_a_coulomb_mode_is_refused_with_the_accepted_ones(mode):
    scf = scftypes.scfclass(geometry.chain().get_hamiltonian())
    with pytest.raises(ValueError, match="accepted ones") as err:
        scf.setup_interaction(mode=mode)
    accepted = str(err.value).split("accepted ones")[1]
    assert "Coulomb" not in accepted
    assert "'V'" in accepted
