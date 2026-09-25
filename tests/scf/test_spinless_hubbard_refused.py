import pytest

from pyqula import geometry, meanfield

# A local Hubbard U is the interaction between the up and down densities on
# the same site, so it has no meaning for spinless fermions. Vinteraction
# and Vinteraction_kpm refused it already; the two Hubbard wrappers built a
# spinless onsite term instead, which only shifts the chemical potential.


def _spinless_chain():
    return geometry.chain().get_hamiltonian(has_spin=False)


@pytest.mark.parametrize("engine", [meanfield.hubbardscf,
                                    meanfield.hubbardscf_kpm])
def test_a_spinless_local_U_is_refused(engine):
    with pytest.raises(ValueError, match="requires the spin degree"):
        engine(_spinless_chain(), U=2.0, nk=4, maxite=2, verbose=0)
