import numpy as np
import pytest

from pyqula import geometry
from pyqula import ldos

# 'ldos' is one of the names operatorlist.get_operator_names() advertises,
# and the user guide promises that every routine taking an operator accepts
# any registered name. The projector was built from get_ldos_tb's *written*
# profile, which replicates the cell nrep times (nrep defaults to 5), so on
# any periodic Hamiltonian the matrix came out nrep^dim times too large and
# every consumer died with a shape error instead. The invariant asserted
# here is the one that makes an operator an operator at all: it has to act
# on the Hilbert space of the Hamiltonian it was built from.


def _hamiltonians():
    return [
      # the 0d case is the one that always worked, because there the
      # replicated grid and the unit cell coincide -- keep it as the control
      ("0d fractal", geometry.sierpinski(n=2,
              mode="triangular").get_hamiltonian(has_spin=False)),
      ("spinless chain", geometry.chain().get_hamiltonian(has_spin=False)),
      ("spinful honeycomb", geometry.honeycomb_lattice().get_hamiltonian()),
      ("2x2 supercell", geometry.honeycomb_lattice().get_hamiltonian(
              ).get_supercell(2)),
            ]


@pytest.mark.parametrize("name,h", _hamiltonians())
def test_the_ldos_operator_lives_in_the_hilbert_space(name, h):
    m = h.get_operator("ldos").get_matrix()
    assert m.shape == h.intra.shape


def test_the_ldos_operator_can_weight_a_band_structure():
    """The consumer that used to raise 'matmul: dimension mismatch'."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    out = h.get_bands(nk=3, operator="ldos", write=False)
    assert len(out) == 3
    assert np.all(np.isfinite(np.array(out[2])))


def test_the_ldos_density_profile_has_one_entry_per_site():
    """ldos_density is documented as a normalized profile and is consumed
    by magneticexchange.NN_exchange as one value per site of the unit
    cell; it inherited the same nrep replication."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    d = ldos.ldos_density(h, nk=6, delta=1e-2)
    assert len(d) == len(h.geometry.r)
    assert abs(np.sum(d) - 1.0) < 1e-10


def test_ldos_potential_says_it_is_not_built():
    """Its whole body was `return # not finished yet`, so it handed back
    None for every input and the failure surfaced somewhere else."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(NotImplementedError):
        ldos.ldos_potential(h)
