import numpy as np
import pytest

from pyqula import geometry
from pyqula import films
from pyqula import algebra
from testutils import temporary_attr


def _af_sc_film(af=0.5, sc=0.4, mu=0.7, nz=6):
    """Diamond-lattice film with a sharp domain wall at z=0: the z<0 half is
    an antiferromagnet of amplitude af, the z>0 half an s-wave
    superconductor of gap sc shifted by mu. Kane-Mele SOC everywhere."""
    g = geometry.diamond_lattice_minimal()
    g = films.geometry_film(g, nz=nz)
    h = g.get_hamiltonian(is_multicell=True, is_sparse=False)

    def step(z, width=0.00001):
        return (-np.tanh(z / width) + 1.0) / 2.

    h.add_antiferromagnetism(lambda r: af * step(r[2]))
    h.shift_fermi(lambda r: mu * (-step(r[2], width=0.0001) + 1.0))
    h.add_swave(lambda r: sc * (-step(r[2]) + 1.0))
    h.add_kane_mele(0.1)
    return h


@pytest.mark.slow
def test_af_sc_junction_binds_its_subgap_states_to_the_domain_wall(tmp_path,
                                                                   monkeypatch):
    """Both halves of this film are gapped -- the z<0 half by the
    antiferromagnetic exchange, the z>0 half by the s-wave pairing -- so the
    only states left inside the gap are bound to the domain wall at z=0.
    The film spans |z| <= 3.5, and every subgap state must sit within half a
    lattice spacing of the wall.

    Replaces recorded sum(e) and sum(c) constants. For a BdG Hamiltonian
    sum(e) = sum_k Tr H(k) = 0 identically (that is just particle-hole
    symmetry, asserted directly below), and sum(c) over a full band
    structure is nk*Tr(sz) = 0 whatever the Hamiltonian is -- neither
    responds to the exchange, the pairing, the SOC or the thickness.
    Marked slow: stays a few seconds even at reduced thickness/mesh."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    with temporary_attr(algebra, "accelerate", True):
        h = _af_sc_film()
        (k, e, z) = h.get_bands(operator="zposition", nk=20)
        e, z = np.array(e), np.array(z)
        # the identity that made the old sum(e) reference zero
        assert np.allclose(np.sort(e), -np.sort(e)[::-1], atol=1e-8)
        subgap = np.abs(e) < 0.1
        assert np.sum(subgap) > 0
        assert np.max(np.abs(z[subgap])) < 0.5  # the film spans |z| <= 3.5

        # without the exchange the magnetic half is no longer gapped and the
        # junction state is gone
        h0 = _af_sc_film(af=0.)
        (k, e0, z0) = h0.get_bands(operator="zposition", nk=20)
        assert np.sum(np.abs(np.array(e0)) < 0.1) == 0
