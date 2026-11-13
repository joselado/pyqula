"""ScreenedInteraction.get_dict off the mesh.

On an even k-mesh the lattice vector -nk/2 and the lattice vector +nk/2
give the same phase at every mesh q, so the inverse transform cannot tell
them apart. Keeping only -nk/2 broke W(d) = W(-d)^dagger, and W(q) at a q
off the mesh came out non-Hermitian by 0.14 on a two-site chain at nk=4.
A density-density interaction is Hermitian at every q, on the mesh or
not, so that is the invariant here; the round trip on the mesh must stay
exact."""
import numpy as np
import pytest

from testutils import gapped_ionic_chain, gapped_honeycomb
from pyqula.bsetk.interaction import density_interaction, interaction_at_q


def _models():
    return [("chain", gapped_ionic_chain()),
            ("honeycomb", gapped_honeycomb(mass=1.0))]


@pytest.mark.parametrize("nk", [4, 5])
@pytest.mark.parametrize("name,h", _models())
def test_get_dict_is_hermitian_off_the_mesh(name, h, nk):
    W = h.get_screened_interaction(V=density_interaction(h, U=0.6, V1=0.4),
                                   nk=nk)
    d = W.get_dict(tol=0.)
    g = h.geometry
    # every lattice vector has its partner, with W(-d) = W(d)^dagger
    for key, m in d.items():
        mkey = tuple(-np.array(key))
        assert mkey in d
        assert np.max(np.abs(d[mkey] - m.conj().T)) < 1e-10
    # W(q) is Hermitian away from the mesh
    rng = np.random.default_rng(3)
    for _ in range(4):
        q = np.zeros(3)
        q[:g.dimensionality] = rng.random(g.dimensionality)
        Wq = interaction_at_q(d, g, q)
        assert np.max(np.abs(Wq - Wq.conj().T)) < 1e-10
    # and the dictionary still reproduces the table on the mesh
    for q in W.qs:
        assert np.max(np.abs(interaction_at_q(d, g, q) - W.at(q))) < 1e-10
