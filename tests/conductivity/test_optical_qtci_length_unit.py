"""The qtci optical conductivity does not depend on the unit of length.

In 2D the conductivity is (1/A) sum_k v v / (omega - dE): the velocity
goes as a length and the cell area as a length squared, so scaling every
position and every lattice vector by the same factor leaves sigma exactly
unchanged. The weights W = ratio v^a v^b that the qtci path interpolates
scale as a length squared while the transition energies dE do not, so the
stacked interpolation must normalize the two families apart: on a shared
scale a lattice constant of 100 resolves dE only to 1e-6 times 1e4 of the
weights.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import conductivity


def _gapped_honeycomb(lam):
    """Honeycomb lattice with sublattice imbalance 0.3, every length
    multiplied by lam"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    g = h.geometry
    g.r = g.r*lam
    g.a1, g.a2, g.a3 = g.a1*lam, g.a2*lam, g.a3*lam
    return h


@pytest.mark.parametrize("lam", [100., 0.01])
def test_qtci_does_not_depend_on_the_unit_of_length(lam):
    kw = dict(energies=np.linspace(0.3, 3., 28), nk=64, T=0.01, delta=0.02)
    _, t1 = conductivity.optical_conductivity(_gapped_honeycomb(1.),
            integration="triangle", **kw)
    _, tl = conductivity.optical_conductivity(_gapped_honeycomb(lam),
            integration="triangle", **kw)
    scale = np.max(np.abs(t1))
    # the triangle sum on the full mesh is exactly invariant
    assert np.max(np.abs(tl-t1)) < 1e-12*scale
    _, q1 = conductivity.optical_conductivity(_gapped_honeycomb(1.),
            integration="qtci", tolerance=1e-6, **kw)
    _, ql = conductivity.optical_conductivity(_gapped_honeycomb(lam),
            integration="qtci", tolerance=1e-6, **kw)
    assert np.max(np.abs(q1-t1)) < 1e-5*scale
    assert np.max(np.abs(ql-tl)) < 1e-5*scale
    assert np.max(np.abs(ql-q1)) < 1e-5*scale
