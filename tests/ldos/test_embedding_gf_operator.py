import numpy as np

from pyqula import geometry
from pyqula import embedding
from pyqula.increase_hilbert import full2profile

# Embedding.get_gf declared an `operator` argument and never referenced it,
# so eb.get_gf(operator="sz") returned the plain Green's function while
# eb.get_ldos(operator="sz") -- the sibling method of the same class, built
# on top of that very Green's function -- did honour it. The invariant used
# here is that relation between the two: the operator-resolved LDOS is
# -Im(diag(A G))/pi, so whatever get_gf returns for a given operator has to
# reproduce get_ldos for the same one.

ARGS = dict(energy=0.2, delta=0.1, nk=6)


def _embedding():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_exchange([0., 0., 0.5])
    hv = h.copy()
    hv.add_onsite(lambda r: 3.0 if np.abs(r[0]) < 1e-6 else 0.0)
    return embedding.Embedding(h, m=hv.intra)


def test_the_operator_reaches_the_greens_function():
    eb = _embedding()
    for op in ["sz", "sx"]:
        g = np.array(eb.get_gf(operator=op, **ARGS))
        d = full2profile(eb.H, -np.diag(g).imag/np.pi, check=False)
        ref = eb.get_ldos(operator=op, write=False, **ARGS)[2]
        assert np.max(np.abs(np.array(d)-np.array(ref))) < 1e-10


def test_an_operator_actually_changes_the_greens_function():
    """The exchange field polarizes the lattice, so the sz-resolved
    Green's function cannot coincide with the charge one."""
    eb = _embedding()
    plain = np.array(eb.get_gf(**ARGS))
    sz = np.array(eb.get_gf(operator="sz", **ARGS))
    assert np.max(np.abs(plain-sz)) > 1e-3
