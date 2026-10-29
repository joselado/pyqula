"""The convolution apply of the BSE block is EXACT.

bsetk/convolution.py applies the resonant block at momentum Q by writing every
pair vector as a matrix in orbital space and doing the direct term as a
convolution over the k-mesh, with the interaction tabulated at the mesh
differences. That is the same matrix kernel.build_blocks writes down, not
an approximation of it, so it is tested against the dense block to machine
precision, for a real-space dictionary and for a tabulated screened
interaction, which the low-rank factorization cannot take.
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain, gapped_honeycomb
from pyqula import specialhamiltonian
from pyqula.bsetk.pairbasis import PairBasis
from pyqula.bsetk.interaction import bare_interaction, density_interaction
from pyqula.bsetk.kernel import build_blocks
from pyqula.bsetk.convolution import ConvolutionKernel
from pyqula.bsetk.solve import get_direct_interaction


def _model(name):
    if name == "chain":
        h = gapped_ionic_chain()
        return h, dict(U=1.0, V1=0.5, V2=0.3), 9
    if name == "honeycomb":
        return gapped_honeycomb(spinful=False), dict(V1=0.6, V2=0.2), 6
    if name == "spinful U": # an onsite U only couples part of the channels
        return gapped_honeycomb(spinful=True), dict(U=1.0), 5
    h = specialhamiltonian.three_band_tmdc("MoS2") # six orbitals, SOC
    return h, dict(Vr=lambda r1, r2: 2./np.sqrt((r1-r2).dot(r1-r2)+1.)), 4


@pytest.mark.parametrize("name", ["chain", "honeycomb", "spinful U", "tmd"])
@pytest.mark.parametrize("kernel", ["full", "direct", "exchange", "none"])
def test_convolution_reproduces_the_dense_block(name, kernel):
    h, kw, nk = _model(name)
    W = bare_interaction(h, V=density_interaction(h, **kw))
    pb = PairBasis(h, nk=nk)
    A = build_blocks(pb, W, Wx=W, kernel=kernel)[0]
    f = ConvolutionKernel(pb, W, Wx=W, kernel=kernel)
    scale = max(1., np.max(np.abs(A)))
    assert np.max(np.abs(f.to_dense() - A)) / scale < 1e-13
    # a single vector, and two calls with the same number of columns, the
    # second one reusing the buffers of the first
    rng = np.random.default_rng(1)
    x = rng.normal(size=(pb.npair, 3)) + 1j*rng.normal(size=(pb.npair, 3))
    for _ in range(2):
        assert np.max(np.abs(f.matvec(x) - A@x)) / scale < 1e-12
    xf = np.asfortranarray(x) # a column-major input
    assert np.max(np.abs(f.matvec(xf) - A@x)) / scale < 1e-12
    assert np.allclose(f.matvec(x[:, 0]), A@x[:, 0], atol=1e-12*scale)


@pytest.mark.parametrize("nkW", [6, 12])
def test_convolution_takes_a_tabulated_screened_interaction(nkW):
    """The RPA-screened direct term is tabulated on a mesh, on the BSE mesh
    or on a multiple of it, and enters the convolution as it is"""
    h = gapped_honeycomb(spinful=False)
    V = density_interaction(h, V1=0.6, V2=0.2)
    pb = PairBasis(h, nk=6)
    W = get_direct_interaction(h, pb, V=V, nk=6, screening="rpa", nkW=nkW)
    assert hasattr(W, "at") # tabulated, not a dictionary
    Wx = bare_interaction(h, V=V)
    A = build_blocks(pb, W, Wx=Wx)[0]
    f = ConvolutionKernel(pb, W, Wx=Wx)
    assert np.max(np.abs(f.to_dense() - A)) / np.max(np.abs(A)) < 1e-13


@pytest.mark.parametrize("name", ["chain", "honeycomb", "spinful U"])
@pytest.mark.parametrize("kernel", ["full", "direct", "exchange"])
@pytest.mark.parametrize("onmesh", [True, False])
def test_convolution_at_finite_momentum(name, kernel, onmesh):
    """The convolution does not see Q: the pair (k,k+Q) couples to
    (k',k'+Q) through W(k-k'), so the same table serves a Q on the mesh
    and one off it, and only the exchange term takes W(Q)"""
    h, kw, nk = _model(name)
    W = bare_interaction(h, V=density_interaction(h, **kw))
    Q = [1./nk, 2./nk, 0.] if onmesh else [0.13, 0.27, 0.]
    if h.dimensionality == 1: Q[1] = 0.
    pb = PairBasis(h, Q=Q, nk=nk)
    A = build_blocks(pb, W, Wx=W, kernel=kernel)[0]
    f = ConvolutionKernel(pb, W, Wx=W, kernel=kernel)
    assert np.max(np.abs(f.to_dense() - A)) / np.max(np.abs(A)) < 1e-13


def test_convolution_takes_a_screened_interaction_at_finite_momentum():
    """A tabulated W enters the direct term at the mesh differences, which
    do not depend on Q, while the exchange term keeps the bare W(Q)"""
    h = gapped_honeycomb(spinful=False)
    V = density_interaction(h, V1=0.6, V2=0.2)
    pb = PairBasis(h, Q=[0.13, 0.27, 0.], nk=6)
    W = get_direct_interaction(h, pb, V=V, nk=6, screening="rpa")
    Wx = bare_interaction(h, V=V)
    A = build_blocks(pb, W, Wx=Wx)[0]
    f = ConvolutionKernel(pb, W, Wx=Wx)
    assert np.max(np.abs(f.to_dense() - A)) / np.max(np.abs(A)) < 1e-13
