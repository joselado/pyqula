"""Cross-path invariants of the batched k-mesh machinery in
src/pyqula/conductivitytk/kubo.py.

kubo evaluates the Kubo formula on a whole k-mesh at once: it stacks the
Bloch Hamiltonians with htk.eigenvectors.hk_matrix_batch, diagonalizes
them with numba's batched eigh, and builds every dH/dk with a single
einsum over precomputed Bloch phases. None of that changes the physics,
so the tests here assert exactly that -- that the batched machinery
agrees with the module's own single-k reference implementations, which go
through scipy's eigh and through current.hk_derivative, the shared
correctly-normalized k-derivative.

Two things make this worth testing separately from the physics
benchmarks in test_optical_conductivity.py:

1) The einsum derivative re-derives, by hand, the prefactor that
current.hk_derivative applies (a factor 2*pi per derivative order times
(i R_i)^order_i per hopping). A wrong prefactor or a dropped hopping
would be invisible in a single-band model and is caught here on a
multiorbital supercell.

2) The velocity matrix elements themselves are GAUGE dependent: scipy's
eigh and numba's eigh return different eigenvector phases and different
rotations inside a degenerate multiplet, so |v_nm| can differ by O(1)
between the two paths even when both are right. The oracle therefore has
to be the gauge-invariant output -- sigma, the Drude weight, the sum-rule
weight -- never the intermediate. test_sigma_is_invariant_... below
verifies directly that the output really does have that invariance, i.e.
that the oracle is legitimate.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import algebra
from pyqula import current
from pyqula import conductivity
from pyqula.conductivitytk import kubo


def _model2d(nsuper=2):
    """Spinful honeycomb with Rashba and an exchange field, on a
    nsuper x nsuper supercell: a multiorbital (n = 8*nsuper^2), fully
    complex, non-orthogonal-lattice model with no symmetry left to hide a
    wrong Cartesian direction or a wrong hopping prefactor, and -- via
    get_supercell -- with its hoppings stored as the legacy numpy.matrix."""
    g = geometry.honeycomb_lattice()
    if nsuper>1: g = g.get_supercell(nsuper)
    h = g.get_hamiltonian()
    h.add_rashba(0.15)
    h.add_exchange([0.1,0.05,0.2])
    return h


def _model1d():
    """Spinful chain with Rashba and an exchange field: the 1D branch of
    the derivative, which uses a different code path in
    current.derivative (and a one-component order list)."""
    h = geometry.chain().get_hamiltonian()
    h.add_rashba(0.1)
    h.add_exchange([0.,0.,0.3])
    return h


def _serial_bands_and_velocities(h,ks):
    """The single-k reference implementation of
    kubo._bands_and_velocities: scipy's algebra.eigh one matrix at a
    time, and kubo._velocities, which builds dH/dk with
    current.hk_derivative rather than with the batched einsum. The
    comparisons below use nk=9 in 2D, i.e. 81 k-points, so that the
    batched path really has to stitch more than one kubo._kbatch batch
    together."""
    hm,orders,hkgen,jac,dr,cellvol,scale = kubo._setup(h)
    n = hm.intra.shape[0]
    es = np.zeros((len(ks),n),dtype=np.float64)
    vs = np.zeros((len(ks),3,n,n),dtype=np.complex128)
    for ik,k in enumerate(ks):
        hk = kubo._hk(hkgen,k)
        (e,w) = algebra.eigh(hk)
        wc = np.conjugate(w)
        v = kubo._velocities(hm,orders,jac,dr,hk,k)
        es[ik] = e
        for a in range(3): vs[ik,a] = wc.T@v[a]@w
    return es,vs,cellvol,scale


def _serial_sum_rule_weight(h,nk,T):
    """The single-k reference implementation of kubo.sum_rule_weight,
    built on kubo._second_derivatives (i.e. on current.hk_derivative)."""
    hm,orders,hkgen,jac,dr,cellvol,scale = kubo._setup(h)
    ks = kubo._kmesh(h,nk)
    W = np.zeros((3,3),dtype=np.float64)
    for k in ks:
        hk = kubo._hk(hkgen,k)
        (e,w) = algebra.eigh(hk)
        f = kubo._fermi(e,T)
        wc = np.conjugate(w)
        d2 = kubo._second_derivatives(hm,orders,jac,dr,hk,k)
        for a in range(3):
            for b in range(a,3):
                di = np.real(np.einsum("in,ij,jn->n",wc,d2[a][b],w,
                        optimize=True))
                W[a,b] += np.dot(f,di)
    for a in range(3):
        for b in range(a+1,3): W[b,a] = W[a,b]
    return W/(len(ks)*cellvol)


@pytest.mark.parametrize("model,orders",
        [(_model2d,[[1,0],[0,1],[2,0],[1,1],[0,2]]), (_model1d,[[1],[2]])])
def test_batched_derivative_matches_the_shared_reference(model,orders):
    """The einsum built from the stacked hoppings and the Bloch phases
    must reproduce current.hk_derivative -- the shared, benchmarked
    k-derivative that the single-k path uses -- at every k-point and for
    every derivative order the module asks for: the first derivatives
    that make the velocity, and the mixed second derivatives that make
    the diamagnetic weight. This is the one piece of the batched path
    that is a hand re-derivation rather than a rearrangement, so it gets
    its own direct check against the reference."""
    h = model()
    hm,ords,hkgen,jac,dr,cellvol,scale = kubo._setup(h)
    ks = kubo._kmesh(h,4)
    tms,dirs = kubo._hopping_arrays(hm)
    ph = kubo._bloch_phases(dirs,ks)
    for order in orders:
        batch = kubo._derivative_batch(tms,dirs,ph,order)
        for ik,k in enumerate(ks):
            ref = current.hk_derivative(hm,k,order=order)
            assert np.allclose(batch[ik],ref,atol=1e-10)


def test_optical_conductivity_matches_the_serial_reference():
    """sigma(omega) from the batched path and from the single-k
    reference must agree. The two use different eigensolvers (numba's
    eigh vs scipy's) and different k-derivatives (einsum vs
    current.hk_derivative), so their velocity matrix elements do NOT
    agree elementwise -- only the gauge-invariant output does."""
    h = _model2d()
    ks = kubo._kmesh(h,9)
    ws = np.linspace(0.,4.,12)
    T,delta,tol = 0.1,0.1,1e-6
    es,vs,cellvol,scale = _serial_bands_and_velocities(h,ks)
    ratio,dE = kubo._response_weights(es,T,tol*scale)
    ref = kubo._sigma_jit(dE,ratio,vs,ws,delta)/(len(ks)*cellvol)
    out = conductivity.optical_conductivity(h,energies=ws,nk=9,T=T,
            delta=delta,degeneracy_tol=tol)[1]
    assert np.max(np.abs(out-ref)) < 1e-12*np.max(np.abs(ref))


def test_drude_weight_matches_the_serial_reference():
    """Same oracle for the intraband weight, which sums over the
    (near-)degenerate band pairs only -- the part of the formula that is
    most exposed to the degenerate-subspace rotation differing between
    the two eigensolvers."""
    h = _model2d()
    ks = kubo._kmesh(h,9)
    T,tol = 0.1,1e-6
    es,vs,cellvol,scale = _serial_bands_and_velocities(h,ks)
    ratio,dE = kubo._response_weights(es,T,tol*scale,
            intraband=True,interband=False)
    ref = np.einsum("knm,kanm,kbmn->ab",ratio,vs,vs,optimize=True)
    ref = ref.real/(len(ks)*cellvol)
    out = conductivity.drude_weight(h,nk=9,T=T,degeneracy_tol=tol)
    assert np.max(np.abs(out-ref)) < 1e-12*np.max(np.abs(ref))


def test_sum_rule_weight_matches_the_serial_reference():
    """Same oracle for the diamagnetic weight, which is the only
    consumer of the mixed second derivatives."""
    h = _model2d()
    T = 0.1
    ref = _serial_sum_rule_weight(h,9,T)
    out = conductivity.sum_rule_weight(h,nk=9,T=T)
    assert np.max(np.abs(out-ref)) < 1e-12*np.max(np.abs(ref))


def test_sigma_is_invariant_under_a_degenerate_subspace_rotation():
    """Why the tests above may compare sigma but never the velocity
    matrix elements: the eigenbasis inside a degenerate multiplet is
    arbitrary, and two eigensolvers will pick different ones. The Kubo
    sum is built so that this cannot matter -- within a degenerate block
    the occupation factor is constant, so every term collapses to a
    trace Tr[P v_a P v_b] over the block projector P. Verified here by
    rotating the velocity matrix elements of a spin-degenerate model
    with a random unitary inside each degenerate block and checking that
    sigma does not move."""
    h = geometry.honeycomb_lattice().get_hamiltonian() # spin degenerate
    h.add_haldane(0.2)
    h.shift_fermi(0.3)
    ks = kubo._kmesh(h,6)
    ws = np.linspace(0.,4.,12)
    T,delta,tol = 0.05,0.1,1e-6
    es,vs,cellvol,scale = kubo._bands_and_velocities(h,ks)
    n = es.shape[1]
    # spin degeneracy: eigenvalues come out sorted, so every band pairs up
    # with its neighbour at every k-point
    assert np.max(np.abs(es[:,0::2]-es[:,1::2])) < 1e-10
    rng = np.random.default_rng(0)
    vsr = np.zeros_like(vs)
    for ik in range(len(ks)):
        u = np.zeros((n,n),dtype=np.complex128) # block diagonal unitary
        i = 0
        while i<n: # walk the degenerate blocks of this k-point
            j = i+1
            while j<n and abs(es[ik,j]-es[ik,i])<1e-9: j += 1
            m = rng.normal(size=(j-i,j-i)) + 1j*rng.normal(size=(j-i,j-i))
            u[i:j,i:j] = np.linalg.qr(m)[0] # random unitary of the block
            i = j
        for a in range(3): vsr[ik,a] = np.conjugate(u).T@vs[ik,a]@u
    def sigma(v):
        ratio,dE = kubo._response_weights(es,T,tol*scale)
        return kubo._sigma_jit(dE,ratio,v,ws,delta)/(len(ks)*cellvol)
    s0,s1 = sigma(vs),sigma(vsr)
    assert np.max(np.abs(vsr-vs)) > 0.1 # the rotation really did something
    assert np.max(np.abs(s1-s0)) < 1e-12*np.max(np.abs(s0))
