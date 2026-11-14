"""The matrix elements of the triangle susceptibility on multi-band models.

chi_ij(q,w) = int dk sum_nm M^ij_nm(k) W_nm(k,w), with
M^ij_nm(k) = <n,k|A_i|m,k+q><m,k+q|B_j|n,k> and W_nm(k,w) the simplex
weight of the pair (n at k, m at k+q). On a one-band model M is one at
every k, so the orientation of the two factors, the conjugation of the
eigenvectors and the (i,j) layout of mode="matrix" are invisible there.
Here they are checked on models with several bands and complex
eigenvectors, against two routes that do not share the matrix elements of
chi_triangle: the same weights (simplexchi.pair_weights) contracted with
M written out from its definition, which agrees to rounding, and the
converged grid of chiAB_q at a temperature where the grid converges.
"""
import numpy as np
import pytest

from pyqula import algebra
from pyqula import geometry
from pyqula import operators
from pyqula.chitk import chiAB
from pyqula.chitk import simplexchi
from pyqula.dostk.simplexdos import uniform_mesh


def _spinful_honeycomb():
    """no degeneracies left: the exchange field along a generic direction
    and the Rashba coupling split the spin, the imbalance the sublattices"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_onsite(-0.6)
    h.add_sublattice_imbalance(0.2)
    h.add_rashba(0.3)
    h.add_exchange([0.1,0.2,0.4])
    return h


def _operators(h):
    """site projectors times local spin operators, one of them not
    Hermitian, and a dense operator with every row nonzero, so that the
    operators have different numbers of nonzero rows. A and B differ, so
    that exchanging them, or i and j, changes the response"""
    def local(name): return algebra.todense(h.get_operator(name).get_matrix())
    sx,sy,sz = local("sx"),local("sy"),local("sz")
    p = [algebra.todense(operators.index(h,n=[i]).get_matrix())
         for i in range(len(h.geometry.r))]
    n = h.intra.shape[0]
    rng = np.random.default_rng(7)
    r1 = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    r2 = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    pAs = np.array([p[0]@sx,p[1]@(sx+1j*sy),r1],dtype=np.complex128)
    pBs = np.array([p[0]@sy,p[1]@sz,r2],dtype=np.complex128)
    return pAs,pBs


def _definition(h,q,nk,pAs,pBs,energies,delta,T):
    """chi_ij from the definition of M, with the weights of pair_weights"""
    ks = uniform_mesh(2,nk)
    hk = h.get_hk_gen()
    def eigh(kk):
        return np.linalg.eigh(np.array([algebra.todense(hk(k)) for k in kk]))
    E1,U1 = eigh(ks)
    E2,U2 = eigh(ks+np.array(q)[None,:])
    W = simplexchi.pair_weights(2,nk,E1,E2,np.array(energies),delta,T)
    MA = np.einsum("kan,iab,kbm->kinm",U1.conj(),pAs,U2) # <n,k|A_i|m,k+q>
    MB = np.einsum("kbm,jbc,kcn->kjmn",U2.conj(),pBs,U1) # <m,k+q|B_j|n,k>
    return np.einsum("knmw,kinm,kjmn->wij",W,MA,MB)


@pytest.mark.parametrize("kernel",["sources","simplices"])
@pytest.mark.parametrize("lowmem",[False,True])
@pytest.mark.parametrize("q",[[0.137,0.211,0.],[1./8,2./8,0.]])
def test_matrix_elements_are_those_of_the_definition(kernel,lowmem,q,
        monkeypatch):
    """mode="matrix", "diagonal" and "trace" against the weights contracted
    with M from its definition, at a generic q and at a mesh q, with the
    eigenvectors kept and, lowmem, recomputed at every pass with one
    component per pass"""
    h = _spinful_honeycomb()
    pAs,pBs = _operators(h)
    nk = 8
    kw = dict(energies=[0.,0.7],delta=2e-2,T=0.)
    ref = _definition(h,q,nk,pAs,pBs,**kw)
    assert np.min(np.abs(ref)) > 1e-3*np.max(np.abs(ref)) # every entry seen
    if lowmem: monkeypatch.setattr(simplexchi,"max_memory_gb",1e-9)
    def run(mode):
        return simplexchi.chi_triangle(h,q=q,nk=nk,pAs=pAs,pBs=pBs,
                mode=mode,kernel=kernel,**kw)[1]
    scale = np.max(np.abs(ref))
    out = run("matrix")
    assert out.shape==ref.shape
    assert np.max(np.abs(out-ref)) < 1e-9*scale
    diag = np.array([np.diag(r) for r in ref])
    assert np.max(np.abs(run("diagonal")-diag)) < 1e-9*scale
    assert np.max(np.abs(run("trace")-np.mean(diag,axis=1))) < 1e-9*scale


def test_blocks_of_two_components_take_their_own_operators(monkeypatch):
    """the source kernel takes the components (i,j) of mode="matrix" by
    blocks, and builds in every block the matrix elements of the operators
    of that block only. With two components per block and three operators
    the second block is (0,2),(1,0): its B operators are not in order and
    skip one, so the map from the components to the operators of the block
    is not the identity, as it is with one block or one component per
    block. The memory limit is lowered until the blocks hold two
    components, and the response has to stay that of the definition"""
    h = _spinful_honeycomb()
    pAs,pBs = _operators(h)
    q,nk = [0.137,0.211,0.],8
    kw = dict(energies=[0.,0.7],delta=2e-2,T=0.)
    ref = _definition(h,q,nk,pAs,pBs,**kw)
    scale = np.max(np.abs(ref))
    sizes = [] # the components of every call of the source kernel
    sources = simplexchi.response_sources
    def counted(dim,nk,E1,E2,G,*args):
        sizes.append(G.shape[3])
        return sources(dim,nk,E1,E2,G,*args)
    monkeypatch.setattr(simplexchi,"response_sources",counted)
    def run(mode):
        del sizes[:]
        return simplexchi.chi_triangle(h,q=q,nk=nk,pAs=pAs,pBs=pBs,
                mode=mode,kernel="sources",**kw)[1]
    mem = simplexchi.max_memory_gb
    out = run("matrix")
    assert sizes==[9] # one block at the default limit
    while sizes[0]>2: # a block is the limit over a fixed size, rounded down
        mem *= 0.8
        monkeypatch.setattr(simplexchi,"max_memory_gb",mem)
        out = run("matrix")
    assert sizes==[2,2,2,2,1]
    assert np.max(np.abs(out-ref)) < 1e-9*scale
    diag = np.array([np.diag(r) for r in ref])
    assert np.max(np.abs(run("diagonal")-diag)) < 1e-9*scale
    assert sizes==[2,1]


def test_off_diagonal_entries_match_the_converged_grid():
    """the honeycomb metal at T = 0.1 t, where the grid of chiAB_q converges
    exponentially (to 3e-6 of the largest entry at nk = 48): the triangle
    has to reach it entry by entry in mode="matrix", the off-diagonal
    entries to a fraction of their own size. Their exchange, chi_ij against
    chi_ji, moves them by more than that at the finite frequency"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_onsite(-0.6)
    h.add_sublattice_imbalance(0.2)
    kw = dict(q=[0.37,0.21,0.],energies=[0.,1.],delta=0.3,T=0.1,
              mode="matrix")
    grid = np.array(chiAB.chiAB_q(h,nk=48,**kw)[1])
    tri = np.array(chiAB.chiAB_q(h,nk=32,integration="triangle",**kw)[1])
    scale = np.max(np.abs(grid))
    assert np.max(np.abs(tri-grid)) < 4e-3*scale
    for i,j in ((0,1),(1,0)):
        off = grid[:,i,j]
        assert np.min(np.abs(off)) > 3e-2*scale
        assert np.max(np.abs(tri[:,i,j]-off)/np.abs(off)) < 0.1
    assert np.abs(grid[1,0,1]-grid[1,1,0]) > 0.5*np.abs(grid[1,0,1])


def test_row_restricted_product_is_the_dense_product():
    """U^dag A_i V from the nonzero rows of each A_i only, for operators
    with the rows of one site, a dense operator, a single entry and a zero
    operator, against the dense product"""
    rng = np.random.default_rng(3)
    n,nkb = 6,4
    def unitary():
        z = rng.normal(size=(nkb,n,n)) + 1j*rng.normal(size=(nkb,n,n))
        return np.linalg.qr(z)[0]
    U,V = unitary(),unitary()
    local = rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2))
    ops = np.zeros((6,n,n),dtype=np.complex128)
    for i in range(3): ops[i,2*i:2*i+2] = local @ rng.normal(size=(2,n))
    ops[3] = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    ops[4,5,1] = 0.3-0.2j
    nz = np.any(ops!=0.,axis=2)
    dense = np.einsum("kan,iab,kbm->kinm",U.conj(),ops,V)
    for idx in (np.arange(6),np.array([4,0,5]),np.array([5])):
        rows,sub = simplexchi._nonzero_rows(ops,nz,idx)
        out = simplexchi._sandwich(U,rows,sub,V)
        assert out.shape==(nkb,len(idx),n,n)
        assert np.max(np.abs(out-dense[:,idx])) < 1e-14
