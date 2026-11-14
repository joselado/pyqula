# The finite-delta simplex sum of dostk/simplexdos.py adds the simplex
# average of the resolvent onto the vertices of each simplex first, and
# then contracts it with the weights of every mesh point in one matrix
# product. The first form gathered the weights of the p vertices of every
# simplex at every energy. The two are the same sum in another order, so
# they agree to rounding; the reference below is the first form, written
# out with the same simplex average of the module.
import numpy as np
import pytest

from pyqula import geometry
from pyqula.dostk import simplexdos


def _gather_sum(dim,nk,es,W,energies,delta):
    """sum over simplices and bands of the vertex-averaged weight times the
    simplex average, as the first kernel did it"""
    sim,frac = simplexdos.simplices(dim,nk)
    p = sim.shape[1]
    t = np.zeros(p+1)
    hw = np.zeros(simplexdos._nterms+1)
    d = np.zeros(p+1,dtype=np.complex128)
    es = np.ascontiguousarray(es)
    out = np.zeros((len(energies),W.shape[2]),dtype=np.complex128)
    for ie,E in enumerate(energies):
        for it in range(len(sim)):
            for n in range(es.shape[1]):
                a = simplexdos._simplex_average(es,sim,it,n,E,-delta,t,
                        simplexdos._harm,simplexdos._fact,hw,d)
                out[ie] += a*W[sim[it],n,:].mean(axis=0)
    return out*frac


def _rashba():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.1,0.2,0.3])
    h.add_sublattice_imbalance(0.2)
    return h


def _models():
    chain = geometry.bichain().get_hamiltonian(has_spin=False)
    chain.add_onsite(lambda r: 0.3*np.sign(r[0]+1e-3))
    diamond = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    diamond.add_sublattice_imbalance(0.3) # every band a spin pair
    return [(chain,1,9),(_rashba(),2,7),(diamond,3,3)]


@pytest.mark.parametrize("kind",["dos","operator","site"])
def test_the_vertex_sum_is_the_gather_sum(kind):
    """the DOS, the operator-weighted DOS and the LDOS with an operator, in
    1D, 2D and 3D, at energies in and out of the bands"""
    energies = np.array([-2.3,-0.4,0.17,1.9])
    for h,dim,nk in _models():
        m = np.diag(np.arange(h.intra.shape[0])%3-1.)+0.2 # a generic matrix
        op = None if kind=="dos" else m+m.T
        f = simplexdos._Fields(h,kind=kind,operator=op)
        es,W = f(simplexdos.uniform_mesh(dim,nk))
        new = simplexdos.simplex_sum(dim,nk,es,W,energies,0.05)
        ref = _gather_sum(dim,nk,es,W,energies,0.05)
        assert np.max(np.abs(ref)) > 1e-2
        assert np.max(np.abs(new-ref)) < 1e-12*np.max(np.abs(ref))


def _projectors(es,U):
    """the band projectors averaged over the multiplets, as kind='green'
    of the first form kept them on the whole mesh"""
    n = U.shape[1]
    W = np.einsum("kin,kjn->knij",U,U.conj())
    W = np.ascontiguousarray(W.reshape(len(es),n,n*n))
    simplexdos._multiplet_average(es,W,simplexdos._degeneracy_tol(es))
    return W


def test_the_green_function_from_the_eigenvectors_is_the_projector_sum():
    """green_triangle keeps the eigenvectors and averages the vertex
    resolvent over each multiplet, which is the sum of the
    multiplet-averaged projectors of the first form. The spinful diamond
    lattice has a degenerate pair in every band, so the average matters"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.3)
    nk = 3
    energies = np.array([-1.7,0.1,1.2])
    es,U = simplexdos._Fields(h,kind="green")(simplexdos.uniform_mesh(3,nk))
    ref = _gather_sum(3,nk,es,_projectors(es,U),energies,0.05)
    g = simplexdos.green_triangle(h,energies,nk=nk,delta=0.05)
    n = h.intra.shape[0]
    assert np.max(np.abs(ref)) > 1e-2
    assert np.max(np.abs(g-ref.reshape(-1,n,n))) < 1e-12*np.max(np.abs(ref))


def test_blocks_of_energies_and_of_the_mesh_do_not_change_the_result(
        monkeypatch):
    """the energies are split in blocks, and the projectors of the Green's
    function are built for chunks of the mesh, so that each array stays
    near a fixed size; one energy per block and a few k-points per chunk
    give the same numbers"""
    h = _rashba()
    energies = np.linspace(-2.,2.,5)
    f = simplexdos._Fields(h,kind="site",operator="sz")
    es,W = f(simplexdos.uniform_mesh(2,8))
    d0 = simplexdos.simplex_sum(2,8,es,W,energies,0.05)
    g0 = simplexdos.green_triangle(h,energies,nk=8,delta=0.05)
    monkeypatch.setattr(simplexdos,"_block",50)
    d1 = simplexdos.simplex_sum(2,8,es,W,energies,0.05)
    g1 = simplexdos.green_triangle(h,energies,nk=8,delta=0.05)
    assert np.max(np.abs(d1-d0)) < 1e-12*np.max(np.abs(d0))
    assert np.max(np.abs(g1-g0)) < 1e-12*np.max(np.abs(g0))


@pytest.mark.parametrize("kind",["dos","operator","site","green"])
def test_the_diagonalization_batch_does_not_change_the_fields(kind):
    """the mesh is diagonalized in batches whose size goes down as 1/n^2,
    so that a many-orbital cell does not hold 2^15 Bloch matrices at once;
    the batches are independent, so any size gives the same fields"""
    h = _rashba()
    ks = simplexdos.uniform_mesh(2,9)
    op = None if kind in ("dos","green") else "sx"
    es0,W0 = simplexdos._Fields(h,kind=kind,operator=op)(ks)
    es1,W1 = simplexdos._Fields(h,kind=kind,operator=op,batch=7)(ks)
    assert np.max(np.abs(es1-es0)) < 1e-12
    if kind=="green": # eigenvectors, compared through the projectors
        W0,W1 = _projectors(es0,W0),_projectors(es1,W1)
    assert np.max(np.abs(W1-W0)) < 1e-10
    big = geometry.honeycomb_lattice().get_supercell(5).get_hamiltonian(
            has_spin=False)
    n = big.intra.shape[0]
    assert simplexdos._Fields(big,kind=kind).batch*n*n <= 2**22


def test_the_green_function_at_a_point_degeneracy_of_the_mesh():
    """The chain written with two sites per cell has its two bands
    degenerate at the zone boundary, a mesh point for an even nk, and not
    at the neighbouring points. So the vertex resolvent of the two bands
    differs there, while the basis the diagonalization returns there is
    arbitrary: the average over the pair is what makes the sum well
    defined. green_triangle takes it on the resolvent and the first form
    took it on the projectors, and the two agree. The spin pairs of the
    diamond lattice above are degenerate at every k, where the resolvents
    of a pair are equal and the average does nothing. Without the average
    the sum is another matrix, and with it the two sites of the cell, which
    a translation of the chain exchanges, have the same local Green's
    function"""
    h = geometry.bichain().get_hamiltonian(has_spin=False)
    nk = 8
    energies = np.array([-1.7,-0.2,0.3,1.1])
    es,U = simplexdos._Fields(h,kind="green")(simplexdos.uniform_mesh(1,nk))
    assert np.sum(es[:,1]-es[:,0]<1e-10)==1 # the zone boundary only
    ref = _gather_sum(1,nk,es,_projectors(es,U),energies,0.05)
    W = np.einsum("kin,kjn->knij",U,U.conj()).reshape(len(es),2,4)
    raw = _gather_sum(1,nk,es,np.ascontiguousarray(W),energies,0.05)
    g = simplexdos.green_triangle(h,energies,nk=nk,delta=0.05)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(raw-ref)) > 1e-3*scale # the average matters here
    assert np.max(np.abs(g-ref.reshape(-1,2,2))) < 1e-12*scale
    assert np.max(np.abs(g[:,0,0]-g[:,1,1])) < 1e-12*scale


@pytest.mark.parametrize("dim,nk",[(1,9),(2,7),(3,3)])
def test_slices_of_the_simplices_do_not_change_the_vertex_resolvent(dim,nk):
    """when a block has fewer bands times energies than there are threads,
    the simplices are cut in slices so that every thread has a task, each
    slice adding into its own row of mesh points. The sum over the slices
    is the sum over all the simplices, for any number of slices, more
    slices than simplices included"""
    h = [m for m,d,_ in _models() if d==dim][0]
    es,_ = simplexdos._Fields(h,kind="dos")(simplexdos.uniform_mesh(dim,nk))
    esT = np.ascontiguousarray(es.T)
    sim,_ = simplexdos.simplices(dim,nk)
    energies = np.array([-0.4,0.17,1.9])
    def resolvent(nrep):
        A = simplexdos._vertex_resolvent(sim,esT,energies,0.05,nrep,
                simplexdos._harm,simplexdos._fact)
        assert A.shape==(es.shape[1],len(energies),nrep,len(es))
        return A.sum(axis=2)
    A1 = resolvent(1)
    assert np.max(np.abs(A1)) > 1e-2
    for nrep in (2,5,len(sim)+3):
        assert np.max(np.abs(resolvent(nrep)-A1)) < 1e-12*np.max(np.abs(A1))


def test_the_number_of_threads_does_not_change_the_sum():
    """the slices follow the number of threads: one band at one energy is
    one task on one thread and one slice per thread on several, and both
    give the same vertex resolvent to rounding"""
    import numba
    h = geometry.square_lattice().get_hamiltonian(has_spin=False)
    es,_ = simplexdos._Fields(h,kind="dos")(simplexdos.uniform_mesh(2,8))
    esT = np.ascontiguousarray(es.T)
    sim,_ = simplexdos.simplices(2,8)
    nt = numba.get_num_threads()
    if nt==1: pytest.skip("needs two numba threads")
    out = []
    try:
        for n in (1,nt):
            numba.set_num_threads(n)
            (b,A), = simplexdos._vertex_resolvent_blocks(sim,esT,
                    np.array([0.3]),0.05)
            assert b==slice(0,1) and A.shape==(1,1,len(es))
            out.append(A)
    finally: numba.set_num_threads(nt)
    assert np.max(np.abs(out[0])) > 1e-2
    assert np.max(np.abs(out[1]-out[0])) < 1e-12*np.max(np.abs(out[0]))
