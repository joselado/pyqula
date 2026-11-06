"""integration="triangle" of the mean-field density matrix
(dmtk/simplexdm.py): the linear simplex (triangle, tetrahedron) integration
of the occupations on the uniform k-mesh."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.dmtk import simplexdm
from pyqula.scftk import densitydensity
from pyqula.scftk.densitydensity import Vinteraction


def _haldane():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(0.1) # complex eigenvectors, so a conjugation error shows
    h.add_sublattice_imbalance(0.2)
    return h


def _directions(h):
    return [(0,0,0)]+[tuple(int(x) for x in d) for d in h.get_dict()
            if tuple(d)!=(0,0,0)]


@pytest.mark.parametrize("p", [2,3,4])
def test_step_weights_match_sampling(p):
    """The T=0 vertex weights of one simplex are the integral of the
    barycentric coordinates over its occupied part, which a uniform
    sampling of the simplex reproduces, also with two equal energies"""
    rng = np.random.default_rng(p)
    lam = rng.dirichlet(np.ones(p),size=200000)
    for e in (np.sort(rng.normal(size=p)),np.array([0.,0.,0.5,1.2])[:p]):
        E = lam@e
        for mu in np.linspace(e[0]-0.1,e[-1]+0.1,9):
            out = np.zeros(p)
            c = simplexdm._step_weights(e,p,mu,out)
            ref = (lam*(E<mu)[:,None]).mean(axis=0)
            assert abs(c-np.sum(out))<1e-14
            assert np.max(np.abs(out-ref))<5e-3


@pytest.mark.parametrize("dim", [1,2,3])
def test_full_band_has_the_uniform_mesh_weight(dim):
    """Summed over the simplices that share it, every state of a fully
    occupied band weighs 1/nk^dim, the weight of the uniform mesh, at T=0
    and at T>0 far from the band"""
    nk = 6
    es = -1.-np.random.default_rng(0).random((nk**dim,2))
    for T in (0.,1e-2):
        w = simplexdm.simplex_weights(dim,nk,es,mu=0.,T=T)
        assert np.max(np.abs(w-1./nk**dim))<1e-13


def test_finite_temperature_limits():
    """The weights at small T go to the step weights, and their sum is the
    charge simplex_charge counts without forming them"""
    rng = np.random.default_rng(3)
    nk = 8
    es = rng.normal(size=(nk*nk,2))
    w0 = simplexdm.simplex_weights(2,nk,es,mu=0.1,T=0.)
    w1 = simplexdm.simplex_weights(2,nk,es,mu=0.1,T=1e-9)
    assert np.max(np.abs(w0-w1))<1e-8
    for T in (0.,1e-3,0.1):
        w = simplexdm.simplex_weights(2,nk,es,mu=0.1,T=T)
        assert abs(np.sum(w)-simplexdm.simplex_charge(2,nk,es,mu=0.1,T=T))<1e-12


def test_bloechl_keeps_the_charge_of_every_band():
    """sum_j dw_j = 0 in every simplex, so the Bloechl term moves no charge
    between bands and does not move the Fermi level"""
    rng = np.random.default_rng(4)
    nk = 10
    es = np.sort(rng.normal(size=(nk*nk,3)),axis=1)
    for T in (0.,1e-2):
        w = simplexdm.simplex_weights(2,nk,es,mu=0.,T=T)
        wb = simplexdm.simplex_weights(2,nk,es,mu=0.,T=T,bloechl=True)
        assert np.max(np.abs(wb-w))>1e-6 # it does something
        assert np.max(np.abs(np.sum(wb-w,axis=0)))<1e-13


def test_bloechl_band_energy_converges_faster():
    """With c_2 = 1/24 the Bloechl term removes the leading 1/nk^2 error
    of the band energy of a curved band at fixed filling"""
    h = geometry.square_lattice().get_hamiltonian(has_spin=False)
    h.add_onsite(0.9) # a metal
    ref = simplexdm.band_energy(h,nk=512,bloechl=True)
    err = abs(simplexdm.band_energy(h,nk=32)-ref)
    errb = abs(simplexdm.band_energy(h,nk=32,bloechl=True)-ref)
    assert err>1e-4 # the plain linear method is visibly off at nk=32
    assert errb<err/20.


def test_insulator_density_matrix_is_the_uniform_mesh_one():
    """With the Fermi level in a gap every simplex is full or empty, so the
    density matrix is exactly the one of exact diagonalization on the same
    mesh, for every lattice vector (the Bloch phase lives in the
    interpolated field)"""
    h = _haldane()
    ds = _directions(h)
    nk = 12
    dm = simplexdm.get_dm(h,ds,nk=nk,T=0.)
    ed = h.get_density_matrix(ds=ds,nk=nk,T=1e-10)
    for d in ds:
        assert np.max(np.abs(dm[d]-ed[d]))<1e-12, d


def test_metal_density_matrix_3d_agrees_with_dense_ed():
    """The tetrahedron integration of a cubic-lattice metal at a finite
    temperature converges to the dense uniform mesh at second order"""
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    h.add_onsite(0.7)
    ds = [(0,0,0),(1,0,0),(0,1,1)]
    T = 0.1
    ed = h.get_density_matrix(ds=ds,nk=40,T=T) # converged at this T
    errs = []
    for nk in (8,16):
        dm = simplexdm.get_dm(h,ds,nk=nk,T=T)
        errs.append(max(abs(dm[d][0,0]-ed[d][0,0]) for d in ds))
    assert errs[1]<3e-3
    assert errs[1]<errs[0]/3. # second order: about 4


def test_unknown_integration_lists_the_registry():
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    with pytest.raises(ValueError) as e:
        Vinteraction(h,U=1.,filling=0.3,nk=4,integration="tetra",
                load_mf=False,write=False,verbose=0)
    for name in densitydensity.get_integration_names():
        assert repr(name) in str(e.value)
    assert "triangle" in densitydensity.get_integration_names()


def test_refuses_nambu_and_finite():
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.setup_nambu_spinor()
    with pytest.raises(NotImplementedError):
        Vinteraction(h,U=-1.,filling=0.3,nk=4,integration="triangle",
                load_mf=False,write=False,verbose=0)
    g = geometry.chain().supercell(4)
    g.set_finite() # a finite system has no Brillouin zone
    with pytest.raises(ValueError):
        Vinteraction(g.get_hamiltonian(has_spin=True),U=1.,filling=0.3,
                integration="triangle",load_mf=False,write=False,verbose=0)


# ------------------------------------------------------------ the SCF loop

def _ferro(integration,nk,T,mf="ferro",seed=0,**kw):
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    np.random.seed(seed)
    scf = Vinteraction(h,U=FERRO_U,filling=FERRO_FILLING,nk=nk,
            integration=integration,T=T,mf=mf,maxerror=1e-9,maxite=500,
            solver="broyden_mixing",
            load_mf=False,write=False,verbose=0,**kw)
    assert scf.converged
    dm = scf.dm[(0,0,0)]
    mz = (dm[0,0]-dm[1,1]).real; mx = 2*dm[0,1].real; my = 2*dm[0,1].imag
    return scf,np.sqrt(mz**2+mx**2+my**2)


FERRO_U, FERRO_FILLING = 6.0, 0.4 # a partly polarized metal


def test_filling_is_conserved_in_a_metal():
    """The Fermi level is located on the same simplices, where the charge
    is continuous in mu, so the converged density matrix holds the
    requested filling to rounding, at T=0 and at T>0"""
    for T in (1e-7,1e-2):
        scf,m = _ferro("triangle",24,T)
        n = scf.dm[(0,0,0)].shape[0]
        assert abs(np.trace(scf.dm[(0,0,0)]).real/n-FERRO_FILLING)<1e-9
        assert m>0.05 # the state is magnetic


def test_metal_agrees_with_dense_ed():
    """A partly polarized Hubbard ferromagnet at T=0.05: the triangle mean
    field converges at second order in 1/nk to the one of exact
    diagonalization on a mesh dense enough for that T (nk=96 holds the
    magnetization to 1e-5)"""
    scf24,m24 = _ferro("triangle",24,5e-2)
    scf48,m48 = _ferro("triangle",48,5e-2)
    # from the triangle state, which only saves iterations
    scfe,me = _ferro("ed",96,5e-2,mf=scf48.mf)
    assert abs(m48-me)<3e-3
    assert abs(m48-me)<abs(m24-me)/3. # about 4
    e24 = abs(scf24.total_energy-scfe.total_energy)
    e48 = abs(scf48.total_energy-scfe.total_energy)
    assert e48<3e-3 and e48<e24/3.
    assert abs(scf48.hamiltonian.fermi-scfe.hamiltonian.fermi)<2e-3


def test_two_random_guesses_give_the_same_state():
    """The self-consistent state does not depend on the random initial
    guess: the magnitude of the magnetization and the total energy agree"""
    a,ma = _ferro("triangle",24,1e-3,mf="random",seed=1)
    b,mb = _ferro("triangle",24,1e-3,mf="random",seed=2)
    assert abs(ma-mb)<1e-6
    assert abs(a.total_energy-b.total_energy)<1e-8


def test_insulator_agrees_with_ed():
    """A gapped honeycomb magnet at half filling: every simplex is full or
    empty, so the triangle mean field is the exact diagonalization one on
    the same mesh, up to the Fermi level, which is anywhere in the gap"""
    g = geometry.honeycomb_lattice()
    out = []
    for integration in ("triangle","ed"):
        h = g.get_hamiltonian(has_spin=True)
        h.add_sublattice_imbalance(0.5)
        np.random.seed(0)
        scf = Vinteraction(h,U=1.,filling=0.5,nk=12,integration=integration,
                T=1e-7,mf="ferro",maxerror=1e-10,load_mf=False,write=False,
                verbose=0)
        out.append(scf)
    for d in out[0].dm:
        assert np.max(np.abs(out[0].dm[d]-out[1].dm[d]))<1e-8
    assert abs(out[0].total_energy-out[1].total_energy)<1e-8


def test_mean_field_hamiltonian_entry_point():
    """h.get_mean_field_hamiltonian routes integration="triangle" to the
    density-density engine, spinless (V1) included"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h1 = h.get_mean_field_hamiltonian(V1=1.,filling=0.35,nk=12,
            integration="triangle",mf="random",load_mf=False,write=False,
            verbose=0)
    assert h1.intra.shape==h.intra.shape
