# Tests of the triangle (segment, tetrahedron) and qtci integrations of the
# local density of states of a crystal (dostk/simplexdos.py, reached from
# ldos.get_ldos, ldos.multi_ldos and ldos.ldosmap). Each one checks the
# simplex LDOS against an independent path -- the Lorentzian sum of
# mode="arpack" on a fine mesh, the triangle DOS of h.get_dos, the
# spinless copy of a spinful Hamiltonian -- never a recorded number.
import numpy as np
import pytest

from pyqula import geometry
from pyqula import ldos
from pyqula.dostk import simplexdos


def _graphene(has_spin=False):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_sublattice_imbalance(0.3) # the two sites differ
    return h


def _ldos(h,**kwargs):
    return np.array(h.get_ldos(nrep=1,write=False,**kwargs)[2])


def test_triangle_ldos_agrees_with_the_diagonalization_at_large_delta():
    """At a broadening a fine grid resolves, the triangle LDOS on a coarse
    mesh is the LDOS of mode='arpack' on a fine one, site by site, on a
    lattice with three inequivalent sites"""
    h = geometry.lieb_lattice().get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 0.2*np.cos(r[0])+0.1*np.sin(2*r[1]))
    for e in [-1.7,-0.4,0.15,1.2]:
        tri = _ldos(h,e=e,delta=0.1,nk=60,integration="triangle")
        ref = _ldos(h,e=e,delta=0.1,nk=150,mode="arpack")
        assert np.max(np.abs(tri-ref)) < 3e-3*np.max(ref)
        assert tri.shape==ref.shape


def test_triangle_ldos_with_an_operator_agrees_with_the_diagonalization():
    """With an operator each band is weighted with <u_n|A|u_n> times its
    density on the site, the convention of mode='arpack'. The exchange
    field is along a generic direction, so that sz varies with k"""
    h = _graphene(has_spin=True)
    h.add_rashba(0.2)
    h.add_exchange([0.1,0.2,0.3])
    for e in [-1.1,0.6]:
        tri = _ldos(h,e=e,delta=0.1,nk=120,integration="triangle",
                operator="sz")
        ref = _ldos(h,e=e,delta=0.1,nk=150,mode="arpack",operator="sz")
        assert np.max(np.abs(ref)) > 1e-3 # a nonzero spin polarization
        assert np.max(np.abs(tri-ref)) < 3e-3*np.max(np.abs(ref))+1e-5


@pytest.mark.parametrize("delta",[0.,1e-2])
@pytest.mark.parametrize("name",["chain2","kagome","diamond"])
def test_the_sites_add_up_to_the_triangle_dos(name,delta):
    """The weights of a band on the sites add up to one, so the sum of the
    simplex LDOS over the sites is the triangle DOS of h.get_dos on the
    same mesh, to rounding, in 1D, 2D and 3D"""
    if name=="chain2":
        h = geometry.bichain().get_hamiltonian(has_spin=False)
        h.add_onsite(lambda r: 0.3*np.sign(r[0]+1e-3)); nk = 101
    elif name=="kagome":
        h = geometry.kagome_lattice().get_hamiltonian(has_spin=False)
        h.add_onsite(lambda r: 0.2*r[0]); nk = 33
    else:
        h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=False)
        h.add_sublattice_imbalance(0.3); nk = 10
    es = np.linspace(-3.5,3.5,23)+0.0037
    _,ld,_,_ = simplexdos.ldos_triangle_counted(h,es,nk=nk,delta=delta)
    _,d = h.get_dos(integration="triangle",energies=es,nk=nk,delta=delta,
            write=False)
    assert np.max(d) > 0.1
    assert np.max(np.abs(ld.sum(axis=1)-d)) < 1e-10*np.max(d)


def test_a_spinful_hamiltonian_sums_the_spin():
    """Without a spin-dependent term the spinful LDOS of a site is twice
    the spinless one, at finite delta and at delta=0"""
    for delta in [0.,0.05]:
        d0 = _ldos(_graphene(),e=0.7,delta=delta,nk=60,
                integration="triangle")
        d1 = _ldos(_graphene(has_spin=True),e=0.7,delta=delta,nk=60,
                integration="triangle")
        assert d1.shape==d0.shape
        assert np.max(np.abs(d1-2*d0)) < 1e-10*np.max(d0)


def test_the_two_sublattices_differ_at_zero_delta():
    """With a sublattice imbalance m the weight of the band at energy E on
    the site at +m is (E+m)/(2E), so at delta=0 the LDOS of the two sites
    are in the ratio (E+m)/(E-m). The vertex average of the weights over a
    triangle makes it exact only up to the mesh, at second order"""
    h = _graphene()
    m = 0.3
    onsite = np.diag(h.intra).real
    ip = int(np.argmax(onsite)); im = 1-ip
    for e in [1.4,-1.6,-2.1]: # away from the band edges at +-m
        d = _ldos(h,e=e,delta=0.,nk=99,integration="triangle")
        ratio = (e+m)/(e-m)
        assert abs(d[ip]/d[im]-ratio) < 1e-3*ratio


def test_qtci_ldos_agrees_with_the_triangle_ldos():
    """The qtci LDOS reconstructs the band energies and the site weights on
    the 2^R mesh, so it is the triangle LDOS on that mesh to the tolerance
    of the interpolation"""
    h = _graphene()
    for e in [-1.3,0.5]:
        tri = _ldos(h,e=e,delta=1e-2,nk=64,integration="triangle")
        qt = _ldos(h,e=e,delta=1e-2,nk=64,integration="qtci",tolerance=1e-8)
        assert np.max(np.abs(tri-qt)) < 1e-4*np.max(tri)


def test_multi_ldos_writes_the_same_files(tmp_path,monkeypatch):
    """multi_ldos with the triangle writes the files of the diagonalization
    mode, with the same shape, the map at each energy is what get_ldos
    returns there, and MULTILDOS/DOS.OUT is the triangle DOS"""
    monkeypatch.chdir(tmp_path)
    h = _graphene()
    energies = np.array([-0.8,0.4])
    h.get_multildos(energies=energies,delta=0.05,nrep=2,nk=12)
    ref = {f:np.genfromtxt("MULTILDOS/"+f) for f in
            ["LDOS_-0.8_.OUT","LDOS_0.4_.OUT","DOS.OUT"]}
    refmap = np.genfromtxt("DOSMAP.OUT")
    out = h.get_multildos(energies=energies,delta=0.05,nrep=2,nk=60,
            integration="triangle")
    assert np.array(out[3]).shape==(2,2)
    for f in ref: assert np.genfromtxt("MULTILDOS/"+f).shape==ref[f].shape
    assert np.genfromtxt("DOSMAP.OUT").shape==refmap.shape
    for e in energies:
        m = np.genfromtxt("MULTILDOS/LDOS_"+str(e)+"_.OUT")
        assert np.max(np.abs(m[:,:2]-ref["LDOS_"+str(e)+"_.OUT"][:,:2]))<1e-12
        d = _ldos(h,e=e,delta=0.05,nk=60,integration="triangle")
        assert np.max(np.abs(m[:,2]-np.tile(d,4))) < 1e-8*np.max(d)
    dosout = np.genfromtxt("MULTILDOS/DOS.OUT").T
    _,d = h.get_dos(integration="triangle",energies=dosout[0],nk=60,
            delta=0.05,write=False)
    assert np.max(np.abs(dosout[1]-d)) < 1e-6*np.max(d)


def test_get_ldos_writes_the_replicated_map(tmp_path,monkeypatch):
    """LDOS.OUT has the rows and positions of the diagonalization mode,
    with the nrep supercell replication"""
    monkeypatch.chdir(tmp_path)
    h = _graphene()
    h.get_ldos(e=0.5,delta=0.05,nk=10,nrep=3)
    ref = np.genfromtxt("LDOS.OUT")
    h.get_ldos(e=0.5,delta=0.05,nk=60,nrep=3,integration="triangle")
    out = np.genfromtxt("LDOS.OUT")
    assert out.shape==ref.shape
    assert np.max(np.abs(out[:,:2]-ref[:,:2])) < 1e-12


def test_ldosmap_triangle_is_the_ldos_at_each_energy():
    h = _graphene()
    es = np.array([-1.,0.45])
    _,ds = ldos.ldosmap(h,energies=es,nk=48,delta=0.05,integration="triangle")
    for e,d in zip(es,ds):
        ref = _ldos(h,e=e,delta=0.05,nk=48,integration="triangle")
        assert np.max(np.abs(d-ref)) < 1e-12*np.max(ref)


def test_the_integration_is_checked():
    h = _graphene()
    with pytest.raises(ValueError,match="triangle"):
        h.get_ldos(e=0.,integration="tetra",write=False)
    for name in ldos.get_integration_names():
        assert name in ["grid","triangle","qtci"]
    with pytest.raises(ValueError,match="mode"):
        h.get_ldos(e=0.,delta=0.1,mode="green",integration="triangle",
                write=False)
    with pytest.raises(ValueError,match="ks"):
        h.get_ldos(e=0.,ks=[[0.,0.,0.]],integration="triangle",write=False)
    with pytest.raises(TypeError,match="num_bands"):
        h.get_ldos(e=0.,integration="triangle",num_bands=4,write=False)
    with pytest.raises(ValueError,match="delta"): # grid needs delta > 0
        h.get_ldos(e=0.,delta=0.,write=False)
    with pytest.raises(NotImplementedError,match="2D"):
        geometry.chain().get_hamiltonian().get_ldos(e=0.,
                integration="qtci",write=False)
