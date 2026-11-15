"""The mean field of the ultracell, integration="ultracell" of scftk.

With every state kept, the loop sees the same supercell Hamiltonian in
another basis, so it must converge to the mean field of the supercell
itself, integration="ed", from the same initial guess. Its response to a
weak modulation must be the RPA response about the converged mean-field
state, which chitk computes on the unit cell alone. With a cut, the basis
rebuilt from the lattice-periodic part of the mean field must converge to
the full result as emax grows, and do better than the basis of the clean
unit cell when the mean field has a uniform part.
"""

import numpy as np
import pytest

from pyqula import geometry
from pyqula.ultracelltk.basis import dense


def fractional(g):
    B = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)
    return lambda r: B@np.array(r)


def same_state(u2,h3,k=(0.21,0.33,0.)):
    m2 = dense(u2.hs.get_hk_gen()(np.array(k)))
    m3 = dense(h3.get_hk_gen()(np.array(k)))
    return np.max(np.abs(m2-m3))


@pytest.mark.parametrize("rebuild",[True,False])
def test_hubbard_every_state_is_the_supercell(rebuild):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian() # spinful
    hs = h.get_supercell([3,2])
    x = fractional(hs.geometry)
    hs.add_onsite(lambda r: 0.2*np.cos(2*np.pi*x(r)[0]))
    u = h.get_ultracell(hs)
    kw = dict(U=3.0,filling=0.5,mf="antiferro",nk=2,maxerror=1e-9,verbose=0,
            write=False)
    u2,e2 = u.get_mean_field_hamiltonian(return_total_energy=True,
            rebuild_basis=rebuild,**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3)<1e-7
    m = u2.get_vev(operator="sz",nk=2)
    assert np.min(np.abs(m))>0.3 # an antiferromagnet, not the trivial state
    assert np.max(np.abs(m-h3.get_vev(operator="sz",nk=2)))<1e-7


def test_bond_interaction_every_state_is_the_supercell():
    """A first-neighbor V1 has Fock terms on the bonds, so the mean field
    modulates the hoppings and not only the onsite energies"""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    n = 6
    hs = h.get_supercell(n)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*r[0]/n))
    u = h.get_ultracell(hs)
    kw = dict(V1=1.5,filling=0.5,mf="random",nk=4,maxerror=1e-10,verbose=0,
            write=False)
    np.random.seed(3)
    u2,e2 = u.get_mean_field_hamiltonian(return_total_energy=True,**kw)
    np.random.seed(3)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3,k=(0.37,0.,0.))<1e-7


def test_linear_response_is_the_rpa_about_the_mean_field():
    """A weak potential eps cos(2 pi x1/n) on a gapped honeycomb lattice:
    the change of the occupations, Fourier transformed over the cells, is
    chi V at q=Q, with chi the bare response at U=0 and the RPA response
    about the mean-field state at finite U, on the same k-mesh"""
    from pyqula.chitk.chiAB import chiAB_q
    from pyqula.chitk.densitychi import densitychi_RPA
    n,nk,eps = 4,2,1e-4
    g = geometry.honeycomb_lattice()
    x = fractional(g)
    V = lambda r: eps*np.cos(2*np.pi*x(r)[0]/n)
    for U in [0.,1.0]:
        h = g.get_hamiltonian()
        h.add_sublattice_imbalance(1.0)
        hs = h.get_supercell([n,n])
        hs1 = hs.copy() ; hs1.add_onsite(V)
        u0,u1 = h.get_ultracell(hs),h.get_ultracell(hs1)
        if U==0.:
            dn = u1.get_vev(nk=nk) - u0.get_vev(nk=nk)
            es,chis = chiAB_q(h,energies=[0.],q=[1./n,0.,0.],nk=n*nk,
                    delta=1e-7)
        else:
            kw = dict(U=U,filling=0.5,nk=nk,maxerror=1e-12,verbose=0,
                    mf="random",write=False)
            np.random.seed(1)
            m1 = u1.get_mean_field_hamiltonian(**kw).get_vev(nk=nk)
            np.random.seed(2)
            m0 = u0.get_mean_field_hamiltonian(**kw).get_vev(nk=nk)
            dn = m1 - m0
            np.random.seed(3)
            hmf = h.get_mean_field_hamiltonian(U=U,filling=0.5,nk=n*nk,
                    maxerror=1e-12,verbose=0,mf="random",integration="ed",
                    write=False)
            es,chis = densitychi_RPA(hmf,U=U,energies=[0.],q=[1./n,0.,0.],
                    nk=n*nk,delta=1e-7)
        v = np.array([V(r) for r in hs.geometry.r])
        Q,f = u1.get_harmonics(dn)
        Q,fv = u1.get_harmonics(v) # the potential in the same cell gauge
        i = np.where((np.round(Q[:,0]*n)==1)&(np.round(Q[:,1]*n)==0))[0][0]
        pred = np.array(chis[0])@fv[i]
        assert np.max(np.abs(f[i]-pred))<1e-3*np.max(np.abs(pred))
        assert np.max(np.abs(pred))>1e-7 # the response is not empty


def test_cut_rebuilt_basis_converges():
    """The antiferromagnet of graphene under a slow potential, cut at emax:
    the rebuilt basis converges to the full result as emax grows, and does
    better than the basis of the clean unit cell, which misses the uniform
    moment"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    n = 4
    hs = h.get_supercell([n,1])
    x = fractional(g)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*x(r)[0]/n))
    kw = dict(U=3.0,filling=0.5,mf="antiferro",nk=2,maxerror=1e-8,verbose=0,
            return_total_energy=True,write=False)
    u0,e0 = h.get_ultracell(hs).get_mean_field_hamiltonian(**kw)
    m0 = u0.get_vev(operator="sz",nk=2)
    err = []
    for emax in [1.0,1.5,2.5]:
        u2,e = h.get_ultracell(hs,emax=emax).get_mean_field_hamiltonian(**kw)
        err.append(abs(e-e0))
        assert e>=e0-1e-8 # the cut cannot go below the full basis
    assert err[0]>err[1]>err[2]
    assert err[2]<1e-3
    u3,e3 = h.get_ultracell(hs,emax=1.0).get_mean_field_hamiltonian(
            rebuild_basis=False,**kw)
    assert abs(e3-e0)>10*err[0]
    m3 = u3.get_vev(operator="sz",nk=2)
    assert np.max(np.abs(m3-m0))>0.1 # the clean basis loses the moment


def test_mean_field_wrong_inputs_raise():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    hs = h.get_supercell([2,2])
    u = h.get_ultracell(hs,emax=0.)
    with pytest.raises(ValueError): # the cut leaves no state above zero
        u.get_mean_field_hamiltonian(U=1.,filling=0.5,nk=1,verbose=0,
                write=False)
    with pytest.raises(ValueError):
        h.get_ultracell(hs).get_mean_field_hamiltonian(U=1.,integration="ed")
    with pytest.raises(ValueError): # a plain supercell has no ultracell basis
        hs.get_mean_field_hamiltonian(U=1.,integration="ultracell",verbose=0,
                write=False)
    with pytest.raises(NotImplementedError): # no spin-spin exchange
        h.get_ultracell(hs).get_mean_field_hamiltonian(J1=1.,write=False)
    hn = h.copy() ; hn.setup_nambu_spinor()
    with pytest.raises(NotImplementedError):
        hn.get_ultracell(hn.get_supercell([2,2])).get_mean_field_hamiltonian(
                U=-1.)
