"""Physical invariants of the second-harmonic conductivity
(src/pyqula/conductivitytk/shg.py, entry point conductivity.shg_conductivity).

The absolute normalization and the sign are pinned by the static limit of
a crystal of isolated molecules, against the second derivative of the
exact polarization in a finite field. The velocity gauge (the production
path) is checked against the independent length-gauge sum over states, and
the symmetry content against the D3h pattern of a gapped honeycomb lattice,
against a rotation of the crystal, and against inversion. The length gauge
at two frequencies (shg.sigma2_length) is checked against the direct
density-matrix response of isolated molecules, against a velocity-gauge
two-frequency sum on a strained crystal, against the permutation of
the two fields, against Kleinman symmetry in the static limit, and through
its diagonal part against the closed-form injection rate; the 3D grid
against stacks of honeycomb layers.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import sculpt
from pyqula import conductivity
from pyqula import specialhamiltonian
from pyqula import multihopping
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import shg


def _gapped_honeycomb(m=0.5):
    """Spinless honeycomb lattice with a staggered onsite mass, the two-band
    model of hBN. pyqula's two sites sit at x = -1/2 and +1/2, so x is the
    armchair direction"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(m)
    return h


def _triangular_three_sublattice():
    """Triangular lattice with three inequivalent sublattices (onsite -3,
    0.5 and 3): three bands with no inversion and no degeneracy anywhere,
    so that the sum over intermediate states of the length gauge is
    exercised and well defined"""
    g = geometry.honeycomb_lattice()
    g.x = np.array([-0.5,0.5,-1.0])
    g.y = np.array([0.,0.,np.sqrt(3)/2])
    g.z = np.zeros(3)
    g.xyz2r()
    g.has_sublattice = False
    g.get_fractional()
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([-3.,0.5,3.])
    return h


def _bilayer(m1=0.5,m2=0.5,V=0.):
    """Two gapped honeycomb layers at z = +-1, the second one the
    inversion image of the first (sites at -r), coupled by a vertical
    hopping 0.3. With m1 = m2 and V = 0 the bilayer is centrosymmetric;
    m1 != m2 makes the two layers different and V is a potential step
    between them"""
    g = geometry.honeycomb_lattice()
    g.x = np.array([-0.5,0.5,0.5,-0.5])
    g.y = np.zeros(4)
    g.z = np.array([1.,1.,-1.,-1.])
    g.xyz2r()
    g.has_sublattice = False
    g.get_fractional()
    def tij(r1,r2):
        dr = r1 - r2
        if abs(dr[2])<1e-3 and abs(np.linalg.norm(dr)-1.)<1e-3: return 1.0
        if abs(np.linalg.norm(dr[:2]))<1e-3 and abs(abs(dr[2])-2.)<1e-3:
            return 0.3
        return 0.
    h = g.get_hamiltonian(has_spin=False,tij=tij)
    h.add_onsite([m1+V,-m1+V,m2-V,-m2-V])
    return h


def test_derivatives_match_kubo():
    """The first and second Cartesian derivatives of H built by shg.py
    from the bond vectors are kubo.py's velocity and inverse-mass
    operators, built there by a different route (lattice-gauge derivative
    plus intracell commutator)"""
    h = _triangular_three_sublattice()
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ks = np.random.default_rng(1).random((6,3))
    ks[:,2] = 0.
    ph = kubo._bloch_phases(dirs,ks)
    hks = np.einsum("kh,hnm->knm",ph,tms)
    hm,orders,_,jac,dr,_,_ = kubo._setup(h)
    tm2,di2 = kubo._hopping_arrays(hm)
    ph2 = kubo._bloch_phases(di2,ks)
    dhs = [kubo._derivative_batch(tm2,di2,ph2,o) for o in orders]
    v = kubo._velocities_batch(orders,jac,dr,hks,dhs)
    d2 = kubo._second_derivatives_batch(orders,jac,dr,hks,tm2,di2,ph2)
    for a in range(3):
        assert np.allclose(v[:,a],shg._cartesian_derivative(tms,d,ph,[a]),
                atol=1e-12)
        for b in range(3):
            assert np.allclose(d2[a][b],
                    shg._cartesian_derivative(tms,d,ph,[a,b]),atol=1e-12)


@pytest.mark.parametrize("gauge",["velocity","length"])
def test_static_limit_matches_finite_field(gauge):
    """A 1D crystal of isolated three-site molecules has no intercell
    hopping, so its static second-order polarization is that of one
    molecule, (1/2) d2P/dE2 with P = q sum_occ <x>/L from the exact
    eigenstates of H0 - q E x. At w = 0 with a small broadening,
    chi = sigma/(-2 i (w + i delta)) must reproduce it, which fixes the
    prefactor, the sign of q and the symmetrization"""
    L = 6.0
    g = geometry.chain()
    g.a1 = np.array([L,0.,0.])
    g.x = np.array([0.0,1.0,2.3]) ; g.y = np.zeros(3) ; g.z = np.zeros(3)
    g.xyz2r()
    g.has_sublattice = False
    g.update_reciprocal() ; g.get_fractional()
    H0 = np.array([[-1.0,-1.0,0.0],[-1.0,0.4,-0.7],[0.0,-0.7,1.3]])
    h = g.get_hamiltonian(has_spin=False,tij=lambda r1,r2: 0.)
    h.intra = H0.astype(np.complex128)
    delta = 1e-3
    (w,s) = h.get_shg_conductivity(energies=[0.],nk=4,delta=delta,T=0.,
            components="xxx",gauge=gauge)
    chi = s[0]/(-2j*(w[0]+1j*delta))
    q = shg._charge
    def P(E):
        es,vs = np.linalg.eigh(H0 - q*E*np.diag(g.x))
        occ = vs[:,es<0.]
        return q*np.sum(np.abs(occ)**2*g.x[:,None])/L
    dE = 1e-3
    ref = (P(dE) - 2*P(0.) + P(-dE))/dE**2/2.
    assert abs(ref)>1e-2 # a noncentrosymmetric molecule, not a vacuous zero
    assert abs(chi-ref)<1e-5*abs(ref) + 1e-8


@pytest.mark.parametrize("model",[_gapped_honeycomb,
                                  _triangular_three_sublattice])
def test_velocity_and_length_gauges_agree(model):
    """For a cold insulator with a nondegenerate spectrum the two gauges
    are two independent evaluations of the same tight-binding response
    and agree on the same k-mesh. The two-band case would be identically
    zero in a velocity gauge without the second- and third-derivative
    vertices"""
    h = model()
    comps = ["xxx","xyy","yxy","yyx","yyy","xxy"]
    es = [0.3,1.0,1.7,2.5,3.5]
    (w,sv) = conductivity.shg_conductivity(h,energies=es,nk=24,delta=0.1,
            T=0.,components=comps)
    (w,sl) = conductivity.shg_conductivity(h,energies=es,nk=24,delta=0.1,
            T=0.,components=comps,gauge="length")
    scale = max(np.max(np.abs(sv[c])) for c in comps)
    assert scale>1e-3
    for c in comps: assert np.allclose(sv[c],sl[c],atol=1e-10*scale)


def test_d3h_pattern_of_gapped_honeycomb():
    """D3h with x along the armchair direction allows only
    xxx = -xyy = -yxy = -yyx; every component with an odd number of y
    vanishes by the mirror y -> -y"""
    h = _gapped_honeycomb()
    comps = ["xxx","xyy","yxy","yyx","yyy","yxx","xxy","xyx"]
    (w,s) = h.get_shg_conductivity(energies=[0.4,1.3,2.2],nk=24,delta=0.1,
            components=comps)
    ref = np.max(np.abs(s["xxx"]))
    assert ref>1e-3
    for c in ["xyy","yxy","yyx"]:
        assert np.allclose(s[c],-s["xxx"],atol=1e-10*ref)
    for c in ["yyy","yxx","xxy","xyx"]:
        assert np.allclose(s[c],0.,atol=1e-10*ref)


def test_rotation_of_the_crystal_rotates_the_tensor():
    """Rotating the geometry by an angle that is not a symmetry rotates
    sigma_abc as a rank-3 tensor, sigma' = R R R sigma, which checks every
    in-plane component at a generic orientation rather than along an
    axis"""
    comps = [x+y+z for x in "xy" for y in "xy" for z in "xy"]
    def tensor(h):
        (w,s) = h.get_shg_conductivity(energies=[0.7,1.9],nk=24,delta=0.1,
                components=comps)
        t = np.zeros((2,2,2,2),dtype=np.complex128)
        for c in comps: t[:,"xy".index(c[0]),"xy".index(c[1]),
                          "xy".index(c[2])] = s[c]
        return t
    h0 = _triangular_three_sublattice()
    theta = 0.37
    h1 = h0.copy()
    h1.geometry = sculpt.rotate(h0.geometry,theta)
    R = np.array([[np.cos(theta),np.sin(theta)], # sculpt.rotate turns the
                  [-np.sin(theta),np.cos(theta)]]) # crystal clockwise
    t0,t1 = tensor(h0),tensor(h1)
    rot = np.einsum("ai,bj,ck,wijk->wabc",R,R,R,t0)
    assert np.max(np.abs(t0))>1e-3
    assert np.allclose(t1,rot,atol=1e-10*np.max(np.abs(t0)))


def test_inversion_symmetric_bilayer_has_no_shg():
    """The centrosymmetric bilayer gives zero in every component, and the
    response appears as soon as inversion is broken: by two different
    layers (in-plane) or by a potential step between them (out-of-plane),
    so the zero is not an artefact of a code path that returns nothing"""
    comps = ["xxx","xyy","zxx","zzz","xxz"]
    es = [0.6,1.2]
    (w,s0) = _bilayer().get_shg_conductivity(energies=es,nk=24,delta=0.1,
            components=comps)
    (w,sm) = _bilayer(m2=0.8).get_shg_conductivity(energies=es,nk=24,
            delta=0.1,components=comps)
    (w,sV) = _bilayer(V=0.1).get_shg_conductivity(energies=es,nk=24,
            delta=0.1,components=comps)
    for c in comps: assert np.allclose(s0[c],0.,atol=1e-12)
    assert np.min(np.abs(sm["xxx"]))>1e-2
    assert np.min(np.abs(sV["zxx"]))>1e-3


def test_options_are_checked():
    """An unknown gauge lists the accepted ones, the length gauge refuses a
    finite temperature, where it would silently drop the Fermi-surface
    terms, a zero broadening is refused, and a Nambu Hamiltonian is
    rejected with a message naming this routine"""
    h = _gapped_honeycomb()
    with pytest.raises(ValueError,match="length"):
        h.get_shg_conductivity(energies=[1.],nk=4,gauge="lenght")
    with pytest.raises(ValueError,match="T=0"):
        h.get_shg_conductivity(energies=[1.],nk=4,gauge="length",T=0.1)
    with pytest.raises(ValueError,match="xyy"):
        h.get_shg_conductivity(energies=[1.],nk=4,components="xy")
    with pytest.raises(ValueError,match="delta"):
        h.get_shg_conductivity(energies=[0.],nk=4,delta=0.)
    hn = _gapped_honeycomb()
    hn.setup_nambu_spinor()
    with pytest.raises(NotImplementedError,match="second-harmonic"):
        hn.get_shg_conductivity(energies=[1.],nk=4)


def test_susceptibility_is_the_converted_conductivity():
    """chi = i sigma/(2 (w + i delta) epsilon_0) in nm^2/V for a 2D
    geometry, with the lattice and energy units passed in"""
    h = _gapped_honeycomb()
    kw = dict(energies=[0.,0.8],nk=12,delta=0.1,components="xxx")
    (w,s) = h.get_shg_conductivity(**kw)
    (w,chi) = h.get_shg_susceptibility(length_unit=1.45,energy_unit=2.33,
            **kw)
    fac = 1.602176634e-19/8.8541878128e-12*1.45e-10/2.33**2/1e-18
    assert np.allclose(chi,1j*s/(2.*(w+0.1j))*fac,rtol=1e-12)


def test_monolayer_mos2_against_trolle():
    """Monolayer MoS2 from the three-band model of Liu et al. follows the
    D3h pattern with x along the zigzag direction, chi_yyy = -chi_yxx =
    -chi_xxy and chi_xxx = 0, and its static sheet susceptibility is of
    the order of the 0.3 nm^2/V of Trolle, Seifert and Pedersen, PRB 89,
    235410 (2014), arXiv:1310.0674, computed with an sp3d5 model. A three-
    band model with a different gap can only agree in order of magnitude,
    which is what is asserted (it gives 0.59 nm^2/V)"""
    h = specialhamiltonian.three_band_tmdc("MoS2")
    comps = ["yyy","yxx","xxy","xxx"]
    (w,chi) = h.get_shg_susceptibility(energies=[0.,1.0],nk=30,T=0.,
            delta=0.05,components=comps)
    ref = np.max(np.abs(chi["yyy"]))
    for c in ["yxx","xxy"]:
        assert np.allclose(chi[c],-chi["yyy"],atol=1e-8*ref)
    assert np.allclose(chi["xxx"],0.,atol=1e-8*ref)
    assert 0.1<abs(chi["yyy"][0])<1.


def _haldane_gapped_honeycomb(m=0.3,t2=0.2):
    """Spinless honeycomb lattice with a staggered mass and a Haldane
    term: no inversion and no time reversal, two nondegenerate bands"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.add_sublattice_imbalance(m)
    return h


def _strained_gapped_honeycomb(m=0.5,eps=0.3):
    """The gapped honeycomb lattice with the bond to one neighbouring cell
    stretched by a factor 1 + eps: the C3 rotation is gone and only the
    mirror y -> -y is left, so that no symmetry ties yxy to yyx"""
    h = _gapped_honeycomb(m)
    hd = {R:np.array(t.todense()) if hasattr(t,"todense") else np.array(t)
          for R,t in h.get_hopping_dict().items()}
    R = (1,0,0) if (1,0,0) in hd else [R for R in hd if R!=(0,0,0)][0]
    Rm = tuple(-x for x in R)
    hd[R] = hd[R]*(1. + eps) ; hd[Rm] = hd[Rm]*(1. + eps)
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _injection_closed_form(h,w,nk,delta,comp):
    """The golden-rule injection rate for E(t) = Re E0 exp(-iwt),

      eta_abc = (pi q^3/2) (1/(N_k V)) sum_k sum_{n!=m} f_nm Delta^a_mn
                r^b_mn r^c_nm L_delta(w - (E_m - E_n))

    with Delta^a_mn = v^a_mm - v^a_nn and L_delta the normalized
    Lorentzian, written directly from the band energies and velocities"""
    a,b,c = [kubo._index(s) for s in comp]
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ks = kubo._kmesh(h,nk)
    es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,ks)
    f = kubo._fermi(es,0.)
    n = es.shape[1]
    eps = es[:,None,:] - es[:,:,None] # eps[k,n,m] = E_m - E_n
    off = ~np.eye(n,dtype=bool)
    eps_safe = np.where(off,eps,1.)
    rb_mn = np.swapaxes(h1[:,b],1,2)/(1j*eps_safe) # r^b_mn at [k,n,m]
    rc_nm = h1[:,c]/(-1j*eps_safe) # r^c_nm at [k,n,m]
    v = np.real(np.diagonal(h1[:,a],axis1=1,axis2=2)) # v^a_nn
    Da = v[:,None,:] - v[:,:,None] # v^a_mm - v^a_nn at [k,n,m]
    fnm = f[:,:,None] - f[:,None,:]
    L = (delta/np.pi)/((w - eps)**2 + delta**2)
    s = np.sum(np.where(off,fnm*Da*rb_mn*rc_nm*L,0.))
    return np.pi*shg._charge**3/2.*s/(len(ks)*cellvol)


def _stack(h2,c=2.5,tm=0.,tab=0.):
    """3D crystal of copies of the 2D h2 stacked along z with spacing c,
    with the interlayer hopping diag(tm,-tm), which makes the staggered
    mass m + 2 tm cos(2 pi k3) depend on kz, and a hopping tab from each A
    site to its three in-plane B neighbours of the layer above, which keeps
    the C3 rotation but breaks the mirror z -> -z, so that the response
    has a polar part along z"""
    def dense(m): return np.array(m.todense()) if hasattr(m,"todense") \
            else np.array(m)
    h = h2.copy()
    h.geometry.a3 = np.array([0.,0.,c])
    h.geometry.dimensionality = 3
    hd = {(R[0],R[1],0):dense(m) for R,m in h2.get_hopping_dict().items()}
    if tm!=0.:
        M = np.diag([tm,-tm]).astype(np.complex128)
        hd[(0,0,1)] = M
        hd[(0,0,-1)] = M
    if tab!=0.:
        M = np.array([[0.,tab],[0.,0.]],dtype=np.complex128)
        for R,m in h2.get_hopping_dict().items():
            if abs(dense(m)[0,1])<1e-9: continue # not an A-B bond
            up,down = (R[0],R[1],1),(-R[0],-R[1],-1)
            hd[up] = hd.get(up,0.) + M
            hd[down] = hd.get(down,0.) + M.T.conj()
    h.dimensionality = 3
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def test_length_gauge_two_frequency_reduces_to_shg():
    """At w1 = w2 the two-frequency length gauge is the second-harmonic
    conductivity of the length gauge, component by component"""
    h = _gapped_honeycomb()
    comps = ["xxx","xyy","yxy","yyx","xxy"]
    es = np.array([0.4,1.3,2.5])
    (w,s) = conductivity.shg_conductivity(h,energies=es,nk=24,delta=0.1,
            T=0.,components=comps,gauge="length")
    s2 = shg.sigma2_length(h,es,es,nk=24,delta=0.1,components=comps)
    scale = np.max(np.abs(s["xxx"]))
    assert scale>1e-3
    for c in comps: assert np.allclose(s2[c],s[c],rtol=0.,atol=1e-14*scale)


def test_length_gauge_two_frequency_permutation():
    """sigma_abc(w1,w2) = sigma_acb(w2,w1), the permutation of the two
    fields together with their frequencies, while at w1 != w2 the tensor
    is not symmetric in b and c alone, so the average is over the right
    pairs"""
    h = _strained_gapped_honeycomb()
    w1s = np.array([0.7,1.1,2.3])
    w2s = np.array([1.9,-0.4,0.5])
    comps = ["yyx","yxy","xyy","xxx"]
    s12 = shg.sigma2_length(h,w1s,w2s,nk=18,delta=0.1,components=comps)
    s21 = shg.sigma2_length(h,w2s,w1s,nk=18,delta=0.1,components=comps)
    scale = np.max(np.abs(s12["xxx"]))
    assert scale>1e-3
    for (c,cp) in [("yyx","yxy"),("xyy","xyy"),("xxx","xxx")]:
        assert np.allclose(s12[c],s21[cp],rtol=0.,atol=1e-14*scale)
    assert np.min(np.abs(s12["yyx"]))>1e-3*scale
    assert np.max(np.abs(s12["yyx"]-s12["yxy"]))>1e-2*scale


def test_length_gauge_two_frequency_matches_isolated_molecules():
    """A 2D crystal of isolated three-site molecules has no intercell
    hopping, so its response at (w1,w2) is that of one molecule per cell,
    which the density-matrix recursion with the full position operator x
    (diagonal elements included) gives directly: rho1 at w2 from -q x^c,
    rho2 at w1 + w2 from -q x^b, and J = -i(w1 + w2) P with P = q Tr[x^a
    rho2]/A. The Bloch length gauge builds the same thing from r^a_nm and
    the generalized derivative instead, so this checks the two-frequency
    denominators and the pairing of each field with its frequency"""
    g = geometry.honeycomb_lattice()
    g.a1 = np.array([6.,0.,0.]) ; g.a2 = np.array([1.,5.,0.])
    g.x = np.array([0.0,1.0,0.3]) ; g.y = np.array([0.0,0.2,1.1])
    g.z = np.zeros(3)
    g.xyz2r()
    g.has_sublattice = False
    g.update_reciprocal() ; g.get_fractional()
    H0 = np.array([[-1.0,-1.0,-0.4],[-1.0,0.4,-0.7],[-0.4,-0.7,1.3]])
    h = g.get_hamiltonian(has_spin=False,tij=lambda r1,r2: 0.)
    h.intra = H0.astype(np.complex128)
    area = np.linalg.norm(np.cross(g.a1,g.a2))
    delta = 0.05
    q = shg._charge
    E,V = np.linalg.eigh(H0)
    f = (E<0.).astype(float)
    X = [np.conjugate(V.T)@np.diag(p)@V for p in (g.x,g.y)]
    eps = E[:,None] - E[None,:]
    def ordered(a,b,c,w1,w2): # field b at w1, field c at w2
        z1,z2 = w1 + 1j*delta,w2 + 1j*delta
        rho1 = -q*X[c]*(f[None,:] - f[:,None])/(z2 - eps)
        rho2 = -q*(X[b]@rho1 - rho1@X[b])/(z1 + z2 - eps)
        return -1j*(z1 + z2)*q*np.trace(X[a]@rho2)/area
    w1s = np.array([0.3,1.2,2.1,0.9])
    w2s = np.array([1.7,-0.5,0.8,-0.9])
    comps = [x+y+z for x in "xy" for y in "xy" for z in "xy"]
    s = shg.sigma2_length(h,w1s,w2s,nk=2,delta=delta,components=comps)
    ref = {}
    for comp in comps:
        a,b,c = ["xy".index(l) for l in comp]
        ref[comp] = np.array([(ordered(a,b,c,w1,w2) + ordered(a,c,b,w2,w1))/2.
                              for (w1,w2) in zip(w1s,w2s)])
    scale = max(np.max(np.abs(ref[c])) for c in comps)
    assert scale>1e-2
    for comp in comps:
        assert np.allclose(s[comp],ref[comp],rtol=0.,atol=1e-10*scale)


def _velocity_two_frequency(h,w1s,w2s,nk,delta,comps):
    """The velocity-gauge sigma_abc(w1,w2) of a cold insulator, averaged
    over the two orderings of the fields, written directly from the
    density-matrix recursion of the Peierls Hamiltonian H(K - qA), with
    A_b = E_b/(i w1) and A_c = E_c/(i w2):

      rho1^c  = -q h^c_nm (f_m - f_n)/(w2 - (E_n - E_m))
      rho2    = ([-q h^b,rho1^c] + [-q h^c,rho1^b] + q^2 [h^bc,rho0])_nm
                /(w1 + w2 - (E_n - E_m))
      K       = q Tr h^a rho2 - q^2 (Tr h^ab rho1^c + Tr h^ac rho1^b)
                + q^3 Tr h^abc rho0
      sigma   = -K/(2 w1 w2)

    with rho1^b at w1 and every frequency carrying + i delta"""
    q = shg._charge
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ks = kubo._kmesh(h,nk)
    out = {c:np.zeros(len(w1s),dtype=np.complex128) for c in comps}
    def tr(A,B): return np.einsum("knm,kmn->",A,B)
    def comm(A,B): return A@B - B@A
    for i0 in range(0,len(ks),kubo._kbatch):
        es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,
                ks[i0:i0+kubo._kbatch])
        f = kubo._fermi(es,0.)
        eps = es[:,:,None] - es[:,None,:] # E_n - E_m
        fmn = f[:,None,:] - f[:,:,None] # f_m - f_n
        rho0 = np.zeros(h1[:,0].shape,dtype=np.complex128)
        for i in range(es.shape[1]): rho0[:,i,i] = f[:,i]
        for (iw,(w1,w2)) in enumerate(zip(w1s,w2s)):
            z1,z2 = w1 + 1j*delta,w2 + 1j*delta
            for comp in comps:
                a,b,c = [kubo._index(s) for s in comp]
                r1c = -q*h1[:,c]*fmn/(z2 - eps)
                r1b = -q*h1[:,b]*fmn/(z1 - eps)
                r2 = (comm(-q*h1[:,b],r1c) + comm(-q*h1[:,c],r1b)
                      + q**2*comm(h2[:,b,c],rho0))/(z1 + z2 - eps)
                K = q*tr(h1[:,a],r2) \
                    - q**2*(tr(h2[:,a,b],r1c) + tr(h2[:,a,c],r1b)) \
                    + q**3*tr(h3[:,a,b,c],rho0)
                out[comp][iw] += -K/(2.*z1*z2)
    return {c:out[c]/(len(ks)*cellvol) for c in comps}


def test_length_gauge_two_frequency_matches_velocity_gauge():
    """On a dispersing crystal with no symmetry tying b to c (the strained
    honeycomb lattice), the length gauge at w1 != w2 matches the velocity
    gauge at two frequencies written in this file, once nk has converged.
    The bc-antisymmetric part, which the generalized derivative r^c;b
    carries and which vanishes at w1 = w2, is nonzero here, so this
    checks its index order. The velocity sum is itself anchored to the
    production velocity gauge at w1 = w2"""
    comps = ["yxy","yyx","xxy"]
    h = _strained_gapped_honeycomb()
    es = np.array([0.8,1.6])
    (w,s) = conductivity.shg_conductivity(h,energies=es,nk=12,delta=0.1,
            T=0.,components=comps)
    sv = _velocity_two_frequency(h,es,es,12,0.1,comps)
    for c in comps:
        assert np.allclose(sv[c],s[c],rtol=0.,atol=1e-12*np.max(np.abs(s[c])))
    w1s = np.array([0.7,1.1])
    w2s = np.array([1.9,-0.4])
    sl = shg.sigma2_length(h,w1s,w2s,nk=96,delta=0.3,components=comps)
    sv = _velocity_two_frequency(h,w1s,w2s,96,0.3,comps)
    scale = max(np.max(np.abs(sv[c])) for c in comps)
    assert scale>1e-2
    assert np.min(np.abs(sv["yxy"]-sv["yyx"]))>1e-2*scale # not bc-symmetric
    for c in comps:
        assert np.allclose(sl[c],sv[c],rtol=0.,atol=1e-5*scale)


def test_length_gauge_static_limit_is_kleinman_symmetric():
    """In the static limit the second-order response of an insulator is a
    third derivative of the energy in the field, so it is symmetric under
    any permutation of its three indices. On the strained honeycomb lattice
    no crystal symmetry ties xyy to yxy, and the term of the generalized
    derivative with the band velocities, which sums to zero on a C3-
    symmetric mesh, is what makes them equal (without it they differ by
    15 %)"""
    h = _strained_gapped_honeycomb()
    comps = ["xyy","yxy","yyx","yxx","xxy","xyx"]
    s = shg.sigma2_length(h,[0.],[0.],nk=48,delta=1e-3,components=comps)
    for (c1,c2) in [("xyy","yxy"),("xyy","yyx"),("yxx","xxy"),("yxx","xyx")]:
        assert abs(s[c1][0])>1e-6
        assert abs(s[c1][0]-s[c2][0])<1e-4*abs(s[c1][0])


@pytest.mark.parametrize("delta",[0.1,0.05])
def test_length_gauge_diagonal_is_the_golden_rule(delta):
    """delta times the diagonal part [D^b,Y^c]_nn of the length gauge at
    (w,-w) is the golden-rule injection rate at the same delta, exactly,
    for the linear (yxx) and for the complex mixed (xxy, xyx) components,
    on a Haldane-gapped honeycomb lattice where time reversal is broken;
    on the time-reversal symmetric gapped honeycomb lattice the diagonal
    of every bc-symmetric component vanishes while the rectified current
    itself, the shift current, does not"""
    h = _haldane_gapped_honeycomb()
    w = 2.0
    comps = ["yxx","xxy","xyx","yyy"]
    diag = shg.sigma2_length_diagonal(h,[w],[-w],nk=48,delta=delta,
            components=comps)
    for c in comps:
        eta = _injection_closed_form(h,w,48,delta,c)
        assert abs(eta)>1e-3
        assert abs(delta*diag[c][0] - eta)<1e-10*abs(eta)
    h0 = _gapped_honeycomb()
    comps = [x+y+z for x in "xy" for y in "xy" for z in "xy"]
    tot = shg.sigma2_length(h0,[w],[-w],nk=24,delta=delta,components=comps)
    diag = shg.sigma2_length_diagonal(h0,[w],[-w],nk=24,delta=delta,
            components=comps)
    assert np.max(np.abs(tot["xxx"]))>1e-3
    for c in comps:
        sym = (diag[c] + diag[c[0]+c[2]+c[1]])/2.
        assert np.allclose(sym,0.,atol=1e-12)


def test_shg_3d_decoupled_layers():
    """Decoupled gapped honeycomb layers stacked along a3 with spacing c
    give the 2D sheet response divided by c, and with an interlayer
    hopping diag(tm,-tm) the 3D response is the average over kz of the 2D
    one at the mass m + 2 tm cos(2 pi k3), which checks the kz mesh and the
    cell volume independently of each other"""
    comps = ["xxx","xyy"]
    es = [0.6,1.4,2.2]
    kw = dict(energies=es,delta=0.1,T=0.,components=comps)
    c = 2.5
    (w,s2) = _gapped_honeycomb().get_shg_conductivity(nk=16,**kw)
    (w,s3) = conductivity.shg_conductivity(_stack(_gapped_honeycomb(),c=c),
            nk=[16,16,3],**kw)
    scale = np.max(np.abs(s2["xxx"]))
    assert scale>1e-3
    for comp in comps:
        assert np.allclose(s3[comp],s2[comp]/c,rtol=0.,atol=1e-10*scale)
    tm,nz = 0.2,4
    (w,s3) = conductivity.shg_conductivity(
            _stack(_gapped_honeycomb(),c=c,tm=tm),nk=[16,16,nz],**kw)
    ref = {comp:0. for comp in comps}
    for j in range(nz):
        m = 0.5 + 2.*tm*np.cos(2.*np.pi*j/nz)
        (w,sj) = _gapped_honeycomb(m=m).get_shg_conductivity(nk=16,**kw)
        for comp in comps: ref[comp] = ref[comp] + sj[comp]/(nz*c)
    for comp in comps:
        assert np.allclose(s3[comp],ref[comp],rtol=0.,atol=1e-10*scale)
    assert np.max(np.abs(ref["xxx"]-s2["xxx"]/c))>1e-2*scale # kz matters


def test_shg_3d_gauges_agree():
    """On a dispersing 3D stack without the mirror z -> -z the velocity
    and the length gauge agree on the same mesh, including zzz, whose
    vertices are the kz derivatives. The mixed components with a z index
    are left out: there the two gauges differ by mesh sums of total
    k-derivatives that no symmetry of this stack cancels, so they agree
    only once nk has converged"""
    h = _stack(_gapped_honeycomb(),c=2.5,tm=0.2,tab=0.3)
    comps = ["xxx","xyy","yxy","zzz"]
    kw = dict(energies=[0.5,1.5,2.5],nk=[12,12,6],delta=0.1,T=0.,
            components=comps)
    (w,sv) = conductivity.shg_conductivity(h,**kw)
    (w,sl) = conductivity.shg_conductivity(h,gauge="length",**kw)
    scale = np.max(np.abs(sv["xxx"]))
    assert np.max(np.abs(sv["zzz"]))>1e-1*scale
    for c in comps: assert np.allclose(sv[c],sl[c],rtol=0.,atol=1e-10*scale)


@pytest.mark.parametrize("integration",["qtci","triangle"])
def test_shg_3d_rejects_qtci(integration):
    """The simplex integrations are built for 1D and 2D meshes; in 3D they
    refuse with a message naming 3D rather than using a 2D normalization"""
    h = _stack(_gapped_honeycomb())
    with pytest.raises(NotImplementedError,match="3D"):
        h.get_shg_conductivity(energies=[1.],nk=4,integration=integration)
