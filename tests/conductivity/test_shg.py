"""Physical invariants of the second-harmonic conductivity
(src/pyqula/conductivitytk/shg.py, entry point conductivity.shg_conductivity).

The absolute normalization and the sign are pinned by the static limit of
a crystal of isolated molecules, against the second derivative of the
exact polarization in a finite field. The velocity gauge (the production
path) is checked against the independent length-gauge sum over states, and
the symmetry content against the D3h pattern of a gapped honeycomb lattice,
against a rotation of the crystal, and against inversion.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import sculpt
from pyqula import conductivity
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
