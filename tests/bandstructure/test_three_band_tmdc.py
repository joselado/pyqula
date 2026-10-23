"""The three-band TMD model of Liu et al., PRB 88, 085433 (2013)
(specialhamiltonian.three_band_tmdc).

The real-space hoppings are built from the paper's closed-form H(k), so the
first test checks the stored Hamiltonian against that formula, and the
second checks the formula itself, through its bands, against the energies
at Gamma, K and M that the paper derives separately (its Table I). The
spin-orbit splitting at K and the Fermi level are checked as well.
"""
import numpy as np
import pytest

from pyqula import specialhamiltonian
from pyqula.specialhamiltoniantk import threebandtmdc


def _hk(h,kr):
    m = h.get_hk_gen()([kr[0],kr[1],0.])
    if hasattr(m,"todense"): m = m.todense()
    return np.asarray(m)


def _reciprocal(h):
    g = h.geometry
    A = np.array([g.a1[:2],g.a2[:2]])
    return A,2.*np.pi*np.linalg.inv(A).T


@pytest.mark.parametrize("material",threebandtmdc.get_material_names())
@pytest.mark.parametrize("model",["NN","TNN"])
@pytest.mark.parametrize("functional",["GGA","LDA"])
def test_hamiltonian_matches_the_paper(material,model,functional):
    """At random k-points the Bloch Hamiltonian equals the paper's H(k)
    element by element, which pins the hoppings, their phases and the
    orientation of the lattice"""
    h = specialhamiltonian.three_band_tmdc(material,model=model,
            functional=functional,soc=False,fermi=None)
    a,p = threebandtmdc._parameters(material,model,functional)
    A,B = _reciprocal(h)
    for kr in np.random.default_rng(3).random((6,2)):
        K = kr[0]*B[0] + kr[1]*B[1]
        assert np.allclose(_hk(h,kr),threebandtmdc._hk_liu(K,a,p),atol=1e-12)


@pytest.mark.parametrize("material",threebandtmdc.get_material_names())
def test_high_symmetry_energies_of_table_one(material):
    """The NN bands at Gamma, K and M are the closed-form energies of the
    paper's Table I, an independent derivation from the parameters"""
    h = specialhamiltonian.three_band_tmdc(material,model="NN",soc=False,
            fermi=None)
    a,p = threebandtmdc._parameters(material,"NN","GGA")
    e1,e2,t0,t1,t2,t11,t12,t22 = [p[n] for n in
            ["e1","e2","t0","t1","t2","t11","t12","t22"]]
    f1 = (e1+e2)/2. - t0 - 1.5*t11 + 0.5*t22
    f2 = 0.5*np.sqrt((e1-e2-2*t0+3*t11-t22)**2 + 64*t2**2)
    s3 = np.sqrt(3.)
    ref = {"G":[e1+6*t0,e2+3*(t11+t22),e2+3*(t11+t22)],
           "K":[e2-1.5*(t11+t22)-3*s3*t12,e1-3*t0,
                e2-1.5*(t11+t22)+3*s3*t12],
           "M":[f1-f2,e2+t11-3*t22,f1+f2]}
    pts = {"G":[0.,0.],"K":[4*np.pi/(3*a),0.],
           "M":[np.pi/a,np.pi/(s3*a)]}
    A,B = _reciprocal(h)
    for n in pts:
        kr = A@np.array(pts[n])/(2.*np.pi) # reduced coordinates
        es = np.linalg.eigvalsh(_hk(h,kr))
        assert np.allclose(es,sorted(ref[n]),atol=1e-10)


@pytest.mark.parametrize("material",threebandtmdc.get_material_names())
def test_valence_band_splitting_at_K(material):
    """At K the valence band is an orbital with L_z = +-2, so the onsite
    lambda L.S splits it by exactly 2 lambda, with opposite signs at K and
    -K (the spin-valley locking)"""
    h = specialhamiltonian.three_band_tmdc(material,model="NN")
    lamb = threebandtmdc._soc[material]
    for kr in [[2./3.,2./3.],[1./3.,1./3.]]: # K and -K
        es = np.linalg.eigvalsh(_hk(h,kr))
        assert abs((es[1]-es[0])-2.*lamb)<1e-10


@pytest.mark.parametrize("model",["NN","TNN"])
def test_fermi_level_in_the_gap(model):
    """With fermi="midgap" one band per spin is below zero at every k, the
    undoped monolayer, and the gap is around 1.5 eV"""
    h = specialhamiltonian.three_band_tmdc("MoS2",model=model)
    for kr in np.random.default_rng(5).random((20,2)):
        es = np.linalg.eigvalsh(_hk(h,kr))
        assert np.sum(es<0.)==2
    assert 1.2<h.get_gap()<1.8


def test_options_are_checked():
    """Each string-selected option lists its accepted values"""
    with pytest.raises(ValueError,match="MoS2"):
        specialhamiltonian.three_band_tmdc("MoS3")
    with pytest.raises(ValueError,match="TNN"):
        specialhamiltonian.three_band_tmdc(model="NNN")
    with pytest.raises(ValueError,match="LDA"):
        specialhamiltonian.three_band_tmdc(functional="PBE")


def test_one_layer_is_the_monolayer():
    """A multilayer of one layer is the monolayer"""
    h1 = specialhamiltonian.three_band_tmdc_multilayer(["WSe2"],0.1)
    h0 = specialhamiltonian.three_band_tmdc("WSe2")
    for kr in np.random.default_rng(7).random((4,2)):
        assert np.allclose(_hk(h1,kr),_hk(h0,kr),atol=1e-12)


@pytest.mark.parametrize("stacking",["H","R"])
def test_interlayer_hopping_at_gamma_and_K(stacking):
    """In a homobilayer the d_z2 hopping t_perp to the three nearest metal
    atoms adds to zero at K, so the bilayer has there the monolayer
    energies twice, and at Gamma it splits the d_z2 band by 6|t_perp|"""
    tp = 0.1
    h = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],tp,
            stacking=stacking,soc=False,fermi=None)
    h0 = specialhamiltonian.three_band_tmdc("MoS2",soc=False,fermi=None)
    eK = np.linalg.eigvalsh(_hk(h,[2./3.,2./3.]))
    e0 = np.linalg.eigvalsh(_hk(h0,[2./3.,2./3.]))
    assert np.allclose(eK,np.sort(np.concatenate([e0,e0])),atol=1e-10)
    eG = np.linalg.eigvalsh(_hk(h,[0.,0.]))
    assert abs((eG[1]-eG[0])-6.*tp)<1e-10


def test_2H_bilayer_is_centrosymmetric():
    """With spin-orbit coupling, the 2H homobilayer has inversion and
    time reversal, so every band is doubly degenerate, and its second-
    harmonic response vanishes; the R bilayer has no inversion and
    responds"""
    h = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],0.1)
    for kr in np.random.default_rng(9).random((4,2)):
        es = np.linalg.eigvalsh(_hk(h,kr))
        assert np.allclose(es[0::2],es[1::2],atol=1e-10)
    comps = ["yyy","zxx","yyz"]
    kw = dict(energies=[0.5,1.0],nk=18,delta=0.05,components=comps)
    (w,s) = h.get_shg_conductivity(**kw)
    hr = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],0.1,
            stacking="R")
    (w,sr) = hr.get_shg_conductivity(**kw)
    ref = np.max(np.abs(sr["yyy"]))
    assert ref>1e-2
    for c in comps: assert np.allclose(s[c],0.,atol=1e-10*ref)


def test_heterostructure_cell():
    """A MoS2/MoS2/WSe2/WSe2 stack shares the average lattice constant,
    and with fermi="midgap" one band per spin and layer is occupied"""
    h = specialhamiltonian.three_band_tmdc_multilayer(
            ["MoS2","MoS2","WSe2","WSe2"],0.1,onsite=[0.,0.,0.5,0.5])
    a = (2*3.190 + 2*3.325)/4.
    assert np.allclose(h.geometry.a1,[a,0.,0.])
    for kr in np.random.default_rng(11).random((10,2)):
        assert np.sum(np.linalg.eigvalsh(_hk(h,kr))<0.)==8


def test_multilayer_options_are_checked():
    with pytest.raises(ValueError,match="H, R"):
        specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],0.1,
                stacking="AB")
    with pytest.raises(ValueError,match="one value per interface"):
        specialhamiltonian.three_band_tmdc_multilayer(["MoS2"]*3,0.1,
                stacking=["H"])
    with pytest.raises(ValueError,match="one energy per layer"):
        specialhamiltonian.three_band_tmdc_multilayer(["MoS2"]*2,0.1,
                onsite=[0.])


def test_t_perp_per_interface():
    """With t_perp = [0.1,0.] the third layer is decoupled, so the
    trilayer has at every k the levels of the 2H bilayer and of the
    monolayer (the third layer has the orientation of the first)"""
    h3 = specialhamiltonian.three_band_tmdc_multilayer(["MoS2"]*3,[0.1,0.],
            soc=False,fermi=None)
    h2 = specialhamiltonian.three_band_tmdc_multilayer(["MoS2"]*2,0.1,
            soc=False,fermi=None)
    h1 = specialhamiltonian.three_band_tmdc("MoS2",soc=False,fermi=None)
    for kr in np.random.default_rng(13).random((5,2)):
        ref = np.concatenate([np.linalg.eigvalsh(_hk(h2,kr)),
                              np.linalg.eigvalsh(_hk(h1,kr))])
        assert np.allclose(np.linalg.eigvalsh(_hk(h3,kr)),np.sort(ref),
                atol=1e-10)
    with pytest.raises(ValueError,match="t_perp needs one value"):
        specialhamiltonian.three_band_tmdc_multilayer(["MoS2"]*3,[0.1])


def _sk_table(d,V):
    """The Slater-Koster table of d-d two-center integrals as written in
    Slater and Koster, Phys. Rev. 94, 1498 (1954), Table I, in the basis
    d_z2, d_xy, d_x2-y2"""
    l,m,n = np.array(d)/np.linalg.norm(d)
    s,p,dd = V
    r3 = np.sqrt(3.)
    zz = (n*n-(l*l+m*m)/2)**2*s + 3*n*n*(l*l+m*m)*p + 0.75*(l*l+m*m)**2*dd
    xyxy = 3*l*l*m*m*s + (l*l+m*m-4*l*l*m*m)*p + (n*n+l*l*m*m)*dd
    xyx2 = 1.5*l*m*(l*l-m*m)*s + 2*l*m*(m*m-l*l)*p + 0.5*l*m*(l*l-m*m)*dd
    xyz2 = (r3*l*m*(n*n-(l*l+m*m)/2)*s - 2*r3*l*m*n*n*p
            + r3/2*l*m*(1+n*n)*dd)
    x2x2 = (0.75*(l*l-m*m)**2*s + (l*l+m*m-(l*l-m*m)**2)*p
            + (n*n+(l*l-m*m)**2/4)*dd)
    x2z2 = (r3/2*(l*l-m*m)*(n*n-(l*l+m*m)/2)*s + r3*n*n*(m*m-l*l)*p
            + r3/4*(1+n*n)*(l*l-m*m)*dd)
    return np.array([[zz,xyz2,x2z2],[xyz2,xyxy,xyx2],[x2z2,xyx2,x2x2]])


def test_slater_koster_matches_the_table():
    """The two-center matrix, built by rotating the d orbitals into the
    frame of the bond, equals the written Slater-Koster table for random
    bonds and integrals"""
    rng = np.random.default_rng(17)
    for i in range(10):
        d,V = rng.normal(size=3),rng.normal(size=3)
        assert np.allclose(threebandtmdc._slater_koster(d,V),_sk_table(d,V),
                atol=1e-12)


def test_slater_koster_bilayer():
    """With the Slater-Koster coupling the bilayer keeps the C3 symmetry
    of the monolayer (equal energies at k and at k rotated by 120
    degrees), the 2H bilayer keeps inversion (doubly degenerate bands
    with spin-orbit coupling, no second harmonic), and unlike the d_z2
    coupling it changes the energies at K"""
    V = (-0.3,0.1,0.02)
    kw = dict(interlayer="slater_koster")
    h = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],V,
            soc=False,fermi=None,**kw)
    A,B = _reciprocal(h)
    c,s = np.cos(2*np.pi/3),np.sin(2*np.pi/3)
    R = np.array([[c,-s],[s,c]])
    for kr in np.random.default_rng(19).random((5,2)):
        kr2 = A@(R@(kr@B))/(2.*np.pi)
        assert np.allclose(np.linalg.eigvalsh(_hk(h,kr)),
                np.linalg.eigvalsh(_hk(h,kr2)),atol=1e-10)
    h0 = specialhamiltonian.three_band_tmdc("MoS2",soc=False,fermi=None)
    e0 = np.linalg.eigvalsh(_hk(h0,[2./3.,2./3.]))
    eK = np.linalg.eigvalsh(_hk(h,[2./3.,2./3.]))
    assert np.max(np.abs(eK-np.sort(np.concatenate([e0,e0]))))>1e-3
    hs = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],V,**kw)
    for kr in np.random.default_rng(21).random((4,2)):
        es = np.linalg.eigvalsh(_hk(hs,kr))
        assert np.allclose(es[0::2],es[1::2],atol=1e-10)
    (w,chi) = hs.get_shg_conductivity(energies=[0.5],nk=18,delta=0.05,
            components="yyy")
    assert np.allclose(chi,0.,atol=1e-10)
    with pytest.raises(ValueError,match="slater_koster"):
        specialhamiltonian.three_band_tmdc_multilayer(["MoS2"]*2,0.1,
                interlayer="sk")
    with pytest.raises(ValueError,match="triple"):
        specialhamiltonian.three_band_tmdc_multilayer(["MoS2"]*2,0.1,**kw)
