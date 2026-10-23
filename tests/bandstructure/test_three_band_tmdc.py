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
