"""Invariants of the general n-th order nonlinear conductivity, the
velocity-gauge density-matrix recursion of
src/pyqula/conductivitytk/nonlinearkernel.py.

Every test pairs two independent paths or checks an exact symmetry: the
first order against the Kubo-Greenwood formula, the second harmonic
against the numba kernel of shg.py, the one-band case against the band
derivatives of nonlineardrude.py, isolated molecules against a length-gauge
recursion at any order and frequency tuple, the third-order static limit
against the finite-field polarization, the rotation of a rank-4
tensor, the D6h relation of graphene's third harmonic, the permutation
symmetry of the fields, a stack of decoupled layers in 3D, and exact spin
degeneracy.
"""
import itertools
import math
import warnings

import numpy as np
import pytest

from pyqula import geometry
from pyqula import gpu
from pyqula import multihopping
from pyqula import sculpt
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import shg
from pyqula.conductivitytk import nonlinearkernel as nlk
from pyqula.conductivitytk import nonlineardrude


def _gapped_honeycomb(m=0.5,has_spin=False):
    """Honeycomb lattice with a staggered onsite mass, gap 2m"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_sublattice_imbalance(m)
    return h


def _haldane(t2=0.2,m=0.3):
    """Haldane-gapped honeycomb lattice: no time reversal, so the Hall
    components are finite"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.add_sublattice_imbalance(m)
    return h


def _triangular_three_sublattice():
    """Triangular lattice with three inequivalent sublattices, no inversion
    and no degeneracy anywhere (the fixture of test_shg.py)"""
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


def _low_symmetry():
    """The three-sublattice triangular lattice with the hoppings along a1
    made stronger: no C3, so that sigma_abc(w1,w2) is not symmetric in its
    last two labels alone (with C3v the only invariant rank-3 tensor in the
    plane is fully symmetric)"""
    h = _triangular_three_sublattice()
    hd = {R:_dense(m) for (R,m) in h.get_hopping_dict().items()}
    for R in [(1,0,0),(-1,0,0)]: hd[R] = 1.4*hd[R]
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _bilayer(m1=0.5,m2=0.8,V=0.1):
    """Two different gapped honeycomb layers at z = +-1 with a potential
    step between them (the fixture of test_shg.py with inversion broken in
    plane and out of plane)"""
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


def _dense(m):
    return np.array(m.todense()) if hasattr(m,"todense") else np.array(m)


def _one_band_square():
    """One-band square lattice whose band has no inversion center:
    eps = -2 cos kx - 2 cos ky - 0.6 sin 2kx - 0.4 sin(kx + 2ky), from two
    imaginary hoppings on longer bonds. A single complex first-neighbor
    hopping would only shift k, and the odd derivatives would then sum to
    nearly zero on the mesh"""
    h = geometry.square_lattice().get_hamiltonian(has_spin=False)
    hd = {R:_dense(m) for (R,m) in h.get_hopping_dict().items()}
    for (R,t) in [((2,0,0),0.3j),((1,2,0),0.2j)]:
        hd[R] = np.array([[t]])
        hd[tuple(-np.array(R))] = np.array([[np.conjugate(t)]])
    h.set_multihopping(multihopping.MultiHopping(hd))
    h.shift_fermi(0.7)
    return h


def _molecule_crystal():
    """A 1D crystal of isolated three-site molecules (no intercell
    hopping), the fixture of test_shg.py's static limit"""
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
    return h,H0,g.x,L


def _molecule_crystal_2d():
    """A 2D crystal of isolated four-site molecules with no symmetry, far
    apart (no intercell hopping), insulating at zero chemical potential
    (levels -1.84, -0.24, 0.35, 1.73)"""
    L = 7.0
    g = geometry.square_lattice()
    g.a1 = np.array([L,0.,0.]) ; g.a2 = np.array([0.,L,0.])
    g.x = np.array([0.0,1.0,1.9,0.3]) ; g.y = np.array([0.,0.2,1.1,1.4])
    g.z = np.zeros(4)
    g.xyz2r()
    g.has_sublattice = False
    g.update_reciprocal() ; g.get_fractional()
    H0 = np.array([[-1.2,-1.0,0.,-0.4],[-1.0,0.3,-0.7,0.],
                   [0.,-0.7,1.1,-0.5],[-0.4,0.,-0.5,-0.2]])
    h = g.get_hamiltonian(has_spin=False,tij=lambda r1,r2: 0.)
    h.intra = H0.astype(np.complex128)
    return h,H0,[g.x,g.y],L**2


def _stack(h2,c=2.):
    """3D crystal of decoupled copies of the 2D h2 stacked along z with
    spacing c"""
    h = h2.copy()
    h.geometry.a3 = np.array([0.,0.,c])
    h.geometry.dimensionality = 3
    hd = {(R[0],R[1],0):_dense(m) for (R,m) in h2.get_hopping_dict().items()}
    h.dimensionality = 3
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _close(a,b,rtol):
    """a and b agree to rtol of the largest entry of b, which must not be
    a vacuous zero"""
    a,b = np.array(a),np.array(b)
    scale = np.max(np.abs(b))
    assert scale>1e-4
    return np.max(np.abs(a-b))<rtol*scale


@pytest.mark.parametrize("model",[_gapped_honeycomb,_haldane])
def test_first_order_matches_kubo(model):
    """At first order the recursion is the velocity-gauge Kubo formula with
    the diamagnetic term, Passos et al. Eq. (sigmaA), which equals the
    Kubo-Greenwood sum over band pairs of kubo.py at the same complex
    frequency up to the aliasing of the mesh sum rule. The difference
    falls exponentially with nk: on the gapped honeycomb lattice it is
    8e-7 of the signal at nk=24 and 6e-13 at nk=48. The Haldane model
    makes the Hall component xy finite"""
    h = model()
    es = [0.3,1.2,2.5]
    comps = ["xx","xy","yx","yy"]
    (w,s) = nlk.nonlinear_conductivity(h,order=1,energies=es,nk=48,T=0.1,
            delta=0.1,components=comps)
    (w,sk) = kubo.optical_conductivity(h,energies=es,nk=48,T=0.1,delta=0.1)
    ref = np.array([sk[:,"xy".index(c[0]),"xy".index(c[1])] for c in comps])
    assert _close([s[c] for c in comps],ref,1e-10)


def test_second_harmonic_is_shg():
    """At second order with equal frequencies the recursion is term by term
    the numba kernel of shg.py, on the same mesh, for every in-plane
    component and for the out-of-plane zxx of a bilayer without
    inversion"""
    h = _bilayer()
    comps = [x+y+z for x in "xy" for y in "xy" for z in "xy"] + ["zxx"]
    es = [0.4,1.1,2.3]
    (w,s) = nlk.nonlinear_conductivity(h,order=2,energies=es,nk=16,
            delta=0.1,components=comps)
    (w,ss) = shg.shg_conductivity(h,energies=es,nk=16,delta=0.1,
            components=comps)
    assert np.min(np.abs(ss["zxx"]))>1e-4 # not a vacuous zero
    assert _close([s[c] for c in comps],[ss[c] for c in comps],1e-12)


@pytest.mark.parametrize("order",[2,3,4])
def test_one_band_tadpole_identity(order):
    """For one band every commutator vanishes, so only the p = n term
    survives and sigma = (i^n q^(n+1)/n!) (1/V) sum_k f d^(n+1)eps/
    prod_j(w_j + i delta), with the band derivatives of
    nonlineardrude._Channel, an independent construction. This pins the
    1/p! factors, the power of the charge and the vertices to any order"""
    h = _one_band_square()
    comps = {2:["xxx","yxy","xyy"],3:["xxxx","xxyy","yxyx"],
             4:["xxxxx","xyyxy","yxxyy"]}[order]
    es = np.array([0.4,1.3])
    delta,T,nk = 0.1,0.2,24
    (w,s) = nlk.nonlinear_conductivity(h,order=order,energies=es,nk=nk,
            T=T,delta=delta,components=comps)
    ks = kubo._kmesh(h,nk)
    ch = nonlineardrude._Channel(h.get_multicell(),ks)
    f = kubo._fermi(ch.es,T)
    q = nlk._charge
    for c in comps:
        der = ch.derivative([kubo._index(x) for x in c])
        ref = (1j**order*q**(order+1)/math.factorial(order)
               *np.sum(f*der)/len(ks)/(es+1j*delta)**order)
        assert _close(s[c],ref,1e-12)


def test_third_order_static_limit_matches_finite_field():
    """A crystal of isolated molecules has the static third-order
    polarization of one molecule, (1/6) d3P/dE3 with P = q sum_occ <x>/L
    from the exact eigenstates of H0 - q E x. At w = 0, P = sigma E^3/
    (-i Omega) with Omega = 3 i delta, so chi3 = sigma/(3 delta) up to
    O(delta^2), and the 1/delta^3 prefactor amplifies the rounding as
    1/delta^4 (measured: 3.8e-4 at delta=1e-2, 6e-5 at 4e-3, and 3.3e-4
    again at 1e-3, where rounding dominates). The delta^2 term is removed
    by combining delta = 5e-3 and 1e-2, which fixes the absolute prefactor
    and the sign of the third order"""
    h,H0,x,L = _molecule_crystal()
    q = nlk._charge
    def P(E):
        es,vs = np.linalg.eigh(H0 - q*E*np.diag(x))
        occ = vs[:,es<0.]
        return q*np.sum(np.abs(occ)**2*x[:,None])/L
    dE = 1e-3
    ref = (P(2*dE) - 2*P(dE) + 2*P(-dE) - P(-2*dE))/(2*dE**3)/6.
    assert abs(ref)>1e-2 # not a vacuous zero
    def chi3(delta):
        (w,s) = nlk.nonlinear_conductivity(h,order=3,energies=[0.],nk=4,
                delta=delta,T=0.,components="xxxx")
        return s[0]/(3.*delta)
    c1,c2 = chi3(5e-3),chi3(1e-2)
    assert abs(c1-ref)<5e-4*abs(ref) # the bare value, O(delta^2) off
    extrapolated = (4.*c1 - c2)/3.
    assert abs(extrapolated-ref)<1e-5*abs(ref)


def _length_gauge_molecule(H0,r,area,lab,ws):
    """Conductivity of a crystal of isolated molecules in the length gauge,
    H = H0 - q E.r, an independent recursion with no velocity vertices and
    no 1/(w1...wn) prefactor: rho^{a1..ap}_nm = -q [x^a1,rho^{a2..ap}]_nm/
    (w1 + ... + wp - dE_nm), P_b = q Tr[x^b rho^(n)]/area and J = -i Omega
    P, averaged over the orderings of the pairs (a_j,w_j)"""
    q = nlk._charge
    e,U = np.linalg.eigh(H0)
    X = [U.T@np.diag(c)@U for c in r] # positions in the eigenbasis
    dE = e[:,None] - e[None,:]
    n = len(ws)
    tot = 0.
    for P in itertools.permutations(range(n)):
        a = [lab[1+P[i]] for i in range(n)]
        w = [ws[P[i]] for i in range(n)]
        rho = np.diag((e<0.).astype(np.complex128)) # cold insulator
        for j in range(n-1,-1,-1): # innermost field first
            rho = -q*(X[a[j]]@rho - rho@X[a[j]])/(sum(w[j:]) - dE)
        tot += q*np.trace(X[lab[0]]@rho)
    return -1j*sum(ws)*tot/math.factorial(n)/area


@pytest.mark.parametrize("order,mult,comps",[
        (1,(1,),["xx","xy"]),
        (2,(1,2),["xxy","yxy","yyx","xxx"]),
        (3,(1,-1,0),["xxyy","xyyx","yxxx"]),
        (3,(1,2,-0.5),["xxyy","xyxy","yyyx"]),
        (4,(1,1,-1,0.5),["xxyxy","yxxxy"])])
def test_velocity_gauge_is_length_gauge_for_molecules(order,mult,comps):
    """For isolated molecules the position operator is bounded and the
    Peierls coupling with a diagonal position is an exact gauge
    transformation, so the velocity-gauge recursion, with its vertices up
    to h^(n+1) and its 1/(w1...wn) prefactor, equals the length-gauge
    recursion term by term at any order and any frequency tuple
    (measured: 1e-13 at third order). This is the check of the frequency
    sums of the denominators for unequal frequencies, of the average over
    the orderings, and of the absolute prefactor at every order"""
    h,H0,r,area = _molecule_crystal_2d()
    E,delta = np.array([0.7,1.3]),0.1
    (w,s) = nlk.nonlinear_conductivity(h,order=order,energies=E,
            frequencies=mult,nk=2,T=0.,delta=delta,components=comps)
    for c in comps:
        lab = ["xy".index(x) for x in c]
        ref = [_length_gauge_molecule(H0,r,area,lab,
                [cj*e+1j*delta for cj in mult]) for e in E]
        assert _close(s[c],ref,1e-10)


def test_rotation_covariance_third_order():
    """Rotating the crystal by an angle that is not a symmetry rotates the
    third-order conductivity as a rank-4 tensor, sigma' = R R R R sigma,
    which checks every in-plane component at a generic orientation"""
    comps = ["".join(c) for c in itertools.product("xy",repeat=4)]
    def tensor(h):
        (w,s) = nlk.nonlinear_conductivity(h,order=3,energies=[0.7,1.9],
                nk=18,delta=0.1,components=comps)
        t = np.zeros((2,2,2,2,2),dtype=np.complex128)
        for c in comps: t[(slice(None),)+tuple("xy".index(x) for x in c)] = s[c]
        return t
    h0 = _triangular_three_sublattice()
    theta = 0.37
    h1 = h0.copy()
    h1.geometry = sculpt.rotate(h0.geometry,theta)
    R = np.array([[np.cos(theta),np.sin(theta)], # sculpt.rotate turns the
                  [-np.sin(theta),np.cos(theta)]]) # crystal clockwise
    t0,t1 = tensor(h0),tensor(h1)
    rot = np.einsum("ai,bj,ck,dl,wijkl->wabcd",R,R,R,R,t0)
    assert _close(t1,rot,1e-10)


def test_graphene_thg_d6h():
    """The third harmonic of doped graphene obeys the D6h relations
    sigma_xxxx = sigma_xxyy + sigma_xyxy + sigma_xyyx and sigma_xxxx =
    sigma_yyyy, exact on the C6-symmetric mesh at any resolution, since the
    point group acts on the reduced mesh by integer matrices. The feature
    at 3w = 2 mu is not asserted, the mesh does not resolve it"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.shift_fermi(-0.1) # chemical potential at mu = 0.1
    comps = ["xxxx","xxyy","xyxy","xyyx","yyyy"]
    (w,s) = nlk.nonlinear_conductivity(h,order=3,energies=[0.3,0.9,1.6],
            nk=60,T=0.05,delta=0.1,components=comps)
    assert _close(s["xxyy"]+s["xyxy"]+s["xyyx"],s["xxxx"],1e-8)
    assert _close(s["yyyy"],s["xxxx"],1e-8)


def test_permutation_symmetry():
    """The conductivity is symmetric under the exchange of the pairs
    (label, frequency): sigma_abc(w,2w) = sigma_acb(2w,w), and at third
    order with a static field, (1,1,0) and (1,-1,0), the component with the
    static label moved first equals the one with the multipliers permuted.
    The exchange of labels alone is not a symmetry, so the test would fail
    without the average over the orderings of the fields"""
    h = _low_symmetry()
    kw = dict(energies=[0.6,1.5],nk=12,delta=0.1)
    (w,s12) = nlk.nonlinear_conductivity(h,order=2,frequencies=(1,2),
            components=["yxy","yyx"],**kw)
    (w,s21) = nlk.nonlinear_conductivity(h,order=2,frequencies=(2,1),
            components=["yxy","yyx"],**kw)
    assert not _close(s12["yxy"],s12["yyx"],1e-2) # labels alone: no symmetry
    assert _close(s12["yxy"],s21["yyx"],1e-12)
    assert _close(s12["yyx"],s21["yxy"],1e-12)
    for (c,cp) in [((1,1,0),(0,1,1)),((1,-1,0),(0,1,-1))]:
        (w,s) = nlk.nonlinear_conductivity(h,order=3,frequencies=c,
                components=["xxyy","xyyx"],**kw)
        (w,sp) = nlk.nonlinear_conductivity(h,order=3,frequencies=cp,
                components="xyxy",**kw)
        assert not _close(s["xxyy"],s["xyyx"],1e-3) # static label matters
        assert _close(sp,s["xxyy"],1e-12)


def test_complex_frequencies_accepted():
    """The internal entry point takes the complex frequencies as they are,
    including Im w < 0 (half of the circle of the Laurent extraction), and
    at w_j = c_j E + i delta it is the public call"""
    h = _triangular_three_sublattice()
    om = np.array([[0.5-0.05j,0.7+0.1j],[0.03j,-0.2-0.04j]])
    s = nlk.nonlinear_response(h,2,om,["xyy","xxx"],nk=8,T=0.1)
    assert s.shape==(2,2) and np.all(np.isfinite(s))
    E,delta = np.array([0.4,1.1]),0.1
    om = E[:,None]*np.array([1.,-1.])[None,:] + 1j*delta
    s = nlk.nonlinear_response(h,2,om,["xyy"],nk=8,T=0.1)
    (w,sp) = nlk.nonlinear_conductivity(h,order=2,energies=E,
            frequencies=(1,-1),nk=8,T=0.1,delta=delta,components="xyy")
    assert _close(s[:,0],sp,1e-14)


def test_3d_decoupled_layers():
    """A 3D stack of decoupled gapped honeycomb layers with spacing c has
    the conductivity of one layer divided by c, at first and second order:
    the kz sum of a kz-independent integrand is the 2D mesh sum and the
    cell volume is the area times c"""
    c = 2.
    h2 = _gapped_honeycomb()
    h3 = _stack(h2,c=c)
    for (order,comps) in [(1,["xx","yy"]),(2,["xxx","xyy","yxy"])]:
        kw = dict(order=order,energies=[0.5,1.4],delta=0.1,components=comps)
        (w,s2) = nlk.nonlinear_conductivity(h2,nk=12,**kw)
        (w,s3) = nlk.nonlinear_conductivity(h3,nk=[12,12,3],**kw)
        assert _close([s3[x] for x in comps],[s2[x]/c for x in comps],1e-10)


@pytest.mark.parametrize("order",[2,3])
def test_degenerate_bands_are_safe(order):
    """With exact spin degeneracy the eigenbasis inside every degenerate
    pair is arbitrary, and the result must still be twice the spinless
    one, since the recursion never divides by a band energy difference.
    The doped lattice makes the Fermi-surface terms finite"""
    comps = {2:["xxx","xyy"],3:["xxxx","xxyy"]}[order]
    kw = dict(order=order,energies=[0.5,1.4],nk=12,delta=0.1,T=0.1,
            components=comps)
    hs = _gapped_honeycomb(has_spin=False) ; hs.shift_fermi(-0.8)
    hf = _gapped_honeycomb(has_spin=True) ; hf.shift_fermi(-0.8)
    (w,s1) = nlk.nonlinear_conductivity(hs,**kw)
    (w,s2) = nlk.nonlinear_conductivity(hf,**kw)
    assert _close([s2[c] for c in comps],[2.*s1[c] for c in comps],1e-12)


def test_k_batches_do_not_change_the_result(monkeypatch):
    """The k-sum is linear, so splitting the mesh into many batches under a
    small memory budget gives the same result as one batch"""
    h = _triangular_three_sublattice()
    kw = dict(order=3,energies=[0.6],frequencies=(1,-1,1),nk=10,delta=0.1,
            components=["xxyy","yxxx"])
    (w,s1) = nlk.nonlinear_conductivity(h,**kw)
    monkeypatch.setattr(nlk,"_batch_bytes",3e4) # a few k-points per batch
    assert nlk._batch_size(3,30,10)<20
    (w,s2) = nlk.nonlinear_conductivity(h,**kw)
    assert _close([s2[c] for c in s2],[s1[c] for c in s1],1e-13)


def test_susceptibility_reduces_to_shg():
    """At second order with equal frequencies chi = sigma/(-i Omega
    epsilon_0) with Omega = 2(w + i delta) is shg.to_susceptibility"""
    h = _gapped_honeycomb()
    E,delta = np.array([0.,0.8,1.7]),0.1
    (w,s) = nlk.nonlinear_conductivity(h,order=2,energies=E,nk=8,
            delta=delta,components="xxx")
    om = E[:,None]*np.ones(2)[None,:] + 1j*delta
    chi = nlk.to_susceptibility(h,om,s,length_unit=1.45,energy_unit=2.33)
    (w,ref) = shg.to_susceptibility(h,E,s,delta,1.45,2.33)
    assert np.allclose(chi,ref,rtol=1e-14,atol=0.)
    (w,chi2) = nlk.nonlinear_susceptibility(h,order=2,energies=E,nk=8,
            delta=delta,components="xxx",length_unit=1.45,energy_unit=2.33)
    assert np.allclose(chi2,ref,rtol=1e-14,atol=0.)


def test_options():
    """Every requirement of the routine is checked with a message naming
    it: the number of labels of a component, the number of multipliers,
    a positive broadening, no Nambu, a periodic system, a known precision,
    and the warning for single precision below the broadening"""
    h = _gapped_honeycomb()
    kw = dict(energies=[1.],nk=4)
    with pytest.raises(ValueError,match="4 Cartesian labels"):
        nlk.nonlinear_conductivity(h,order=3,components="xyy",**kw)
    with pytest.raises(ValueError,match="multipliers"):
        nlk.nonlinear_conductivity(h,order=3,frequencies=(1,1),**kw)
    with pytest.raises(ValueError,match="delta"):
        nlk.nonlinear_conductivity(h,order=2,delta=0.,**kw)
    with pytest.raises(ValueError,match="positive integer"):
        nlk.nonlinear_conductivity(h,order=0,**kw)
    with pytest.raises(ValueError,match=r"\(nw,2\)"):
        nlk.nonlinear_response(h,2,[[0.1j]],["xxx"],nk=4)
    with pytest.raises(ValueError,match="single.*double"):
        nlk.nonlinear_conductivity(h,order=2,components="xxx",prec="half",
                **kw)
    with pytest.raises(NotImplementedError,match="single"):
        nlk.nonlinear_conductivity(h,order=2,components="xxx",
                prec="single",**kw)
    hn = _gapped_honeycomb()
    hn.setup_nambu_spinor()
    with pytest.raises(NotImplementedError,match="Nambu"):
        nlk.nonlinear_conductivity(hn,order=2,components="xxx",**kw)
    h0 = geometry.chain().get_hamiltonian(has_spin=False).set_finite_system(n=4)
    assert h0.dimensionality==0
    with pytest.raises(NotImplementedError,match="dimensionality"):
        nlk.nonlinear_conductivity(h0,order=2,components="xxx",**kw)
    try: # single precision on the device, below the broadening
        with warnings.catch_warnings(): # a machine with no GPU warns here
            warnings.simplefilter("ignore")
            gpu.set_gpu(True)
        with pytest.warns(UserWarning,match="single"):
            (w,s) = nlk.nonlinear_conductivity(h,order=2,energies=[0.05],
                    nk=4,delta=0.1,components="xxx",prec="single")
        assert np.all(np.isfinite(s))
    finally: gpu.set_gpu(False)
