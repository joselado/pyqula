"""Invariants of the second-order photocurrents and of the sum and
difference frequency conductivity (src/pyqula/conductivitytk/photocurrent.py).

The totals come from the velocity-gauge recursion of nonlinearkernel.py and
are checked against the independent length gauge of shg.sigma2_length at
(w,-w) and (w,2w): to rounding on time-reversal symmetric insulators, and
once the mesh has converged on a Haldane-gapped lattice, where the 1/delta
pole of the rectified current amplifies the aliasing. The injection rate is
checked against the diagonal of the length gauge, the shift current for the
absence of that pole, and the symmetry content against time reversal,
inversion, the D3h pattern, a rotation of the crystal, exact spin
degeneracy and a supercell, whose folded bands are degenerate on lines of
the zone. As an absolute reference outside the frequency-domain formulas,
the rectified current and its injection part of a doped 1D chain at T > 0
are compared with a real-time propagation of the density matrix.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import multihopping
from pyqula import sculpt
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import shg
from pyqula.conductivitytk import photocurrent


def _gapped_honeycomb(m=0.5):
    """Spinless honeycomb lattice with a staggered onsite mass, x along the
    armchair direction: D3h and time reversal"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(m)
    return h


def _triangular_three_sublattice():
    """Triangular lattice with three inequivalent sublattices (onsite -3,
    0.5 and 3): three nondegenerate bands with no inversion"""
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


def _haldane(m=0.3,t2=0.2,has_spin=False):
    """Honeycomb lattice with a staggered mass and a Haldane term: no
    inversion and no time reversal"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_haldane(t2)
    h.add_sublattice_imbalance(m)
    return h


def _strained_gapped_honeycomb(m=0.5,eps=0.3):
    """The gapped honeycomb lattice with the bond to one neighbouring cell
    stretched by 1 + eps: time reversal is kept and the C3 rotation is
    gone, so an in-plane current under circular light is allowed"""
    h = _gapped_honeycomb(m)
    hd = {R:np.array(t.todense()) if hasattr(t,"todense") else np.array(t)
          for R,t in h.get_hopping_dict().items()}
    R = (1,0,0) if (1,0,0) in hd else [R for R in hd if R!=(0,0,0)][0]
    Rm = tuple(-x for x in R)
    hd[R] = hd[R]*(1. + eps) ; hd[Rm] = hd[Rm]*(1. + eps)
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _chain():
    """Two-site chain with a staggered onsite and complex hoppings to the
    next cell: no inversion and no time reversal in 1D"""
    g = geometry.chain().get_supercell(2)
    g.x = np.array([0.,0.37])
    g.y = np.zeros(2)
    g.z = np.zeros(2)
    g.xyz2r()
    g.get_fractional()
    h = g.get_hamiltonian(has_spin=False)
    t = np.array([[0.2*np.exp(0.7j),0.],[0.6,-0.15*np.exp(-0.3j)]])
    hd = {(0,0,0):np.array([[0.4,1.],[1.,-0.4]],dtype=np.complex128),
          (1,0,0):t,(-1,0,0):t.conj().T}
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _real_time_rectified(h,w,delta,nk,T,amp=1e-3,dt=0.05,tmul=15.):
    """dc current at t = 0 of a 1D crystal under the field E(t) = Re E0
    exp(-i(w + i delta)t) along x, switched on adiabatically from t =
    -tmul/delta, from the propagation of the density matrix rho_k(t) under
    the Peierls Hamiltonian H(k - qA(t)), E = -dA/dt, with a fourth-order
    Magnus step. Four phases of E0 cancel the odd orders and the second
    harmonic. Returns, per unit |E0|^2, the total current (1/2) sigma_xxx
    and the part carried by the change of the band populations,
    eta_xxx/(2 delta)"""
    q = -1. # electrons
    g = h.geometry
    ks = np.array(kubo._kmesh(h,nk))[:,0]
    Rs,Ts,Ds = [],[],[]
    for (R,t) in h.get_hopping_dict().items():
        Ts.append(np.array(t.todense() if hasattr(t,"todense") else t,
                dtype=np.complex128))
        Rs.append(R[0])
        Ds.append(R[0]*g.a1[0] + g.x[None,:] - g.x[:,None]) # R + x_j - x_i
    Rs,Ts,Ds = np.array(Rs),np.array(Ts),np.array(Ds)
    Tk = np.exp(2j*np.pi*ks[:,None]*Rs[None,:])[:,:,None,None]*Ts[None]
    def H(A,velocity=False): # A[s] -> H[s,k], or dH/dk_x
        ph = np.exp(-1j*q*A[:,None,None,None]*Ds[None])
        if velocity: ph = ph*1j*Ds[None]
        return np.einsum("kRij,sRij->skij",Tk,ph)
    def dagger(m): return np.conj(np.swapaxes(m,-1,-2))
    def step(m): # exp(-i m dt)
        e,U = np.linalg.eigh(m)
        return (U*np.exp(-1j*dt*e)[...,None,:])@dagger(U)
    Om = w + 1j*delta
    E0 = amp*np.exp(0.5j*np.pi*np.arange(4))
    A = lambda t: np.real(E0*np.exp(-1j*Om*t)/(1j*Om))
    nt = int(round(tmul/(delta*dt)))
    e0,U0 = np.linalg.eigh(H(np.zeros(1))[0])
    f0 = kubo._fermi(e0,T)
    rho0 = (U0*f0[:,None,:])@dagger(U0)
    rho = np.array([rho0]*len(E0))
    c1,c2 = 0.5 - np.sqrt(3)/6,0.5 + np.sqrt(3)/6 # Gauss points
    a1,a2 = (3 - 2*np.sqrt(3))/12,(3 + 2*np.sqrt(3))/12
    for i in range(nt):
        t = (i - nt)*dt
        H1,H2 = H(A(t + c1*dt)),H(A(t + c2*dt))
        U = step(a1*H1 + a2*H2)@step(a2*H1 + a1*H2)
        rho = U@rho@dagger(U)
    A0 = A(0.)
    norm = q/(len(ks)*abs(g.a1[0]))
    V = H(A0,velocity=True)
    J0 = norm*np.real(np.einsum("kij,kji->",rho0,
            H(np.zeros(1),velocity=True)[0])) # current of the mesh at E=0
    Jt = norm*np.real(np.einsum("skij,skji->s",rho,V)) - J0
    e,U = np.linalg.eigh(H(A0)) # the bands at t = 0, nondegenerate here
    pop = np.real(np.diagonal(dagger(U)@rho@U,axis1=2,axis2=3)) - f0[None]
    vnn = np.real(np.diagonal(dagger(U)@V@U,axis1=2,axis2=3))
    Jp = norm*np.sum(vnn*pop,axis=(1,2))
    return np.mean(Jt)/amp**2,np.mean(Jp)/amp**2


def _bilayer(m1=0.5,m2=0.5):
    """Two gapped honeycomb layers at z = +-1, the second one the
    inversion image of the first, coupled by a vertical hopping 0.3:
    centrosymmetric for m1 = m2"""
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
    h.add_onsite([m1,-m1,m2,-m2])
    return h


_comps8 = [a+b+c for a in "xy" for b in "xy" for c in "xy"]


@pytest.mark.parametrize("model",[_gapped_honeycomb,
        _triangular_three_sublattice])
def test_velocity_and_length_gauges_agree_at_two_frequencies(model):
    """On time-reversal symmetric insulators at T=0 the velocity-gauge
    totals at (w,-w) and (w,2w) equal the length gauge of shg.sigma2_length
    to rounding, for six components at three frequencies"""
    h = model()
    comps = ["xxx","xyy","yxy","yyx","yxx","yyy"]
    ws = np.array([0.7,1.5,2.4])
    (w,r) = photocurrent.rectified_conductivity(h,energies=ws,nk=24,T=0.,
            delta=0.1,components=comps)
    (w,s2) = photocurrent.sum_frequency_conductivity(h,(1,2),energies=ws,
            nk=24,T=0.,delta=0.1,components=comps)
    L1 = shg.sigma2_length(h,ws,-ws,nk=24,delta=0.1,components=comps)
    L2 = shg.sigma2_length(h,ws,2*ws,nk=24,delta=0.1,components=comps)
    for (s,L) in [(r["total"],L1),(s2,L2)]:
        ref = max(np.max(np.abs(L[c])) for c in comps)
        assert ref>1e-4
        for c in comps: assert np.allclose(s[c],L[c],atol=1e-10*ref)


def test_gauges_agree_with_broken_time_reversal_when_converged():
    """With a Haldane term the rectified totals of the two gauges differ
    on a coarse mesh, the 1/delta pole amplifying the aliasing, and agree
    to 1 % at nk=96, the difference shrinking from nk=48"""
    h = _haldane()
    diff = []
    for nk in [48,96]:
        (w,r) = photocurrent.rectified_conductivity(h,energies=[2.],nk=nk,
                T=0.,delta=0.1,components="yxx")
        L = shg.sigma2_length(h,[2.],[-2.],nk=nk,delta=0.1,components="yxx")
        assert abs(L[0])>1e-2
        diff.append(abs(r["total"][0] - L[0])/abs(L[0]))
    assert diff[1]<1e-2
    assert diff[1]<diff[0]/4.


def test_injection_is_the_length_gauge_diagonal():
    """eta/delta equals the n = m part of the length gauge at (w,-w) at
    two deltas, for the linear (yxx, yyy) and the mixed (xxy, xyx)
    components of the Haldane-gapped lattice. And the shift current carries
    no pole: on a converged mesh it moves far less between delta = 0.1 and
    0.05 than eta/delta does (it would move by half of that if the pole
    were 2 delta sigma)"""
    h = _haldane()
    comps = ["yxx","xxy","xyx","yyy"]
    for delta in [0.1,0.05]:
        (w,eta) = photocurrent.injection_current(h,energies=[2.],nk=48,T=0.,
                delta=delta,components=comps)
        D = shg.sigma2_length_diagonal(h,[2.],[-2.],nk=48,delta=delta,
                components=comps)
        for c in comps:
            assert abs(eta[c][0])>1e-3
            assert abs(eta[c][0]/delta - D[c][0])<1e-10*abs(D[c][0])
    hs = _strained_gapped_honeycomb() # complex eta, the circular part
    (w,eta) = photocurrent.injection_current(hs,energies=[1.3,2.],nk=24,
            T=0.,delta=0.1,components=["yxy","yyx","xxy"])
    D = shg.sigma2_length_diagonal(hs,[1.3,2.],[-1.3,-2.],nk=24,delta=0.1,
            components=["yxy","yyx","xxy"])
    for c in eta:
        assert np.min(np.abs(eta[c].imag))>1e-3
        assert np.allclose(eta[c]/0.1,D[c],atol=1e-10*np.max(np.abs(D[c])))
    r = [photocurrent.rectified_conductivity(h,energies=[2.],nk=192,T=0.,
            delta=delta,components="yxx")[1] for delta in [0.1,0.05]]
    dshift = abs(r[0]["shift"][0] - r[1]["shift"][0])
    dpole = abs(r[0]["injection"][0]/0.1 - r[1]["injection"][0]/0.05)
    assert dpole>1e-2
    assert dshift<0.05*dpole


def test_injection_with_more_than_two_coupled_bands():
    """eta/delta equals the n = m part of the length gauge on the
    noncentrosymmetric bilayer with a Haldane term, four bands coupled at
    the same k. With two coupled bands [r^c, P h^a P]_nm and [r^c, h^a]_nm
    are the same, since r^c_nn = 0, so only a third band coupled to both
    sees that injection_rate restricts the velocity to the multiplets: the
    full velocity moved yxx by 60 to 390 % here, changed its sign at
    w = 1.2 and 2.8, and gave a finite xxx, xyy and yxy where the length
    gauge is zero"""
    hb = _bilayer(m2=0.8)
    hb.add_haldane(0.1)
    ws = np.array([1.2,2.1,2.8])
    finite = ["yxx","yyy","xxy"]
    zero = ["xxx","xyy","yxy"]
    (w,eta) = photocurrent.injection_current(hb,energies=ws,nk=24,T=0.,
            delta=0.1,components=finite+zero)
    D = shg.sigma2_length_diagonal(hb,ws,-ws,nk=24,delta=0.1,
            components=finite+zero)
    ref = max(np.max(np.abs(D[c])) for c in finite)
    for c in finite:
        assert np.min(np.abs(eta[c]/0.1))>1e-3
        assert np.allclose(eta[c]/0.1,D[c],rtol=0.,atol=1e-10*ref)
    for c in zero:
        assert np.max(np.abs(D[c]))<1e-12*ref
        assert np.max(np.abs(eta[c]/0.1))<1e-10*ref


def test_supercell_gives_the_same_photocurrents():
    """The 2x2 supercell of the Haldane-gapped lattice at nk=24 is the same
    set of k-points as the primitive cell at nk=48, so the total, the
    injection and the shift current agree to rounding. The folded bands of
    the supercell are degenerate along whole lines of the zone, where the
    velocity inside a multiplet is not a multiple of the identity: the
    injection must not depend on the basis the diagonalization picks
    there (the diagonal of the velocity alone gave 4 % on eta and 24 % on
    the shift current, and a finite xxx forbidden by the mirror)"""
    comps = ["yxx","xxx","xxy","yyy","yxy"]
    h = _haldane()
    out = [photocurrent.rectified_conductivity(hh,energies=[1.3,2.],nk=nk,
            T=0.,delta=0.1,components=comps)[1]
           for (hh,nk) in [(h,48),(h.get_supercell(2),24)]]
    for key in ["total","injection","shift"]:
        ref = max(np.max(np.abs(out[0][key][c])) for c in comps)
        assert ref>1e-3
        for c in comps:
            assert np.allclose(out[1][key][c],out[0][key][c],atol=1e-10*ref)


def test_real_time_propagation_of_a_doped_chain():
    """The rectified current and the population (injection) current of a
    doped chain with no inversion and no time reversal, at the default
    temperature T = delta, equal a real-time propagation of the density
    matrix under the Peierls Hamiltonian, which uses none of the
    eigenbasis derivatives, closed forms or sum rules of the
    frequency-domain code. The occupations are Pauli-blocked near the
    Fermi level, so eta at T = 0 differs and the check sees T"""
    h = _chain()
    h.shift_fermi(0.8)
    w,delta = 1.5,0.1
    (Jt,Jp) = _real_time_rectified(h,w,delta,nk=48,T=delta)
    assert abs(Jp)>1e-2
    (e,r) = photocurrent.rectified_conductivity(h,energies=[w],nk=48,
            delta=delta,components="xxx")
    (e,eta) = photocurrent.injection_current(h,energies=[w],nk=48,
            delta=delta,components="xxx")
    (e,eta0) = photocurrent.injection_current(h,energies=[w],nk=48,T=0.,
            delta=delta,components="xxx")
    assert abs(r["total"][0]/2. - Jt)<1e-4*abs(Jt) # J = sigma |E0|^2/2
    assert abs(r["injection"][0]/(2.*delta) - Jp)<1e-4*abs(Jp)
    assert abs(eta[0]/(2.*delta) - Jp)<1e-4*abs(Jp)
    assert abs(eta0[0] - eta[0])>0.1*abs(eta[0])


def test_circular_part_carries_its_pole_in_eta():
    """With time reversal and no C3 (the strained lattice) the
    bc-antisymmetric part of the rectified current, the response to
    circular light, is dominated by the injection pole Im eta/delta. The
    shift current keeps no pole there either: between delta = 0.1 and
    0.05 its imaginary part moves far less than Im eta/delta, which
    doubles (taking only Re eta out of the total, or the conjugate total,
    would leave the pole in it)"""
    hs = _strained_gapped_honeycomb()
    comps = ["xxy","yxy"]
    r = [photocurrent.rectified_conductivity(hs,energies=[1.3,2.],nk=192,
            T=0.,delta=delta,components=comps)[1] for delta in [0.1,0.05]]
    for c in comps:
        pole = [r[0]["injection"][c].imag/0.1,r[1]["injection"][c].imag/0.05]
        dpole = np.abs(pole[1] - pole[0])
        dshift = np.abs(r[1]["shift"][c].imag - r[0]["shift"][c].imag)
        assert np.min(dpole)>1e-2
        assert np.all(dshift<0.05*dpole)
        assert np.all(np.abs(r[0]["shift"][c].imag)<0.3*np.abs(pole[0]))


def test_sum_frequency_assigns_the_fields_to_their_frequencies():
    """On the strained lattice sigma_abc(w,2w) differs from sigma_acb(w,2w),
    so which field sits at which frequency is visible, and the velocity
    gauge at multipliers (1,2) equals the length gauge of
    shg.sigma2_length(w,2w), with the field along b at w and the one
    along c at 2w. With a larger mass the two fields lie below the gap,
    where the length gauge converges at nk=48"""
    hs = _strained_gapped_honeycomb(m=1.0)
    ws = np.array([0.5,0.7])
    comps = ["xxy","xyx","yxy","yyx"]
    (w,s) = photocurrent.sum_frequency_conductivity(hs,(1,2),energies=ws,
            nk=48,T=0.,delta=0.1,components=comps)
    L = shg.sigma2_length(hs,ws,2*ws,nk=48,delta=0.1,components=comps)
    for c in comps:
        asym = np.abs(L[c] - L[c[0]+c[2]+c[1]])
        assert np.min(asym)>5e-4
        assert np.all(np.abs(s[c] - L[c])<1e-8*asym)


def test_injection_vanishes_with_time_reversal():
    """With time reversal the linear injection, Re eta, is zero in every
    component while the shift current is finite; the circular injection,
    Im eta, is zero on the gapped honeycomb lattice by C3 and finite once
    the strain breaks it (converged by doubling nk); a Haldane term
    breaks time reversal and gives a finite Re eta"""
    ws = [1.3,2.0]
    for model in [_gapped_honeycomb,_strained_gapped_honeycomb]:
        (w,r) = photocurrent.rectified_conductivity(model(),energies=ws,
                nk=24,T=0.,delta=0.1,components=_comps8)
        assert max(np.max(np.abs(r["shift"][c])) for c in _comps8)>1e-3
        for c in _comps8:
            assert np.allclose(r["injection"][c].real,0.,atol=1e-12)
        if model is _gapped_honeycomb:
            for c in _comps8:
                assert np.allclose(r["injection"][c].imag,0.,atol=1e-12)
    hs = _strained_gapped_honeycomb()
    circ = [photocurrent.injection_current(hs,energies=ws,nk=nk,T=0.,
            delta=0.1,components="yxy")[1].imag for nk in [96,192]]
    assert np.min(np.abs(circ[1]))>1e-3
    assert np.allclose(circ[0],circ[1],rtol=1e-2)
    (w,eta) = photocurrent.injection_current(hs,energies=ws,nk=96,T=0.,
            delta=0.1,components=["yxy","yyx"])
    assert np.allclose(eta["yyx"],np.conj(eta["yxy"]),atol=1e-14)
    lin = [photocurrent.injection_current(_haldane(),energies=ws,nk=nk,
            T=0.,delta=0.1,components="yxx")[1].real for nk in [48,96]]
    assert np.min(np.abs(lin[1]))>1e-3
    assert np.allclose(lin[0],lin[1],rtol=3e-2)


def test_shift_current_d3h_pattern():
    """D3h with x along the armchair direction allows only
    xxx = -xyy = -yxy = -yyx in the shift current, and every component
    with an odd number of y vanishes by the mirror y -> -y"""
    (w,s) = photocurrent.shift_current(_gapped_honeycomb(),
            energies=[1.3,2.2],nk=24,T=0.,delta=0.1,components=_comps8)
    ref = np.max(np.abs(s["xxx"]))
    assert ref>1e-3
    for c in ["xyy","yxy","yyx"]:
        assert np.allclose(s[c],-s["xxx"],atol=1e-10*ref)
    for c in ["yyy","yxx","xxy","xyx"]:
        assert np.allclose(s[c],0.,atol=1e-10*ref)


def test_inversion_kills_everything():
    """The centrosymmetric bilayer gives zero shift current, injection and
    sum frequency conductivity, and two different layers give all three
    finite, so the zero is not an artefact of a path that returns
    nothing"""
    comps = ["xxx","xyy","yxx","yyy"]
    ws = [1.2,2.1]
    out = []
    for h in [_bilayer(),_bilayer(m2=0.8)]:
        h.add_haldane(0.1) # breaks time reversal, so that eta can appear
        (w,r) = photocurrent.rectified_conductivity(h,energies=ws,nk=24,
                T=0.,delta=0.1,components=comps)
        (w,sfg) = photocurrent.sum_frequency_conductivity(h,(1,2),
                energies=ws,nk=24,T=0.,delta=0.1,components=comps)
        out.append([max(np.max(np.abs(x[c])) for c in comps)
                for x in [r["shift"],r["injection"],sfg]])
    assert np.max(out[0])<1e-12
    assert np.min(out[1])>1e-3


def test_sum_frequency_reduces_to_shg():
    """Multipliers (1,1) are the second harmonic of shg_conductivity, the
    numba kernel; exchanging the multipliers exchanges the two fields,
    sigma_abc(w,2w) = sigma_acb(2w,w); and the rectified tensor at (-1,1)
    is both the swapped and the complex conjugate one at (1,-1), on a
    lattice with no time reversal"""
    h = _haldane()
    ws = [0.6,1.4,2.3]
    (w,s11) = photocurrent.sum_frequency_conductivity(h,(1,1),energies=ws,
            nk=24,T=0.05,delta=0.1,components=_comps8)
    (w,shg2) = h.get_shg_conductivity(energies=ws,nk=24,T=0.05,delta=0.1,
            components=_comps8)
    ref = max(np.max(np.abs(shg2[c])) for c in _comps8)
    assert ref>1e-3
    for c in _comps8: assert np.allclose(s11[c],shg2[c],atol=1e-12*ref)
    for (m1,m2) in [((1,2),(2,1)),((1,-1),(-1,1)),((2,-1),(-1,2))]:
        (w,s1) = photocurrent.sum_frequency_conductivity(h,m1,energies=ws,
                nk=24,T=0.05,delta=0.1,components=_comps8)
        (w,s2) = photocurrent.sum_frequency_conductivity(h,m2,energies=ws,
                nk=24,T=0.05,delta=0.1,components=_comps8)
        ref = max(np.max(np.abs(s1[c])) for c in _comps8)
        assert ref>1e-3
        for c in _comps8:
            assert np.allclose(s1[c],s2[c[0]+c[2]+c[1]],atol=1e-12*ref)
            if m1==(1,-1):
                assert np.allclose(s2[c],np.conj(s1[c]),atol=1e-12*ref)


def test_rotation_covariance():
    """Rotating the geometry by an angle that is not a symmetry rotates
    the shift current and eta as rank-3 tensors, every in-plane component
    at a generic orientation"""
    def tensors(h):
        (w,r) = photocurrent.rectified_conductivity(h,energies=[1.3,2.0],
                nk=24,T=0.,delta=0.1,components=_comps8)
        out = []
        for key in ["shift","injection"]:
            t = np.zeros((2,2,2,2),dtype=np.complex128)
            for c in _comps8:
                t[:,"xy".index(c[0]),"xy".index(c[1]),"xy".index(c[2])] = \
                        r[key][c]
            out.append(t)
        return out
    h0 = _haldane()
    theta = 0.37
    h1 = h0.copy()
    h1.geometry = sculpt.rotate(h0.geometry,theta)
    R = np.array([[np.cos(theta),np.sin(theta)], # sculpt.rotate turns the
                  [-np.sin(theta),np.cos(theta)]]) # crystal clockwise
    for (t0,t1) in zip(tensors(h0),tensors(h1)):
        rot = np.einsum("ai,bj,ck,wijk->wabc",R,R,R,t0)
        assert np.max(np.abs(t0))>1e-3
        assert np.allclose(t1,rot,atol=1e-10*np.max(np.abs(t0)))


def test_spin_degeneracy_doubles_everything():
    """A spinful copy of a doped Haldane lattice at T > 0, every band
    doubly degenerate, gives exactly twice the spinless total, shift and
    injection: the pairs inside a degenerate multiplet are left out of the
    closed form instead of dividing by a zero energy difference"""
    def run(has_spin):
        h = _haldane(has_spin=has_spin)
        h.shift_fermi(0.6)
        return photocurrent.rectified_conductivity(h,energies=[1.3,2.0],
                nk=24,T=0.05,delta=0.1,components=["yxx","xxx","xxy"])[1]
    r0,r1 = run(False),run(True)
    for key in ["total","shift","injection"]:
        ref = max(np.max(np.abs(r0[key][c])) for c in r0[key])
        assert ref>1e-3
        for c in r0[key]:
            assert np.all(np.isfinite(r1[key][c]))
            assert np.allclose(r1[key][c],2.*r0[key][c],atol=1e-10*ref)


def test_options():
    """Multipliers of the wrong length, a component of the wrong length, a
    zero broadening and a Nambu Hamiltonian are refused with a message"""
    h = _gapped_honeycomb()
    with pytest.raises(ValueError,match="multipliers"):
        photocurrent.sum_frequency_conductivity(h,(1,2,3),energies=[1.],nk=4)
    with pytest.raises(ValueError,match="multipliers"):
        photocurrent.sum_frequency_conductivity(h,1,energies=[1.],nk=4)
    with pytest.raises(ValueError,match="xyy"):
        photocurrent.injection_current(h,energies=[1.],nk=4,components="xy")
    with pytest.raises(ValueError,match="xyy"):
        photocurrent.shift_current(h,energies=[1.],nk=4,components="xyyx")
    for f in [photocurrent.injection_current,photocurrent.shift_current,
              photocurrent.rectified_conductivity]:
        with pytest.raises(ValueError,match="delta"):
            f(h,energies=[1.],nk=4,delta=0.)
    with pytest.raises(ValueError,match="delta"):
        photocurrent.sum_frequency_conductivity(h,(1,2),energies=[1.],nk=4,
                delta=0.)
    hn = _gapped_honeycomb()
    hn.setup_nambu_spinor()
    with pytest.raises(NotImplementedError,match="Nambu"):
        photocurrent.injection_current(hn,energies=[1.],nk=4)
    with pytest.raises(NotImplementedError,match="Nambu"):
        photocurrent.shift_current(hn,energies=[1.],nk=4)
    with pytest.raises(NotImplementedError,match="Nambu"):
        photocurrent.sum_frequency_conductivity(hn,(1,2),energies=[1.],nk=4)


def test_injection_rate_shares_its_commutators():
    """injection_rate builds r^b once per b and [r^c, P h^a P] once per
    (a,c) for all the triplets at once. Each triplet alone, where nothing
    is shared, gives the same weights, on a supercell whose folded bands
    are degenerate along lines (so the multiplet blocks are there); only
    the final product with the Lorentzians may round differently when it
    has one column instead of 27"""
    h = _haldane().get_supercell(2)
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ks = kubo._kmesh(h,6)
    es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,ks)
    f = kubo._fermi(es,0.05)
    ws = np.linspace(0.5,4.,7)
    trip = photocurrent._triplets(photocurrent._all_components())
    tol = 1e-6*scale
    eta = photocurrent.injection_rate(es,h1,f,ws,0.1,trip,tol)
    assert np.max(np.abs(eta))>0.
    for (t,abc) in enumerate(trip):
        one = photocurrent.injection_rate(es,h1,f,ws,0.1,[abc],tol)
        assert np.max(np.abs(one[:,0]-eta[:,t]))<=1e-14*np.max(np.abs(eta))


def test_injection_current_does_not_depend_on_the_batch():
    """The k-points of injection_current go in batches sized by their
    memory, thousands for a few bands. The k-sum is linear, so batches of
    7 k-points, which do not divide the mesh, give the same rate to
    rounding, and the batch stays within its budget"""
    h = _haldane()
    ws = np.linspace(0.5,4.,9)
    comps = ["xxy","yxy","xyx"]
    (w,e0) = photocurrent.injection_current(h,energies=ws,nk=12,T=0.05,
            components=comps)
    old = photocurrent._batch_size
    try:
        photocurrent._batch_size = lambda n,nw,nt: 7
        (w,e1) = photocurrent.injection_current(h,energies=ws,nk=12,T=0.05,
                components=comps)
    finally: photocurrent._batch_size = old
    scale = max(np.max(np.abs(e0[c])) for c in comps)
    assert scale>0.
    for c in comps: assert np.max(np.abs(e1[c]-e0[c]))<1e-13*scale
    for (n,nw,nt) in [(2,351,3),(18,50,27),(200,100,27)]:
        nb = photocurrent._batch_size(n,nw,nt)
        assert nb>=1 and (nb==1 or nb*(16.*(40+nt)+24.*nw)*n*n<=2e8)
    assert photocurrent._batch_size(2,351,3)>1000 # not a fixed 64
