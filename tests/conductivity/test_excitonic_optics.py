"""Excitonic linear and second-harmonic response from the Tamm-Dancoff BSE
(src/pyqula/conductivitytk/excitonicoptics.py).

Without interaction both reduce to the single-particle responses, which
are pinned independently (the Kubo conductivity, and the second-harmonic
conductivity checked against a finite-field calculation), so that limit
fixes every prefactor and sign. With the interaction on, the result must
not depend on the gauge of the Bloch eigenvectors, must keep the D3h
pattern of the lattice, and the excitons must enhance the response at
their resonance.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction


def _gapped_honeycomb():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(1.0) # gap 2, at K and K'
    return h


def _zero(h):
    """A vanishing interaction, for the kernel="none" limit"""
    return {(0,0,0):np.zeros(h.intra.shape,dtype=np.complex128)}


def _coulomb(h):
    """The Coulomb tail of the excitons page of the guide; on the gapped
    honeycomb lattice it binds an exciton at 1.845, below the gap 2"""
    return density_interaction(h,Vr=lambda r1,r2:
            0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))


@pytest.mark.parametrize("method",["dense","kpm"])
@pytest.mark.parametrize("haldane",[0.,0.15])
def test_linear_noninteracting_limit_is_kubo(haldane,method):
    """Without interaction the exciton sum is the interband Kubo formula
    at T=0, exactly and at any broadening; the Haldane case checks the
    Hall component as well"""
    h = _gapped_honeycomb()
    if haldane: h.add_haldane(haldane)
    es = [0.3,1.5,2.2,3.0]
    kw = dict(energies=es,nk=12,delta=0.1)
    (w,s0) = h.get_optical_conductivity(T=0.,intraband=False,**kw)
    (w,s1) = h.get_excitonic_optical_conductivity(V=_zero(h),
            kernel="none",method=method,**kw)
    assert np.max(np.abs(s0[:,:2,:2]))>1e-2
    assert np.allclose(s1,s0,atol=1e-8)


@pytest.mark.parametrize("method",["dense","kpm"])
@pytest.mark.parametrize("haldane",[0.,0.05])
def test_shg_noninteracting_limit_is_the_single_particle_shg(haldane,method):
    """Without interaction the exciton sum reproduces shg_conductivity,
    with time reversal and without it (Haldane term, gap 1.55). Below the
    resonances the two agree on the same mesh; on a resonance the
    k-derivative of the pair amplitudes needs the mesh to resolve the
    broadening, and the difference falls from 20% at nk=24 to 0.3% at
    nk=48 (measured at w=2.2, delta=0.1, no Haldane term)"""
    h = _gapped_honeycomb()
    if haldane: h.add_haldane(haldane)
    es = [0.,0.5,1.0] if haldane==0. else [0.,0.3,0.6]
    comps = ["xxx","xyy","yyy","yxx"]
    kw = dict(energies=es,nk=24,delta=0.1,components=comps)
    (w,s0) = h.get_shg_conductivity(T=0.,**kw)
    (w,s1) = h.get_excitonic_shg_conductivity(V=_zero(h),kernel="none",
            method=method,**kw)
    ref = max(np.max(np.abs(s0[c])) for c in comps)
    assert ref>1e-2
    for c in comps: assert np.allclose(s1[c],s0[c],atol=2e-4*ref)


@pytest.mark.parametrize("method",["dense","kpm"])
def test_interacting_shg_is_gauge_invariant_and_symmetric(method):
    """With the Coulomb tail, the response does not change when the
    eigenvectors are put in a different gauge, the components forbidden by
    the mirror y -> -y vanish, and xyy = -xxx below the gap, the C3
    relation, exactly on the mesh since the position operator is built on
    the bond vectors of the lattice"""
    h = _gapped_honeycomb()
    W = _coulomb(h)
    comps = ["xxx","xyy","yyy","yxx"]
    kw = dict(energies=[0.3,0.9],nk=21,delta=0.05,components=comps,V=W,
            method=method)
    (w,s) = h.get_excitonic_shg_conductivity(**kw)
    (w,sg) = h.get_excitonic_shg_conductivity(gauge="phase",**kw)
    ref = np.max(np.abs(s["xxx"]))
    assert ref>1e-2
    for c in comps: assert np.allclose(sg[c],s[c],atol=1e-10*ref)
    for c in ["yyy","yxx"]: assert np.allclose(s[c],0.,atol=1e-10*ref)
    assert np.allclose(s["xyy"],-s["xxx"],atol=1e-10*ref)


def test_shg_of_a_stacked_bilayer_keeps_c3():
    """In an R-stacked bilayer the second layer sits at a third of the
    cell, and the C3 rotation about a metal atom maps it to itself only up
    to a lattice vector. The components that C3v forbids (xxx, xyy, yxy for
    this orientation) vanish in the velocity gauge, and must vanish here
    too, on a coarse mesh where the resonances are not resolved; a
    k-derivative in reduced coordinates gave xxx at 5% of the signal at
    nk=9"""
    from pyqula import specialhamiltonian
    h = specialhamiltonian.three_band_tmdc_multilayer(["MoS2","MoS2"],0.1,
            stacking="R",soc=False)
    comps = ["xxx","xyy","yxy","yyy","yxx","xxy"]
    (w,s) = h.get_excitonic_shg_conductivity(V=_zero(h),kernel="none",
            energies=[0.95,1.05],nk=9,delta=0.1,components=comps)
    ref = np.max(np.abs(s["yyy"]))
    assert ref>1e-2
    for c in ["xxx","xyy","yxy"]: assert np.allclose(s[c],0.,atol=1e-10*ref)
    for c in ["yxx","xxy"]: assert np.allclose(s[c],-s["yyy"],atol=1e-10*ref)


def test_excitons_enhance_the_shg_at_their_resonance():
    """The lowest bright exciton, at E_1 = 1.845 below the gap 2, gives a
    second-harmonic resonance at w = E_1/2 that the single-particle
    response does not have; there the excitonic |chi_xxx| is about four
    times the independent-particle one"""
    h = _gapped_honeycomb()
    w0 = 1.8446/2.
    kw = dict(energies=[w0],nk=21,delta=0.05,components="xxx")
    (w,s) = h.get_excitonic_shg_conductivity(V=_coulomb(h),**kw)
    (w,s0) = h.get_excitonic_shg_conductivity(V=_zero(h),kernel="none",**kw)
    assert abs(s[0])>3.*abs(s0[0])


@pytest.mark.parametrize("spinful",[False,True])
def test_chebyshev_route_equals_the_dense_one(spinful):
    """The Chebyshev expansion of the resolvent at complex energy converges
    to the Lorentzian-broadened dense result itself, so the two routes
    agree on the same mesh, below the gap, on the bound exciton and above
    the gap. The spinful case has every exciton level four-fold degenerate,
    which a Krylov eigensolver cannot resolve and a polynomial of the
    matrix does not need to"""
    h = _gapped_honeycomb()
    if spinful: h.turn_spinful()
    kw = dict(energies=[0.,0.6,0.9225,1.845,2.5],nk=9 if spinful else 15,
            delta=0.05,V=_coulomb(h))
    (w,s0) = h.get_excitonic_optical_conductivity(**kw)
    (w,s1) = h.get_excitonic_optical_conductivity(method="kpm",**kw)
    assert np.max(np.abs(s0))>1e-2
    assert np.allclose(s1,s0,atol=1e-8*np.max(np.abs(s0)))
    comps = ["xxx","yxy"]
    (w,s0) = h.get_excitonic_shg_conductivity(components=comps,**kw)
    (w,s1) = h.get_excitonic_shg_conductivity(components=comps,
            method="kpm",**kw)
    ref = np.max(np.abs(s0["xxx"]))
    assert ref>1e-2
    for c in comps: assert np.allclose(s1[c],s0[c],atol=1e-8*ref)


def test_chebyshev_route_takes_a_screened_interaction():
    """The RPA-screened direct term is tabulated on the mesh, and the
    Chebyshev route takes it as it is and agrees with the dense one, for
    the absorption and for the second harmonic"""
    h = _gapped_honeycomb()
    kw = dict(energies=[0.9225,1.845,2.5],nk=12,delta=0.05,V=_coulomb(h))
    (w,s0) = h.get_excitonic_optical_conductivity(**kw)
    (w,s1) = h.get_excitonic_optical_conductivity(screening="rpa",**kw)
    (w,s2) = h.get_excitonic_optical_conductivity(screening="rpa",
            method="kpm",**kw)
    ref = np.max(np.abs(s1))
    assert np.max(np.abs(s1-s0))>1e-3*ref # the screening is there
    assert np.allclose(s2,s1,atol=1e-8*ref)
    (w,s1) = h.get_excitonic_shg_conductivity(screening="rpa",
            components="xxx",**kw)
    (w,s2) = h.get_excitonic_shg_conductivity(screening="rpa",
            components="xxx",method="kpm",**kw)
    assert np.allclose(s2,s1,atol=1e-8*np.max(np.abs(s1)))


def test_chebyshev_resolvent_and_scale_guard():
    """The expanded resolvent solves (z - A) x = D, its brakets (from
    vectors of half the order) are D^+ x, and an interval that misses part
    of the spectrum makes the Chebyshev vectors grow past their bound, and
    the expansion refuses"""
    from pyqula.bsetk.pairbasis import PairBasis
    from pyqula.bsetk.interaction import bare_interaction
    from pyqula.bsetk.convolution import ConvolutionKernel
    from pyqula.bsetk import chebyshev
    h = _gapped_honeycomb()
    pb = PairBasis(h,nk=6)
    W = bare_interaction(h,V=_coulomb(h))
    f = ConvolutionKernel(pb,W,Wx=W)
    c,s = chebyshev.spectral_bounds(f)
    D = np.random.default_rng(1).normal(size=(pb.npair,2))+0j
    zs = np.array([1.5+0.1j,-0.4+0.05j])
    x = chebyshev.resolvent_apply(f.matvec,D,zs,c,s)
    for i,z in enumerate(zs):
        assert np.allclose(z*x[...,i] - f.matvec(x[...,i]),D,atol=1e-8)
    G = chebyshev.resolvent_braket(f.matvec,D,zs,c,s)
    assert np.allclose(G,np.einsum("pa,pbz->zab",np.conj(D),x),atol=1e-8)
    for route in [chebyshev.resolvent_apply,chebyshev.resolvent_braket]:
        with pytest.raises(ValueError,match="diverges"):
            route(f.matvec,D,zs,c,0.5*s)


def test_options_are_checked():
    h = _gapped_honeycomb()
    with pytest.raises(ValueError,match="Q=0"):
        h.get_excitonic_shg_conductivity(energies=[1.],nk=6,V=_zero(h),
                Q=[0.1,0.,0.])
    with pytest.raises(ValueError,match="delta"):
        h.get_excitonic_optical_conductivity(energies=[1.],nk=6,
                V=_zero(h),delta=0.)
    with pytest.raises(ValueError,match="'dense', 'kpm'"):
        h.get_excitonic_optical_conductivity(energies=[1.],nk=6,
                V=_zero(h),method="lanczos")
    with pytest.raises(TypeError,match="max_memory"):
        h.get_excitonic_optical_conductivity(energies=[1.],nk=6,
                V=_zero(h),method="kpm",max_memory=4.)
