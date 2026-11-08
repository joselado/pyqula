"""Physical invariants of the third-order dc conductivity in powers of tau
(src/pyqula/conductivitytk/nonlineardc.py, order=3): the nonlinear Drude
term (tau^3) and the Berry connection polarizability term (tau^1) of Liu et
al., arXiv:2106.04931.

The Drude term is pinned against nonlinear_drude_conductivity on a
spin-conserving magnet and on a honeycomb cell of area 3 sqrt(3)/2, and
against a Fermi-surface form built from the vertices of shg.py on a Rashba
metal. The BCP term, computed in a
Fermi-surface form after two integrations by parts and a symmetrization, is
pinned against the semiclassical current of Liu et al., Eq. (j3), evaluated
directly with spectral k-derivatives of the band-resolved polarizability, and
against their Fermi-sea form, Eq. (chiI). The rest are symmetries: rotation
covariance, C3 isotropy and the mirror zero of the transverse response, a
one-band zero, time-reversal and inversion evenness, the block trace in
degenerate multiplets, the spin doubling and the 3D cell volume.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import sculpt
from pyqula import strain
from pyqula.multihopping import MultiHopping
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import shg
from pyqula.conductivitytk import nonlineardc
from pyqula.conductivitytk import nonlineardrude
from pyqula.specialhamiltoniantk import xwave
from pyqula.dostk.simplexdos import uniform_mesh


s0 = np.eye(2,dtype=np.complex128)
sx = np.array([[0.,1.],[1.,0.]],dtype=np.complex128)
sy = np.array([[0.,-1j],[1j,0.]],dtype=np.complex128)
sz = np.array([[1.,0.],[0.,-1.]],dtype=np.complex128)

LAB4 = [a+b+c+d for a in "xy" for b in "xy" for c in "xy" for d in "xy"]


def _two_orbital_square(intra,tx,ty):
    """One site of the square lattice carrying two orbitals, with onsite
    matrix intra and hoppings tx along +x and ty along +y"""
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=True)
    dd = {(0,0,0):intra,(1,0,0):tx,(-1,0,0):np.conjugate(tx).T,
          (0,1,0):ty,(0,-1,0):np.conjugate(ty).T}
    h.set_multihopping(MultiHopping(dd))
    return h


def _tilted_dirac(m=0.4,t=0.3,mu=0.6):
    """Shibata's tilted massive Dirac lattice model (gap 0.8),
    H = t sin kx s0 + sin kx sx + sin ky sy + [m + 2 - cos kx - cos ky] sz"""
    h = _two_orbital_square((m+2.)*sz,-0.5j*t*s0 - 0.5j*sx - 0.5*sz,
                            -0.5j*sy - 0.5*sz)
    h.shift_fermi(-mu)
    return h


def _ezawa_altermagnet(phi=0.,J=0.05,lam=0.3,mu=-2.5):
    """Ezawa's d-wave altermagnet with Rashba coupling, arXiv:2409.09241"""
    n = [np.sin(phi),np.cos(phi),0.]
    ns = n[0]*sx + n[1]*sy + n[2]*sz
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=True)
    dd = h.get_multihopping().get_dict()
    for (R,s) in [((1,0,0),-1.),((-1,0,0),-1.),((0,1,0),1.),((0,-1,0),1.)]:
        dd[R] = np.asarray(dd[R]) + s*J*ns
    h.set_multihopping(MultiHopping(dd))
    h.add_rashba(lam)
    h.shift_fermi(-mu)
    return h


def _rashba_square_metal(mu=-2.5):
    """Ezawa's altermagnet at a strong exchange along a generic in-plane
    direction plus a Zeeman field along z: two bands that never touch, spin
    not conserved, so nonlinear_drude_conductivity refuses it"""
    h = _ezawa_altermagnet(phi=0.4,J=0.3,mu=mu)
    h.add_zeeman([0.,0.,0.2])
    return h


def _dwave_altermagnet(mu=-2.):
    """The d-wave altermagnet of nonlineardrude's tests: spin conserved,
    the two spin bands cross on the diagonals"""
    h = xwave.xwave_magnet(wave="d",J=0.3,t=-1.)
    h.shift_fermi(-mu)
    return h


def _strain(h,s=0.15):
    """Uniaxial strain along a generic in-plane direction, which breaks C3
    and the mirrors and keeps inversion and time reversal"""
    strain.uniaxial_strain(h,d=np.array([1.,0.4,0.]),s=s)
    return h


def _gapped_honeycomb(mass=0.5,haldane=0.,s=0.,mu=0.9):
    """Spinless honeycomb lattice with a staggered mass (gap 1.0, C3v and
    time reversal), optionally with a Haldane term and uniaxial strain"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(mass)
    if haldane!=0.: h.add_haldane(haldane)
    if s!=0.: _strain(h,s)
    h.shift_fermi(-mu)
    return h


def _pt_antiferromagnet(mu=-1.4,s=0.15):
    """PT-symmetric metal: honeycomb Neel state with Kane-Mele coupling and
    strain, Kramers degenerate at every k"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_antiferromagnetism(0.5)
    h.add_kane_mele(0.1)
    if s!=0.: _strain(h,s)
    h.shift_fermi(-mu)
    return h


def _tensor(h,term,**kw):
    """The in-plane (2,2,2,2) tensor of one third-order term"""
    s = nonlineardc.nonlinear_dc_conductivity(h,order=3,term=term,
            components=LAB4,**kw)
    t = np.zeros((2,2,2,2))
    for c in LAB4: t[tuple("xy".index(x) for x in c)] = s[c]
    return t


def _all(h,**kw):
    """The in-plane tensors of the two terms, from one term="all" call"""
    s = nonlineardc.nonlinear_dc_conductivity(h,order=3,components=LAB4,**kw)
    out = dict()
    for term in ["drude","bcp","total"]:
        t = np.zeros((2,2,2,2))
        for c in LAB4: t[tuple("xy".index(x) for x in c)] = s[term][c]
        out[term] = t
    return out


def _rotate(R,t):
    """A rank-4 tensor rotated by the 2x2 matrix R"""
    return np.einsum("ai,bj,ck,dl,ijkl->abcd",R,R,R,R,t)


def _mesh_bands(h,ks,degeneracy_tol=1e-6):
    """Band energies and the band-diagonal u_a, u_ab and Gt_ab of
    nonlineardc._band_blocks on the k-points ks"""
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,ks)
    b = nonlineardc._band_blocks(es,h1,h2,degeneracy_tol*scale)
    def diag(X): return np.real(np.einsum("...nn->...n",X))
    return es,diag(b["u"]),diag(b["ubc"]),diag(b["Gt"]),cellvol


def _spectral_derivative(h,X,nk):
    """Cartesian derivatives [d_x X, d_y X] of an array X[k,...] on the
    uniform nk x nk mesh, from the spectral derivative in reduced
    coordinates and the Jacobian of kubo._cell_factors"""
    jac = kubo._cell_factors(h)[1]
    freq = 2.*np.pi*1j*np.fft.fftfreq(nk,d=1./nk)
    freq[nk//2] = 0. # the Nyquist component has no derivative
    shape = X.shape[1:]
    x = np.fft.fft2(X.reshape((nk,nk)+shape),axes=(0,1))
    f1 = freq.reshape((nk,1)+(1,)*len(shape))
    f2 = freq.reshape((1,nk)+(1,)*len(shape))
    dk = [np.real(np.fft.ifft2(x*f1,axes=(0,1))).reshape(X.shape),
          np.real(np.fft.ifft2(x*f2,axes=(0,1))).reshape(X.shape)]
    return [jac[0,a]*dk[0] + jac[1,a]*dk[1] for a in range(2)]


def test_third_order_drude_matches_nonlinear_drude():
    """tau^3 <f d_a d_b d_c d_d eps> on the whole Hamiltonian against
    nonlinear_drude_conductivity at l = 3, which runs the two spin blocks
    separately, on the d-wave altermagnet whose spin bands cross on the
    diagonals (the Kato block form on the whole Hamiltonian against two
    nondegenerate problems), and on the strained spinful honeycomb metal,
    whose cell is not of unit area. On a Rashba metal, where spin is not
    conserved and nonlinear_drude_conductivity refuses, against the
    Fermi-surface form <f'' u_a u_d u_bc> - <w u_ad u_bc> (two integrations
    by parts) built from the vertices of shg.py, which shares nothing with
    the Rayleigh-Schroedinger recursion"""
    T,tau = 0.2,1.3
    comps = ["xxxx","xxyy","yxxy","yyyy","xyyy","yxxx"]
    h = _dwave_altermagnet()
    s1 = nonlineardc.nonlinear_dc_conductivity(h,order=3,term="drude",
            tau=tau,nk=60,T=T,components=comps)
    s2 = nonlineardc.nonlinear_dc_conductivity(h,order=3,term="drude",
            tau=tau,nk=120,T=T,components=comps)
    scale = max(abs(s2[c]) for c in comps)
    assert scale>1e-3
    for c in comps:
        assert abs(s1[c]-s2[c])<1e-6*scale # converged in nk
        ref = nonlineardrude.nonlinear_drude_conductivity(h,field=c[1:],
                current=c[0],channel="charge",nk=120,T=T,tau=tau)
        assert abs(s2[c]-ref)<1e-8*scale
    assert abs(s2["xxxx"])>1e-1*scale # a longitudinal response
    # the same identity on the honeycomb lattice, whose cell has the area
    # 3 sqrt(3)/2: every other magnitude check of the tau^3 term is on a
    # cell of unit area, where a missing 1/V_cell goes unnoticed
    h = _gapped_honeycomb(s=0.15)
    h.turn_spinful()
    cs = ["xxxx","yxxy","xyyy"]
    s = nonlineardc.nonlinear_dc_conductivity(h,order=3,term="drude",
            tau=tau,nk=30,T=T,components=cs)
    scale = max(abs(s[c]) for c in cs)
    assert scale>1e-2
    for c in cs:
        ref = nonlineardrude.nonlinear_drude_conductivity(h,field=c[1:],
                current=c[0],channel="charge",nk=30,T=T,tau=tau)
        assert abs(s[c]-ref)<1e-8*scale
    h = _rashba_square_metal()
    nk = 240
    s = nonlineardc.nonlinear_dc_conductivity(h,order=3,term="drude",tau=tau,
            nk=nk,T=T,components=comps)
    s2 = nonlineardc.nonlinear_dc_conductivity(h,order=3,term="drude",
            tau=tau,nk=nk//2,T=T,components=comps)
    with pytest.raises(ValueError,match="spin is not a good quantum number"):
        nonlineardrude.nonlinear_drude_conductivity(h,field="xxx",nk=8)
    ks = uniform_mesh(2,nk)
    es,u,ubc,Gt,cellvol = _mesh_bands(h,ks)
    assert np.min(np.diff(es,axis=1))>1e-2 # no band touching
    w,f2 = nonlineardc._weights(es,T)
    scale = max(abs(s[c]) for c in comps)
    assert scale>1e-3
    for c in comps:
        a,b,cc,d = [kubo._index(x) for x in c]
        assert abs(s[c]-s2[c])<1e-6*scale
        ref = np.sum(f2*u[:,a]*u[:,d]*ubc[:,b,cc] - w*ubc[:,a,d]*ubc[:,b,cc])
        ref *= tau**3/(len(ks)*cellvol)
        assert abs(s[c]-ref)<1e-6*scale


def test_bcp_term_is_the_semiclassical_current():
    """The BCP term contracted with a field, sum sigma_abcd E_b E_c E_d,
    against the tau-linear current of Liu et al. Eq. (j3) evaluated
    directly, with no integration by parts and no symmetrization:

      j = -tau int [ v (E.d)(eps2 f0') + d eps2 (E.d) f0
                     + E x Omega1 (E.d) f0 ],

    eps2 = -(1/2) E.Gt.E, A1 = Gt.E and Omega1_ab = d_a A1_b - d_b A1_a,
    with the k-derivatives of the band-resolved Gt spectral on the mesh and
    those of f0 analytic, and against their Fermi-sea form, Eq. (chiI),
    with the second derivative of Gt spectral. On a strained gapped
    honeycomb metal, nondegenerate everywhere, with no rotation or mirror
    left to constrain the tensor; three field directions"""
    h = _gapped_honeycomb(s=0.15)
    T,tau,nk = 0.2,0.7,120
    sig = _tensor(h,"bcp",tau=tau,nk=nk,T=T)
    sig2 = _tensor(h,"bcp",tau=tau,nk=nk//2,T=T)
    big = np.max(np.abs(sig))
    assert big>1e-4
    assert np.allclose(sig,sig2,atol=1e-6*big) # converged in nk
    ks = uniform_mesh(2,nk)
    es,u,ubc,Gt,cellvol = _mesh_bands(h,ks)
    assert np.min(np.diff(es,axis=1))>0.5
    f = kubo._fermi(es,T)
    w,f2 = nonlineardc._weights(es,T)
    Gt = Gt[:,0:2,0:2]
    u = u[:,0:2]
    dG = _spectral_derivative(h,Gt,nk) # dG[c][k,a,b,n] = d_c Gt_ab
    ddG = [_spectral_derivative(h,dG[c],nk) for c in range(2)]
    norm = 1./(len(ks)*cellvol)
    for theta in [0.3,1.1,2.4]:
        E = np.array([np.cos(theta),np.sin(theta)])
        eps2 = -0.5*np.einsum("a,kabn,b->kn",E,Gt,E)
        deps2 = [-0.5*np.einsum("a,kabn,b->kn",E,dG[c],E) for c in range(2)]
        A1 = np.einsum("kabn,b->kan",Gt,E)
        dA1 = [np.einsum("kabn,b->kan",dG[c],E) for c in range(2)]
        Eu = np.einsum("a,kan->kn",E,u) # (E.d) f0 = -w E.u
        Edeps2 = E[0]*deps2[0] + E[1]*deps2[1]
        j = np.zeros(2)
        for a in range(2):
            ja = u[:,a]*(f2*Eu*eps2 - w*Edeps2) # v_a (E.d)(eps2 f0')
            ja += deps2[a]*(-w*Eu) # d_a eps2 (E.d) f0
            for b in range(2): # (E x Omega1)_a = E_b Omega1_ab
                Om = dA1[a][:,b] - dA1[b][:,a]
                ja += E[b]*Om*(-w*Eu)
            j[a] = -tau*np.sum(ja)*norm
        jt = np.einsum("abcd,b,c,d->a",sig,E,E,E)
        assert np.allclose(jt,j,atol=1e-6*big)
        assert np.max(np.abs(j))>1e-2*big
        # Eq. (chiI): tau int [(-d_a d_b G_cd + d_a d_d G_bc - d_b d_d G_ac)
        # f0 + (1/2) v_a v_b G_cd f0'']
        jc = np.zeros(2)
        EGE = np.einsum("c,kcdn,d->kn",E,Gt,E)
        for a in range(2):
            x = 0.
            for b in range(2):
                for d in range(2):
                    for c in range(2):
                        x = x + E[b]*E[c]*E[d]*(-ddG[b][a][:,c,d]
                                + ddG[d][a][:,b,c] - ddG[d][b][:,a,c])*f
            x = x + 0.5*u[:,a]*Eu*EGE*f2
            jc[a] = tau*np.sum(x)*norm
        assert np.allclose(jt,jc,atol=1e-6*big)


def test_bcp_term_symmetries():
    """Rotating the crystal by an angle that is not a symmetry rotates both
    terms as rank-4 tensors; on the C3v gapped honeycomb metal both are
    isotropic (invariant under any rotation, the only rank-4 pattern C3
    allows in 2D) and the transverse response chi_perp(theta) =
    (z x e).sigma.eee is zero, while strain makes it finite; a one-band
    model has no interband geometry and no BCP term; time reversal and
    inversion allow the term, so it is finite on the gapped honeycomb metal
    and on the untilted Dirac model, which is centrosymmetric and so has
    no second-order response at all"""
    theta = 0.37
    R = np.array([[np.cos(theta),np.sin(theta)], # sculpt.rotate turns the
                  [-np.sin(theta),np.cos(theta)]]) # crystal clockwise
    kw = dict(nk=60,T=0.2)
    h0 = _gapped_honeycomb(haldane=0.03,s=0.15)
    h1 = h0.copy()
    h1.geometry = sculpt.rotate(h0.geometry,theta)
    t0,t1 = _all(h0,**kw),_all(h1,**kw)
    for term in ["drude","bcp"]:
        big = np.max(np.abs(t0[term]))
        assert big>1e-4
        assert np.allclose(t1[term],_rotate(R,t0[term]),atol=1e-10*big)
    def perp(t,th):
        e = np.array([np.cos(th),np.sin(th)])
        n = np.array([-np.sin(th),np.cos(th)])
        return np.einsum("a,abcd,b,c,d->",n,t,e,e,e)
    ths = [0.2,0.7,1.3]
    tc = _all(_gapped_honeycomb(),**kw) # C3v, time reversal
    ts = _all(_gapped_honeycomb(s=0.15),**kw) # strained
    for term in ["drude","bcp"]:
        big = np.max(np.abs(tc[term]))
        assert big>1e-4 # time reversal allows both terms
        assert np.allclose(_rotate(R,tc[term]),tc[term],atol=1e-10*big)
        assert max(abs(perp(tc[term],th)) for th in ths)<1e-10*big
        bigs = np.max(np.abs(ts[term]))
        assert max(abs(perp(ts[term],th)) for th in ths)>1e-3*bigs
    hi = _tilted_dirac(t=0.) # inversion sz H(-k) sz = H(k)
    second = nonlineardc.nonlinear_dc_conductivity(hi,term="all",nk=60,T=0.2)
    assert max(abs(v) for v in second["total"].values())<1e-12
    tb = _tensor(hi,"bcp",**kw) # inversion allows the third order
    tb2 = _tensor(hi,"bcp",nk=120,T=0.2)
    big = np.max(np.abs(tb2))
    assert big>1e-4
    assert np.allclose(tb,tb2,atol=1e-5*big)
    h = geometry.square_lattice().get_hamiltonian(has_spin=False)
    h.shift_fermi(1.)
    t1 = _tensor(h,"bcp",**kw)
    assert np.max(np.abs(t1))<1e-14
    assert np.max(np.abs(_tensor(h,"drude",**kw)))>1e-3


def _random_block_unitary(es,tol,rng):
    """A random unitary inside every degenerate multiplet, the identity
    elsewhere, for each k-point"""
    same = nonlineardc._multiplets(es,tol)
    nk,n = es.shape
    V = np.zeros((nk,n,n),dtype=np.complex128)
    nblock = 0
    for ik in range(nk):
        i = 0
        while i<n:
            j = i
            while j+1<n and same[ik,i,j+1]: j += 1
            m = j-i+1
            if m>1: nblock += 1
            z = rng.normal(size=(m,m)) + 1j*rng.normal(size=(m,m))
            V[ik,i:j+1,i:j+1] = np.linalg.qr(z)[0]
            i = j+1
    return V,nblock


def test_block_trace_in_degenerate_multiplets():
    """A random unitary inside every Kramers pair of the PT antiferromagnet
    leaves the block traces of the BCP term unchanged, and the spinful copy
    of a spinless metal, exactly spin degenerate, gives twice the spinless
    result for both terms"""
    rng = np.random.default_rng(7)
    h = _pt_antiferromagnet()
    ks = uniform_mesh(2,16)
    T = 0.1
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,ks)
    tol = 1e-6*scale
    V,nblock = _random_block_unitary(es,tol,rng)
    assert nblock>=len(ks)
    Vc = np.conjugate(np.swapaxes(V,-1,-2))[:,None]
    h1r = Vc@h1@V[:,None]
    h2r = Vc[:,None]@h2@V[:,None,None]
    t0 = nonlineardc._batch_tensors3(es,h1,h2,T,tol)
    t1 = nonlineardc._batch_tensors3(es,h1r,h2r,T,tol)
    for key in ["M","N"]:
        big = np.max(np.abs(t0[key]))
        assert big>1e-3
        assert np.allclose(t0[key],t1[key],atol=1e-10*big)
    hs = _gapped_honeycomb(haldane=0.03,s=0.15)
    hf = hs.copy()
    hf.turn_spinful()
    kw = dict(nk=40,T=0.2)
    ts,tf = _all(hs,**kw),_all(hf,**kw)
    for term in ["drude","bcp"]:
        big = np.max(np.abs(ts[term]))
        assert big>1e-4
        assert np.allclose(tf[term],2.*ts[term],atol=1e-10*big)


def test_tau_powers_and_3d():
    """term="all" carries tau^3 and tau^1, total is their sum and the
    symmetric field indices give equal components; layers stacked along a3
    with no hopping between them give the 2D BCP and Drude terms divided by
    the layer spacing, with the z components zero"""
    h = _tilted_dirac()
    kw = dict(nk=40,T=0.2,components=["xxyy","xyxy","xyyx","yxxx"])
    a1 = nonlineardc.nonlinear_dc_conductivity(h,order=3,tau=1.,**kw)
    a2 = nonlineardc.nonlinear_dc_conductivity(h,order=3,tau=2.,**kw)
    for c in kw["components"]:
        assert abs(a2["drude"][c]-8.*a1["drude"][c])<1e-12*abs(a1["drude"][c])
        assert abs(a2["bcp"][c]-2.*a1["bcp"][c])<1e-12*abs(a1["bcp"][c])
        assert abs(a2["total"][c]-a2["drude"][c]-a2["bcp"][c])<1e-14
    for term in ["drude","bcp"]:
        x = a1[term]
        assert abs(x["xxyy"])>1e-5
        assert abs(x["xxyy"]-x["xyxy"])<1e-12*abs(x["xxyy"])
        assert abs(x["xxyy"]-x["xyyx"])<1e-12*abs(x["xxyy"])
    c = 2.5
    g = geometry.cubic_lattice()
    g.a3 = np.array([0.,0.,c])
    h3 = g.get_hamiltonian(has_spin=True)
    dd = {R:np.asarray(v) for (R,v) in h.get_multihopping().get_dict().items()}
    h3.set_multihopping(MultiHopping(dd))
    comps = ["xxxx","xxyy","yxxx","yyyy"]
    s2 = nonlineardc.nonlinear_dc_conductivity(h,order=3,term="bcp",nk=30,
            T=0.2,components=comps+["xxxy"])
    s3 = nonlineardc.nonlinear_dc_conductivity(h3,order=3,term="bcp",nk=30,
            T=0.2,components=comps+["zxxx","xzzz"])
    big = max(abs(s2[x]) for x in comps)
    assert big>1e-4
    for x in comps: assert abs(s3[x]-s2[x]/c)<1e-10*big
    for x in ["zxxx","xzzz"]: assert abs(s3[x])<1e-12*big
    d2 = nonlineardc.nonlinear_dc_conductivity(h,order=3,term="drude",nk=30,
            T=0.2,components=comps)
    d3 = nonlineardc.nonlinear_dc_conductivity(h3,order=3,term="drude",
            nk=30,T=0.2,components=comps+["zzzz","xxzz"])
    big = max(abs(d2[x]) for x in comps)
    assert big>1e-4
    for x in comps: assert abs(d3[x]-d2[x]/c)<1e-10*big
    for x in ["zzzz","xxzz"]: assert abs(d3[x])<1e-12*big


def test_options():
    """At order=3 an unknown term lists the order-3 registry, a spin channel
    is refused, a component of three labels is refused naming four, an
    order outside the registry lists the accepted ones; the order-2
    registry is unchanged"""
    h = _tilted_dirac()
    f = nonlineardc.nonlinear_dc_conductivity
    with pytest.raises(ValueError,
            match=", ".join(nonlineardc.get_term_names(3))):
        f(h,order=3,term="bcd",nk=4)
    for channel in ["spin","up","dn"]:
        with pytest.raises(NotImplementedError,match="charge channel"):
            f(_dwave_altermagnet(),order=3,channel=channel,nk=4)
    with pytest.raises(ValueError,match="xxyy"):
        f(h,order=3,components="xyy",nk=4)
    with pytest.raises(ValueError,match="xyy"):
        f(h,order=2,components="xxyy",nk=4)
    with pytest.raises(NotImplementedError,match="order=2 and order=3"):
        f(h,order=4,nk=4)
    with pytest.raises(ValueError,match="Fermi-surface"):
        f(h,order=3,T=0.,nk=4)
    assert nonlineardc.get_orders()==[2,3]
    assert set(nonlineardc.get_term_names(3))=={"drude","bcp","all"}
    assert set(nonlineardc.get_term_names())=={"drude","bcd","qmd",
            "qmd_hall","qmd_disp","qmd_intra","all"}
    s = f(h,order=3,nk=8,T=0.2)
    assert set(s)=={"drude","bcp","total"}
    assert len(s["bcp"])==16 # every in-plane component
