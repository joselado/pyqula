"""Physical invariants of the second-order dc conductivity in powers of tau
(src/pyqula/conductivitytk/nonlineardc.py): the nonlinear Drude term, the
Berry curvature dipole and the quantum-metric terms.

Each building block is pinned against an independent code path (the
Rayleigh-Schroedinger recursion of nonlineardrude.py, the quantum geometric
tensor of topologytk/qgt.py, the Berry curvature of anomaloushall.py, the
static polarizability of kubo.py), each Fermi-surface form against a
Fermi-sea form, and the terms together against the selection rules of PT,
time reversal, C3 and inversion, a rotation of the crystal, the Neel-angle
law of Ezawa's altermagnet and Shibata's benchmark value of the intra term.
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
from pyqula.topologytk import anomaloushall
from pyqula.topologytk import qgt
from pyqula.dostk.simplexdos import uniform_mesh


s0 = np.eye(2,dtype=np.complex128)
sx = np.array([[0.,1.],[1.,0.]],dtype=np.complex128)
sy = np.array([[0.,-1j],[1j,0.]],dtype=np.complex128)
sz = np.array([[1.,0.],[0.,-1.]],dtype=np.complex128)

TERMS = ["drude","bcd","qmd_hall","qmd_disp","qmd_intra"]


def _two_orbital_square(intra,tx,ty):
    """One site of the square lattice carrying two orbitals (the spin index
    used as an orbital), with onsite matrix intra and hoppings tx along +x
    and ty along +y, their Hermitian partners along -x and -y"""
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=True)
    dd = {(0,0,0):intra,(1,0,0):tx,(-1,0,0):np.conjugate(tx).T,
          (0,1,0):ty,(0,-1,0):np.conjugate(ty).T}
    h.set_multihopping(MultiHopping(dd))
    return h


def _check_bloch(h,hk):
    """H(k) of h against the closed form hk(Kx,Ky) at a few k-points"""
    hkgen = h.get_hk_gen()
    for k in np.random.default_rng(3).random((4,3)):
        K = 2.*np.pi*k # the square lattice has a = 1
        assert np.allclose(np.asarray(hkgen([k[0],k[1],0.])),hk(K[0],K[1]),
                atol=1e-13)


def _shibata_real(mu=0.5):
    """Shibata's real two-band model, arXiv:2606.22359,
    H = t0 sin kx s0 + (m + tx cos kx + ty cos ky) sx + lam sin ky sz.
    Real, so the Berry curvature is zero at every k while the metric is
    finite; gap 1.0. The chemical potential is placed at mu"""
    m,tx,ty,lam,t0 = 2.2,1.0,0.7,0.9,0.5
    h = _two_orbital_square(m*sx,tx/2.*sx - 0.5j*t0*s0,
                            ty/2.*sx - 0.5j*lam*sz)
    _check_bloch(h,lambda kx,ky: t0*np.sin(kx)*s0
            + (m + tx*np.cos(kx) + ty*np.cos(ky))*sx + lam*np.sin(ky)*sz)
    h.shift_fermi(-mu)
    return h


def _dirac_d(kx,ky,m=0.4):
    """d-vector of the tilted Dirac model, vx = vy = bx = by = 1"""
    return np.array([np.sin(kx),np.sin(ky),m + 2. - np.cos(kx) - np.cos(ky)])


def _tilted_dirac(m=0.4,t=0.3,mu=None):
    """Shibata's tilted massive Dirac lattice model,
    H = t sin kx s0 + vx sin kx sx + vy sin ky sy
        + [m + bx (1 - cos kx) + by (1 - cos ky)] sz,
    vx = vy = bx = by = 1: gap 0.8, finite Berry curvature and metric"""
    h = _two_orbital_square((m+2.)*sz,-0.5j*t*s0 - 0.5j*sx - 0.5*sz,
                            -0.5j*sy - 0.5*sz)
    def hk(kx,ky):
        d = _dirac_d(kx,ky,m)
        return t*np.sin(kx)*s0 + d[0]*sx + d[1]*sy + d[2]*sz
    _check_bloch(h,hk)
    if mu is not None: h.shift_fermi(-mu)
    return h


def _ezawa_altermagnet(phi=0.,J=0.05,lam=0.3,mu=-2.5,n=None):
    """Ezawa's d-wave altermagnet with Rashba coupling, arXiv:2409.09241, on
    the square lattice: first-neighbor hopping, the exchange
    2J(cos ky - cos kx) n.sigma, a d_{x^2-y^2} form factor, and the Rashba
    term. n = (sin phi, cos phi, 0) unless given"""
    if n is None: n = [np.sin(phi),np.cos(phi),0.]
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


def _strain(h,s=0.15):
    """Uniaxial strain along a generic in-plane direction, which breaks C3
    and keeps inversion and time reversal"""
    strain.uniaxial_strain(h,d=np.array([1.,0.4,0.]),s=s)
    return h


def _pt_antiferromagnet(mu=-1.4,s=0.15):
    """PT-symmetric metal: honeycomb Neel state along z with Kane-Mele
    coupling and uniaxial strain. Kramers degenerate at every k, broken
    inversion, broken C3"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_antiferromagnetism(0.5)
    h.add_kane_mele(0.1)
    if s!=0.: _strain(h,s)
    h.shift_fermi(-mu)
    return h


def _gapped_honeycomb(mass=0.5,haldane=0.,s=0.,mu=None):
    """Spinless honeycomb lattice with a staggered mass, optionally with a
    Haldane term (breaks T) and uniaxial strain (breaks C3)"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(mass)
    if haldane!=0.: h.add_haldane(haldane)
    if s!=0.: _strain(h,s)
    if mu is not None: h.shift_fermi(-mu)
    return h


def _bilayer(mu=0.9):
    """The centrosymmetric bilayer of test_shg.py, two gapped honeycomb
    layers at z = +-1, the second the inversion image of the first,
    doped into the conduction band"""
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
    h.add_onsite([0.5,-0.5,0.5,-0.5])
    h.shift_fermi(-mu)
    return h


def _fwave_zeeman(mu=-4.,B=0.3):
    """The f-wave magnet of nonlineardrude's tests with a Zeeman field along
    z: spin is conserved and each spin band lacks inversion, so the
    second-order Drude term is finite in the spin and in the charge
    channel (the d-wave altermagnet alone is inversion symmetric and gives
    zero in both)"""
    h = xwave.xwave_magnet(wave="f",J=0.3,t=-1.)
    h.add_zeeman([0.,0.,B])
    h.shift_fermi(-mu)
    return h


def _rashba_exchange_honeycomb(mu=-1.0):
    """A four-band honeycomb metal with Rashba coupling, a canted exchange
    field and a staggered mass: spin is not conserved and the bands are
    nondegenerate at a generic k"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.2,0.1,0.3])
    h.add_sublattice_imbalance(0.2)
    h.shift_fermi(-mu)
    return h


def _rashba_square_metal(mu=-2.5):
    """Ezawa's altermagnet at a strong exchange along a generic in-plane
    direction, plus a Zeeman field along z that gaps its crossings at Gamma
    and M: two bands that never touch, spin not conserved, no symmetry that
    kills the Drude term"""
    h = _ezawa_altermagnet(phi=0.4,J=0.3,mu=mu)
    h.add_zeeman([0.,0.,0.2])
    return h


def _tensor(h,term,**kw):
    """The in-plane (2,2,2) tensor of one term"""
    labels = [a+b+c for a in "xy" for b in "xy" for c in "xy"]
    s = nonlineardc.nonlinear_dc_conductivity(h,term=term,components=labels,
            **kw)
    t = np.zeros((2,2,2))
    for c in labels: t["xy".index(c[0]),"xy".index(c[1]),"xy".index(c[2])] = s[c]
    return t


def _mesh_blocks(h,ks,degeneracy_tol=1e-6):
    """Band energies and the block matrices of nonlineardc._band_blocks on
    the k-points ks"""
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,ks)
    return es,nonlineardc._band_blocks(es,h1,h2,degeneracy_tol*scale)


def _diag(X):
    """Band-diagonal part of a block matrix, X[...,n,n]"""
    return np.real(np.einsum("...nn->...n",X))


def _spectral_derivative(h,X,nk):
    """Cartesian derivatives [d_x X, d_y X] of an array X[k,...] on the
    uniform nk x nk mesh of uniform_mesh, from the spectral derivative in
    the reduced coordinates and the Jacobian of kubo._cell_factors. Exact
    up to aliasing for a smooth periodic X"""
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


def test_drude_matches_nonlinear_drude():
    """The Fermi-surface form -tau^2 <w u_a u_bc> against the Fermi-sea form
    -tau^2 <f d_a d_b d_c eps> of nonlineardrude.py, whose band derivatives
    come from a Rayleigh-Schroedinger recursion that shares nothing with
    the h^ab vertices here. In the charge and spin channel of a
    spin-conserving magnet through nonlinear_drude_conductivity, and on a
    Rashba metal, where the spin channels do not exist, through
    nonlineardrude._Channel on the whole Hamiltonian. The Rashba metal has
    no band touching anywhere: the Rashba-exchange honeycomb lattice,
    whose bands cross, converges only as a power of nk in both forms"""
    T,tau = 0.2,1.3
    comps = ["xxx","xyy","yxy","yyy","yxx"]
    h = _fwave_zeeman()
    for channel in ["charge","spin"]:
        s1 = nonlineardc.nonlinear_dc_conductivity(h,term="drude",tau=tau,
                channel=channel,nk=120,T=T,components=comps)
        s2 = nonlineardc.nonlinear_dc_conductivity(h,term="drude",tau=tau,
                channel=channel,nk=240,T=T,components=comps)
        scale = max(abs(s2[c]) for c in comps)
        assert scale>1e-3
        for c in comps:
            assert abs(s1[c]-s2[c])<1e-6*scale # converged in nk
            ref = nonlineardrude.nonlinear_drude_conductivity(h,field=c[1:],
                    current=c[0],channel=channel,nk=240,T=T,tau=tau)
            assert abs(s2[c]-ref)<1e-6*scale
    h = _rashba_square_metal()
    nk = 240
    ks = kubo._kmesh(h,nk)
    chan = nonlineardrude._Channel(h.get_multicell(),ks)
    f = kubo._fermi(chan.es,T)
    norm = 1./(len(ks)*nonlineardrude._cell_volume(h))
    s = nonlineardc.nonlinear_dc_conductivity(h,term="drude",tau=tau,nk=nk,
            T=T,components=comps)
    s2 = nonlineardc.nonlinear_dc_conductivity(h,term="drude",tau=tau,
            nk=nk//2,T=T,components=comps)
    scale = max(abs(s[c]) for c in comps)
    assert scale>1e-3
    for c in comps:
        assert abs(s[c]-s2[c])<1e-6*scale
        axes = [kubo._index(x) for x in c]
        ref = -tau**2*np.sum(f*chan.derivative(axes))*norm
        assert abs(s[c]-ref)<1e-6*scale


def test_inverse_mass_matches_rs_recursion():
    """u^n_ab = h^ab_nn + 2 sum_m Re(h^a_nm h^b_mn)/eps_nm, second-order
    perturbation theory on the vertices, against the second derivative of
    the band energy from the Taylor recursion of nonlineardrude._Channel,
    on a four-band model with no degeneracy"""
    h = _rashba_exchange_honeycomb()
    ks = np.random.default_rng(5).random((7,3))
    ks[:,2] = 0.
    es,b = _mesh_blocks(h,ks)
    chan = nonlineardrude._Channel(h.get_multicell(),ks)
    assert np.allclose(es,chan.es,atol=1e-12)
    assert np.min(np.diff(es,axis=1))>1e-3 # no degeneracy
    for (a,c) in [(0,0),(0,1),(1,1)]:
        ref = chan.derivative([a,c])
        assert np.max(np.abs(ref))>1e-1
        assert np.allclose(_diag(b["ubc"][:,a,c]),ref,atol=1e-10)
        assert np.allclose(_diag(b["u"][:,a]),chan.derivative([a]),atol=1e-10)


def test_metric_and_curvature_match_qgt_and_anomaloushall():
    """G^n_ab and Omega^n_ab of the block matrices against the quantum
    geometric tensor of qgt.py (Cartesian coordinates), against the Kubo
    curvature of anomaloushall.py on a mesh, and against the closed forms
    of a two-band d.sigma model, Omega^pm = -/+ (1/2) dhat.(d_x dhat x
    d_y dhat) and G^pm_ij = (1/4) d_i dhat.d_j dhat"""
    m = 0.4
    h = _tilted_dirac(m=m)
    ks = np.random.default_rng(7).random((6,3))
    ks[:,2] = 0.
    es,b = _mesh_blocks(h,ks)
    for ik,k in enumerate(ks):
        K = 2.*np.pi*k
        d = _dirac_d(K[0],K[1],m)
        dd = [np.array([np.cos(K[0]),0.,np.sin(K[0])]), # d_x d
              np.array([0.,np.cos(K[1]),np.sin(K[1])])] # d_y d
        nd = np.linalg.norm(d)
        dh = d/nd
        ddh = [(x - dh*dh.dot(x))/nd for x in dd] # derivatives of dhat
        om = 0.5*dh.dot(np.cross(ddh[0],ddh[1]))
        G = 0.25*np.array([[ddh[i].dot(ddh[j]) for j in range(2)]
                           for i in range(2)])
        assert abs(om)>1e-4
        for n,sign in [(0,1.),(1,-1.)]: # lower band +, upper band -
            assert abs(_diag(b["Omega"][ik,0,1])[n]-sign*om)<1e-10
            assert np.allclose(_diag(b["G"][ik,:2,:2])[...,n],G,atol=1e-10)
            Q = qgt.quantum_geometric_tensor_k(h,k=k,occ_idxs=[n],
                    coordinates="cartesian")
            assert np.allclose(_diag(b["G"][ik])[...,n],Q.real,atol=1e-10)
            assert np.allclose(_diag(b["Omega"][ik])[...,n],-2.*Q.imag,
                    atol=1e-10)
    nk = 8
    es0,om0,cellvol,tol = anomaloushall.band_curvature_mesh(h,nk=nk)
    es1,b1 = _mesh_blocks(h,uniform_mesh(2,nk))
    assert np.allclose(es0,es1,atol=1e-12) # same mesh, same order
    assert np.allclose(om0,_diag(b1["Omega"][:,0,1]),atol=1e-10)


def test_normalized_metric_is_the_polarizability():
    """The second-order Stark shift of a band is (q^2/2) E.Gt.E, so the
    static polarizability of an insulator is chi_ab = -q^2 <f Gt_ab>, and
    it must equal lim_{w->0} i sigma_ab(w)/w of the Kubo conductivity,
    which is built from the same velocities but summed over pairs of
    bands in a different way (the occupied-occupied pairs cancel in the
    first and are absent in the second)"""
    h = _gapped_honeycomb(s=0.15) # strained, so that xx != yy
    nk = 30
    ks = uniform_mesh(2,nk)
    es,b = _mesh_blocks(h,ks)
    f = kubo._fermi(es,0.)
    cellvol = kubo._cell_factors(h)[2]
    q = shg._charge
    chi = -q**2*np.einsum("kn,kabn->ab",f,_diag(b["Gt"]))/(len(ks)*cellvol)
    w,delta = 1e-4,1e-8
    (e,sigma) = kubo.optical_conductivity(h,energies=[w],nk=nk,T=0.,
            delta=delta)
    ref = np.real(1j*sigma[0]/(w + 1j*delta))
    assert chi[0,0]>1e-2 and abs(chi[0,0]-chi[1,1])>1e-3
    assert np.allclose(chi[:2,:2],ref[:2,:2],atol=1e-6*chi[0,0])


def test_bcd_fermi_surface_equals_fermi_sea():
    """D_a = <w u_a Omega> against two Fermi-sea forms: <f d_a Omega>, with
    Omega from anomaloushall.band_curvature_mesh and d_a a spectral
    derivative on the periodic mesh, and the derivative of the anomalous
    Hall sum sum_k f(eps(k + dk)) Omega(k) with respect to a rigid shift
    of the Fermi sea alone, built by multiplying every hopping by
    exp(i dk.R). The BCD term itself is tau D_y in xyy, -tau D_x in yxx,
    tau D_x/2 in xxy and xyx, -tau D_y/2 in yxy and yyx, and zero in xxx
    and yyy"""
    h = _tilted_dirac(mu=0.6)
    T,nk = 0.05,200
    D = nonlineardc.berry_curvature_dipole(h,nk=nk,T=T)
    D2 = nonlineardc.berry_curvature_dipole(h,nk=2*nk,T=T)
    assert np.max(np.abs(D))>1e-3
    assert np.allclose(D,D2,atol=1e-6*np.max(np.abs(D))) # converged
    es,om,cellvol,tol = anomaloushall.band_curvature_mesh(h,nk=nk)
    f = kubo._fermi(es,T)
    dom = _spectral_derivative(h,om,nk)
    sea = np.array([np.sum(f*dom[a]) for a in range(2)])/(len(es)*cellvol)
    assert np.allclose(D,sea,atol=1e-6*np.max(np.abs(D)))
    def shifted(dK): # every hopping times exp(i dK.R), so H(k+dK)
        dd = h.get_multihopping().get_dict()
        dd = {R:np.asarray(m)*np.exp(1j*np.dot(dK,R)) for (R,m) in dd.items()}
        hs = h.copy()
        hs.set_multihopping(MultiHopping(dd))
        return anomaloushall.band_curvature_mesh(hs,nk=nk)[0]
    eps = 1e-4
    for a in range(2):
        dK = np.zeros(3) ; dK[a] = eps
        F = [np.sum(kubo._fermi(shifted(s*dK),T)*om) for s in [1.,-1.]]
        fd = -(F[0]-F[1])/(2.*eps)/(len(es)*cellvol)
        assert abs(fd-D[a])<1e-4*np.max(np.abs(D))
    tau = 0.7
    labels = [a+b+c for a in "xy" for b in "xy" for c in "xy"]
    s = nonlineardc.nonlinear_dc_conductivity(h,term="bcd",tau=tau,nk=nk,
            T=T,components=labels)
    # the whole 2D tensor, (tau/2)(<w u_c Omega_ab> + <w u_b Omega_ac>) with
    # Omega_xy = -Omega_yx = Omega: the mixed components carry half the
    # dipole, so they pin the symmetrization in b,c
    ref = {"xyy":tau*D[1],"yxx":-tau*D[0],
           "xxy":tau*D[0]/2.,"xyx":tau*D[0]/2.,
           "yxy":-tau*D[1]/2.,"yyx":-tau*D[1]/2.,
           "xxx":0.,"yyy":0.}
    assert abs(D[0])>1e-3 # the tilt along x gives D_x, which xxy and xyx carry
    for c in labels:
        assert abs(s[c]-ref[c])<1e-12*np.max(np.abs(D))


def test_intra_qmd_two_forms_agree():
    """The intra term in production form, -(1/2) <f'' u_a G_bc>, against
    the covariant derivative of the metric, -(1/2) <w d_a G_bc>, built
    from the generalized derivative of the Berry connection; the two are
    one integration by parts apart, so they agree once the mesh resolves T"""
    h = _tilted_dirac(mu=0.6)
    T,nk = 0.05,200
    labels = [a+b+c for a in "xy" for b in "xy" for c in "xy"]
    s = nonlineardc.nonlinear_dc_conductivity(h,term="qmd_intra",nk=nk,T=T,
            components=labels)
    cov = nonlineardc._intra_covariant(h,nk=nk,T=T)
    scale = max(abs(v) for v in s.values())
    assert scale>1e-3
    for c in labels:
        assert abs(s[c]-cov[tuple(kubo._index(x) for x in c)])<1e-8*scale


def test_quantum_metric_pieces_match_the_fermi_sea_forms():
    """The three quantum-metric pieces against the Fermi-sea forms as the
    papers write them, with e^3 = 1 and the k-derivatives of the band
    metrics taken spectrally on the mesh: Shibata's Eq. (interQMD-band),
    -<f [d_a Gt_bc - (1/2)(d_b Gt_ca + d_c Gt_ab)]>, for qmd_hall; the
    dispersion part of Kaplan-Holder-Yan's Eq. (fullcurrent2-metric),
    -<f d_a Gt_bc>, for qmd_disp; Shibata's Eq. (intraQMD-band),
    (1/2)<f' d_a G_bc>, for qmd_intra. This pins the sign of each piece,
    which the selection rules cannot see"""
    h = _tilted_dirac(mu=0.6)
    T,nk = 0.05,200
    es,b = _mesh_blocks(h,uniform_mesh(2,nk))
    f = kubo._fermi(es,T)
    fp = -f*(1.-f)/T # f'
    norm = 1./(len(es)*kubo._cell_factors(h)[2])
    dGt = _spectral_derivative(h,_diag(b["Gt"][:,:2,:2]),nk) # [a][k,b,c,n]
    dG = _spectral_derivative(h,_diag(b["G"][:,:2,:2]),nk)
    def sea(wt,dX): # sea[a,b,c] = <wt d_a X_bc>
        return np.array([np.einsum("kn,kbcn->bc",wt,dX[a]) for a in range(2)])\
                *norm
    fGt = sea(f,dGt)
    ref = {"qmd_hall":-(fGt - 0.5*(np.transpose(fGt,(2,0,1))
                                  + np.transpose(fGt,(1,2,0)))),
           "qmd_disp":-fGt,
           "qmd_intra":0.5*sea(fp,dG)}
    for term in ref:
        t = _tensor(h,term,nk=nk,T=T)
        big = np.max(np.abs(t))
        assert big>1e-4
        assert np.allclose(t,ref[term],atol=1e-6*big)


def test_shibata_real_two_band_benchmark():
    """Shibata's diagnostic on his real two-band model, arXiv:2606.22359:
    sigma^intra_xyy = +1.16e-2 at mu = 0.5, T = 0.05, converged in nk; and
    the Berry curvature dipole is zero in every component of a real
    Hamiltonian"""
    h0 = _shibata_real(mu=0.)
    h = _shibata_real(mu=0.5)
    k = [0.13,0.31,0.]
    e0 = np.linalg.eigvalsh(np.asarray(h0.get_hk_gen()(k)))
    e1 = np.linalg.eigvalsh(np.asarray(h.get_hk_gen()(k)))
    assert np.allclose(e1,e0-0.5) # shift_fermi(-mu) puts mu at zero energy
    s = [nonlineardc.nonlinear_dc_conductivity(h,term="qmd_intra",nk=nk,
            T=0.05,components="xyy") for nk in [250,500]]
    assert abs(s[0]-s[1])<1e-6*abs(s[1])
    assert abs(s[1]-1.16e-2)<0.01*1.16e-2
    labels = [a+b+c for a in "xy" for b in "xy" for c in "xy"]
    bcd = nonlineardc.nonlinear_dc_conductivity(h,term="bcd",nk=100,T=0.05,
            components=labels)
    for c in labels: assert abs(bcd[c])<1e-12


def _all_terms(h,**kw):
    return {t:_tensor(h,t,**kw) for t in TERMS}


def test_selection_rules():
    """PT makes Omega vanish at every k, so the dipole is zero while the
    Drude and quantum-metric terms are allowed; time reversal kills the
    Drude and every quantum-metric term and allows the dipole, which a C3
    axis kills again; with C3 the transverse hall piece vanishes as well and
    the other tau-even terms follow the two C3 patterns; inversion kills
    everything. Each zero is paired with a fixture where the same term is
    finite"""
    kw = dict(T=0.1)
    # PT antiferromagnet with strain
    t1 = _all_terms(_pt_antiferromagnet(),nk=120,**kw)
    t2 = _all_terms(_pt_antiferromagnet(),nk=240,**kw)
    for term in ["drude","qmd_hall","qmd_intra"]:
        big = np.max(np.abs(t2[term]))
        assert big>1e-4
        assert np.allclose(t1[term],t2[term],atol=1e-2*big) # converged
    assert np.max(np.abs(t2["bcd"]))<1e-12*np.max(np.abs(t2["drude"]))
    # time-reversal symmetric metal, with and without strain
    t1 = _all_terms(_gapped_honeycomb(s=0.15,mu=0.9),nk=120,**kw)
    t2 = _all_terms(_gapped_honeycomb(s=0.15,mu=0.9),nk=240,**kw)
    big = np.max(np.abs(t2["bcd"]))
    assert big>1e-4
    assert np.allclose(t1["bcd"],t2["bcd"],atol=1e-2*big)
    for term in ["drude","qmd_hall","qmd_disp","qmd_intra"]:
        assert np.max(np.abs(t2[term]))<1e-10*big
    t0 = _tensor(_gapped_honeycomb(mu=0.9),"bcd",nk=60,**kw) # C3: no dipole
    assert np.max(np.abs(t0))<1e-12*big
    # C3 metal with broken time reversal
    t = _all_terms(_gapped_honeycomb(haldane=0.1,mu=0.9),nk=60,**kw)
    big = max(np.max(np.abs(t[x])) for x in ["drude","qmd_disp","qmd_intra"])
    assert big>1e-4
    for term in ["bcd","qmd_hall"]: assert np.max(np.abs(t[term]))<1e-12*big
    for term in ["drude","qmd_disp","qmd_intra"]:
        s = t[term]
        x,y = 0,1
        assert max(abs(s[x,x,x]),abs(s[y,y,y]))>1e-3*big
        for (i,j,l) in [(x,y,y),(y,x,y),(y,y,x)]:
            assert abs(s[i,j,l]+s[x,x,x])<1e-10*big
        for (i,j,l) in [(y,x,x),(x,y,x),(x,x,y)]:
            assert abs(s[i,j,l]+s[y,y,y])<1e-10*big
    # inversion-symmetric bilayer
    t = _all_terms(_bilayer(),nk=40,**kw)
    for term in TERMS: assert np.max(np.abs(t[term]))<1e-12


def test_rotation_covariance():
    """Rotating the crystal by an angle that is not a symmetry rotates every
    term as a rank-3 tensor, sigma' = R R R sigma, on a spinless metal with
    no symmetry left (Haldane term, staggered mass and strain), where all
    five terms are finite at once"""
    h0 = _gapped_honeycomb(haldane=0.1,s=0.15,mu=0.9)
    theta = 0.37
    h1 = h0.copy()
    h1.geometry = sculpt.rotate(h0.geometry,theta)
    R = np.array([[np.cos(theta),np.sin(theta)], # sculpt.rotate turns the
                  [-np.sin(theta),np.cos(theta)]]) # crystal clockwise
    t0 = _all_terms(h0,nk=40,T=0.1)
    t1 = _all_terms(h1,nk=40,T=0.1)
    for term in TERMS:
        big = np.max(np.abs(t0[term]))
        assert big>1e-4
        rot = np.einsum("ai,bj,ck,ijk->abc",R,R,R,t0[term])
        assert np.allclose(t1[term],rot,atol=1e-10*big)


def test_ezawa_neel_angle_law():
    """Ezawa's law for the d-wave altermagnet with Rashba coupling,
    arXiv:2409.09241: at first order in J, sigma_xxx of the Drude and of
    the quantum-metric term goes as cos(Phi) for n = (sin Phi, cos Phi, 0),
    is exactly zero for n along x and along z (the mirror M_y T), is linear
    in J and flips sign with J. The deviation from cos(Phi) is a cos(3 Phi)
    harmonic of third order in J: at J = 0.05 it is 11 % of the amplitude
    for the quantum-metric term (2 % for the Drude term), so the law is
    tested at J = 0.01, and the J^2 scaling of the harmonic is tested too"""
    kw = dict(nk=100,T=0.1,components="xxx")
    def sig(term,**k): return nonlineardc.nonlinear_dc_conductivity(
            _ezawa_altermagnet(**k),term=term,**kw)
    phis = np.linspace(0.,2.*np.pi,8,endpoint=False)
    for term in ["drude","qmd"]:
        harm = []
        for J in [0.01,0.02]:
            s = np.array([sig(term,phi=p,J=J) for p in phis])
            amp = np.sum(s*np.cos(phis))/np.sum(np.cos(phis)**2)
            amp3 = np.sum(s*np.cos(3.*phis))/np.sum(np.cos(3.*phis)**2)
            assert abs(amp)>1e-6
            rest = s - amp*np.cos(phis) - amp3*np.cos(3.*phis)
            assert np.max(np.abs(rest))<1e-10*abs(amp) # odd harmonics only
            harm.append(amp3/amp)
            if J==0.01:
                assert np.sqrt(np.mean((s-amp*np.cos(phis))**2))<0.02*abs(amp)
                assert abs(sig(term,phi=np.pi/2.,J=J))<1e-10*abs(amp) # n || x
                assert abs(sig(term,n=[0.,0.,1.],J=J))<1e-10*abs(amp) # n || z
                s0 = sig(term,phi=0.,J=J)
                assert abs(sig(term,phi=0.,J=J/2.)-s0/2.)<0.02*abs(s0)
                assert abs(sig(term,phi=0.,J=-J)+s0)<1e-8*abs(s0) # odd in J
        assert abs(harm[1]/harm[0]-4.)<0.2*4. # cos(3 Phi) goes as J^3


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


def _rotated_tensors(h,ks,T,rng):
    """nonlineardc._batch_tensors on the k-points ks, in the eigenbasis of
    the diagonalization and in a basis rotated by a random unitary inside
    every degenerate multiplet; also the band-diagonal product
    sum w u^n_a u^n_bc in the two bases, and the number of multiplets"""
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,ks)
    tol = 1e-6*scale
    V,nblock = _random_block_unitary(es,tol,rng)
    Vc = np.conjugate(np.swapaxes(V,-1,-2))[:,None]
    h1r = Vc@h1@V[:,None]
    h2r = Vc[:,None]@h2@V[:,None,None]
    w = nonlineardc._weights(es,T)[0]
    out = []
    for (x1,x2) in [(h1,h2),(h1r,h2r)]:
        t = nonlineardc._batch_tensors(es,x1,x2,T,tol)
        b = nonlineardc._band_blocks(es,x1,x2,tol)
        t["diag"] = np.einsum("kn,kan,kbcn->abc",w,_diag(b["u"]),
                _diag(b["ubc"]))
        t["Omega_n"] = _diag(b["Omega"])
        out.append(t)
    return out[0],out[1],nblock


def test_block_trace_is_basis_independent():
    """A random unitary inside every degenerate multiplet leaves every term
    unchanged: the block traces do not depend on the basis the
    diagonalization picks inside a multiplet. On the Kramers-degenerate PT
    fixture the velocity is proportional to the identity inside each pair,
    so only the band-resolved curvature, not a product, sees the rotation;
    at the Rashba crossing of Ezawa's model at Gamma, put on the Fermi
    level, the velocity inside the pair is not, and the band-diagonal
    product sum w u^n_a u^n_bc does change while the block trace does not"""
    rng = np.random.default_rng(11)
    ks = uniform_mesh(2,16)
    t0,t1,nblock = _rotated_tensors(_pt_antiferromagnet(),ks,0.1,rng)
    assert nblock>=len(ks) # degenerate pairs at every k-point
    assert np.max(np.abs(t0["uu"]))>1e-3
    for key in ["uu","uGt","uG2","uOmega"]:
        assert np.allclose(t0[key],t1[key],atol=1e-10*np.max(np.abs(t0["uu"])))
    assert np.max(np.abs(t0["Omega_n"]-t1["Omega_n"]))>1e-3
    h = _ezawa_altermagnet(phi=0.4,J=0.3,mu=0.)
    egamma = np.linalg.eigvalsh(np.asarray(h.get_hk_gen()([0.,0.,0.])))
    assert abs(egamma[1]-egamma[0])<1e-12 # the crossing at Gamma
    h.shift_fermi(-egamma[0])
    t0,t1,nblock = _rotated_tensors(h,ks,0.1,rng)
    assert nblock>=1
    big = np.max(np.abs(t0["uu"]))
    assert big>1e-3
    for key in ["uu","uGt","uG2","uOmega"]:
        assert np.allclose(t0[key],t1[key],atol=1e-10*big)
    assert np.max(np.abs(t0["diag"]-t1["diag"]))>1e-3*big


def test_channels():
    """In a spin-conserving magnet the spin channel is (up - dn)/2 and the
    charge channel up + dn of the two spin-resolved runs, for every term;
    with spin-orbit coupling a spin channel is refused"""
    h = _fwave_zeeman()
    kw = dict(nk=40,T=0.1,components=["xxx","xyy","yxy","yyy"])
    out = {ch:nonlineardc.nonlinear_dc_conductivity(h,channel=ch,**kw)
           for ch in ["up","dn","spin","charge"]}
    big = max(abs(out[ch]["total"][c]) for ch in ["up","dn"]
              for c in kw["components"])
    for term in ["drude","bcd","qmd","total"]:
        for c in kw["components"]:
            up,dn = out["up"][term][c],out["dn"][term][c]
            assert abs(out["spin"][term][c]-(up-dn)/2.)<1e-12*big
            assert abs(out["charge"][term][c]-(up+dn))<1e-12*big
    assert max(abs(out["spin"]["drude"][c]) for c in kw["components"])>1e-4
    assert max(abs(out["charge"]["drude"][c]) for c in kw["components"])>1e-4
    with pytest.raises(ValueError,match="spin is not a good quantum number"):
        nonlineardc.nonlinear_dc_conductivity(_ezawa_altermagnet(),
                channel="spin",nk=8)


def test_tau_powers_and_total():
    """term="all" carries tau^2, tau^1 and tau^0 on the three terms, the
    qmd conventions are sums of the pieces, and total is their sum"""
    h = _gapped_honeycomb(haldane=0.1,s=0.15,mu=0.9)
    kw = dict(nk=24,T=0.1,components="xyy")
    a1 = nonlineardc.nonlinear_dc_conductivity(h,tau=1.,**kw)
    a2 = nonlineardc.nonlinear_dc_conductivity(h,tau=2.,**kw)
    assert abs(a2["drude"]-4.*a1["drude"])<1e-12*abs(a1["drude"])
    assert abs(a2["bcd"]-2.*a1["bcd"])<1e-12*abs(a1["bcd"])
    assert abs(a2["qmd"]-a1["qmd"])<1e-12*abs(a1["qmd"])
    assert abs(a2["total"]-a2["drude"]-a2["bcd"]-a2["qmd"])<1e-14
    p = {t:nonlineardc.nonlinear_dc_conductivity(h,term=t,**kw)
         for t in ["qmd_hall","qmd_disp","qmd_intra"]}
    for (conv,pieces) in [("shibata",["qmd_hall","qmd_intra"]),
                          ("gao",["qmd_hall"]),
                          ("kaplan",["qmd_hall","qmd_disp"])]:
        q = nonlineardc.nonlinear_dc_conductivity(h,term="qmd",qmd=conv,**kw)
        assert abs(q-sum(p[x] for x in pieces))<1e-14
    assert abs(nonlineardc.nonlinear_dc_conductivity(h,term="qmd_hall",
            nk=24,T=0.1,components="xxx"))<1e-14 # hall is transverse


def test_decoupled_layers_in_3d():
    """Layers of the tilted Dirac model stacked along a3 with no hopping
    between them: the 3D conductivity per unit volume is the 2D one per
    unit area divided by the layer spacing, for every term, and the
    components with a z index vanish"""
    m,t,c = 0.4,0.3,2.5
    h2 = _tilted_dirac(mu=0.6)
    g = geometry.cubic_lattice()
    g.a3 = np.array([0.,0.,c])
    h3 = g.get_hamiltonian(has_spin=True)
    dd = {R:np.asarray(v) for (R,v) in h2.get_multihopping().get_dict().items()}
    h3.set_multihopping(MultiHopping(dd))
    kw = dict(nk=40,T=0.1,components=["xyy","xxx","yxx","zxx","xzz"])
    s2 = nonlineardc.nonlinear_dc_conductivity(h2,**kw)
    s3 = nonlineardc.nonlinear_dc_conductivity(h3,**kw)
    for term in ["drude","bcd","qmd"]:
        big = max(abs(s2[term][x]) for x in ["xyy","xxx","yxx"])
        assert big>1e-4
        for x in ["xyy","xxx","yxx"]:
            assert abs(s3[term][x]-s2[term][x]/c)<1e-10*big
        for x in ["zxx","xzz"]: assert abs(s3[term][x])<1e-12*big


def test_options():
    """Unknown names list the accepted ones (derived from the registries),
    T=0 is refused naming the Fermi-surface weight, as are a Nambu
    Hamiltonian, a component of the wrong length and a dipole in 1D"""
    h = _tilted_dirac(mu=0.6)
    f = nonlineardc.nonlinear_dc_conductivity
    with pytest.raises(ValueError,match=", ".join(nonlineardc.get_term_names())):
        f(h,term="qmd_metric",nk=4)
    with pytest.raises(ValueError,
            match=", ".join(nonlineardc.get_qmd_conventions())):
        f(h,qmd="ulrich",nk=4)
    with pytest.raises(ValueError,
            match=", ".join(nonlineardc.get_channel_names())):
        f(h,channel="valley",nk=4)
    with pytest.raises(ValueError,match="Fermi-surface"):
        f(h,T=0.,nk=4)
    with pytest.raises(ValueError,match="xyy"):
        f(h,components="xy",nk=4)
    with pytest.raises(NotImplementedError,match="order=2"):
        f(h,order=4,nk=4)
    hn = _gapped_honeycomb(mu=0.9)
    hn.turn_spinful()
    hn.setup_nambu_spinor()
    with pytest.raises(NotImplementedError,match="Nambu"):
        f(hn,nk=4)
    h1 = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError,match="1D"):
        nonlineardc.berry_curvature_dipole(h1,nk=4)
    assert set(nonlineardc.get_term_names())=={"drude","bcd","qmd",
            "qmd_hall","qmd_disp","qmd_intra","all"}
