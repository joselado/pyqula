"""Laurent coefficients of the dc limit of the general nonlinear recursion
(src/pyqula/conductivitytk/nonlineardcfit.py), extracted on a Cauchy
contour in the complex broadening, against the tau-decomposed dc formulas
of nonlineardc.py and the Fermi-sea Drude term of nonlineardrude.py.

With the adiabatic switching every intraband pole of the recursion is
p i delta, so tau^n maps to 1/(n! delta^n) and a one-pole term to
1/delta: A_-2 = sigma^ND/(2 tau^2) and A_-1 = sigma^BCD/tau at second
order, A_-3 = sigma^ND3/(6 tau^3) at third. The two sides share nothing
but the Hamiltonian: the recursion never forms a band velocity, a Berry
curvature or a band derivative. A_0 matches none of the quantum-metric
conventions and is only recorded (future_development/nonlinear_response.md,
"The dc limit and its Laurent structure in delta").
"""
import functools

import numpy as np
import pytest

from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import nonlineardc
from pyqula.conductivitytk import nonlineardcfit
from pyqula.conductivitytk import nonlineardrude


s0 = np.eye(2,dtype=np.complex128)
sx = np.array([[0.,1.],[1.,0.]],dtype=np.complex128)
sy = np.array([[0.,-1j],[1j,0.]],dtype=np.complex128)
sz = np.array([[1.,0.],[0.,-1.]],dtype=np.complex128)

T = 0.05
COMPS2 = ["xxx","yxx","xyy","xxy"]
COMPS3 = ["xxxx","xxyy"]


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


def _tilted_dirac(mu=0.6,m=0.4,t=0.3):
    """Shibata's tilted massive Dirac lattice model, arXiv:2606.22359,
    H = t sin kx s0 + sin kx sx + sin ky sy + [m + 2 - cos kx - cos ky] sz:
    gap 2m = 0.8 at Gamma, finite Berry curvature dipole and metric"""
    h = _two_orbital_square((m+2.)*sz,-0.5j*t*s0 - 0.5j*sx - 0.5*sz,
                            -0.5j*sy - 0.5*sz)
    _check_bloch(h,lambda kx,ky: t*np.sin(kx)*s0 + np.sin(kx)*sx
            + np.sin(ky)*sy + (m + 2. - np.cos(kx) - np.cos(ky))*sz)
    h.shift_fermi(-mu)
    return h


def _shibata_real(mu=0.5):
    """Shibata's real two-band model, arXiv:2606.22359,
    H = t0 sin kx s0 + (m + tx cos kx + ty cos ky) sx + lam sin ky sz.
    Real, so the Berry curvature is zero at every k; gap 1.0"""
    m,tx,ty,lam,t0 = 2.2,1.0,0.7,0.9,0.5
    h = _two_orbital_square(m*sx,tx/2.*sx - 0.5j*t0*s0,
                            ty/2.*sx - 0.5j*lam*sz)
    _check_bloch(h,lambda kx,ky: t0*np.sin(kx)*s0
            + (m + tx*np.cos(kx) + ty*np.cos(ky))*sx + lam*np.sin(ky)*sz)
    h.shift_fermi(-mu)
    return h


def _doubled_tilted_dirac(mu=0.6,m=0.4,t=0.3):
    """Two copies of the tilted Dirac model on two sites at the same
    position, H4(k) = U [1 x H(k)] U^dagger with a fixed random unitary U:
    every band is doubly degenerate at every k, as in a spinful model with
    inversion and time reversal, and the rotation mixes the two copies so
    that no basis vector belongs to one of them"""
    g = geometry.square_lattice()
    g.r = np.array([[0.,0.,0.],[0.,0.,0.]])
    g.r2xyz()
    h = g.get_hamiltonian(has_spin=True)
    X = np.random.default_rng(7).normal(size=(4,4,2))
    U = np.linalg.qr(X[:,:,0] + 1j*X[:,:,1])[0]
    big = lambda M: U@np.kron(s0,M)@np.conjugate(U).T
    tx = -0.5j*t*s0 - 0.5j*sx - 0.5*sz
    ty = -0.5j*sy - 0.5*sz
    dd = {(0,0,0):big((m+2.)*sz),(1,0,0):big(tx),
          (-1,0,0):big(np.conjugate(tx).T),(0,1,0):big(ty),
          (0,-1,0):big(np.conjugate(ty).T)}
    h.set_multihopping(MultiHopping(dd))
    _check_bloch(h,lambda kx,ky: big(t*np.sin(kx)*s0 + np.sin(kx)*sx
            + np.sin(ky)*sy + (m + 2. - np.cos(kx) - np.cos(ky))*sz))
    h.shift_fermi(-mu)
    return h


@functools.lru_cache(maxsize=None)
def _dirac_coefficients(nk,radius=0.12):
    """Contour coefficients of the second-order recursion on the tilted
    Dirac model, powers -3..0"""
    return nonlineardcfit.laurent_coefficients(_tilted_dirac(),order=2,
            radius=radius,npoints=24,nk=nk,T=T,components=COMPS2)


@functools.lru_cache(maxsize=None)
def _real_coefficients(nk,radius=0.1):
    """Contour coefficients of the third-order recursion on the real
    two-band model, powers -4..0"""
    return nonlineardcfit.laurent_coefficients(_shibata_real(),order=3,
            radius=radius,npoints=24,nk=nk,T=T,components=COMPS3)


def _coefficient(powers,A,k):
    """A_k out of the array returned for one component"""
    return A[list(powers).index(k)]


def _scale(A):
    """max_k |A_k| over every component, the reference of the zeros"""
    return max(float(np.max(np.abs(a))) for a in A.values())


def test_second_order_contour_leading_coefficients():
    """A_-2 = sigma^ND/(2 tau^2) and A_-1 = sigma^BCD/tau on the tilted
    Dirac model, for two longitudinal and two transverse components, with
    A_-3 and every imaginary part at rounding. Converged: nk=150 and
    nk=300 give the same coefficients, and so does a smaller contour"""
    h = _tilted_dirac()
    p,A = _dirac_coefficients(150)
    assert list(p)==[-3,-2,-1,0]
    scale = _scale(A)
    nd = nonlineardc.nonlinear_dc_conductivity(h,term="drude",tau=1.,nk=150,
            T=T,components=COMPS2)
    bcd = nonlineardc.nonlinear_dc_conductivity(h,term="bcd",tau=1.,nk=150,
            T=T,components=COMPS2)
    assert abs(nd["xxx"])>1e-3 and abs(nd["xyy"])>1e-3 # Drude present
    assert abs(bcd["yxx"])>1e-3 and abs(bcd["xxy"])>1e-3 # BCD present
    for c in COMPS2:
        a = A[c]
        assert abs(_coefficient(p,a,-2)-nd[c]/2.)<1e-6*scale
        assert abs(_coefficient(p,a,-1)-bcd[c])<1e-6*scale
        assert abs(_coefficient(p,a,-3))<1e-12*scale # no pole beyond n
        assert np.max(np.abs(a.imag))<1e-12*scale # real for a real field
    p2,A2 = _dirac_coefficients(300)
    p3,A3 = _dirac_coefficients(150,radius=0.08)
    for c in COMPS2:
        assert np.max(np.abs(A[c]-A2[c]))<1e-8*scale # converged in nk
        assert np.max(np.abs(A[c]-A3[c]))<1e-10*scale # free of the radius


def test_symmetry_zeros_against_max_coefficient():
    """The zeros of the tau-decomposed terms seen by the recursion: the
    Berry curvature dipole of xxx is zero because Omega_xx = 0, and the
    Drude term of yxx because eps(kx,ky) = eps(kx,-ky). The recursion
    builds neither, so its A_-1 of xxx and A_-2 of yxx are zero only if it
    is right; the partner coefficients are finite"""
    p,A = _dirac_coefficients(150)
    scale = _scale(A)
    assert abs(_coefficient(p,A["xxx"],-1))<1e-12*scale
    assert abs(_coefficient(p,A["yxx"],-2))<1e-12*scale
    assert abs(_coefficient(p,A["xxx"],-2))>1e-1*scale
    assert abs(_coefficient(p,A["yxx"],-1))>1e-1*scale


def test_third_order_contour_drude():
    """A_-3 = sigma^ND3/(6 tau^3) on the real two-band model, with the
    Fermi-sea Drude term sigma^ND3 = tau^3 <f d_a d_b d_c d_d eps> of
    nonlineardrude (the prefactor (-1)^(l+1) tau^l at l = 3), its fourth
    band derivative from the Rayleigh-Schroedinger recursion of
    nonlineardrude._Channel on the whole Hamiltonian, since the orbital
    index mixes the two blocks that nonlinear_drude_conductivity would take
    as spin channels. A_-4 and the imaginary parts are at rounding, and
    the coefficient is converged in nk"""
    h = _shibata_real()
    nk = 150
    p,A = _real_coefficients(nk)
    assert list(p)==[-4,-3,-2,-1,0]
    scale = _scale(A)
    ks = kubo._kmesh(h,nk)
    chan = nonlineardrude._Channel(h.get_multicell(),ks)
    f = kubo._fermi(chan.es,T)
    norm = 1./(len(ks)*nonlineardrude._cell_volume(h))
    p2,A2 = _real_coefficients(2*nk)
    for c in COMPS3:
        axes = [kubo._index(x) for x in c]
        nd3 = np.sum(f*chan.derivative(axes))*norm # tau = 1
        assert abs(nd3)>1e-3
        a = A[c]
        assert abs(_coefficient(p,a,-3)-nd3/6.)<1e-6*abs(nd3)
        assert abs(_coefficient(p,a,-4))<1e-12*scale
        assert np.max(np.abs(a.imag))<1e-12*scale
        assert abs(_coefficient(p,a,-3)-_coefficient(p2,A2[c],-3))<1e-6*abs(nd3)


def test_a0_is_recorded():
    """The delta^0 coefficient against the three quantum-metric
    conventions, printed for the record: the adiabatic switching gives a
    fourth value, so nothing is asserted on a convention"""
    nk = 150
    rows = [("tilted Dirac",_tilted_dirac(),_dirac_coefficients(nk),
             ["xxx","xyy"])]
    p,A = nonlineardcfit.laurent_coefficients(_shibata_real(),order=2,
            radius=0.12,nk=nk,T=T,components=["xyy"])
    rows.append(("real two-band",_shibata_real(),(p,A),["xyy"]))
    for (name,h,(p,A),comps) in rows:
        for c in comps:
            a0 = _coefficient(p,A[c],0)
            qmd = {q:nonlineardc.nonlinear_dc_conductivity(h,term="qmd",qmd=q,
                    nk=nk,T=T,components=c)
                   for q in nonlineardc.get_qmd_conventions()}
            assert np.isfinite(a0)
            assert all(np.isfinite(v) for v in qmd.values())
            print(name,c,"A_0 = %.3e"%a0.real,
                  " ".join("%s %.3e"%(q,v) for (q,v) in qmd.items()))


def test_convergence_radius_and_options():
    """The nearest pole off the origin is the gap over the order: the
    tilted Dirac model has its minimum gap 2m = 0.8 at Gamma, on the mesh,
    so 0.4 at second order and 0.8/3 at third. A contour that encloses it
    or aliases a coefficient is refused"""
    h = _tilted_dirac()
    assert abs(nonlineardcfit.convergence_radius(h,order=2,nk=20)-0.4)<1e-12
    assert abs(nonlineardcfit.convergence_radius(h,order=3,nk=20)-0.8/3)<1e-12
    with pytest.raises(ValueError,match="encloses a pole"):
        nonlineardcfit.laurent_coefficients(h,order=2,radius=0.5,nk=20,T=T,
                components="xxx")
    with pytest.raises(ValueError,match="aliases"):
        nonlineardcfit.laurent_coefficients(h,order=2,radius=0.1,npoints=4,
                nk=20,T=T,components="xxx",powers=[0,2])
    with pytest.raises(ValueError,match="positive"):
        nonlineardcfit.laurent_coefficients(h,order=2,radius=-0.1,nk=20,T=T,
                components="xxx")
    with pytest.raises(ValueError,match="positive integer"):
        nonlineardcfit.laurent_coefficients(h,order=0,nk=20,T=T)
    with pytest.raises(ValueError,match="Cartesian labels"):
        nonlineardcfit.laurent_coefficients(h,order=2,radius=0.1,nk=20,T=T,
                components="xx")
    # the default radius gives the same coefficients as an explicit one
    p,a = nonlineardcfit.laurent_coefficients(h,order=2,nk=20,T=T,
            components="xxx")
    p2,a2 = nonlineardcfit.laurent_coefficients(h,order=2,radius=0.08,nk=20,
            T=T,components="xxx")
    assert np.max(np.abs(a-a2))<1e-10*np.max(np.abs(a))


def test_degenerate_multiplets():
    """Doubly degenerate bands, the usual spinful input: the zero gap
    inside a multiplet feeds the pole at the origin, not a pole off it, so
    the convergence radius is still the gap 0.8 over the order and the
    coefficients of the two copies are twice those of one"""
    h2 = _tilted_dirac()
    h4 = _doubled_tilted_dirac()
    for order in (2,3):
        assert abs(nonlineardcfit.convergence_radius(h4,order=order,nk=20)
                - nonlineardcfit.convergence_radius(h2,order=order,nk=20))<1e-12
    assert abs(nonlineardcfit.convergence_radius(h4,order=2,nk=20)-0.4)<1e-12
    for radius in (0.12,None): # explicit radius and the default one
        p,A = nonlineardcfit.laurent_coefficients(h2,order=2,radius=0.12,
                nk=20,T=T,components=COMPS2)
        p4,A4 = nonlineardcfit.laurent_coefficients(h4,order=2,radius=radius,
                nk=20,T=T,components=COMPS2)
        assert list(p4)==list(p)
        scale = _scale(A)
        assert scale>1e-3
        for c in COMPS2:
            assert np.max(np.abs(A4[c]-2.*A[c]))<1e-12*scale
