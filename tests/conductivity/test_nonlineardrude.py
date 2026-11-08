"""Frequency dependence of the nonlinear Drude conductivity.

The static limit and the tau^l scaling are pinned in
tests/nonlinearconductivity/test_nonlinear_drude.py. What is pinned here is
the finite-frequency prefactor: for a field E(t) = E0 cos(w t) the l-th
order current at the harmonic l w comes from the recursive solution of the
relaxation-time Boltzmann equation,

  (1/tau - i p w) f_p = -q E d_k f_{p-1},  p = 1..l,

so the prefactor is prod_{p=1..l} 1/(1/tau - i p w), in the exp(-i w t)
convention of kubo.py and nonlinearkernel.py. The reference below does not
use that recursion: it solves the Boltzmann equation in the time domain,
along characteristics, at a small finite field, and takes the harmonics of
the current."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.conductivitytk import nonlineardrude
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import nonlinearkernel


def _chain():
    """Spinful chain with a complex second-neighbor hopping, which breaks
    k -> -k so that the even orders do not vanish. One band per spin, the
    two spins degenerate."""
    h = geometry.chain().get_hamiltonian(has_spin=True)
    hd = {R:np.array(m.todense() if hasattr(m,"todense") else m,
        dtype=complex) for (R,m) in h.get_hopping_dict().items()}
    t2 = 0.4*np.exp(0.7j)*np.eye(2)
    hd[(2,0,0)] = t2
    hd[(-2,0,0)] = np.conjugate(t2).T
    h.set_multihopping(MultiHopping(hd))
    h.shift_fermi(0.6)
    return h


def _time_domain_harmonics(h,T,tau,omega,E0,nk=400,ns=120,nt=16):
    """Charge current of the relaxation-time Boltzmann equation under
    E(t) = E0 cos(w t), q = -1, a = 1, at the harmonics 1w, 2w and 3w.

    The exact solution is f(k,t) = int_0^inf ds/tau e^{-s/tau}
    f0(eps(k - q [A(t) - A(t-s)])) with A(t) = E0 sin(w t)/w, the integral
    over s done with Gauss-Laguerre nodes. Returns c_l, the coefficient of
    e^{-i l w t} in j(t)."""
    hk = h.get_hk_gen()
    ks = np.arange(nk)/nk
    eps = np.array([np.linalg.eigvalsh(np.array(hk([k,0.,0.])))[0]
        for k in ks])
    c = np.fft.fft(eps)/nk
    m = np.fft.fftfreq(nk,d=1./nk)
    def band(dk): return np.real(np.fft.ifft(c*np.exp(1j*m*dk))*nk)
    vel = np.real(np.fft.ifft(1j*m*c)*nk) # d eps/dk on the mesh
    x,wx = np.polynomial.laguerre.laggauss(ns)
    ss = tau*x
    q = -1.
    def A(t): return E0*np.sin(omega*t)/omega
    ts = 2.*np.pi/omega*np.arange(nt)/nt
    J = []
    for t in ts:
        f = np.zeros(nk)
        for (s,w) in zip(ss,wx):
            f += w/(1.+np.exp(band(-q*(A(t)-A(t-s)))/T))
        J.append(2.*q*np.mean(vel*f)) # two spins, cell length 1
    J = np.array(J)
    return [np.mean(J*np.exp(1j*l*omega*ts)) for l in (1,2,3)]


def test_time_domain_reference_is_converged():
    """The quadrature of the reference is converged far below the tolerance
    of the test that uses it"""
    h = _chain()
    kw = dict(T=0.1,tau=4.,omega=0.5,E0=5e-3)
    c1 = _time_domain_harmonics(h,ns=120,nt=16,**kw)
    c2 = _time_domain_harmonics(h,ns=160,nt=32,**kw)
    for (a,b) in zip(c1,c2): assert abs(a-b)<1e-4*abs(b)


@pytest.mark.parametrize("l",[1,2,3])
def test_harmonics_match_time_domain_boltzmann(l):
    """sigma at order l and frequency w is the amplitude of the current at
    l w: the coefficient of e^{-i l w t} is sigma (E0/2)^l. The finite field
    corrects this at relative order (E0 tau)^2 times ratios of the
    derivatives, 2e-4 here, while the old 1/(i w + 1/tau)^l was off
    by more than 50% at l = 2 and 3 and was the complex conjugate at l = 1"""
    h = _chain()
    T,tau,omega,E0 = 0.1,4.,0.5,5e-3
    cs = _time_domain_harmonics(h,T,tau,omega,E0)
    ref = cs[l-1]*(2./E0)**l
    s = nonlineardrude.nonlinear_drude_conductivity(h,field="x"*l,
            current="x",channel="charge",nk=400,T=T,tau=tau,omega=omega)
    assert abs(s-ref)<2e-3*abs(ref)


def test_first_order_matches_kubo():
    """At l = 1 the nonlinear Drude conductivity is the intraband
    (Drude) part of the optical conductivity, so at finite frequency it has
    to agree with kubo.optical_conductivity(interband=False) and with
    nonlinear_conductivity(order=1), both with the broadening delta =
    1/tau. This pins the time convention: the two references use
    exp(-i w t), and the opposite convention gives the complex conjugate"""
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.shift_fermi(1.0)
    T,nk,tau,omega = 0.1,120,5.,0.5
    s = nonlineardrude.nonlinear_drude_conductivity(h,field="x",current="x",
            channel="charge",nk=nk,T=T,tau=tau,omega=omega)
    assert abs(np.imag(s))>0.1*abs(s)
    (w,sk) = kubo.optical_conductivity(h,energies=[omega],nk=nk,T=T,
            delta=1./tau,interband=False)
    assert abs(s-sk[0,0,0])<1e-6*abs(s)
    (w,sn) = nonlinearkernel.nonlinear_conductivity(h,order=1,
            energies=[omega],nk=nk,T=T,delta=1./tau,components="xx")
    assert abs(s-sn[0])<1e-6*abs(s)


def test_components_and_orders_use_the_same_prefactor():
    """nonlinear_drude_components and nonlinear_drude_orders go through the
    same prefactor as nonlinear_drude_conductivity"""
    h = _chain()
    kw = dict(channel="charge",nk=200,T=0.1,tau=4.,omega=0.5)
    for l in (1,2,3):
        comps = nonlineardrude.nonlinear_drude_components(h,l,**kw)
        s = nonlineardrude.nonlinear_drude_conductivity(h,field="x"*l,
                current="x",**kw)
        assert abs(comps["x"*l+";x"]-s)<1e-12*abs(s)
    orders = nonlineardrude.nonlinear_drude_orders(h,lmax=3,**kw)
    for l in (1,2,3):
        s = nonlineardrude.nonlinear_drude_conductivity(h,field="x"*l,
                current="x",**kw)
        assert np.isclose(orders[l],abs(s),rtol=1e-12)
