"""Quantized circular photogalvanic effect of a lattice Weyl semimetal
(the circular injection of src/pyqula/conductivitytk/photocurrent.py).

The one absolute, convention-free benchmark of the injection rate eta is the
result of de Juan, Grushin, Morimoto and Moore (arXiv:1611.05887): when the
two Weyl nodes of opposite chirality sit at different energies, there is a
frequency window where only one node absorbs (the other one is Pauli
blocked), and there the trace of the circular injection tensor is the
monopole charge C of that node times a universal constant.

The constant in the convention of photocurrent.py. The paper writes
dj_i/dt = beta_ij [E(w) x E(w)^*]_j for E(t) = E(w) exp(-iwt) + c.c., and
finds Tr beta = i pi e^3 C/h^2. pyqula writes dJ_a/dt = eta_abc E0_b
E0_c^* for E(t) = Re E0 exp(-iwt), so E(w) = E0/2, and the bc-antisymmetric
part of eta is i eta^circ_abc with eta^circ_abc = Im eta_abc. Matching the
two rates gives beta_ij/4 = (i/2) eps_jbc eta^circ_ibc, so

  Tr beta = 4i S,   S = eta^circ_xyz + eta^circ_yzx + eta^circ_zxy,

and with hbar = 1 (h = 2 pi) the quantized trace is

  S = q^3 C/(16 pi).

The same number follows from the closed form of injection_rate directly,
without the paper: for two bands at a cold node, sum_bc eps_dbc r^b_mn
r^c_nm = i Omega^d_n, the Berry curvature of the occupied band n (Omega =
curl i<n|grad n>), so 2S = (pi q^3/2) int d^3k/(2pi)^3 grad(E_m - E_n).Omega_n
delta(w - E_m + E_n) = (pi q^3/2)(2 pi C_n)/(2 pi)^3, the Berry flux of the
occupied band out of the resonant surface. For H = d(k).sigma with d =
M k near the node, C_n = sign det M, the chirality. This is the sign the
calculation gives: with q = -1, S = -chi/(16 pi) at the absorbing node,
chi = sign det(d d_b/d k_a). Because the paper's sign of e and its own
circular example are not consistent with each other, only the magnitude is
taken from the paper; the sign is tested as the exact flip under a mirror,
and the monopole charge independently as the Chern number of a 2D slice of
the zone between the nodes. With get_chern's orientation the slice kz = 0
gives C = chi of the node at kz = -pi/2, so also S = q^3 C/(16 pi).

The mesh. At a single frequency the Lorentzian of width delta is a shell of
radial width delta/2 in k, which a uniform mesh does not resolve. So the
trace is averaged over the frequency window with a smooth (Hann) weight and
a frequency step delta/2: the integrand on the mesh is then the smooth
weight at E_m - E_n, and the average converges as a smooth integral. The
Lorentzian tails outside the window leave a bias of order delta/(pi w), so
delta is small (0.004). The weighted mean of 16 pi |S| over
w in [0.15, 0.85] was 0.9768, 0.9932 and 0.9924 at nk = 30, 60 and 120
(point by point at delta = 0.02 it was 0.87 to 1.18 at nk = 80, and still
off by 4 to 6 % at nk = 160, the tails); with both nodes absorbing,
w in [1.1, 1.9], it was 0.0015 at nk = 30, 60 and 120.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import multihopping
from pyqula.conductivitytk import photocurrent
from pyqula.conductivitytk import shg


_s0 = np.eye(2,dtype=np.complex128)
_sx = np.array([[0.,1.],[1.,0.]],dtype=np.complex128)
_sy = np.array([[0.,-1j],[1j,0.]],dtype=np.complex128)
_sz = np.array([[1.,0.],[0.,-1.]],dtype=np.complex128)

_cyclic = ["xyz","yzx","zxy"] # the components of the trace S


def _weyl(gamma=0.25,chirality=1,inversion=False):
    """Two-band Weyl semimetal on the cubic lattice (the two components are
    an orbital pseudospin, has_spin=True only sizes the cell to 2x2):

      H(k) = d(k).sigma + eps(k) sigma_0
      d = -(c sin kx, sin ky, cos kx + cos ky + cos kz - 2)

    with two nodes at k = (0,0,-pi/2) and (0,0,pi/2), of velocity one and
    no tilt, and chirality sign det(d d_b/d k_a) = -c and +c. The term
    eps = gamma (1 + sin kz) puts the first node at zero energy and the
    second one at 2 gamma, breaking inversion and every mirror that maps
    one node into the other. So at the chemical potential zero, the first
    node absorbs at every w > 0, the second one is Pauli blocked for w <
    4 gamma (up to the curvature of eps, about 0.94 at gamma = 0.25), and
    the two resonant surfaces merge at w = 2. c = -1 is the mirror image
    kx -> -kx. With inversion=True eps = gamma (1 + cos kz) instead, even
    in k, so sigma_z H(k) sigma_z = H(-k)."""
    c = chirality
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=True)
    tz = 0.5*gamma*_s0 if inversion else gamma*_s0/(2j) # the eps along z
    hd = {(0,0,0): 2.*_sz + gamma*_s0,
          (1,0,0): -0.5*_sz - c*_sx/(2j), (-1,0,0): -0.5*_sz + c*_sx/(2j),
          (0,1,0): -0.5*_sz - _sy/(2j), (0,-1,0): -0.5*_sz + _sy/(2j),
          (0,0,1): -0.5*_sz + tz, (0,0,-1): -0.5*_sz + np.conj(tz)}
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _slice_between_nodes(chirality=1):
    """The 2D slice kz = 0 of _weyl, a square-lattice Chern insulator whose
    Chern number is the monopole charge of one node (the slice kz = pi is
    trivial, |d_z| >= 1 there). The constant eps does not enter"""
    c = chirality
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    hd = {(0,0,0): 1.*_sz,
          (1,0,0): -0.5*_sz - c*_sx/(2j), (-1,0,0): -0.5*_sz + c*_sx/(2j),
          (0,1,0): -0.5*_sz - _sy/(2j), (0,-1,0): -0.5*_sz + _sy/(2j)}
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _pauli_vector(h,k):
    """d(k) of a two-band Bloch Hamiltonian, k reduced"""
    m = np.asarray(h.get_hk_gen()(k))
    return np.array([np.trace(m@s).real/2. for s in (_sx,_sy,_sz)])


def _chirality(h,k0,dk=1e-5):
    """sign det(d d_b/d K_a) at the node k0 (reduced), K = 2 pi k on the
    cubic lattice of unit spacing, by central differences"""
    jac = np.zeros((3,3))
    for a in range(3):
        e = np.zeros(3) ; e[a] = dk
        jac[a] = (_pauli_vector(h,k0+e) - _pauli_vector(h,k0-e))/(2.*np.pi*2.*dk)
    return int(np.sign(np.linalg.det(jac)))


def _hann_mean(h,window,nk,delta):
    """Hann-weighted mean of S(w) = sum of the cyclic Im eta_abc over the
    frequency window, sampled every delta/2"""
    (a,b) = window
    ws = np.arange(a,b+1e-12,delta/2.)
    (w,eta) = photocurrent.injection_current(h,energies=ws,nk=nk,
            delta=delta,components=_cyclic)
    S = sum(eta[c].imag for c in _cyclic)
    weight = np.sin(np.pi*(ws - a)/(b - a))**2
    return np.sum(weight*S)/np.sum(weight)


_delta = 0.004
_one_node = (0.15,0.85) # only the node at zero energy absorbs
_two_nodes = (1.1,1.9) # both nodes absorb, before the surfaces merge


@pytest.mark.slow
def test_cpge_trace_is_quantized():
    """In the window where one node absorbs, 16 pi |S| is the monopole
    charge of that node, |C| = 1 from the Chern number of a slice of the
    zone, to 2 % at a converged nk (60 and its double agree to 0.5 %),
    with the sign S = q^3 chi/(16 pi), chi the chirality of the absorbing
    node; the mirror image (chirality flipped) gives exactly -S;
    with both nodes absorbing the trace is back near zero"""
    q3 = shg._charge**3
    h = _weyl()
    chi = _chirality(h,np.array([0.,0.,-0.25])) # the node at zero energy
    assert chi==-_chirality(h,np.array([0.,0.,0.25])) # opposite partner
    C = _slice_between_nodes().get_chern(nk=40,write=False)
    assert abs(abs(C) - 1.)<1e-3 # a unit monopole charge
    assert round(C)==chi # get_chern of the slice is the chirality of the node below it
    quantum = q3*chi/(16.*np.pi)
    S60 = _hann_mean(h,_one_node,60,_delta)
    S120 = _hann_mean(h,_one_node,120,_delta)
    assert abs(S120 - S60)<5e-3*abs(S120) # converged in nk
    assert abs(S120 - quantum)<0.02*abs(quantum)
    assert abs(16.*np.pi*abs(S120) - abs(C))<0.02
    # the mirror image kx -> -kx flips the chirality and, on the mirror
    # symmetric Gamma-centered mesh, the trace exactly
    hm = _weyl(chirality=-1)
    assert _chirality(hm,np.array([0.,0.,-0.25]))==-chi
    Sm = _hann_mean(hm,_one_node,60,_delta)
    assert abs(Sm + S60)<1e-10*abs(S60)
    # both nodes absorb: the two Berry fluxes cancel
    S2 = _hann_mean(h,_two_nodes,60,_delta)
    assert abs(S2)<0.02*abs(quantum)


@pytest.mark.slow
def test_cpge_vanishes_with_inversion():
    """With an inversion-even eps every component of eta, linear and
    circular, vanishes to rounding on the inversion-symmetric mesh, while
    the inversion-broken model gives a window-averaged trace of the size
    of the quantum on the same mesh, so the zero is the symmetry and not a
    dead code path"""
    ws = np.linspace(0.3,0.7,5)
    (w,eta) = photocurrent.injection_current(_weyl(inversion=True),
            energies=ws,nk=20,delta=0.05)
    assert max(np.abs(v).max() for v in eta.values())<1e-10
    S = _hann_mean(_weyl(),_one_node,20,_delta)
    assert 16.*np.pi*abs(S)>0.5
