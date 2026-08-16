"""Frequency-dependent (optical) conductivity tensor sigma_ab(omega) of a
periodic tight-binding Hamiltonian, in the Kubo-Greenwood formalism.

Formula
-------
The implemented expression is the standard sum-over-states Kubo-Greenwood
formula, in the form used by Wannier90's postw90 "berry" module
(berry_task = kubo, Yates, Wang, Vanderbilt & Souza, PRB 75, 195121
(2007)) and by WannierBerri (Tsirkin, npj Comput. Mater. 7, 33 (2021)):

  sigma_ab(hbar w) = (i e^2 hbar)/(N_k V_cell) sum_k sum_{n,m}
        (f_nk - f_mk)/(E_mk - E_nk)
        * <n|v_a|m><m|v_b|n> / (hbar w + i eta - (E_mk - E_nk))

with v_a = (1/hbar) dH/dK_a the velocity operator (K the Cartesian crystal
momentum), f the Fermi function at temperature T, and eta = delta a
phenomenological (Lorentzian) broadening. It is the same expression given
in the Kubo-Bastin/Chebyshev literature (Garcia, Covaci & Rappoport, PRL
114, 116602 (2015); KITE, Joao et al., R. Soc. Open Sci. 7, 191809 (2020),
arXiv:1910.05194), specialized to a clean periodic crystal where the
Chebyshev machinery is unnecessary and the Lehmann representation can be
summed directly.

The n = m terms (and, more generally, all pairs with E_m - E_n below the
degeneracy tolerance) are kept: there the occupation factor
(f_n - f_m)/(E_m - E_n) goes to its limit -df/dE(E_n), and the formula
reduces to the intraband/Drude response

  sigma_ab^intra(w) = i D_ab/(w + i eta),
  D_ab = (1/(N_k V_cell)) sum_{k,n} (-df/dE)(E_nk) v^a_nn v^b_nn

whose real part is a Lorentzian of weight pi*D_ab centered at w = 0.
Treating the degenerate pairs by that limit (rather than by the raw
quotient) is also what keeps a spin-degenerate pair or a band crossing
from producing 0/0, so interband and intraband come out of a single sum.

Velocity operator (gauge)
-------------------------
The velocity operator is the current operator of the Peierls-substituted
tight-binding model,

  v_alpha = i [H, r_alpha],   v_alpha,ij(k) = i sum_R t_ij(R) d_ij(R)_alpha
                                                 exp(2 pi i k.R)

with the *full* bond vector d_ij(R) = R + r_j - r_i, i.e. the position
operator taken diagonal in the orbital basis (the standard tight-binding
approximation). pyqula builds H(k) in the lattice gauge (Wannier90's
convention I), whose Bloch phase carries only the lattice vector R, so
dH/dK alone is *not* the velocity: it drops every intracell bond. The
missing piece is exactly a commutator with the position operator,

  v_alpha = dH/dK_alpha + i [H(k), r_alpha]

(elementwise i H_ij (r_j - r_i)_alpha), which is Wannier90/PythTB's
"atomic" gauge (convention II) for the velocity matrix elements. The
correction leaves the diagonal (band velocity) elements, and hence the
Drude weight, untouched, but it is essential for the interband response:
without it the honeycomb lattice -- whose two sites sit at +-x/2, so that
one of the three nearest-neighbour bonds is intracell -- comes out with
sigma_xx != sigma_yy, in open violation of its C3 symmetry, and misses
the universal optical conductivity of graphene.

Nambu (superconducting) Hamiltonians are rejected: there the charge
current is not i[H,r] (the hole block carries the opposite charge), so
this formula does not apply as written.

Units and conventions
---------------------
* hbar = e = 1. Energies (and hbar*omega) are in the Hamiltonian's own
  units (i.e. hopping units), lengths in the geometry's units (i.e.
  lattice-constant units, taken from the actual lattice vectors g.a1,
  g.a2 -- so a rescaled geometry does change the answer, as it must).
* sigma is then returned in units of e^2/hbar times a^(2-d) with d the
  dimensionality: e^2/hbar (a sheet conductance) in 2D, e^2*a/hbar in 1D.
  Note e^2/h = e^2/(2 pi hbar), so a value sigma_xy = 1/(2 pi) here is one
  conductance quantum e^2/h.
* The chemical potential is zero, as everywhere else in pyqula: states
  with E < 0 are occupied (use h.shift_fermi to move the Fermi level).
* Cartesian components: the returned tensor is always 3x3 (x,y,z), built
  from the true lattice vectors, so components along non-periodic
  directions come out identically zero.
* Sign/branch: the retarded (causal) branch omega -> omega + i*eta is
  used, so sigma(omega) is analytic in the upper half plane; Re sigma_aa
  >= 0 is dissipative absorption. With this convention the DC limit of the
  antisymmetric part is the intrinsic anomalous Hall conductivity in the
  standard form sigma_xy = -(e^2/hbar) (1/(2 pi)^2) int d^2k Omega_xy
  (Wang, Yates, Souza & Vanderbilt, PRB 74, 195118 (2006)), i.e.
  sigma_xy(omega -> 0) = -C e^2/h = -C/(2 pi) for a Chern insulator of
  Chern number C. That sign is measured against pyqula's own, independent
  Fukui-Hatsugai-Suzuki implementation (h.get_chern) in
  tests/conductivity/test_optical_conductivity.py -- with the Chern number
  as returned by h.get_chern(), sigma_xy(0)*2*pi = -C.

f-sum rule
----------
Integrating the dissipative part over the whole real frequency axis gives
the diamagnetic (inverse effective mass) weight

  int_{-inf}^{inf} Re sigma_ab(w) dw = pi * W_ab,
  W_ab = (1/(N_k V_cell)) sum_{k,n} f_nk <n|d2H/dK_a dK_b|n>

and W_ab = D_ab + (interband weight) exactly (the identity follows from
integrating sum_n f_n d2E_n/dK_a dK_b by parts over the Brillouin zone).
sum_rule_weight() computes W_ab from the second k-derivative of the
Hamiltonian, independently of the conductivity itself, which is what pins
the absolute normalization of this module: for a nearest-neighbour chain
at half filling W_xx = 2|t|/pi analytically, and there (a single band, no
interband transitions) the whole weight is the Drude weight. The other
absolute-scale check, in 2D, is the universal optical conductivity of
graphene, pi e^2/(4 h) per spin (i.e. 1/8 in these units for a spinless
honeycomb model).

References
----------
* J. Yates, X. Wang, D. Vanderbilt, I. Souza, PRB 75, 195121 (2007)
  -- the Kubo formula above, in exactly this normalization.
* X. Wang, J. Yates, I. Souza, D. Vanderbilt, PRB 74, 195118 (2006)
  -- the anomalous Hall conductivity and its sign convention.
* S. Tsirkin, npj Comput. Mater. 7, 33 (2021) (WannierBerri)
  -- same formula, used here as the structural reference.
* J. H. Garcia, L. Covaci, T. G. Rappoport, PRL 114, 116602 (2015), and
  S. M. Joao et al., arXiv:1910.05194 (KITE) -- Kubo-Bastin formulation.
"""
import numpy as np

from .conductivitytk import kubo


def optical_conductivity(h,energies=None,nk=20,T=None,delta=0.1,
        intraband=True,interband=True,component=None,degeneracy_tol=1e-6):
    """Compute the Kubo-Greenwood optical conductivity tensor
    sigma_ab(omega) of a 1D or 2D periodic Hamiltonian (see the module
    docstring for the formula, the units and the sign conventions).

    Parameters
    ----------
    h : Hamiltonian
      the (1D or 2D) periodic Hamiltonian
    energies : array
      frequencies hbar*omega at which to evaluate the conductivity
      (default np.linspace(0.,4.,100)). Re sigma_aa is even in omega and
      Im sigma_aa is odd, so negative frequencies carry no new information
      for the diagonal components.
    nk : int
      number of k-points per periodic direction (a 2D calculation uses
      nk*nk points)
    T : float
      temperature of the Fermi occupations. Defaults to delta, following
      the same convention as chitk/chiAB.py. T = 0 gives sharp step
      occupations, but then the intraband/Drude channel (which is weighted
      by -df/dE, a Dirac delta at T = 0) vanishes: a metal needs T > 0,
      and in practice T larger than the k-mesh level spacing
      (bandwidth/nk) for the Drude weight to be resolved.
    delta : float
      Lorentzian broadening eta of the retarded response
    intraband, interband : bool
      switch the two families of terms on and off independently, e.g.
      intraband=False for the pure interband (optical) spectrum
    component : str or None
      if given (e.g. "xy"), return only that Cartesian component as a 1D
      array instead of the full tensor
    degeneracy_tol : float
      band pairs closer in energy than degeneracy_tol times the
      Hamiltonian's characteristic hopping scale are treated as
      degenerate, i.e. through the -df/dE limit of the occupation factor

    Returns
    -------
    (energies, sigma) with sigma of shape (len(energies),3,3), complex,
    or of shape (len(energies),) if component was given.
    """
    return kubo.optical_conductivity(h,energies=energies,nk=nk,T=T,
            delta=delta,intraband=intraband,interband=interband,
            component=component,degeneracy_tol=degeneracy_tol)


def drude_weight(h,nk=20,T=0.05,degeneracy_tol=1e-6):
    """Drude (intraband) weight tensor D_ab of a 1D or 2D periodic
    Hamiltonian,

      D_ab = (1/(N_k V_cell)) sum_{k,n} (-df/dE)(E_nk) v^a_nn v^b_nn

    (generalized to the trace over each degenerate block), such that the
    intraband conductivity is sigma_ab^intra(omega) = i D_ab/(omega + i
    delta) and its dissipative part carries a spectral weight pi*D_ab.
    D_ab is real and, for a = b, non-negative; it vanishes for an
    insulator. It is a Fermi-surface quantity, so it needs T > 0 and
    enough k-points (T larger than the k-mesh level spacing) to converge.
    Returns a real 3x3 array, in units of e^2/hbar times energy."""
    return kubo.drude_weight(h,nk=nk,T=T,degeneracy_tol=degeneracy_tol)


def sum_rule_weight(h,nk=20,T=0.05):
    """Diamagnetic (inverse effective mass) weight tensor

      W_ab = (1/(N_k V_cell)) sum_{k,n} f_nk <n|d2H/dK_a dK_b|n>

    of a 1D or 2D periodic Hamiltonian, computed from the exact second
    k-derivative of the Bloch Hamiltonian -- i.e. independently of the
    conductivity itself. It is the right-hand side of the f-sum rule,
    int_{-inf}^{inf} Re sigma_ab(omega) domega = pi*W_ab, and equals the
    Drude weight plus the interband spectral weight. Returns a real,
    symmetric 3x3 array, in units of e^2/hbar times energy."""
    return kubo.sum_rule_weight(h,nk=nk,T=T)
