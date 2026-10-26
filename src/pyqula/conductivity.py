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


from .conductivitytk import shg


def shg_conductivity(h,energies=None,nk=20,T=None,delta=0.1,
        components=None,gauge="velocity",degeneracy_tol=1e-6):
    """Second-harmonic conductivity sigma_abc(2w;w,w) of a 1D or 2D
    periodic Hamiltonian, the second-order current at twice the frequency
    of the field, J_a(2w) = sigma_abc(2w;w,w) E_b(w) E_c(w), symmetrized
    in b and c.

    Formula
    -------
    The density-matrix recursion of Passos, Ventura, Viana Parente Lopes,
    Lopes dos Santos and Peres, PRB 97, 235446 (2018), arXiv:1712.04924,
    in the velocity gauge (hbar = 1, carrier charge q = -e):

      rho1^c_nm  = h^c_nm (f_m - f_n)/(w - (E_n - E_m))
      rho2^bc_nm = ([h^b,rho1^c] + (1/2)[h^bc,rho0])_nm/(2w - (E_n - E_m))
      sigma_abc  = -(q^3/w^2) (1/(N_k V_cell)) sum_k sum_nm
                   [h^a_nm rho2^bc_mn + h^ab_nm rho1^c_mn
                    + (1/2) h^abc_nn f_n delta_nm]

    with h^a, h^ab and h^abc the eigenbasis matrix elements of the first,
    second and third Cartesian k-derivatives of the Peierls-coupled Bloch
    Hamiltonian (the full bond vector R + r_j - r_i, exactly as in
    optical_conductivity, so h^a is its velocity operator). For a tight-
    binding model with the position operator diagonal in the orbitals this
    is exact for any number of bands: the second- and third-derivative
    vertices are what the truncated velocity gauge of a continuum model
    misses (Taghizadeh, Hipolito and Pedersen, PRB 96, 195413 (2017),
    arXiv:1710.01300), and with them the two-band model is not zero. The
    apparent 1/w^2 divergence cancels between the three terms.

    Every frequency carries the same broadening, w -> w + i delta, so the
    2w poles carry 2 i delta. Nothing is dropped: the intraband (Fermi-
    surface) response of a metal enters through the cancellation between
    the three terms rather than through an explicit -df/dE factor, so a
    metal is allowed at T > 0, with the same caveat on T and nk as the
    Drude part of optical_conductivity.

    gauge="length" evaluates the same quantity in the length gauge, as a
    sum over states with the interband Berry connection r^a_nm and its
    generalized derivative r^a_nm;b (Sipe and Shkrebtii, PRB 61, 5337
    (2000); Ibanez-Azpiroz, Tsirkin and Souza, PRB 97, 245143 (2018),
    arXiv:1804.04030), the latter from the exact sum rule of the tight-
    binding model, which includes the second-derivative term. It is an
    independent check, not the production path: it drops the Fermi-
    surface terms, so it needs T=0 and an insulator, and it divides by
    band energy differences, so it needs a nondegenerate spectrum and
    converges slowly in nk wherever two occupied bands come close. Where
    it applies, it agrees with the velocity gauge to machine precision on
    the same k-mesh.

    Units and conventions
    ---------------------
    As in optical_conductivity: hbar = e = 1, energies in the
    Hamiltonian's units t, lengths in the geometry's units a, the
    chemical potential at zero, and the retarded branch with time
    dependence exp(-i w t). sigma is then in units of e^3 a^(3-d)/(hbar t)
    with d the dimensionality. The sign follows from q = -e, since the
    response is odd in the charge. The spin degeneracy is whatever the
    Hamiltonian has: a spinless model gives the response of one spin
    species. The static limit is pinned in tests/conductivity against the
    second derivative of the exact polarization of an isolated molecule
    in a finite field.

    Parameters
    ----------
    h : Hamiltonian
      the (1D or 2D) periodic Hamiltonian, not Nambu
    energies : array
      fundamental frequencies hbar*w (default np.linspace(0.,4.,100))
    nk : int
      number of k-points per periodic direction
    T : float
      temperature of the Fermi occupations, delta by default
    delta : float
      broadening added to every frequency
    components : str, list of str or None
      Cartesian components such as "xxx" or ["xxx","xyy"]; None gives all
      27. Components along a non-periodic direction (z of a 2D slab) are
      meaningful, since the position operator is taken from the geometry
    gauge : str
      "velocity" (default) or "length", see above
    degeneracy_tol : float
      relative tolerance below which the length gauge treats two bands
      as degenerate and refuses to run

    Returns
    -------
    (energies, sigma) with sigma an array of len(energies) if components
    is a single string, and a dict {component: array} otherwise.
    """
    return shg.shg_conductivity(h,energies=energies,nk=nk,T=T,delta=delta,
            components=components,gauge=gauge,degeneracy_tol=degeneracy_tol)


def shg_susceptibility(h,length_unit=1.0,energy_unit=1.0,**kwargs):
    """Second-harmonic susceptibility chi_abc(2w;w,w) in SI units, the
    quantity measured in an SHG experiment, P_a(2w) = epsilon_0 chi_abc
    E_b(w) E_c(w). It is shg_conductivity converted through J = dP/dt,

      chi_abc = i sigma_abc/(2 (w + i delta) epsilon_0),

    with the broadening carried consistently, so that the static limit of
    an insulator is finite. length_unit is the length unit of the geometry
    in Angstrom and energy_unit the energy unit of the Hamiltonian in eV;
    the result is in nm^(4-d)/V, i.e. the sheet susceptibility in nm^2/V
    for a 2D system (divide by an effective thickness for a bulk value in
    nm/V). Every other keyword is passed to shg_conductivity, with the
    same return layout."""
    return shg.shg_susceptibility(h,length_unit=length_unit,
            energy_unit=energy_unit,**kwargs)


from .conductivitytk import excitonicoptics


def excitonic_optical_conductivity(h,energies=None,nk=20,delta=0.1,
        component=None,**bse_kwargs):
    """Optical conductivity sigma_ab(w) of a 1D or 2D insulator with the
    electron-hole interaction, from the Bethe-Salpeter equation (BSE) in
    the Tamm-Dancoff approximation at Q=0.

    Formula
    -------
    With E_n the exciton energies and R^a_0n = <0|R_a|n> =
    sum_cvk A^n_cvk r^a_vc(k) the dipole of exciton n with the ground
    state (A^n the BSE amplitudes, r_vc = v_vc/(i(E_v - E_c)) the
    interband dipoles of the Peierls-coupled model),

      chi_ab(w)   = (q^2/(N_k V_cell)) sum_n [ R^a_0n R^b_n0/(E_n - w)
                                             + R^b_0n R^a_n0/(E_n + w) ]
      sigma_ab(w) = -i w chi_ab(w) ,   w -> w + i delta

    (Taghizadeh and Pedersen, PRB 97, 205432 (2018), arXiv:1804.04869,
    Eq. (TensorFirstB)). This route through J = dP/dt misses the
    frequency-independent antisymmetric part of sigma, the Berry-curvature
    (Hall) term, which in an insulator is fixed by the Chern number and is
    not changed by the excitons; it is added at the independent-particle
    level, -i q^2/(N_k V_cell) sum_vck (r^a_vc r^b_cv - r^a_cv r^b_vc),
    from the bands of the BSE window, so with nv or nc restricting the
    bands it is not the full Chern-number term. Without interaction, kernel="none",
    it equals
    the interband part of optical_conductivity at T=0 exactly, which is
    what tests/conductivity checks. The units are those of
    optical_conductivity, and the same 3x3 tensor (or one component) is
    returned.

    method selects how the sum over excitons is done:
      "dense"  the BSE matrix is diagonalized, so the cost grows as the
               cube of the number of electron-hole pairs, nk^d times nv
               times nc
      "kpm"    the resolvent (w + i delta - A)^-1 of the BSE matrix A is
               expanded in Chebyshev polynomials of A at complex energy
               (bsetk/chebyshev.py), from products of the low-rank
               factorized A with vectors, so the matrix is never formed and
               the cost is linear in the number of pairs. The imaginary
               part of the energy damps the series, so it converges to the
               dense result itself, with about 20 s/delta moments for a
               spectrum of half width s. A screened interaction needs a
               real-space truncation first (see bse.get_bse), as for
               solver="iterative".

    Every other keyword goes to the BSE (see bse.get_bse): the
    interaction V (by default h.V, from a mean field), nv and nc for the
    band window, kernel, screening, nkW, channel and gauge.
    """
    return excitonicoptics.excitonic_optical_conductivity(h,energies=energies,
            nk=nk,delta=delta,component=component,**bse_kwargs)


def excitonic_shg_conductivity(h,energies=None,nk=20,delta=0.1,
        components=None,**bse_kwargs):
    """Second-harmonic conductivity sigma_abc(2w;w,w) of a 1D or 2D
    insulator with the electron-hole interaction, from the Tamm-Dancoff
    BSE at Q=0, in the same units and conventions as shg_conductivity.

    Formula
    -------
    The length-gauge sum over exciton states of Ruan, Chan and Louie,
    arXiv:2310.09674, Eq. (1) (the same as Taghizadeh and Pedersen,
    arXiv:1804.04869),

      chi_abc = (q^3/(2 N_k V_cell)) sum_nm [
          R^a_0n R^b_nm R^c_m0/((2w - E_n)(w - E_m))
        + R^b_0n R^c_nm R^a_m0/((2w + E_m)(w + E_n))
        - R^c_0n R^a_nm R^b_m0/((w - E_m)(w + E_n)) ] + (b <-> c)

    with w -> w + i delta in every frequency (so 2w carries 2 i delta, as
    in shg_conductivity), and sigma = -2 i w chi. R_nm = <n|R|m> is the
    position operator between two excitons. It is evaluated without a
    smooth gauge: a pair vector g is written as the orbital matrix
    Phi_k = sum_vc g_vck |u_ck><u_vk|, which does not depend on the phases
    of the Bloch eigenvectors and is periodic in k. Its Fourier transform
    on the mesh is an electron-hole amplitude phi_ij(R), and the position
    operator, i dPhi/dK + [tau,Phi], multiplies it by the bond vector
    -(R + tau_j - tau_i), with R the shortest image on the nk-periodic
    supercell. The sums over excitons are done through the resolvent
    (z - A)^-1 of the BSE matrix, diagonalized once with method="dense",
    or expanded in Chebyshev polynomials with method="kpm", as in
    excitonic_optical_conductivity, which reaches the meshes that the
    resolution delta >= v dk/2 asks for with every conduction band.

    Without interaction (kernel="none") it reproduces shg_conductivity at
    T=0, with and without time reversal, converging with nk: the pair
    amplitudes vary over delta/v in k near a resonance, with v the band
    velocity, so the mesh spacing dk needs delta >= v dk/2, and above
    the gap the window nc has to hold every conduction band. It is
    invariant under the gauge of the eigenvectors and keeps the point
    group of the crystal exactly on every mesh, also for a stacked
    multilayer, all checked in tests/conductivity.

    Every other keyword (method included) goes as in
    excitonic_optical_conductivity; wbatch is the number of frequencies
    done together. Returns (energies, sigma), with sigma
    an array for one component and a dict for a list.
    """
    return excitonicoptics.excitonic_shg_conductivity(h,energies=energies,
            nk=nk,delta=delta,components=components,**bse_kwargs)


def excitonic_shg_susceptibility(h,length_unit=1.0,energy_unit=1.0,
        **kwargs):
    """Excitonic second-harmonic susceptibility in SI units, nm^(4-d)/V,
    the conversion of excitonic_shg_conductivity described in
    shg_susceptibility"""
    return excitonicoptics.excitonic_shg_susceptibility(h,
            length_unit=length_unit,energy_unit=energy_unit,**kwargs)


from .conductivitytk import nonlineardrude


def nonlinear_drude_conductivity(h,field="x",current="x",channel="spin",
        nk=100,T=0.01,mu=0.,tau=1.,omega=0.,degeneracy_tol=1e-8):
    """l-th order nonlinear Drude conductivity sigma^{x^l1 y^l2 ; b} of a
    collinear magnet, the quantity whose lowest nonvanishing order measures
    the X-wave index of an altermagnet.

    The current is expanded in the electric field as

      j_b = sum_{l1,l2} sigma^{x^l1 y^l2 ; b} (E_x)^l1 (E_y)^l2,

    and solving the semiclassical Boltzmann equation recursively gives, for
    each spin channel s (spin must be a good quantum number),

      sigma_s^{x^l1 y^l2 ; b} = (-e/hbar)^(l+1)/(i omega + 1/tau)^l
            * Int d^Dk/(2 pi)^D f_s^(0) d^(l+1) eps_s
                                / (dk_x^l1 dk_y^l2 dk_b),   l = l1+l2

    with sigma_spin = (sigma_up - sigma_dn)/2 and sigma_charge = sigma_up +
    sigma_dn (Ezawa, Phys. Rev. B 111, 125420 (2025), arXiv:2411.16036).

    The physics: an X-wave form factor is a k-space harmonic of order l+1,
    so its (l+1)-th derivative is the first that is a nonzero constant while
    every lower one integrates to zero over the Brillouin zone. The spin
    response therefore switches on at

      d-wave: l = 1     f-wave: l = 2     g-wave: l = 3     i-wave: l = 5

    and reading off the lowest order at which a nonlinear spin current
    appears identifies the wave index -- a purely electrical measurement of
    altermagnetic order that needs no spin-orbit coupling. Orders *above*
    the threshold are generically nonzero too on a lattice (the lattice form
    factor contains higher harmonics beyond the leading one), so it is the
    lowest nonvanishing order, not the only nonvanishing one, that carries
    the information. Use nonlinear_drude_orders for that sweep.

    Two cases fall outside that counting. The order l = 0 vanishes for
    every band of every Hamiltonian: f(eps) d eps/dk_b = d F(eps)/dk_b with
    F' = f is a total derivative, whose zone integral is zero, so there is
    no current in equilibrium. And a p-wave magnet has no spin current at
    any order: 2t cos kx + s J sin kx = sqrt(4t^2+J^2) cos(kx - s phi) is
    one band shifted rigidly along kx by opposite amounts for the two spins,
    a zone integral does not see the shift, and so sigma_up = sigma_dn at
    every order (arXiv:2411.16036, Sec. V). On a finite k-mesh both zeros
    are reached only as nk grows, exponentially at finite T, because the
    shift is not a translation of the mesh: the p-wave even orders are
    3e-4, 2e-5, 6e-8 and 1e-11 at nk = 48, 96, 192, 384 (J = 0.3, T = 0.02,
    half a hopping above the band bottom), while its odd orders vanish
    exactly on any mesh, since inversion swaps the two spin channels.

    Parameters
    ----------
    h : Hamiltonian
      a 1D or 2D spinful Hamiltonian in which spin is a good quantum number
      (no spin-orbit coupling; this is checked, and a spin-mixing
      Hamiltonian is rejected rather than silently mistreated)
    field : str
      one character per power of the electric field, e.g. "yyyyy" for the
      fifth-order response to E_y. The empty string is l = 0, the current
      with no field applied, which vanishes identically (see above) and
      so only measures how converged the k-mesh is.
    current : str
      Cartesian direction b of the measured current
    channel : "spin", "charge", "up" or "dn"
    nk : int
      k-points per periodic direction (a 2D calculation uses nk*nk)
    T : float
      temperature of the Fermi occupations
    mu : float
      chemical potential (pyqula puts the Fermi level at zero, so this is
      measured from there)
    tau : float
      relaxation time; the l-th order response scales as tau^l
    omega : float
      frequency of the applied field, entering as 1/(i omega + 1/tau)^l
    degeneracy_tol : float
      multiorbital cells only. The band derivatives are obtained exactly,
      by Rayleigh-Schroedinger perturbation theory in a truncated Taylor
      ring, whose reduced resolvent divides by the band spacing; bands
      closer than this on a mesh point raise rather than return nonsense.
      A Dirac point is the usual culprit -- see the error message.

    Units are e = hbar = 1 and the Brillouin-zone integral carries the
    1/(2 pi)^D of a density, as in the rest of this module; see
    conductivitytk/nonlineardrude.py. Returns a complex scalar.
    """
    return nonlineardrude.nonlinear_drude_conductivity(h,field=field,
            current=current,channel=channel,nk=nk,T=T,mu=mu,tau=tau,
            omega=omega,degeneracy_tol=degeneracy_tol)


def nonlinear_drude_components(h,l,**kwargs):
    """Every component of the l-th order nonlinear Drude conductivity, as
    a dict {"x^l1 y^l2;b": value}. Sharing one k-mesh and one set of Bloch
    phases makes this much cheaper than the equivalent individual calls to
    nonlinear_drude_conductivity, which it otherwise matches exactly."""
    return nonlineardrude.nonlinear_drude_components(h,l,**kwargs)


def nonlinear_drude_orders(h,lmax=6,**kwargs):
    """X-wave selection-rule sweep: for each order l = 0..lmax, the largest
    |sigma^{x^l1 y^l2 ; b}| over that order's components. The lowest l with
    a nonzero entry is the wave index readout (d:1, f:2, g:3, i:5). A p-wave
    magnet has no nonzero entry at all, and the l = 0 entry is zero for
    every Hamiltonian; see nonlinear_drude_conductivity. Returns a list of
    floats."""
    return nonlineardrude.nonlinear_drude_orders(h,lmax=lmax,**kwargs)


def fermi_volume(h,nk=100,T=0.01,mu=0.,channel="charge"):
    """Fermi volume V^F = Int d^Dk/(2 pi)^D f^(0), in the same
    normalization as the conductivities above. It is what the analytic
    results of arXiv:2411.16036 are expressed in terms of, so ratios of a
    conductivity to it are free of the (2 pi)^D convention."""
    return nonlineardrude.fermi_volume(h,nk=nk,T=T,mu=mu,channel=channel)
