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
  g.a2, g.a3 -- so a rescaled geometry does change the answer, as it must).
* sigma is then returned in units of e^2/hbar times a^(2-d) with d the
  dimensionality: e^2/hbar (a sheet conductance) in 2D, e^2*a/hbar in 1D,
  e^2/(hbar a) (a bulk conductivity) in 3D. A 3D stack of decoupled layers
  with spacing c gives the 2D sheet conductance of one layer divided by c.
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
        intraband=True,interband=True,component=None,degeneracy_tol=1e-6,
        integration="grid",tolerance=1e-6,ewindow=None,nv=None,nc=None):
    """Compute the Kubo-Greenwood optical conductivity tensor
    sigma_ab(omega) of a 1D, 2D or 3D periodic Hamiltonian (see the module
    docstring for the formula, the units and the sign conventions).

    Parameters
    ----------
    h : Hamiltonian
      the (1D, 2D or 3D) periodic Hamiltonian
    energies : array
      frequencies hbar*omega at which to evaluate the conductivity
      (default np.linspace(0.,4.,100)). Re sigma_aa is even in omega and
      Im sigma_aa is odd, so negative frequencies carry no new information
      for the diagonal components.
    nk : int
      number of k-points per periodic direction (a 2D calculation uses
      nk*nk points, a 3D one nk^3)
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
    integration : str
      how the k-integral is done, one of kubo.get_integration_names():
      "grid" (default) sums the formula over the uniform nk mesh, which
      needs a mesh whose transition-energy spacing is below delta.
      "triangle" splits the mesh into simplices (segments in 1D, triangles
      in 2D, six tetrahedra per cube in 3D), interpolates the transition energies and the weights
      linearly inside each one, and integrates the resolvent over it
      exactly (conductivitytk/kubosimplex.py). The mesh then only has to
      resolve the bands, not the broadening: at delta = 1e-3 t on a gapped
      honeycomb, nk = 256 gives 1e-3 relative accuracy where the grid
      needs nk = 8192. It converges at second order in 1/nk, so for a
      smooth integrand (delta larger than the band features, or a
      frequency inside a gap) the grid, which converges exponentially
      there, is the better choice. The sum over the simplices is built
      once for all frequencies, also in 3D, where the tetrahedra whose
      vertex form would cancel too much (4% to 20% of them on the meshes
      measured) are summed at every frequency, so it is 4 to 14 times
      faster than summing every tetrahedron at every frequency.
      "qtci" (2D only, 3D raises NotImplementedError) does the
      same on a 2^R x 2^R mesh, nk rounded up to a power of two, where the
      transition energies and weights are reconstructed from a quantics
      tensor cross interpolation instead of computed on every point: for
      smooth bands the number of diagonalizations saturates with R (44k
      for nk = 1024 on the gapped honeycomb, against 1.05M). Both need
      delta > 0. Neither removes the mesh requirement set by T: the
      occupation factors of a metal have a step of width T at the Fermi
      surface, which the linear interpolation resolves only to first order
      and which costs rank in the interpolation, and the Drude term is the
      grid sum in all three.
    tolerance : float
      relative tolerance of the tensor cross interpolation of the fields,
      one interpolation of all of them at once, so relative to the
      largest field; only used by integration="qtci"
    ewindow : float, pair or None
      band window, for large unit cells: only the bands close to the Fermi
      level are diagonalized (a subset eigensolver, or a shift-invert one
      above limits.densedimension), the velocity is rotated only on them,
      and only their pairs are stored, so the cost per k-point goes from
      n^3 to about n^2 nw and the memory from n^2 to nw^2, with nw the
      bands kept. ewindow = W keeps every band that comes inside (-W, W)
      at some k-point of the mesh; a pair (emin, emax) gives an asymmetric
      window. Off (None) by default, which keeps all the bands.
    nv, nc : int or None
      the same window given as band counts: the nv bands below and the nc
      above the Fermi level. Not together with ewindow.

      The bands kept are a fixed range of band indices, the same at every
      k-point, because the triangle and qtci integrations interpolate the
      fields by band index: a band that enters the window anywhere on the
      mesh is kept everywhere, and in a metal every band that crosses the
      Fermi level is kept as well. The range is grown until this holds on
      the whole mesh, and it is not allowed to cut a degenerate multiplet
      at a k-point where the window asks for the edge band.
      For integration="qtci" it is fixed on the coarse 32x32 scan, with a
      warning if a later k-point needs more bands. On the sparse path the
      bands are counted from the Fermi level instead, since a shift-invert
      solver does not give the band index; this is the same labelling for
      an insulator, while in a metal it changes where a band crosses the
      Fermi level.

      What is exact and what is not: a pair left out with a nonzero
      occupation factor has one band below the Fermi level and one above,
      and one of them outside (emin, emax), so its transition energy is
      above min(-emin, emax). Re sigma(omega), the absorption, at omega
      below that energy is then exact up to the Lorentzian tails of the
      pairs left out, of order delta/(dE - omega)^2. With nv, nc the
      covered frequency is instead the lowest transition energy of a pair
      left out, which is set by the bands just outside the range on the
      mesh and not by a number given in the call. Im sigma gets a
      Kramers-Kronig contribution from every pair, so the window shifts it
      by a smooth function of omega, roughly linear at small omega for a
      diagonal component, which goes to zero only as the window grows.

    Returns
    -------
    (energies, sigma) with sigma of shape (len(energies),3,3), complex,
    or of shape (len(energies),) if component was given.
    """
    return kubo.optical_conductivity(h,energies=energies,nk=nk,T=T,
            delta=delta,intraband=intraband,interband=interband,
            component=component,degeneracy_tol=degeneracy_tol,
            integration=integration,tolerance=tolerance,ewindow=ewindow,
            nv=nv,nc=nc)


def drude_weight(h,nk=20,T=0.05,degeneracy_tol=1e-6):
    """Drude (intraband) weight tensor D_ab of a 1D, 2D or 3D periodic
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

    of a 1D, 2D or 3D periodic Hamiltonian, computed from the exact second
    k-derivative of the Bloch Hamiltonian -- i.e. independently of the
    conductivity itself. It is the right-hand side of the f-sum rule,
    int_{-inf}^{inf} Re sigma_ab(omega) domega = pi*W_ab, and equals the
    Drude weight plus the interband spectral weight. Returns a real,
    symmetric 3x3 array, in units of e^2/hbar times energy."""
    return kubo.sum_rule_weight(h,nk=nk,T=T)


from .conductivitytk import shg


def shg_conductivity(h,energies=None,nk=20,T=None,delta=0.1,
        components=None,gauge="velocity",degeneracy_tol=1e-6,
        integration="grid",tolerance=1e-6):
    """Second-harmonic conductivity sigma_abc(2w;w,w) of a 1D, 2D or 3D
    periodic Hamiltonian, the second-order current at twice the frequency
    of the field, J_a(2w) = sigma_abc(2w;w,w) E_b(w) E_c(w), symmetrized
    in b and c. The rectified current sigma_abc(0;w,-w), the sum and
    difference frequencies and the higher orders come from
    nonlinear_conductivity, whose n = 2, frequencies=(1,1) case is this
    quantity on the same mesh.

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
    with d the dimensionality, so in 3D it is a bulk conductivity per unit
    cell volume, and a stack of decoupled layers with spacing c gives the
    2D value of one layer divided by c. The sign follows from q = -e, since the
    response is odd in the charge. The spin degeneracy is whatever the
    Hamiltonian has: a spinless model gives the response of one spin
    species. The static limit is pinned in tests/conductivity against the
    second derivative of the exact polarization of an isolated molecule
    in a finite field.

    Parameters
    ----------
    h : Hamiltonian
      the (1D, 2D or 3D) periodic Hamiltonian, not Nambu
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
      as degenerate and refuses to run (and below which the triangle
      integration treats two energy differences as equal)
    integration : str
      how the k-integral is done, one of shg.get_integration_names():
      "grid" (default) sums the formula over the uniform nk mesh, which
      needs a mesh whose transition-energy spacing is below delta.
      "triangle" splits the mesh into simplices, interpolates the band
      energies and the weights linearly inside each one and integrates the
      resonant factors 1/(w - E_nm) and 1/(2w - E_mn) over it exactly
      (conductivitytk/shgsimplex.py), so the mesh only has to resolve the
      bands, not the broadening. In a two-band model every resonant
      factor is integrated exactly on the interpolated bands; the products
      of a resonance at w and one at 2w of three different bands are not
      linear on a triangle and are integrated through a triangulation of
      the prism triangle x [0,1], averaged over the vertex orderings. Both
      converge at second order in 1/nk. The 1/w^2 prefactor of the
      velocity gauge amplifies the O(1/nk^2) error at small w, so below
      the absorption edge the grid, which converges exponentially there,
      is the better choice. "qtci" (2D only) does the same on a 2^R x 2^R
      mesh, nk rounded up to a power of two, whose band energies and
      weights come from a quantics tensor cross interpolation instead of a
      diagonalization at every point. Both need delta > 0 and the velocity
      gauge, and neither removes the mesh requirement set by T at a Fermi
      surface. In 3D only "grid" is implemented, and the other two raise
      NotImplementedError
    tolerance : float
      relative tolerance of the tensor cross interpolation of each field,
      only used by integration="qtci"

    Returns
    -------
    (energies, sigma) with sigma an array of len(energies) if components
    is a single string, and a dict {component: array} otherwise.
    """
    return shg.shg_conductivity(h,energies=energies,nk=nk,T=T,delta=delta,
            components=components,gauge=gauge,degeneracy_tol=degeneracy_tol,
            integration=integration,tolerance=tolerance)


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
        component=None,scissor=0.,**bse_kwargs):
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
               (bsetk/chebyshev.py), from products of A with vectors, so
               the matrix is never formed and the cost is linear in the
               number of pairs. The product is a convolution over the
               k-mesh (bsetk/convolution.py), whose cost does not depend
               on the range of the interaction, and which takes a
               screened interaction tabulated on the mesh as it is. The
               imaginary part of the energy damps the series, so it
               converges to the dense result itself, with about 20 s/delta
               moments for a spectrum of half width s. With
               pyqula.gpu.set_gpu(True) the expansion runs on the GPU
               (bsetk/chebyshevjax.py), in double precision, or in single
               precision with kpm_prec="single", which agrees with double
               to 1e-5 to 5e-5 of the maximum.

    scissor is a rigid shift s of every electron-hole pair energy,
    A -> A + s, with the interband dipoles left at the bands of h: the
    scissor of the length gauge, which moves the gap of a tight-binding
    model to the measured (or GW) one without changing its wavefunctions.
    It is done as the resolvent of A at w - s, the same for both methods,
    and with kernel="none" it moves every resonance E_c - E_v by s.

    Every other keyword goes to the BSE (see bse.get_bse): the
    interaction V (by default h.V, from a mean field), nv and nc for the
    band window, kernel, screening, nkW, channel and gauge.
    """
    return excitonicoptics.excitonic_optical_conductivity(h,energies=energies,
            nk=nk,delta=delta,component=component,scissor=scissor,
            **bse_kwargs)


def excitonic_shg_conductivity(h,energies=None,nk=20,delta=0.1,
        components=None,scissor=0.,**bse_kwargs):
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

    Every other keyword (method and scissor included) goes as in
    excitonic_optical_conductivity, the scissor shifting E_n to E_n + s in
    every denominator with R_0n and R_nm left at the bands; wbatch is the number of frequencies
    done together, by default as many as fit in about 2 GB
    (excitonicoptics.batch_memory), since with method="kpm" every batch
    runs the Chebyshev recursion again. Returns (energies, sigma), with sigma
    an array for one component and a dict for a list.
    """
    return excitonicoptics.excitonic_shg_conductivity(h,energies=energies,
            nk=nk,delta=delta,components=components,scissor=scissor,
            **bse_kwargs)


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

    The routine returns the symmetric tensor component sigma_{b a1..al}
    with l1 of the field indices a_j equal to x and l2 = l - l1 equal to
    y. binom(l,l1) orderings of the field indices give that component, so
    the current at order l is

      j_b = sum_{l1} binom(l,l1) sigma^{x^l1 y^l2 ; b} (E_x)^l1 (E_y)^l2,

    and at l = 2 the mixed term is 2 sigma^{xy;b} E_x E_y. At l = 2 and
    l = 3, in the charge channel and at omega = 0, sigma^{x^l1 y^l2 ; b}
    is the Drude term of nonlinear_dc_conductivity with the labels in any
    order. Solving the semiclassical Boltzmann equation recursively gives,
    for each spin channel s (spin must be a good quantum number),

      sigma_s^{x^l1 y^l2 ; b} = (-e/hbar)^(l+1)
            * prod_{p=1..l} 1/(1/tau - i p omega)
            * Int d^Dk/(2 pi)^D f_s^(0) d^(l+1) eps_s
                                / (dk_x^l1 dk_y^l2 dk_b),   l = l1+l2

    with sigma_spin = (sigma_up - sigma_dn)/2 and sigma_charge = sigma_up +
    sigma_dn (Ezawa, Phys. Rev. B 111, 125420 (2025), arXiv:2411.16036).
    omega is the frequency of the field E(t) = Re[E exp(-i omega t)], in
    the convention of optical_conductivity, and sigma is the current at
    the harmonic l omega: the p-th step of the recursion acts on the part
    of the distribution that oscillates at p omega. At omega = 0 the
    prefactor is tau^l, the static result of Ezawa, and at l = 1 it is the
    intraband part of optical_conductivity. Ezawa writes
    1/(i omega + 1/tau)^l, which agrees with this only at omega = 0.

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
      relaxation time; at omega = 0 the l-th order response scales as
      tau^l
    omega : float
      frequency of the applied field, entering as
      prod_{p=1..l} 1/(1/tau - i p omega); the result is the current at
      the harmonic l omega
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


from .conductivitytk import nonlinearkernel
from .conductivitytk import nonlineardc
from .conductivitytk import photocurrent


# Argument guards of the public nonlinear routines below. They run before
# the conductivitytk routines, so that an input the signatures accept but
# the formulas cannot take is refused with a message, rather than with an
# error from deep inside numpy or with a silently different result.

def _check_order(order,what):
    """order is a positive integer (a float or a bool is refused, rather
    than rounded)"""
    if isinstance(order,bool) or not isinstance(order,(int,np.integer)):
        raise TypeError("the order of "+what+" is a positive integer, got "
                +repr(order))
    if order<1: raise ValueError("the order of "+what+" is a positive "
            "integer, got order="+str(order))


def _check_energies(energies,what):
    """energies is None or a 1D array of frequencies"""
    if energies is None: return
    if np.ndim(energies)!=1: raise ValueError(what+" takes energies as a "
            "1D array of frequencies, got "+repr(energies)+"; for a single "
            "frequency w use energies=[w]")


def _check_components(components,what):
    """components is None, one label string, or a nonempty list of them"""
    if components is None or isinstance(components,str): return
    if len(components)==0: raise ValueError(what+" needs at least one "
            "component, such as \"xyy\", got an empty list; components=None "
            "gives every component")


def _check_temperature(T,what):
    """T is given, since the dc routines are Fermi-surface sums and have no
    broadening to default to (their positivity is checked further in)"""
    if T is None: raise ValueError(what+" is a Fermi-surface sum, whose "
            "weight -df/dE is zero on a k-mesh at T=0: it needs T>0, got "
            "T=None")


def nonlinear_conductivity(h,order=2,energies=None,frequencies=None,nk=20,
        T=None,delta=0.1,components=None,degeneracy_tol=1e-6,
        prec="double"):
    """n-th order conductivity sigma_{b a1..an}(w1,...,wn) of a 1D, 2D or 3D
    periodic Hamiltonian, the current at the sum frequency w1 + ... + wn
    driven by n fields, at any order and for any tuple of frequencies.

    Formula
    -------
    The velocity-gauge density-matrix recursion of Passos et al., PRB 97,
    235446 (2018), arXiv:1712.04924, Eqs. (rho) and (responsefunction),
    with hbar = 1 and carrier charge q = -e. With rho^(0) = diag(f) and
    every w_j already complex,

      rho^{a1..ap}_nm = sum_{m'=1}^{p} (1/m'!) [h^{a1..am'},
                        rho^{a(m'+1)..ap}]_nm / (w1 + ... + wp - (E_n - E_m))
      sigma_{b a1..an} = (i^n q^(n+1)/(w1...wn)) (1/(N_k V_cell)) sum_k
                         sum_{p=0}^{n} (1/p!) Tr[h^{b a1..ap} rho^{a(p+1)..an}]

    averaged over the orderings of the pairs (a_j,w_j), so that sigma is
    symmetric under their exchange. h^{a1..am} are the eigenbasis matrix
    elements of the m-th Cartesian k-derivative of the Peierls-coupled
    Bloch Hamiltonian, with the full bond vector R + r_j - r_i as in
    optical_conductivity, so the result is exact for the tight-binding
    model with any number of bands. The apparent 1/(w1...wn) divergence
    cancels between the terms. At n = 1 this is optical_conductivity, and
    at n = 2 with frequencies=(1,1) it is shg_conductivity, on the same
    mesh.

    Field convention
    ----------------
    J_b(w) = Int prod_j dw_j/(2 pi) sigma_{b a1..an}(w1..wn)
    E_a1(w1)...E_an(wn) 2 pi delta(w - sum_j w_j), with time dependence
    exp(-i w t) (Passos et al., Eq. (defresponsefunction)). For a real
    field E(t) = Re E0 exp(-i w t) this means

      second harmonic   J(t) = (1/2) Re[sigma(w,w) E0 E0 exp(-2 i w t)]
      rectified current J    = (1/2) sigma(w,-w) E0 E0^*
      static field      J    = sigma(0,...,0) E^n

    and at third order the third harmonic is (1/4) Re[sigma(w,w,w) E0^3
    exp(-3 i w t)].

    Broadening
    ----------
    Every field frequency carries the same broadening, w_j = c_j*w +
    i delta, so a pole at a sum of m frequencies carries i m delta. This
    is the adiabatic switching of Passos et al., not a relaxation time.
    In particular energies=[0.] gives the dc value of this phenomenology
    at finite delta, which is not the tau-expanded dc conductivity of
    nonlinear_dc_conductivity and cannot be read at delta = 1/tau: at dc
    the n-field Drude term goes as 1/(n! delta^n) where the relaxation
    time gives tau^n, a term with one intraband pole as 1/delta where the
    relaxation time gives tau, and the delta^0 coefficient matches none of
    the quantum-metric conventions (see future_development/
    nonlinear_response.md, "The dc limit and its Laurent structure").

    The recursion never divides by a band energy difference, so
    degenerate bands, band crossings and metals are allowed. A metal needs
    T > 0 and a mesh whose energy spacing v 2 pi/(nk a) is below both T
    and delta.

    Units
    -----
    hbar = e = 1, energies in units of the hopping t, lengths in the
    units a of the geometry, the chemical potential at zero (use
    h.shift_fermi). sigma is per unit cell volume (area in 2D, length in
    1D), in units of e^(n+1) a^(n+1-d)/(hbar t^(n-1)). The response goes
    as q^(n+1), so the even orders change sign with the charge of the
    carriers and the odd ones do not. The spin degeneracy is whatever the
    Hamiltonian has.

    Parameters
    ----------
    h : Hamiltonian
      the (1D, 2D or 3D) periodic Hamiltonian, not Nambu
    order : int
      the order n of the response, any positive integer
    energies : array
      the frequency hbar*w (default np.linspace(0.,4.,100))
    frequencies : tuple or None
      the multipliers c_j of the n fields, w_j = c_j*w. None is all ones
      (harmonic generation); (1,-1) the rectified current; (1,2) the sum
      frequency w + 2w; (1,1,0) the field-induced second harmonic and
      (1,-1,0) the jerk current, where the static field carries i delta
    nk : int
      number of k-points per periodic direction
    T : float
      temperature of the Fermi occupations, delta by default
    delta : float
      the broadening of every field frequency, positive
    components : str, list of str or None
      strings of n+1 Cartesian labels, the current first, such as "xyy"
      at second order and "xxyy" at third; None gives all 3^(n+1)
    degeneracy_tol : float
      accepted for the uniform signature of the nonlinear routines; the
      recursion does not use it
    prec : str
      "double", or "single" on the GPU (pyqula.gpu.set_gpu(True)). The
      1/(w1...wn) prefactor cancels terms of order one when a field
      frequency is below delta, so single precision is meaningful only
      for |c_j w| above delta; the float32 error then grows as
      (t/|w + i delta|)^n, and the routine warns otherwise

    Returns
    -------
    (energies, sigma) with sigma a complex array of len(energies) if
    components is a single string, and a dict {component: array} otherwise.
    """
    what = "the nonlinear conductivity"
    _check_order(order,what)
    _check_energies(energies,what)
    _check_components(components,what)
    return nonlinearkernel.nonlinear_conductivity(h,order=order,
            energies=energies,frequencies=frequencies,nk=nk,T=T,delta=delta,
            components=components,degeneracy_tol=degeneracy_tol,prec=prec)


def nonlinear_susceptibility(h,order=2,length_unit=1.0,energy_unit=1.0,
        **kwargs):
    """n-th order susceptibility chi^(n) in SI units, from
    nonlinear_conductivity through J = dP/dt: the polarization at the sum
    frequency Omega = w1 + ... + wn (every w_j with its broadening) is
    P = J/(-i Omega), so

      chi^(n) = sigma^(n)/(-i Omega epsilon_0),

    which at n = 2 and frequencies=(1,1) is shg_susceptibility.
    length_unit is the length unit of the geometry in Angstrom and
    energy_unit the energy unit of the Hamiltonian in eV; the result is in
    nm^(n+2-d)/V^(n-1), so nm^2/V for the sheet susceptibility of a 2D
    system at second order and nm^3/V^2 at third. The static limit
    (energies=[0.]) of an insulator is finite, since Omega = n i delta
    there. Every other keyword is that of nonlinear_conductivity, with the
    same return layout."""
    what = "the nonlinear susceptibility"
    _check_order(order,what)
    _check_energies(kwargs.get("energies",None),what)
    _check_components(kwargs.get("components",None),what)
    return nonlinearkernel.nonlinear_susceptibility(h,order=order,
            length_unit=length_unit,energy_unit=energy_unit,**kwargs)


def nonlinear_dc_conductivity(h,order=2,term="all",tau=1.,qmd="shibata",
        channel="charge",nk=100,T=0.05,components=None,degeneracy_tol=1e-6):
    """dc nonlinear conductivity of a metal in the relaxation-time
    approximation, decomposed in powers of tau: at second order
    J_a = sigma_abc E_b E_c, symmetric in bc, and at third order
    J_a = sigma_abcd E_b E_c E_d, symmetric in bcd.

    Second order
    ------------
    With u^n_a = h^a_nn the band velocity, u^n_ab the inverse mass,
    Omega^n_ab the Berry curvature, G^n_ab the quantum metric and
    Gt^n_ab = 2 Re sum_m h^a_nm h^b_mn/(E_n - E_m)^3 the band-normalized
    metric (the Berry connection polarizability), w_n = -f'(E_n) the
    Fermi-surface weight and <X> = (1/(N_k V_cell)) sum_k sum_n X_n,

      drude (tau^2)  -tau^2 < w u_a u_bc >
      bcd   (tau^1)  (tau/2) < w (u_c Omega_ab + u_b Omega_ac) >
      qmd   (tau^0)  the quantum-metric term in the convention qmd=, from
        qmd_hall     - < w [u_a Gt_bc - (1/2)(u_b Gt_ca + u_c Gt_ab)] >
        qmd_disp     - < w u_a Gt_bc >
        qmd_intra    -(1/2) < w d_a G_bc > = -(1/2) < f'' u_a G_bc >

    The three terms are told apart in an experiment by how they scale with
    the linear conductivity, which goes as tau. The Drude term needs broken
    time reversal and broken inversion. The Berry curvature dipole term
    (Sodemann and Fu, arXiv:1508.00571, with their factor 2 for a dc
    field) needs broken inversion and is allowed with time reversal; in
    2D sigma_xyy = tau D_y and sigma_yxx = -tau D_x with the dipole of
    berry_curvature_dipole. The quantum-metric term needs broken time
    reversal and survives where PT symmetry kills the Berry curvature.

    Three conventions of the tau^0 term are in use, and they differ by
    gauge-invariant pieces: qmd="shibata" (default) is hall + intra, the
    clean limit of the velocity-gauge Keldysh calculation of Shibata,
    arXiv:2606.22359; "gao" is hall alone (Gao, Yang and Niu,
    arXiv:1402.2538); "kaplan" is hall + disp, Kaplan, Holder and Yan,
    arXiv:2211.17213, with the coefficient as they write it. Their
    dispersion current uses a field-corrected band energy that is twice
    the second-order Stark shift (q^2/2) E.Gt.E, so the physical
    dispersion current with an unshifted distribution is half of
    qmd_disp. The hall piece is purely transverse, sigma_aaa = 0.

    Third order
    -----------
    order=3 has two terms, in the charge channel only:

      drude (tau^3)  tau^3 < f d_a d_b d_c d_d eps >, 1D and 2D only
      bcp   (tau^1)  the Berry connection polarizability term of Liu et
                     al., arXiv:2106.04931, from Gt and the band
                     velocities and inverse masses

    and term="all" gives a dict with "drude", "bcp" and "total". The tau^2
    (Berry curvature quadrupole) and tau^0 terms at third order are not
    built.

    Degenerate bands
    ----------------
    Every sum over m leaves out the degenerate multiplet of n (bands
    closer than degeneracy_tol times the hopping scale), and a product of
    two band-diagonal quantities inside a multiplet is replaced by the
    trace over the multiplet. This is basis independent, and exact for a
    spin degeneracy without spin-orbit coupling. A PT-symmetric magnet is
    Kramers degenerate at every k and is handled in this way.

    Temperature and mesh
    --------------------
    Every term is a Fermi-surface sum, so T > 0 is required, and nk has to
    make the energy spacing of the mesh, v 2 pi/(nk a), smaller than T.
    The way to check this is to double nk at fixed T.

    Units
    -----
    hbar = e = 1, energies in units of t, lengths in units of a, tau in
    units of hbar/t and the chemical potential at zero (h.shift_fermi(-mu)
    puts it at mu). sigma is per unit cell volume, in units of
    e^(n+1) a^(n+1-d)/(hbar t^(n-1)), the units of nonlinear_conductivity.

    Parameters
    ----------
    h : Hamiltonian
      the (1D, 2D or 3D) periodic Hamiltonian, not Nambu
    order : int
      2 or 3 (nonlineardc.get_orders())
    term : str
      "all" (a dict with each term, each with its power of tau included,
      and their sum "total"), or one term: at second order "drude", "bcd",
      "qmd", "qmd_hall", "qmd_disp" or "qmd_intra", at third order
      "drude" or "bcp"
    tau : float
      relaxation time
    qmd : str
      "shibata" (default), "gao" or "kaplan", the convention of the
      quantum-metric term at second order
    channel : str
      "charge" for any Hamiltonian, or "spin", "up", "dn" for a
      spin-conserving one, at second order
    nk : int
      number of k-points per periodic direction
    T : float
      temperature, positive
    components : str, list of str or None
      labels such as "xyy" (or "xxyy" at third order), the current first;
      None gives every component along the periodic directions
    degeneracy_tol : float
      relative tolerance below which two bands form a multiplet

    Returns
    -------
    For one component string, its value; for a list or None, a dict
    {component: value}. With term="all", a dict keyed by the term names
    whose values have that layout.
    """
    what = "the nonlinear dc conductivity"
    _check_order(order,what)
    _check_temperature(T,what)
    _check_components(components,what)
    return nonlineardc.nonlinear_dc_conductivity(h,order=order,term=term,
            tau=tau,qmd=qmd,channel=channel,nk=nk,T=T,components=components,
            degeneracy_tol=degeneracy_tol)


def berry_curvature_dipole(h,nk=100,T=0.05,channel="charge",
        degeneracy_tol=1e-6):
    """Berry curvature dipole, the Fermi-surface average of the Berry
    curvature weighted with the band velocity,

      D_a = (1/(N_k V_cell)) sum_k sum_n (-f'(E_n)) u^n_a Omega^n_xy
          = (1/(N_k V_cell)) sum_k sum_n f(E_n) d_a Omega^n_xy,

    the quantity behind the nonlinear Hall effect of a time-reversal
    symmetric metal with broken inversion (Sodemann and Fu,
    arXiv:1508.00571). In 2D it is the array [D_x, D_y], with
    sigma^bcd_xyy = tau D_y and sigma^bcd_yxx = -tau D_x in
    nonlinear_dc_conductivity. In 3D it is the tensor D[a,b] with the
    pseudovector Omega_x = Omega_yz, Omega_y = Omega_zx, Omega_z = Omega_xy,
    D[a,b] = <w u_a Omega_b>, so that sigma^bcd_xyy = tau D[y,z] and
    sigma^bcd_yxx = -tau D[x,z], and cyclically. D is an in-plane pseudovector, so it vanishes under a C3 rotation and
    needs a crystal with at most one mirror. In units of a^(3-d), a length
    in 2D and dimensionless in 3D, with the keywords and the mesh rule of
    nonlinear_dc_conductivity, T > 0 included."""
    _check_temperature(T,"the Berry curvature dipole")
    return nonlineardc.berry_curvature_dipole(h,nk=nk,T=T,channel=channel,
            degeneracy_tol=degeneracy_tol)


def shift_current(h,energies=None,nk=20,T=None,delta=0.1,components=None,
        degeneracy_tol=1e-6,prec="double"):
    """Shift-current conductivity of the bulk photovoltaic effect, the
    part of the rectified current under light that stays finite as the
    broadening goes to zero,

      sigma^shift_abc(w) = sigma_abc(0;w,-w) - eta_abc(w)/delta,

    with J_a = (1/2) sigma^shift_abc E0_b E0_c^* for the field
    E(t) = Re E0 exp(-i w t). The rectified total is
    nonlinear_conductivity at frequencies=(1,-1), and eta is the injection
    rate of injection_current, the pole of that total, both at the same
    delta, nk and T. The split is done in this way because only the total
    is gauge invariant; the velocity-gauge diagonal is not the pole. The
    shift current needs broken inversion and is allowed with time
    reversal, where eta vanishes for linear light and the shift current is
    the whole real, bc-symmetric response.

    The total is Hermitian in b and c, sigma_acb = sigma_abc^*. With broken
    time reversal the 1/delta pole amplifies the aliasing of the mesh, so
    a larger nk is needed than for the second harmonic. In a metal the
    total also holds Fermi-surface terms that eta does not, and the
    remainder is then not only the shift current. Units e^3 a^(3-d)/(hbar
    t), keywords as nonlinear_conductivity, T defaulting to delta.
    Returns (energies, array) for one component string and (energies,
    dict) otherwise."""
    _check_energies(energies,"the shift current")
    _check_components(components,"the shift current")
    return photocurrent.shift_current(h,energies=energies,nk=nk,T=T,
            delta=delta,components=components,
            degeneracy_tol=degeneracy_tol,prec=prec)


def injection_current(h,energies=None,nk=20,T=None,delta=0.1,
        components=None,degeneracy_tol=1e-6):
    """Injection rate eta_abc(w) of the bulk photovoltaic effect, the
    current that grows linearly in time under light,
    dJ_a/dt = eta_abc E0_b E0_c^* for E(t) = Re E0 exp(-i w t), until the
    relaxation stops it. From the golden rule,

      eta_abc = (pi q^3/2) (1/(N_k V_cell)) sum_k sum_{n != m} (f_n - f_m)
                (v^a_mm - v^a_nn) r^b_mn r^c_nm L_delta(w - (E_m - E_n))

    with r^a_nm = h^a_nm/(i (E_n - E_m)) the interband Berry connection and
    L_delta a Lorentzian of width delta; inside a degenerate multiplet the
    velocity difference becomes a commutator with the velocity restricted
    to the multiplet, which is basis independent.

    eta is complex, with eta_acb = eta_abc^*. Its real part, symmetric in
    b and c, is the linear injection, under linearly polarized light; it
    vanishes with time reversal. Its imaginary part, antisymmetric in b
    and c, is the circular injection, eta^circ_abc = Im eta_abc, under
    circularly polarized light; it is allowed with time reversal once
    inversion is broken, and its trace over a window where one Weyl node
    absorbs is quantized. For E0 = (1, i, 0)/sqrt(2) the rate is
    dJ_a/dt = (Re eta_axx + Re eta_ayy)/2 + Im eta_axy.

    With the adiabatic switching of nonlinear_conductivity the rate
    integrates to J = eta |E0|^2/(2 delta), so eta = delta sigma^pole with
    sigma^pole the pole part of sigma_abc(0;w,-w). Units e^3
    a^(3-d)/hbar^2 (a rate of the second-order current), T defaulting to
    delta, keywords as nonlinear_conductivity. Returns (energies, array)
    for one component string and (energies, dict) otherwise."""
    _check_energies(energies,"the injection current")
    _check_components(components,"the injection current")
    return photocurrent.injection_current(h,energies=energies,nk=nk,T=T,
            delta=delta,components=components,
            degeneracy_tol=degeneracy_tol)


# integration="kpm": the Chebyshev double expansion of conductivitytk/kpmkubo.py.
# It takes keywords of its own (npol, ntries, kernel...), so it is dispatched
# here, before the band signature, and the accepted names are the union of
# the band registry (kubo) and the KPM one.
from .conductivitytk import kpmkubo
_optical_conductivity_bands = optical_conductivity


def get_integration_names():
    """Accepted values of integration= in optical_conductivity"""
    return kubo.get_integration_names()+kpmkubo.get_integration_names()


def optical_conductivity(h,*args,integration="grid",**kwargs):
    if integration not in get_integration_names():
        raise ValueError("unknown integration "+repr(integration)+" for the "
                "optical conductivity, accepted values are "
                +", ".join(get_integration_names()))
    if integration in kpmkubo.get_integration_names():
        return kpmkubo.optical_conductivity(h,*args,**kwargs)
    return _optical_conductivity_bands(h,*args,integration=integration,
            **kwargs)
optical_conductivity.__doc__ = _optical_conductivity_bands.__doc__ + \
        kpmkubo.optical_conductivity.__doc__
