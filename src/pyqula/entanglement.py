# Free-fermion entanglement entropy and entanglement spectrum, via the
# correlation-matrix method.
#
# For a Slater determinant (a non-interacting or mean-field ground state)
# all many-body correlators factorize, and the reduced density matrix of a
# spatial region A is itself the exponential of a free-fermion operator,
#
#   rho_A = K exp( -sum_{ij in A} h_ij c_i^dag c_j )
#
# with the "entanglement Hamiltonian" h fixed by the requirement that
# rho_A reproduce the one-particle correlations inside A,
#
#   C_ij = <c_i^dag c_j>   (i,j in A)  ==>  h = ln[ (1 - C_A)/C_A ]
#
# (Peschel, J. Phys. A 36, L205 (2003) [cond-mat/0212631]; reviewed with
# this notation in Peschel & Eisler, J. Phys. A 42, 504003 (2009)
# [arXiv:0906.1663], Eqs. (14)-(18), which is the reference followed
# here). Diagonalizing the restricted correlation matrix C_A gives
# eigenvalues zeta_n in [0,1], from which
#
#   S   = -sum_n [ zeta_n ln zeta_n + (1 - zeta_n) ln(1 - zeta_n) ]
#   xi_n = ln[ (1 - zeta_n)/zeta_n ]
#
# are the entanglement entropy and the single-particle entanglement
# spectrum. Only an len(A) x len(A) matrix is ever diagonalized, never the
# exponentially large rho_A.
#
# CONVENTIONS AND SCOPE
#
# - Occupation is a hard T=0 cut: the occupied states are those with
#   E < fermi (default 0, the Fermi-level convention used throughout
#   pyqula). A level sitting AT the Fermi energy makes the ground state
#   degenerate and raises, rather than silently returning the entropy of
#   an arbitrary determinant -- see
#   entanglementtk.correlation.occupied_correlation_matrix.
# - Spin, sublattice and Nambu components are just extra orbitals of a
#   site: a region is always given in terms of SITES and lifted to the
#   orbitals of those sites.
# - Periodic Hamiltonians (dimensionality 1 or 2) are cut by stacking
#   `nsuper` unit cells along one lattice direction and evaluating the
#   Bloch matrix at zero momentum along it, which is a periodic RING of
#   nsuper cells. Region A is then a set of consecutive cells of that
#   ring, so the region has TWO entanglement boundaries, not one. For a 2D
#   Hamiltonian the momentum parallel to the cut stays a good quantum
#   number, and the resulting xi_n(k_par) is the Li-Haldane entanglement
#   spectrum (Li & Haldane, PRL 101, 010504 (2008) [arXiv:0805.0332]):
#   for a Chern insulator its mid-gap branches flow across xi=0 and count
#   2|C| (|C| chiral modes per boundary, two boundaries), mirroring the
#   edge spectrum of the same model. Entropies of a 2D Hamiltonian are
#   likewise per parallel unit cell and count both boundaries.
# - Nambu/BdG Hamiltonians (h.has_eh) are handled with the same formulas
#   applied to the FULL Nambu correlation matrix, which includes the
#   anomalous <c c> blocks (Peschel & Eisler Eq. (18) and the equivalent
#   Majorana form below it). Because the Nambu basis doubles every
#   orbital, the eigenvalues of C_A come in (zeta,1-zeta) pairs and the
#   naive entropy sum counts every physical mode twice, so it is halved
#   here; the entanglement spectrum correspondingly comes in +-xi pairs.
#   The BdG chemical potential lives inside the Hamiltonian, so `fermi`
#   must stay 0 there.
#
# Verified in tests/entanglement: the c=1 CFT log law of the half-filled
# chain (which pins the absolute normalization), the area law of a gapped
# 2D insulator, the A <-> B symmetry of a pure state, S=0 for a filled or
# empty region, the Li-Haldane counting against pyqula's own
# topology.chern, and the BdG factor 1/2 against the normal-state result
# of the same Hamiltonian at zero pairing.
import numpy as np
from .entanglementtk import correlation
from .entanglementtk import region as regiontk
from .entanglementtk import spectra


def get_correlation_generator(h,region=None,nsuper=10,direction=None,
        fermi=0.0,degeneracy_tol=correlation.degeneracy_tol):
    """Return a function kpar -> C_A giving the restricted correlation
    matrix of region A at a parallel momentum, the object every other
    function in this module is built on. The ring, the region indices and
    the Bloch generator are built once here and reused at every momentum,
    instead of being rebuilt inside the k-loop (building the supercell is
    more expensive than diagonalizing it for a typical cut). See
    entanglement_entropy for the arguments."""
    if h.dimensionality>2: raise NotImplementedError(
        "entanglement cuts are implemented for dimensionality 0, 1 and 2")
    if h.has_eh and fermi!=0.0: raise ValueError(
        "fermi must be 0 for a Nambu/BdG Hamiltonian: its chemical "
        "potential is already inside the Hamiltonian, and shifting the "
        "Bogoliubov spectrum instead would break the particle-hole "
        "structure the entanglement formulas rely on")
    if h.dimensionality==0: # finite system, cut it directly
        hs,direction = h,0
    else: # periodic system, build the ring first
        direction = correlation.cut_direction(h,direction=direction)
        hs = correlation.cut_hamiltonian(h,nsuper=nsuper,direction=direction)
    idx = regiontk.orbital_indices(hs,region=region,direction=direction)
    hkgen = hs.get_hk_gen() # Bloch generator of the ring, built once
    def f(kpar=0.0):
        k = correlation.kvector(hs.dimensionality,direction,kpar)
        C = correlation.occupied_correlation_matrix(hkgen(k),fermi=fermi,
                degeneracy_tol=degeneracy_tol)
        return correlation.restrict(C,idx)
    return f


def entanglement_correlation_matrix(h,kpar=0.0,**kwargs):
    """Restricted one-particle correlation matrix C_A of region A at a
    single parallel momentum. See entanglement_entropy for the
    arguments."""
    if kpar is None: kpar = 0.0
    return get_correlation_generator(h,**kwargs)(kpar)


def entanglement_entropy(h,kpar=None,nk=20,tol=spectra.tol,**kwargs):
    """Entanglement entropy of a spatial region A.

    Parameters
    ----------
    h : Hamiltonian
      Dimensionality 0 (finite system), 1 or 2 (periodic, see below).
    region : optional
      The region A, in one of four forms:
        * None (default) -- half of the system: for a finite system the
          half with the smallest x, for a periodic one the first half of
          the cells along the cut direction.
        * a float f in (0,1) -- the cells whose fractional coordinate
          along the cut direction is below f (periodic systems only).
        * a list/array of ints -- site indices, as in sculpt; or a
          boolean mask over the sites, the sculpt "store" convention.
        * a callable r -> bool on the 3D position of a site, the selector
          convention of sculpt.intersected_indexes.
      Site indices and positions always refer to the RING built for a
      periodic Hamiltonian (nsuper cells), not to the original unit cell.
    nsuper : int
      Number of unit cells of the ring, for a periodic Hamiltonian. The
      region must be small enough compared with it that its two
      boundaries are not artificially close.
    direction : int, optional
      Lattice direction the cut is normal to. Default: the last periodic
      one, so a 2D Hamiltonian is cut normal to a2 and the parallel
      momentum runs along a1.
    kpar : float, optional
      For a 2D Hamiltonian, the conserved momentum parallel to the cut
      (reduced coordinates). With the default None the entropy is
      averaged over a uniform mesh of `nk` parallel momenta, giving the
      entropy per parallel unit cell. A finite or 1D system has no
      momentum parallel to its cut, so there kpar must be None or 0 and
      anything else raises.
    nk : int
      Number of parallel momenta in that mesh.
    fermi : float
      Occupied states are those with E < fermi.
    tol : float
      Correlation eigenvalues closer than this to 0 or 1 are treated as
      empty/full, see entanglementtk.spectra.
    degeneracy_tol : float
      Relative tolerance for detecting a level at the Fermi energy, see
      entanglementtk.correlation.occupied_correlation_matrix.

    Returns
    -------
    S : float
      For a 2D Hamiltonian with kpar=None, the entropy per parallel unit
      cell, counting BOTH boundaries of the region.
    """
    f = get_correlation_generator(h,**kwargs) # kpar -> C_A, built once
    def S(k): # entropy at a single parallel momentum
        zeta = spectra.occupations_from_correlation(f(k),tol=tol)
        return spectra.entropy_from_occupations(zeta,has_eh=h.has_eh,tol=tol)
    if h.dimensionality==2 and kpar is None: # average over the parallel BZ
        ks = np.linspace(0.,1.,nk,endpoint=False) # no doubled endpoint
        return np.mean([S(k) for k in ks])
    return S(0.0 if kpar is None else kpar)


def entanglement_spectrum(h,kpar=None,nk=41,tol=spectra.tol,**kwargs):
    """Single-particle entanglement spectrum xi_n = ln[(1-zeta_n)/zeta_n]
    of a spatial region A. Arguments are those of entanglement_entropy.

    Returns
    -------
    xi : ndarray, shape (nA,)
      For a finite (0D) or 1D Hamiltonian, or whenever a single `kpar` is
      given, the sorted entanglement levels of that single cut.
    (kpar,xi) : (ndarray (nk,), ndarray (nk,nA))
      For a 2D Hamiltonian with kpar=None (the default): the Li-Haldane
      k-resolved entanglement spectrum, on a mesh of `nk` parallel momenta
      running over the whole Brillouin zone (both endpoints included, so
      that a crossing pinned at a high-symmetry point is sampled for odd
      nk).
    """
    f = get_correlation_generator(h,**kwargs) # kpar -> C_A, built once
    def xi(k): # entanglement spectrum at a single parallel momentum
        zeta = spectra.occupations_from_correlation(f(k),tol=tol)
        return spectra.spectrum_from_occupations(zeta,tol=tol)
    if h.dimensionality==2 and kpar is None: # sweep the parallel BZ
        ks = np.linspace(0.,1.,nk) # includes both BZ endpoints
        return ks,np.array([xi(k) for k in ks])
    return xi(0.0 if kpar is None else kpar)
