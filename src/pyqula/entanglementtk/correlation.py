# One-particle correlation matrices for the entanglement calculation, and
# the "cut" geometry used for periodic Hamiltonians. See
# pyqula/entanglement.py for the formalism and the references.
import numpy as np
from .. import algebra

# relative tolerance for the gap between the last occupied and the first
# empty level, see occupied_correlation_matrix
degeneracy_tol = 1e-6


def occupied_correlation_matrix(m,fermi=0.0,degeneracy_tol=degeneracy_tol):
    """One-particle correlation matrix of the Slater determinant built
    from the eigenstates of `m` below `fermi`,

        C_ij = <c_i^dag c_j> = sum_{E_n<fermi} conj(psi_n(i)) psi_n(j)

    This is the T=0, hard-cutoff counterpart of
    densitymatrix.occupied_projector / dmtk.fulldm.full_dm_python, which
    weight the same outer product with a Fermi function of finite width
    `delta`. A smeared occupation is deliberately not used here: a level
    occupied to 1-epsilon rather than 1 makes the GLOBAL state mixed, and
    then the "entropy" of region A is no longer entanglement entropy at
    all (it picks up the thermal entropy of the smearing), which is
    exactly the silent error the degeneracy check below guards against.

    Raises if some level sits at `fermi` (within degeneracy_tol relative
    to the spectral width), i.e. if the occupied set is ambiguous: that
    happens for a metal, or for an even-membered ring at half filling
    whose Fermi level is hit exactly by a k-point (a ring of L sites with
    L divisible by 4, for the simple chain). The ground state is then
    degenerate and there is no single Slater determinant to compute the
    entanglement of."""
    m = algebra.todense(m)
    (es,ws) = algebra.eigh(m) # ws[:,n] is the eigenvector of es[n]
    scale = max(np.max(np.abs(es)),1e-12) # spectral width, as an energy scale
    d = np.min(np.abs(es-fermi)) # distance from the Fermi level
    if d<degeneracy_tol*scale: raise ValueError(
        "A level sits at the Fermi energy (distance %g, spectral scale %g): "
        "the occupied set, and hence the ground state, is ambiguous. This is "
        "expected for a gapless/metallic system, or for a ring whose k-mesh "
        "hits the Fermi points (e.g. a half-filled chain of L sites with L "
        "divisible by 4); change the number of cells, the filling or the "
        "Fermi energy."%(d,scale))
    V = ws[:,es<fermi] # occupied eigenvectors, as columns
    return np.conjugate(V)@V.T # <c_i^dag c_j>


def restrict(C,idx):
    """Restriction C_A of a correlation matrix to the orbitals of region A"""
    idx = np.asarray(idx,dtype=int)
    if len(idx)==0: return np.zeros((0,0),dtype=np.complex128)
    return np.asarray(C)[np.ix_(idx,idx)]


def cut_direction(h,direction=None):
    """Lattice direction cut by the entanglement boundary (the direction
    the region is finite along). The default is the last periodic one, so
    for a 2D Hamiltonian the cut is normal to a2 and the conserved
    parallel momentum runs along a1."""
    if direction is None: direction = h.dimensionality-1
    direction = int(direction)
    if not 0<=direction<h.dimensionality: raise ValueError(
        "direction %d out of range for a %dD Hamiltonian"%(direction,
            h.dimensionality))
    return direction


def cut_hamiltonian(h,nsuper=10,direction=0):
    """Supercell Hamiltonian of `nsuper` unit cells stacked along
    `direction`.

    Evaluated at zero Bloch momentum ALONG that direction (see kvector),
    its Bloch matrix is exactly the Hamiltonian of a periodic ring of
    nsuper cells -- the Bloch phase across the supercell boundary is 1, so
    the ring's own boundary condition is periodic, and its spectrum is the
    union of the bulk bands over the nsuper commensurate momenta. Cutting
    that ring in half therefore produces TWO entanglement boundaries, not
    one (see entanglement.py)."""
    if nsuper<2: raise ValueError("nsuper must be at least 2 to cut a ring")
    ns = [1,1,1]
    ns[direction] = int(nsuper)
    return h.get_supercell(ns)


def kvector(dim,direction,kpar):
    """Bloch vector (reduced coordinates) with zero momentum along the cut
    direction -- so the supercell is a ring, see cut_hamiltonian -- and
    `kpar` along the remaining periodic direction. In 1D the cut is a
    point and there is no momentum parallel to it."""
    k = [0.,0.,0.]
    if dim==0: pass # finite system, no momentum at all
    elif dim==1:
        if abs(kpar)>0.: raise ValueError(
            "A cut of a 1D system has no parallel momentum, kpar must be 0")
    elif dim==2: k[1-direction] = kpar # the direction that stays periodic
    else: raise NotImplementedError(
        "entanglement cuts are implemented for dimensionality 0, 1 and 2")
    return k
