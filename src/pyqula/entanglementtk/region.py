# Resolution of the "region A" argument of the entanglement routines into
# the orbital indices of the Hamiltonian matrix. See
# pyqula/entanglement.py for the user-facing description of the accepted
# forms.
import numpy as np
from .. import sculpt


def orbitals_per_site(h):
    """Number of matrix rows belonging to one site of h.geometry (1 for a
    spinless Hamiltonian, 2 for spinful or spinless-Nambu, 4 for
    spinful-Nambu). pyqula stores those orbitals contiguously and
    site-major, so site i owns rows [i*norb,(i+1)*norb) -- the assumption
    that lets a spatial region be lifted to matrix indices."""
    n = h.intra.shape[0] # dimension of the Hamiltonian matrix
    ns = len(h.geometry.r) # number of sites
    if n%ns!=0: raise ValueError(
        "Hamiltonian dimension %d is not a multiple of the number of "
        "sites %d, cannot map a spatial region onto orbitals"%(n,ns))
    return n//ns


def fractional_coordinate(g,direction):
    """Fractional coordinate of every site along a lattice vector, folded
    into [0,1). pyqula's own convention puts them in [-1/2,1/2), so the
    folding is what makes "the first half of the cells" mean the same
    thing regardless of that choice."""
    if g.dimensionality<=direction: raise ValueError(
        "Cannot use a fractional cut along direction %d for a "
        "%d-dimensional geometry"%(direction,g.dimensionality))
    g.get_fractional() # compute (or refresh) the fractional coordinates
    return np.array(g.frac_r)[:,direction]%1.0


def site_indices(h,region=None,direction=0):
    """Sites of h.geometry belonging to region A. See
    entanglement.entanglement_entropy for the accepted forms of
    `region`."""
    g = h.geometry
    ns = len(g.r) # number of sites
    if region is None: # default region
        if g.dimensionality==0: # finite system: the left half
            return np.sort(np.argsort(g.r[:,0])[:ns//2])
        else: region = 0.5 # periodic: half of the cells along the cut
    if callable(region): # a selector on the positions, the sculpt idiom
        return np.array(sculpt.intersected_indexes(g,region),dtype=int)
    if np.isscalar(region): # a fraction of the cells along the cut
        f = float(region)
        if not 0.<f<1.: raise ValueError(
            "A scalar region must be a fraction in (0,1), got %s"%region)
        frac = fractional_coordinate(g,direction)
        return np.where(frac<f)[0]
    region = np.array(region)
    if region.dtype==bool: # a mask over the sites, the sculpt "store" idiom
        return np.where(region)[0]
    return region.astype(int) # an explicit list of site indices


def orbital_indices(h,**kwargs):
    """Matrix indices of the orbitals of region A (see site_indices and
    orbitals_per_site)."""
    sites = site_indices(h,**kwargs)
    ns = len(h.geometry.r)
    if len(sites)>0 and (np.min(sites)<0 or np.max(sites)>=ns):
        raise ValueError("Site index out of range in the entanglement region")
    norb = orbitals_per_site(h)
    if norb==1: return np.sort(sites)
    out = [np.arange(i*norb,(i+1)*norb) for i in np.sort(sites)]
    if len(out)==0: return np.array([],dtype=int)
    return np.concatenate(out)
