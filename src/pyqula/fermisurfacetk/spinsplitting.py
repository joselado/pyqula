import numpy as np
from .. import algebra
from ..check import require_spin

def average_spin_splitting(h,nk=20,tol=algebra.error):
    """Compute the average spin splitting in the BZ.

    The average runs over bands as well as over kpoints, so the result is
    the splitting of a typical band and does not depend on how big a cell
    the same crystal is described in. Summing over bands instead -- which
    is what this did -- makes it grow with the number of bands: a uniform
    Zeeman field splitting every band by 0.6 gave 1.2 on the primitive
    honeycomb cell and 4.8 / 10.8 on its 2x / 3x supercells."""
    check_collinear(h,tol=tol) # refuse rather than return a wrong number
    hup = h.copy() ; hup.remove_spin(channel="up") 
    hdn = h.copy() ; hdn.remove_spin(channel="dn") 
    hkup = hup.get_hk_gen() # get generator
    hkdn = hdn.get_hk_gen() # get generator
    from ..klist import kmesh
    ks = kmesh(h.geometry.dimensionality,nk=nk)
    def am(k):
        # sorted for the same reason as in spin_splitting_vs_energy:
        # algebra.eigvalsh concatenates the two halves of a block diagonal
        # matrix without sorting them, which scrambles the band pairing
        eup = np.sort(algebra.eigvalsh(hkup(k))) # eigenvalues for up
        edn = np.sort(algebra.eigvalsh(hkdn(k))) # eigenvalues for dn
        de = (eup-edn)**2 # square difference
        return np.mean(np.sqrt(de)) # square root, averaged over bands
    out = np.mean([am(k) for k in ks]) # average altermagnetism
    return out # return result





def spin_splitting_density(h,nk=20,energies=None,delta=1e-2,tol=algebra.error):
    """Compute the average spin splitting in the BZ"""
    if energies is None: energies = np.linspace(-3.0,3.0,400)
    check_collinear(h,tol=tol) # refuse rather than return a wrong number
    hup = h.copy() ; hup.remove_spin(channel="up")
    hdn = h.copy() ; hdn.remove_spin(channel="dn")
    hkup = hup.get_hk_gen() # get generator
    hkdn = hdn.get_hk_gen() # get generator
    from ..klist import kmesh
    ks = kmesh(h.geometry.dimensionality,nk=nk)
    from ..dos import calculate_dos
    def am(k):
        # sorted, so that the two channels are paired by band index
        eup = np.sort(algebra.eigvalsh(hkup(k))) # eigenvalues for up
        edn = np.sort(algebra.eigvalsh(hkdn(k))) # eigenvalues for dn
        de = (eup-edn)**2 # square difference
        ea = (eup + edn)/2. # average
        return calculate_dos(ea,energies,delta,w=de)
#        return np.sum(np.sqrt(de)) # square root
    out = np.mean([am(k) for k in ks],axis=0) # average altermagnetism
    return energies,out # return result




def check_collinear(h,tol=algebra.error,ks=None):
    """Raise unless spin up and down are good quantum numbers, i.e. the
    spin off-diagonal block of the Bloch Hamiltonian vanishes everywhere.

    This is not a formality. The spin-resolved quantities below are built
    with remove_spin, which keeps only one spin block and drops the
    off-diagonal one SILENTLY -- with Rashba, any other spin-orbit
    coupling, or non-collinear magnetic order the result is a plausible
    but meaningless number rather than an error. So it is checked here
    instead of trusted."""
    require_spin(h,"the spin splitting")
    if h.has_eh:
        raise ValueError("spin splitting is not defined for a Nambu "
                "(superconducting) Hamiltonian: its spin blocks mix "
                "electrons and holes")
    hk = h.get_hk_gen() # Bloch generator of the FULL spinful Hamiltonian
    if ks is None: # a handful of deliberately unsymmetric points
        ks = [[0.,0.,0.],[0.123,0.456,0.789],[0.317,0.113,0.529]]
    worst = 0. # largest off-diagonal element found
    for k in ks:
        m = algebra.todense(hk(k)) # Bloch matrix at this k
        n = m.shape[0] # dimension
        # spin is the fast index (see increase_hilbert.des_spin), so the
        # up-down block is the even rows against the odd columns
        worst = max(worst,np.max(np.abs(m[0:n:2,1:n:2])))
    if worst>tol:
        raise ValueError("spin up and down are not good quantum numbers "
                "(largest spin off-diagonal element %g > %g). Spin-orbit "
                "coupling or non-collinear order makes the spin-resolved "
                "splitting meaningless; remove_spin would drop that block "
                "without warning."%(worst,tol))


def _bin_max(x,w,centers):
    """Largest w in each bin of x, with bins given by their centers.

    Bins with no entries stay 0.0 rather than becoming NaN, so the result
    plots cleanly. Entries falling outside the outermost bin edges are
    DROPPED, not clamped onto the end bins -- clamping would pile every
    out-of-window state onto the first and last points and invent a peak
    that is not there."""
    centers = np.array(centers)
    out = np.zeros(len(centers)) # empty bins are 0.0, not NaN
    if len(centers)==1: # degenerate mesh, everything in one bin
        if len(w)>0: out[0] = np.max(w)
        return out
    edges = (centers[1:]+centers[:-1])/2. # midpoints between centers
    idx = np.searchsorted(edges,x) # bin index of every entry, 0..n-1
    # searchsorted alone cannot tell "below the first bin" from "in it",
    # so the window is masked explicitly. Half-widths come from the end
    # spacings, which keeps this correct for a non-uniform mesh too.
    lo = centers[0]-(centers[1]-centers[0])/2.
    hi = centers[-1]+(centers[-1]-centers[-2])/2.
    mask = (x>=lo)&(x<=hi)
    np.maximum.at(out,idx[mask],w[mask]) # unbuffered, so repeats accumulate
    return out


def spin_splitting_vs_energy(h,nk=100,energies=None,nbins=400,
        emin=None,emax=None,tol=algebra.error):
    """Energy-resolved MAXIMUM spin splitting over the Brillouin zone.

    For every k of a uniform Gamma-centered mesh spanning the full reduced
    BZ, the spin-up and spin-down bands are diagonalized separately and
    paired by band INDEX (both sorted ascending),

        Delta_n(k) = E^up_n(k) - E^dn_n(k)
        Ebar_n(k)  = (E^up_n(k) + E^dn_n(k))/2

    and each pair is binned at the energy Ebar_n(k) it actually sits at.
    The returned curve is the largest |Delta| found in each bin, so its
    global maximum is the largest spin splitting anywhere in the BZ.

    This is the BZ-wide counterpart of `spin_splitting_density`, which
    instead broadens every pair into a smooth weighted density. Both
    return (energies, values) with the same shape convention, so the two
    are directly comparable on the same axes.

    Unlike a maximum taken along a single k-space cut (a circle of fixed
    radius, say), nothing here depends on choosing where to look: the
    bound covers the whole zone.

    There are two separate things to know about the index pairing, and
    they have different answers.

    WHEN IS THE PAIRING UNAMBIGUOUS? Whenever the two spin channels are
    related by a point-group operation acting on k -- which is what a
    collinear altermagnet is -- the sorted spectra satisfy
    E^up_n(k) = E^dn_n(Rk) exactly, so

        Delta_n(k) = E^dn_n(Rk) - E^dn_n(k)

    compares the same sorted index within ONE channel at two related
    momenta. There is no "which band is which" freedom left, and the
    usual worry (that a splitting exceeding the band spacing makes the
    n-th up and n-th down band different bands) simply does not arise:
    the symmetry supplies the correspondence. Only without such a
    relation does band order become an assumption worth doubting. The
    operation is worth identifying for a new system -- compare
    sorted(E_up(k)) with sorted(E_dn(Rk)) over the point group and look
    for the R that gives ~1e-14 -- since it also tells you the pairing
    is trustworthy.

    WHICH UNIT CELL? This one is a real dependence, and the symmetry
    above does NOT remove it: n indexes whatever band set the cell
    produces, and folding changes that set. On a supercell of the square
    altermagnet the mirror identity still holds to 4e-16, yet the
    reported maximum drops from 4*am to 2*am, because the folded bands
    at one k come from several primitive k-points and index pairing
    compares across them. So give this the true magnetic unit cell: if
    the converged order repeats with a smaller period than the cell it
    was solved in, the result is a lower bound rather than the maximum.
    Both facts are pinned in
    tests/fermisurface/test_spin_splitting_vs_energy.py.

    Parameters
      nk        linear mesh density (nk**d points in d dimensions)
      energies  explicit bin centers; otherwise nbins points spanning
                emin..emax, which default to the range of Ebar actually
                found, so no state falls outside the window
      tol       largest spin off-diagonal element tolerated before this
                refuses to answer (see check_collinear)

    Diagonalization is dense throughout, deliberately: a sparse solver
    returns only the eigenvalues nearest E=0, and the splitting commonly
    peaks far from there, so a sparse shortcut would silently miss the
    maximum this function exists to find."""
    check_collinear(h,tol=tol) # refuse rather than return a wrong number
    hup = h.copy() ; hup.remove_spin(channel="up") # mutates, hence the copy
    hdn = h.copy() ; hdn.remove_spin(channel="dn")
    hkup = hup.get_hk_gen() # get generator
    hkdn = hdn.get_hk_gen() # get generator
    from ..klist import kmesh
    ks = kmesh(h.geometry.dimensionality,nk=nk)
    ebars,deltas = [],[]
    for k in ks:
        # sorted explicitly: algebra.eigvalsh concatenates the two halves
        # of a block-diagonal matrix without sorting them, which would
        # scramble the band-index pairing this function is built on
        eup = np.sort(algebra.eigvalsh(hkup(k))) # spin up bands
        edn = np.sort(algebra.eigvalsh(hkdn(k))) # spin down bands
        deltas.append(eup-edn) # splitting, paired by band index
        ebars.append((eup+edn)/2.) # energy each pair sits at
    ebar = np.concatenate(ebars) # every (k,band) pair
    delta = np.abs(np.concatenate(deltas)) # magnitude of the splitting
    if energies is None: # build the mesh from what was found
        if emin is None: emin = np.min(ebar)
        if emax is None: emax = np.max(ebar)
        energies = np.linspace(emin,emax,nbins)
    else: energies = np.sort(np.array(energies)) # searchsorted needs order
    return energies,_bin_max(ebar,delta,energies) # return result
