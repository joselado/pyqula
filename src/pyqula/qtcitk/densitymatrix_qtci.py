# qtci (quantics/Gauss-Kronrod tensor cross interpolation) alternative to
# scftk/densitydensity.py's get_dm.
#
# get_dm computes a dense density-matrix block per direction by exact
# diagonalization on a k-mesh (densitymatrix.py's full_dm), averaging the
# per-k occupied-state projector with np.mean-like normalization. Here we
# instead replace that discrete k-mesh average by a continuous BZ integral,
# done element by element: for each required (direction d, row i, column j)
# entry, dm[d][i,j] = integral over the BZ of exp(2*pi*i*k.d)*P(k)[i,j],
# where P(k) is the occupied-state projector of the Bloch Hamiltonian H(k)
# (same physics/convention as dmtk/fulldm.py's full_dm_python, just
# evaluated at a continuous k instead of summed over a fixed mesh). Each
# entry's integral is approximated with qutecipy via gkintegrate's shared
# nk-to-GKorder mapping and zero-pivot-safe integration (see topology.
# chern_qtci for the same helpers): GKorder grows only logarithmically
# with nk, since Gauss-Kronrod quadrature converges spectrally -- for a
# smooth integrand, i.e. a gapped system. In a metal at small T the
# projector jumps at the Fermi surface and the rule converges slowly; the
# SCF loop then locates the Fermi level on these same nodes
# (get_fermi4filling_qtci below) so that the charge at least matches.
#
# Only the (direction,i,j) entries actually needed by the interaction
# dictionary "v" are computed (kpmtk.densitymatrix_kpm.required_elements),
# mirroring that module's own "don't build a dense block you don't need"
# strategy -- important here since every entry pays its own qtci overhead.
# H(k) is diagonalized at most once per k-point regardless of how many
# entries/directions need it, via a plain dict cache shared across every
# entry's independent integral (qutecipy's own pivot search, within a
# single entry's integral, revisits the same k-nodes many times too; see
# CachedFunction in fermisurfacetk/singlefs.py for that finer-grained
# case -- here the cache is coarser, shared across entries instead).
#
# Each entry is integrated by its own independent, sequential qutecipy
# call (not parallelized via paralleltk.pcall, unlike kpmtk's per-k KPM
# engine): the cache above is a plain in-process dict keyed by k, and its
# whole point is to be shared across every entry's pivot search within one
# process; handing entries out to separate worker processes would give
# each one its own empty cache, forcing H(k) to be rediagonalized from
# scratch per worker instead of at most once overall.
import numpy as np

from .. import algebra
from ..dmtk.fulldm import full_dm_python
from ..kpmtk.densitymatrix_kpm import required_elements, required_elements_eh
from .gkintegrate import gkorder_from_nk, integrate_robust

DEFAULT_NK = 8


def _dm_qtci_from_needed(h, needed, nk=DEFAULT_NK, fermi=0.0, T=1e-7,
        tolerance=1e-6, **kwargs):
    """Shared per-entry qtci engine: given the (direction, row, col)
    density-matrix entries to compute (see required_elements/
    required_elements_eh), integrate each one over the BZ with qutecipy.
    H(k) is diagonalized at most once per (kx,ky) node visited by any
    entry's pivot search (shared cache), not once per entry."""
    if h.dimensionality != 2:
        raise NotImplementedError("get_dm_qtci only supports 2D "
                "Hamiltonians (the qutecipy BZ integration is over kx,ky "
                "in [0,1]x[0,1]); got dimensionality=%d"%h.dimensionality)
    hk_gen = h.get_hk_gen()
    norb = h.intra.shape[0]
    Tsafe = abs(T) if T!=0. else 1e-15
    GKorder = gkorder_from_nk(nk)

    cache = {} # (kx,ky) -> occupied-state projector matrix, shared across
    def projector(k):                     # every entry computed below
        key = (k[0],k[1])
        if key not in cache:
            hk = hk_gen(np.array([k[0],k[1],0.]))
            es,vs = algebra.eigh(hk)
            es = es-fermi
            cache[key] = full_dm_python(es,vs.T,delta=Tsafe)
        return cache[key]

    ds = sorted({d for (d,i,j) in needed})
    dm = {d: np.zeros((norb,norb),dtype=np.complex128) for d in ds}
    needed_by_d = dict()
    for d,i,j in needed: needed_by_d.setdefault(d,[]).append((i,j))
    for d in ds:
        dvec = np.array(d,dtype=np.float64)
        for (i,j) in needed_by_d[d]:
            def f(k,i=i,j=j): # default args freeze the loop variables
                phase = np.exp(2j*np.pi*(k[0]*dvec[0]+k[1]*dvec[1]))
                return projector(k)[i,j]*phase
            dm[d][i,j] = integrate_robust(np.complex128,f,GKorder,
                    tolerance,**kwargs)
    return dm


def gk_node_grid(nk=None):
    """The tensorized Gauss-Kronrod rule get_dm_qtci integrates with, at
    this nk: returns (kx, ky, w), flat arrays over every node pair, with
    the product weights w summing to one on [0,1]^2. It is the same rule
    integrate_robust hands to qutecipy (same kronrod call, same GKorder
    from gkorder_from_nk), so a sum over it is the quadrature the density
    matrix is built with."""
    from ..qutecipytk.gausskronrod import kronrod
    if nk is None: nk = DEFAULT_NK
    nodes1d,weights1d,_ = kronrod(gkorder_from_nk(nk)//2,-1,1)
    knode = (nodes1d+1)/2 # [-1,1] -> [0,1]
    w1 = weights1d/2 # so that the weights sum to one on [0,1]
    kx,ky = np.meshgrid(knode,knode,indexing="ij")
    w = np.outer(w1,w1)
    return kx.ravel(),ky.ravel(),w.ravel()


def get_fermi4filling_qtci(h, filling, nk=None, T=1e-7):
    """Fermi energy holding `filling` electrons under the quadrature that
    get_dm_qtci integrates the density matrix with.

    spectrum.get_fermi4filling counts eigenvalues on the uniform nk mesh.
    In a metal at small T the occupied-state projector has a step at the
    Fermi surface, and a Fermi level located on one grid holds a
    different number of electrons on another: taking it from the uniform
    mesh and then integrating on the Gauss-Kronrod nodes lost 0.03-0.05
    electrons out of 0.6 on a square-lattice metal. Here the Fermi level
    is located on the Gauss-Kronrod nodes themselves, so the density
    matrix holds the requested charge up to the weight of one level.

    The nodes are not equally weighted, so the count is a staircase with
    uneven steps and at T~0 no Fermi level lands exactly on the target.
    As in filling.get_fermi_energy, the cut goes midway between two
    eigenvalues, at the one whose cumulative weight is closest to the
    target; it never sits on a degenerate level. The steps can be coarse:
    the nodes come in groups related by the lattice symmetries, and on a
    square lattice at nk=8 the level next to the Fermi energy of a 0.3
    filling is a group of 8 states that holds 0.09 electrons, and the
    closest cut misses 0.6 electrons by 0.04. When T is not small against
    the level spacing the Fermi-Dirac count is continuous, and the cut is
    refined by bisection to the requested filling itself.

    For a BdG Hamiltonian the count is done on the normal-state
    Hamiltonian, the same approximation spectrum.get_fermi4filling makes."""
    if h.has_eh:
        h0 = h.copy()
        h0.remove_nambu()
        return get_fermi4filling_qtci(h0,filling,nk=nk,T=T)
    if h.dimensionality != 2:
        raise NotImplementedError("get_fermi4filling_qtci only supports 2D "
                "Hamiltonians, the ones get_dm_qtci integrates; got "
                "dimensionality=%d"%h.dimensionality)
    from ..filling import check_filling
    check_filling(filling)
    kx,ky,w = gk_node_grid(nk)
    hk_gen = h.get_hk_gen()
    norb = h.intra.shape[0]
    es = np.array([algebra.eigvalsh(hk_gen(np.array([x,y,0.])))
        for (x,y) in zip(kx,ky)]) # (nodes,norb)
    ws = np.repeat(w,norb) # the weight of every eigenvalue
    es = es.ravel()
    order = np.argsort(es)
    es,ws = es[order],ws[order]
    target = filling*norb # electrons per unit cell
    cum = np.cumsum(ws) # electrons below each cut
    # a cut is allowed only between two distinct eigenvalues
    gap = np.diff(es) > 1e-10*max(1.,np.max(np.abs(es)))
    icuts = np.nonzero(gap)[0] # cut between es[i] and es[i+1]
    e_reg = 1e-5 # as filling.get_fermi_energy, for an empty or full band
    candidates = [(abs(target),es[0]-e_reg,0.)] # nothing occupied
    candidates += [(abs(cum[i]-target),(es[i]+es[i+1])/2.,cum[i])
            for i in icuts]
    candidates.append((abs(cum[-1]-target),es[-1]+e_reg,cum[-1]))
    _,mu0,ntarget = min(candidates,key=lambda c: c[0])
    if T is None or T<=0.: return mu0
    if ntarget<=0. or ntarget>=norb: return mu0 # empty or full
    from scipy.special import expit
    def nelec(mu): return np.sum(ws*expit(-(es-mu)/T))
    if abs(nelec(mu0)-ntarget)<1e-9*ntarget: return mu0 # T below spacing
    # T smears the staircase into a continuous count, so the requested
    # filling itself is reachable and is what the bisection aims for
    ntarget = target
    from scipy.optimize import brentq
    width = max(4.*T,1e-6)
    wmax = 4.*(es[-1]-es[0]) + 40.*T + 1e-6
    while nelec(mu0-width)>ntarget or nelec(mu0+width)<ntarget:
        if width>wmax: return mu0 # cannot bracket; keep the T=0 cut
        width *= 2.
    return brentq(lambda mu: nelec(mu)-ntarget,mu0-width,mu0+width,
            xtol=1e-12)


def get_dm_qtci(h, v, nk=None, fermi=0.0, T=1e-7, tolerance=1e-6,
        **kwargs):
    """qtci-based analogue of scftk.densitydensity.get_dm: return
    a dictionary {direction: matrix} with the density matrix, computing
    only the entries v actually requires, each one as a BZ integral (via
    qutecipy) of the occupied-state projector instead of a k-mesh average.
    Same {direction: matrix} contract as get_dm, so it is a drop-in
    replacement inside the same (conventional) SCF loop -- see
    scftk.densitydensity.get_dm's integration="qtci" branch.

    A direction with no required entries at all (e.g. an interaction v[d]
    that is exactly all-zero for that d) is backfilled below with an
    all-zero matrix rather than actually integrated -- correct for the
    mean field itself (a zero-everywhere v[d] contributes nothing
    regardless), but note dm[d] is then a placeholder, not a genuinely
    computed occupation/coherence matrix; the same convention is used by
    kpmtk.densitymatrix_kpm.get_dm_kpm.

    For BdG/Nambu Hamiltonians (h.has_eh) the required entries are
    determined by required_elements_eh instead of required_elements,
    exactly as in kpmtk.densitymatrix_kpm.get_dm_kpm (same restriction to
    spinful Nambu Hamiltonians)."""
    if nk is None: nk = DEFAULT_NK
    if getattr(h,"has_eh",False) and not getattr(h,"has_spin",True):
        raise NotImplementedError("get_dm_qtci's BdG/Nambu path only "
                "supports spinful Hamiltonians (h.has_spin=True); "
                "spinless_nambu uses a different Nambu index convention "
                "not implemented here")
    ds = [(0,0,0)] + [d for d in v if d!=(0,0,0)]
    if getattr(h,"has_eh",False):
        needed = required_elements_eh(v)
    else:
        needed = required_elements(v)
    dm = _dm_qtci_from_needed(h,needed,nk=nk,fermi=fermi,T=T,
            tolerance=tolerance,**kwargs)
    # every direction v has a key for must be present in the output, even
    # if it happened to contribute no required entries of its own
    for d in ds:
        if d not in dm:
            dm[d] = np.zeros((h.intra.shape[0],h.intra.shape[0]),
                    dtype=np.complex128)
    return dm
