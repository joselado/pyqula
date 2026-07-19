# qtci (quantics/Gauss-Kronrod tensor cross interpolation) alternative to
# selfconsistency/densitydensity.py's get_dm.
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
# with nk, since Gauss-Kronrod quadrature converges spectrally.
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


def get_dm_qtci(h, v, nk=None, fermi=0.0, T=1e-7, tolerance=1e-6,
        **kwargs):
    """qtci-based analogue of selfconsistency.densitydensity.get_dm: return
    a dictionary {direction: matrix} with the density matrix, computing
    only the entries v actually requires, each one as a BZ integral (via
    qutecipy) of the occupied-state projector instead of a k-mesh average.
    Same {direction: matrix} contract as get_dm, so it is a drop-in
    replacement inside the same (conventional) SCF loop -- see
    selfconsistency.densitydensity.get_dm's integration="qtci" branch.

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
