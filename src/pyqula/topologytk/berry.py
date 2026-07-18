# batched, numba-parallel evaluation of the Wilson-loop Berry curvature.
#
# topology.berry_curvature diagonalizes four Hamiltonians per kpoint (the
# corners of a small plaquette around that kpoint) -- for a kmesh of nk*nk
# points that's 4*nk*nk independent diagonalizations, previously dispatched
# one kpoint at a time through parallel.pcall (process-based, the same shape
# of workload densitymatrix.full_dm_accumulate had). Batch-diagonalize the
# corners in parallel across numba threads instead: no interprocess
# communication, and it composes with the rest of the corner/overlap math,
# which stays in plain numpy since it's comparatively cheap (matrices of
# size n_occ x n_occ, not norb x norb).

import numpy as np
import scipy.linalg as lg

from ..htk.eigenvectors import parallel_diagonalization
from .overlap import uij


def berry_curvature_mesh(h,ks,dk=0.01,batch_size=64):
    """Compute the Wilson-loop Berry curvature at every kpoint in ks.

    ks: array of kpoints (2 or 3 components; only the first two are used,
        matching topology.berry_curvature's 2D-only convention).
    Returns an array of Berry curvatures, one per kpoint, in the same
    order as ks. At any kpoint where the four plaquette corners don't
    have a consistent number of occupied states (a closing gap), the
    curvature is set to 0.0 -- the same fallback topology.berry_curvature
    itself uses.
    """
    hkgen = h.get_hk_gen()
    ks = np.array(ks)
    nk = len(ks)
    dx = np.array([dk,0.])
    dy = np.array([0.,dk])
    offsets = (-dx-dy,dx-dy,dx+dy,-dx+dy) # the four plaquette corners
    bs = np.zeros(nk)
    for i0 in range(0,nk,batch_size): # loop over batches of kpoints
        kb = ks[i0:i0+batch_size]
        nb = len(kb)
        corners = np.array([hkgen(kb[j][0:2]+off) for j in range(nb) for off in offsets],
                dtype=np.complex128) # (4*nb,norb,norb)
        es_all,vs_all = parallel_diagonalization(corners) # diagonalize the whole batch in parallel
        for j in range(nb): # cheap post-processing: plain, serial
            wfs = []
            for c in range(4):
                idx = 4*j+c
                es = es_all[idx]
                vs = np.conjugate(vs_all[idx].T) # rows are conjugated eigenvectors,
                                                  # matching topologytk.occstates.occupied_states
                wfs.append(vs[es<0.]) # occupied states (negative energy)
            dims = [len(w) for w in wfs]
            if max(dims)!=min(dims): continue # inconsistent occupied count, leave as 0.0
            wf1,wf2,wf3,wf4 = wfs
            m = uij(wf1,wf2)@uij(wf2,wf3)@uij(wf3,wf4)@uij(wf4,wf1)
            d = lg.det(m)
            bs[i0+j] = np.arctan2(d.imag,d.real)/(4.*dk*dk)
    return bs
