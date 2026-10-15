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

from ..htk.eigenvectors import peigh_bloch
from .overlap import uij


def use_berry_curvature_mesh(h,mode="Wilson",window=None,max_waves=None):
    """Whether the batched path below reproduces topology.berry_curvature
    for this Hamiltonian and these options. It only covers the plain
    Wilson-loop case with a full diagonalization: an energy window or a
    truncated (ARPACK) diagonalization changes which states count as
    occupied, and a Hamiltonian carrying its own occupied-states generator
    (h.os_gen, set by topologytk.topologicalsector) changes it as well."""
    if mode!="Wilson": return False
    if window is not None or max_waves is not None: return False
    if getattr(h,"os_gen",None): return False # custom occupied states
    return True


def berry_curvature_mesh(h,ks,dk=0.01,batch_size=64):
    """Compute the Wilson-loop Berry curvature at every kpoint in ks.

    ks: array of kpoints (2 or 3 components; only the first two are used,
        matching topology.berry_curvature's 2D-only convention).
    Returns an array of Berry curvatures, one per kpoint, in the same
    order as ks. At any kpoint where the four plaquette corners don't
    have a consistent number of occupied states (a closing gap), or where
    there is no occupied state at all, the curvature is set to 0.0 -- the
    same fallbacks topology.berry_curvature itself uses.
    """
    if h.dimensionality!=2: # only for 2d
        raise NotImplementedError("the Berry curvature mesh is only "
                "implemented for 2d Hamiltonians")
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
        corners = [kb[j][0:2]+off for j in range(nb) for off in offsets]
        # (4*nb,norb,norb): the four plaquette corners of every kpoint
        es_all,vs_all = peigh_bloch(hkgen,corners) # diagonalize the batch in parallel
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
            if max(dims)==0: continue # nothing occupied at this kpoint
            wf1,wf2,wf3,wf4 = wfs
            m = uij(wf1,wf2)@uij(wf2,wf3)@uij(wf3,wf4)@uij(wf4,wf1)
            d = lg.det(m)
            bs[i0+j] = np.arctan2(d.imag,d.real)/(4.*dk*dk)
    return bs


def whole_zone_window(g,maxint=10,tol=1e-8):
    """Window of the Berry curvature map of a 2d geometry, in the Cartesian
    momenta of the map (ki, with reduced momenta k = get_k2K() @ ki).

    Returns (vx,vy,nzones). The window is the cell spanned by the two
    vectors vx and vy, centered at Gamma, and it holds exactly nzones
    Brillouin zones, so that the mean of a periodic quantity over a uniform
    mesh of it is its mean over one zone.

    When the reciprocal lattice has a vector along x and one along y, vx and
    vy are the shortest of them (integer combinations m*B1+n*B2 with |m|,|n|
    up to maxint), and the window is an axis-aligned rectangle: one zone for
    a square or rectangular lattice, two for a hexagonal one. Otherwise (an
    oblique lattice, or a lattice rotated with respect to the axes) vx and
    vy are the two reciprocal lattice vectors themselves, and the window is
    the parallelogram of one zone."""
    if g.dimensionality!=2:
        raise ValueError("the whole-zone window of the Berry curvature map "
                "is only defined for 2d geometries, got dimensionality "
                +str(g.dimensionality))
    R = np.array(g.get_k2K())[:2,:2] # reduced momenta from Cartesian ones
    B = np.linalg.inv(R) # columns: reciprocal lattice vectors, Cartesian
    scale = np.max(np.abs(B))
    vx,vy = None,None
    for m in range(-maxint,maxint+1):
        for n in range(-maxint,maxint+1):
            v = m*B[:,0] + n*B[:,1]
            if np.max(np.abs(v))<tol*scale: continue # zero vector
            if abs(v[1])<tol*scale and v[0]>0: # along +x
                if vx is None or v[0]<vx[0]: vx = np.array([v[0],0.])
            if abs(v[0])<tol*scale and v[1]>0: # along +y
                if vy is None or v[1]<vy[1]: vy = np.array([0.,v[1]])
    if vx is None or vy is None: # no rectangle, take one zone
        return B[:,0].copy(),B[:,1].copy(),1
    nzones = int(round(abs(np.linalg.det(R@np.array([vx,vy]).T))))
    return vx,vy,nzones


def whole_zone_kmesh(g,nk=100,nsuper=1):
    """Uniform nk x nk mesh of the window of whole_zone_window, repeated
    nsuper times along each of its sides, as an array of Cartesian momenta
    [kx,ky,0] (the first index runs along vx, the second along vy)"""
    vx,vy,nzones = whole_zone_window(g)
    ts = np.linspace(-nsuper/2.,nsuper/2.,nk,endpoint=False)
    ks = [t1*vx + t2*vy for t1 in ts for t2 in ts]
    return np.array([[k[0],k[1],0.] for k in ks])
