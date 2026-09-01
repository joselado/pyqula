import numpy as np
from . import extract

def delta2dvector(uu,dd,ud):
    """Transform Deltas to dvectors"""
    out = [(dd + uu)/2.,-1j*(dd-uu)/2,ud] # compute the d-vector
    return np.array(out) # return dvectors (three matrices)


def dvector2delta(ds):
    """Transform a certain dvector into deltauu, deltadd and deltaud"""
    ds = np.array(ds)
    delta = [0.,0.,0.]
    delta[0] = ds[0]-1j*ds[1] # up up
    delta[1] = ds[0]+1j*ds[1] # down down
    delta[2] = ds[2] # up down
    return np.array(delta) # return the delta

dvector2deltas = dvector2delta

def dvector2deltas_old(ds):
    """Transform a certain dvector into deltauu, deltadd and deltaud"""
    raise # this must be checked
    # this function has probably a missing sign somewhere
    deltas = [0.,0.,0.]
    deltas[0] = ds[0]+ds[1]
    deltas[1] = -1j*(ds[0]-ds[1]) # this sign might not be ok
    deltas[2] = ds[2]
    return np.array(deltas)


def extract_dvector_from_hamiltonian(h):
    """Return a function that computes the d-vector matrix at a k-point"""
    hk = h.get_hk_gen() # get Bloch Hamiltonian generator
    def f(k): # define function
        m = hk(k) # compute Bloch Hamiltonian
        (uu,dd,ud) = extract.extract_triplet_pairing(m) # pairing matrices
        return delta2dvector(uu,dd,ud) # return d-vector
    return f # return function

def matrix2dvector(m):
    """Return the dvectors from a matrix"""
    (uu,dd,ud) = extract.extract_triplet_pairing(m) # pairing matrices
    return delta2dvector(uu,dd,ud) # return d-vector


def dvector2nonunitarity(m):
    """Given a matrix of d-vectors, compute the non-unitarity vector

    The convention is q = i*(d x d^*), fixed by

        Delta Delta^dag = |d|^2 * identity + q.sigma

    with Delta = i*(d.sigma)*sigma_y. The state is unitary iff q = 0, and
    q is the spin moment of the Cooper pairs: a pure up-up pairing
    (d proportional to (1,i,0)) gives q along +z. Note this is the
    opposite order of the cross product to i*(d^* x d), which returns -q.
    """
    out = 1j*np.cross(m,np.conjugate(m),axis=0) # q = i*(d x d^*)
    return out.real # it is real by construction


def average_hamiltonian_dvector(h,nk=10,
    spatial_sum=True,
    non_unitarity=False
    ):
    """Compute the average d-vector of a Hamiltonian, as the three
    k-averaged squared components (|dx|^2,|dy|^2,|dz|^2). Optional arguments
       - nk = 10, number of kpoints in each direction
       - spatial_sum = True, return sum over sites
       - non_unitarity = False, average the squared components of the
         non-unitarity vector q = i*(d x d^*) instead of those of d. This
         is a magnitude only, and unlike get_dvector_non_unitarity it
         carries no sign, so use it to ask whether the state is
         non-unitary rather than in which direction.
    """
    if not h.has_eh: raise
    f = extract_dvector_from_hamiltonian(h) # function to extract the d-vector
    ks = h.geometry.get_kmesh(nk=nk) # get k-mesh
    out = np.array([f(k) for k in ks]) # compute d-vector matrices
    # redefine in case you want the non-unitarity
    if non_unitarity: out = [dvector2nonunitarity(o) for o in out]
    out = np.abs(out)**2 # square each term
    out = np.mean(out,axis=0) # average over k-points
    out = np.sum(out,axis=1) # sum over rows
    if spatial_sum: out = np.mean(out,axis=1) # sum over columns
    return out # return a vector

def dvector_times_rij_map(h,nrep=4):
    """Compute the dvector times rij"""
    h = h.supercell(nrep) # create a supercell (if needed)
    hi = h.get_hopping_dict()[(0,0,0)]
    dms = matrix2dvector(hi) # get the dvectors
    rs = h.geometry.r[:,0:3] # get coordinates
    ds = np.zeros(rs.shape,dtype=np.complex128) # array with the result
    for i in range(len(rs)):
        for j in range(len(rs)):
            d = np.cross(dms[:,i,j],rs[i]-rs[j])
            ds[i,:] = ds[i,:] + d # add contribution
    m = np.array([rs[:,0],rs[:,1],rs[:,2],ds[:,0],ds[:,1],ds[:,2]]).T.real
    m = np.round(m,5) # round values
    np.savetxt("DxR_MAP.OUT",m) # write in the file

def dvector_times_mij_map(h,nrep=4):
    """Compute the dvector times rij"""
    h = h.supercell(nrep) # create a supercell (if needed)
    hi = h.get_hopping_dict()[(0,0,0)]
    dms = matrix2dvector(hi) # get the dvectors
    rs = h.geometry.r[:,0:3] # get coordinates
    ds = np.zeros(rs.shape,dtype=np.complex128) # array with the result
    for i in range(len(rs)):
        for j in range(len(rs)):
            d = np.cross(dms[:,i,j],rs[i]-rs[j])
            ds[i,:] = ds[i,:] + d # add contribution
    m = np.array([rs[:,0],rs[:,1],rs[:,2],ds[:,0],ds[:,1],ds[:,2]]).T.real
    m = np.round(m,5) # round values
    np.savetxt("DxR_MAP.OUT",m) # write in the file


def dvector_non_unitarity_map(h,nrep=2,**kwargs):
    """Write a real-space map of the d-vector non-unitarity to
    NON_UNITARITY_MAP.OUT, with columns (x,y,z,qx,qy,qz). Optional arguments
       - nrep = 2, number of replicas written for each direction
       - nk = 10, number of k-points in each direction
    See get_dvector_non_unitarity for the definition of q."""
    ds = dvector_non_unitarity(h,**kwargs) # dvectors
    rs = h.geometry.supercell(nrep).r # supercell positions
    from ..geometry import replicate_array
    ds = replicate_array(h.geometry,ds,nrep=nrep) # replicate
    m = np.array([rs[:,0],rs[:,1],rs[:,2],ds[:,0],ds[:,1],ds[:,2]]).T
    np.savetxt("NON_UNITARITY_MAP.OUT",m) # write in the file


def dvector_non_unitarity(h,nk=10):
    """Compute the non-unitarity vector of the spin-triplet d-vector,
    resolved per site, as an array of shape (number of sites, 3).

    For a spin-triplet pairing matrix Delta = i*(d.sigma)*sigma_y one has

        Delta Delta^dag = |d|^2 * identity + q.sigma,  q = i*(d x d^*)

    so the state is unitary (Delta Delta^dag proportional to the identity)
    iff q = 0. When it is non-zero, q is real and is the spin moment of the
    Cooper pairs: it is parallel to the magnetization in a ferromagnetic
    spin-triplet superconductor, and it points along +z for a pure up-up
    pairing. The d-vector is computed at each k-point of a uniform mesh,
    q is averaged over the mesh, and the pairing partners of each site
    are summed over.

    Optional arguments
       - nk = 10, number of k-points in each direction
    """
    f = extract_dvector_from_hamiltonian(h) # function to extract the d-vector
    ks = h.geometry.get_kmesh(nk=nk) # get k-mesh
    out = np.array([f(k) for k in ks]) # compute d-vector matrices
    # redefine in case you want the non-unitarity
    out = [dvector2nonunitarity(o) for o in out] # non-unitarity
    out = np.mean(out,axis=0) # average over k-points
    ds = np.sum(out,axis=1).T # sum over rows
    return ds

