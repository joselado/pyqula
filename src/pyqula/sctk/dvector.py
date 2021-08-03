import numpy as np
from . import extract

def delta2dvector(uu,dd,ud):
    """Transform Deltas to dvectors"""
    out = [(dd - uu)/2.,(dd+uu)/2j,ud] # compute the d-vector
    return np.array(out) # return dvectors (three matrices)

def dvector2deltas(ds):
  """Transform a certain dvector into deltauu, deltadd and deltaud"""
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


def average_hamiltonian_dvector(h,nk=10,spatial_sum=True):
    """Compute the average d-vector of a Hamiltonian"""
    if not h.has_eh: raise
    f = extract_dvector_from_hamiltonian(h) # function to extract the d-vector
    ks = h.geometry.get_kmesh(nk=nk) # get k-mesh
    out = np.array([f(k) for k in ks]) # compute d-vector matrices
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
    ds = np.zeros(rs.shape,dtype=np.complex) # array with the result
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
    ds = np.zeros(rs.shape,dtype=np.complex) # array with the result
    for i in range(len(rs)):
        for j in range(len(rs)):
            d = np.cross(dms[:,i,j],rs[i]-rs[j])
            ds[i,:] = ds[i,:] + d # add contribution
    m = np.array([rs[:,0],rs[:,1],rs[:,2],ds[:,0],ds[:,1],ds[:,2]]).T.real
    m = np.round(m,5) # round values
    np.savetxt("DxR_MAP.OUT",m) # write in the file

