# routines to perform unfolding of the Brillouin zone

import numpy as np
from scipy.linalg import eigh

def unfolded_bands(hfol,hprim,kpath,inds_super=[]):
  """ Save in file unfolded band structure"""
  hkfol = hfol.get_hk_gen()  # generator
  hkprim = hprim.get_hk_gen() # generator
  nump = hprim.intra.shape[0] # number orbitals of primitive
  numf = hfol.intra.shape[0] # number orbitals of folded
  nn = numf/numfp # number times the hamiltonian is bigger
  for k in kpath:
    kfol = k # kpoint
    kprim = k/3.
    hf = hkfol(k) # get matrix
    hp = hkprim(k) # get matrix
    (ep,wfp) = eigh(hp) # eigenvalues and eigenvectors 
    (ef,wff) = eigh(hf) # eigenvalues and eigenvectors 
    dp = [(w.dot(np.conjugate(w))).real for w in wfp.transpose()] # density prim
    df = [(w.dot(np.conjugate(w))).real for w in wff.transpose()] # density fol
    df = [sum(w.split(nn)) for w in df] # transform into smaller basis
    # now it is time to compare weights of each eigenvalue
    raise





def perturb_bands(hprim,hper,kpath,inds_super=[]):
  """ Save in file perturbed band structure"""
  hkprim = hprim.get_hk_gen() # generator
  for k in kpath: # loop over kpoints
    kprim = k/3.
    hp = hkprim(k) # get matrix
    (ep,wfp) = eigh(hp) # eigenvalues and eigenvectors 
    wfp = wfp.transpose() # transpose eigenvectors
    # now calculate eigenfunctions in a supercell
    wfs = [] # wavefunction of the supercell
    for w in wfp: # loop over functions
      wlist = [] # create empty list
      for ix in range(inds_super[0]): # loop over ix
        for iy in range(inds_super[1]):  # loop over iy
          for iz in range(inds_super[2]):  # loop over iz
            phi = k.dot(np.array([ix,iy,iz])) # phase
            wj = w*np.exp(1j*np.pi*phi) # wavefunction
            wlist.append(wj) # append to the list
      wfs.append(np.concatenate(wlist)) # add this wave to the list
   # now apply perturbation theory
    vm = np.zeros((len(wfs),len(wfs)),dtype=np.complex) # perturbation
    for iw in wfs: 
      for jw in wfs: 
        iw = np.matrix(iw)
        jw = np.matrix(jw).H
        vaw = (iw*hper*jw)[0,0] # matrix element
  
    raise



def bloch_projector(h,g0):
    """Given a certain Hamiltonian and minimal geometry, return
    a projector to the minimal BZ"""
    h0 = g0.get_hamiltonian(has_spin=False) # spinless
    if h.has_spin: h0.turn_spinful() # get spinfull
    if h.dimensionality<3: 
        from .supercell import infer_supercell
        nsuper = infer_supercell(h.geometry,g0) # get the supercell
    else: raise # not implemented
    fs = bloch_phase_matrix(h0,nsuper=nsuper)
    def fun(v,k=0):
        vo = fs(k)@v # return vector
        return v*np.abs(vo.dot(np.conjugate(v)))
    from .operators import Operator
    return Operator(fun) # return operator



def bloch_phase_matrix(self,nsuper=[1,1,1]):
    """Given a Hamiltonian, return the matrix with Bloch phases
    for a supercell"""
    from scipy.sparse import bmat,csc_matrix
    if self.dimensionality>2: raise
    n = self.intra.shape[0] # dimensionality
    iden = csc_matrix(np.identity(n,dtype=np.complex)) # identity
    ns = nsuper[0]*nsuper[1] # number of supercells
    nx = nsuper[0] # supercells in x
    ny = nsuper[1] # supercells in y
    def fun(k): # generator
        out = [[None for i in range(ns)] for j in range(ns)] 
        for ii in range(ns): out[ii][ii] = 0*iden # initialize
        ii = 0 # counter
        for i in range(nx): # loop over x 
          for j in range(ny): # loop over y
            ki = np.array([k[0]/nx,k[1]/ny,0.]) # scale kvector
            out[ii][0] = iden*self.geometry.bloch_phase([i,j,0],ki) 
            ii += 1 # increase counter
        out = bmat(out) # return matrix
        return out
        return np.conjugate(out).T@out
    return fun
