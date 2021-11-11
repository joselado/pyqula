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



def bloch_projector(h,g0=None):
    """Given a certain Hamiltonian and minimal geometry, return
    a projector to the minimal BZ"""
    if g0 is None:
        if h.geometry.primal_geometry is None: 
            print("No primal geometry")
            raise 
        else: g0 = h.geometry.primal_geometry # get the primal geometry
    h0 = g0.get_hamiltonian(has_spin=False)
    if h.has_spin: h0.turn_spinful() # add spin
    if h.has_eh: h0.add_swave(0.0) # add electron hole
    if h.dimensionality<3: 
        from .supercell import infer_supercell
        nsuper = infer_supercell(h.geometry,g0) # get the supercell
    else: raise # not implemented
    fs = bloch_phase_matrix(h0,nsuper=nsuper)
    def fun(v,k=0):
        vo = np.conjugate(fs(k))@v # return vector
        out = np.abs(vo.dot(np.conjugate(vo))) # overlap
        return v*out
    from .operators import Operator
    return Operator(fun) # return operator



def bloch_phase_matrix_simple(self,nsuper=[1,1,1]):
    """Given a Hamiltonian, return the matrix with Bloch phases
    for a supercell. This is a simple non-optimal version
    of this function"""
    from scipy.sparse import bmat,csc_matrix
    if self.dimensionality>2: raise
    n = self.intra.shape[0] # dimensionality
    iden = csc_matrix(np.identity(n,dtype=np.complex)) # identity
    ns = nsuper[0]*nsuper[1] # number of supercells
    nx = nsuper[0] # supercells in x
    ny = nsuper[1] # supercells in y
    def fun(k): # generator
        out = [[None for i in range(ns)] ] 
#        for ii in range(ns): out[ii][ii] = 0*iden # initialize
        ii = 0 # counter
        for i in range(nx): # loop over x 
          for j in range(ny): # loop over y
            ki = np.array([k[0]/nx,k[1]/ny,0.]) # scale kvector
#            frac_r = self.geometry.frac_r # fractional coordinates
#            U = np.diag([self.geometry.bloch_phase(ki,r) for r in frac_r])
#            U = np.array(U) # this is without .H
#            U = np.conjugate(U).T
#            U = self.spinless2full(U) # increase the space if necessary
            phi = np.exp(1j*np.pi*2.*np.array([i,j,0]).dot(ki)) # phase
            out[0][ii] = iden*phi # multiply by phase
            ii += 1 # increase counter
        out = bmat(out) # return matrix
        return out
    return fun






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
    outlist = [None for i in range(ns)] # list with matrices
    dlist = [None for i in range(ns)] # list with vectors
    ii = 0 # counter
    # generate all the matrices
    for i in range(nx): # loop over x 
      for j in range(ny): # loop over y
        out = [[None for ij in range(ns)]]
        for ij in range(ns): out[0][ij] = 0*iden # initialize
        dlist[ii] = np.array([i,j,0]) # list with dvectors
        out[0][ii] = iden*(1.+0j) # identity
        outlist[ii] = bmat(out).todense() # store dense matrix
        ii += 1 # increase counter
    dlist = np.array(dlist) # to array
    outlist = np.array(outlist) # to array
    def fun(k): # function generating the matrix
        ki = np.array([k[0]/nx,k[1]/ny,0.]) # scale kvector
        mout = outlist[0]*0j # initialize
        m = bloch_phase_matrix_jit(outlist,dlist,ki,mout)
        return m
    return fun

from numba import jit

@jit(nopython=True)
def bloch_phase_matrix_jit(ms,ds,k,out):
    """Numba function to generate the matrix"""
    n = len(ds) # loop over replicas
    out = out*0. # initialize
    for ii in range(n): # loop
        d = ds[ii]
        phi0 = d[0]*k[0] + d[1]*k[1] + d[2]*k[2]
        phi = np.exp(1j*np.pi*2.*phi0) # phase
        out = out + ms[ii]*phi # add contribution
    return out # return matrix
