# routines to perform unfolding of the Brillouin zone

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




