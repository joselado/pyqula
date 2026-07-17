from __future__ import print_function, division
import numpy as np
from numba import jit
from . import parallel

delta_dm = 1e-6 # default energy smearing for density matrix
# ds not None does not use it yet

dm_mode = "accumulate" # default mode to compute density matrix

# accumulate is the new mode, it may be worth checking
# if it yields the same results as simultaneous

def full_dm(h,T=delta_dm,dm_mode=dm_mode,**kwargs):
    """Compute the full density matrix"""
    if T==0.: T = 1e-15 # just very small 
    if dm_mode=="accumulate":
        return full_dm_accumulate(h,delta=T,**kwargs)
    elif dm_mode=="simultaneous":
        return full_dm_simultaneous(h,delta=T,**kwargs)
    else: raise # not implemented

# it may be worth to implement some adaptive integration with quad_vec

def full_dm_accumulate(h,nk=10,fermi=0.0,
        delta=delta_dm,
        ds=None,batch_size=16):
    """Compute the full density matrix by adding the
    contributions to the matrix kpoint by kpoint. K-points are processed
    in batches: each batch is diagonalized in parallel across numba
    threads (see htk.eigenvectors.parallel_diagonalization), then every
    kpoint's density-matrix contribution is computed in parallel too
    (see dmtk.fulldm.full_dm_batch_vectorized) and pooled with a single
    sum at the end of the batch -- no interprocess communication anywhere
    in this function, unlike parallel.pcall, so it scales with the number
    of threads instead of being dominated by IPC overhead. batch_size
    bounds how many k-points' eigenvectors are held in memory at once,
    keeping the memory footprint low regardless of how dense the k-mesh
    is."""
    from .htk.eigenvectors import parallel_diagonalization
    hk = h.get_hk_gen() # get the Hamiltonian generator
    ks = np.array(h.geometry.get_kmesh(nk=nk)) # get the mesh
    fac = 1./len(ks) # normalization
    dm = None # accumulator, one slot per batch
    for i0 in range(0,len(ks),batch_size): # loop over batches of kpoints
        kbatch = ks[i0:i0+batch_size]
        mats = np.array([hk(k) for k in kbatch]) # k-Hamiltonians in this batch
        es_batch,vs_batch = parallel_diagonalization(mats) # diagonalize in parallel
        es_batch = es_batch-fermi # substract fermi energy
        if ds is None:
            contribs = full_dm_batch_vectorized(es_batch,vs_batch,delta=delta) # one per kpoint, in parallel
            batch_total = np.sum(contribs,axis=0) # pool the batch's contributions
        else:
            n = vs_batch.shape[1]
            batch_total = np.zeros((len(ds),n,n),dtype=np.complex128)
            for idir,d in enumerate(ds): # each direction, batched over kpoints
                contribs = full_dm_batch_d_vectorized(es_batch,vs_batch,kbatch,
                        np.array(d,dtype=np.float64),delta=delta) # one per kpoint, in parallel
                batch_total[idir] = np.sum(contribs,axis=0) # pool the batch's contributions
        dm = batch_total if dm is None else dm+batch_total # pool across batches
    dm = dm*fac # renormalize
    if ds is None: return dm # return the single array
    else: # if ds were given
        outd = dict() # dictionary
        for i in range(len(ds)): outd[tuple(ds[i])] = dm[i,:,:] # as dictionary
        return outd
    


def full_dm_simultaneous(h,nk=10,fermi=0.0,
        delta=delta_dm,
        ds=None):
    """Compute the full density matrix by first computing all the
    eigenvectors, and after adding all the contributions together.
    This can become memore expesive for large kmesh and moderate
    matrices"""
    if h.dimensionality == 0: fac = 1.
    elif h.dimensionality == 1: fac = 1./nk
    elif h.dimensionality == 2: fac = 1./nk**2
    elif h.dimensionality == 3: fac = 1./nk**3
    else: raise
    if ds is None: # no directions required
      es,vs = h.get_eigenvectors(nk=nk) # get eigenvectors
      es = es - fermi # shift by the Fermi energy
      return np.matrix(full_dm_python(es,np.array(vs),
                             delta=delta))*fac # call the function
    else: # directions required
      es,vs,ks = h.get_eigenvectors(nk=nk,kpoints=True) # get eigenvectors
      es = es - fermi # shift by the Fermi energy
      ks = np.array(ks) # to array
      n = h.intra.shape[0] # dimensionality
      out = parallel.pcall(lambda x: full_dm_python_d(es,vs,ks,x)*fac,ds)
      outd = dict() # dictionary
      for i in range(len(ds)): outd[tuple(ds[i])] = out[i] # as dictionary
      return outd


from .dmtk.fulldm import full_dm_python
from .dmtk.fulldm import full_dm_python_d
from .dmtk.fulldm import full_dm_batch_vectorized
from .dmtk.fulldm import full_dm_batch_d_vectorized




def restricted_dm(h,mode="KPM",pairs=[],
                   scale=10.0,npol=400,ne=None):
  """Calculate certain elements of the density matrix"""
  if h.dimensionality != 0 : raise
  if mode=="full": # full inversion and then select
    dm = full_dm(h) # Full DM
    outm = np.array([dm[j,i] for (i,j) in pairs]) # get the desired ones
    return outm # return elements
  elif mode=="KPM": # use Kernel polynomial method
    if ne is None: ne = npol*4
    from . import kpm
    xin = np.linspace(-.99*scale,0.0,ne) # input x array
    out = np.zeros(len(pairs),dtype=np.complex128)
    ii = 0
    for (i,j) in pairs: # loop over inputs
      (x,y) = kpm.dm_ij_energy(h.intra,i=i,j=j,scale=scale,npol=npol,
                      ne=ne,x=xin)
      out[ii] = np.trapz(y,x=x)/np.pi # pi is here so it normalizes to 0.5
      ii += 1
    return out
  else: raise
       
from . import algebra

def occupied_projector(m,delta=0.0):
    """Return a projector onto the occupied states"""
    (es,vs) = algebra.eigh(m) # diagonalize
    vs = vs.T # transpose
    return np.array(full_dm_python(es,np.array(vs)))

