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

# this mode does not run in parallel
def full_dm_accumulate(h,nk=10,fermi=0.0,
        delta=delta_dm,
        ds=None):
    """Compute the full density matrix by adding the
    contributions to the matrix kpoint by kpoint.
    Good in terms of memory footprint"""
    hk = h.get_hk_gen() # get the Hamiltonian generator
    from .klist import kmesh
    ks = h.geometry.get_kmesh(nk=nk) # get the mesh
    fac = 1./len(ks) # normalization
    # create the storages
    norb = h.intra.shape[0] # size of the matrix
    if ds is None: dm = np.zeros((norb,norb),dtype=np.complex128)
    else: dm = np.zeros((len(ds),norb,norb),dtype=np.complex128)
    for k in ks: # loop over kpoints
        (es,vs) = algebra.eigh(hk(k)) ; vs = vs.T # diagonalize
        es = es-fermi # substract fermi energy
        if ds is None: dm += full_dm_python(es,vs,delta=delta)
        else: 
            kes = np.zeros((len(es),3))
            kes[:,0] = k[0] ; kes[:,1] = k[1] ; kes[:,2] = k[2] # kpoints
            for i in range(len(ds)): # this could be parallelized if needed
                dm[i,:,:] += full_dm_python_d(es,vs,kes,ds[i],delta=delta) # add 
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

