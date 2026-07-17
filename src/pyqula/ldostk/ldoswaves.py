import numpy as np
from .. import algebra
from numba import jit


def ldos_diagonalization(m,e=0.0,**kwargs):
    """Compute the LDOS using exact diagonalization"""
#    if algebra.issparse(m): return ldos_arpack(m,e=e,**kwargs) # sparse
    return ldos_waves(m,es=[e],**kwargs)[0] # dense




def ldos_waves(intra,es = [0.0],delta=0.01,operator=None,
        k=None,delta_discard=None,**kwargs):
  """Calculate the DOS in a set of energies by full diagonalization"""
  emean = np.mean(es) # average energy
  eig,eigvec = get_waves(intra,e0=emean,**kwargs) # eigenvalues and eigenvectors
  return ldos_waves_from_eigsystem(eig,eigvec,es,delta,operator=operator,
          k=k,delta_discard=delta_discard)


def ldos_waves_from_eigsystem(eig,eigvec,es,delta,operator=None,
        k=None,delta_discard=None):
  """Same as ldos_waves, but starting from an already-diagonalized
  (eig, eigvec) pair instead of diagonalizing internally -- lets callers
  batch the diagonalization step themselves (see fermisurface.ldosmap)."""
  es = np.array(es) # array with energies
  ds = [] # empty list
  if operator is None: weights = eig.real*0. + 1.0 # initialize as 1
  else: weights = [operator.braket(v,k=k) for v in eigvec] # weights
  if delta_discard is not None: # discard too far values
      ewin = [np.min(es)-delta_discard*delta,np.min(es)+delta_discard*delta]
      for i in range(len(weights)):
          e = eig[i]
          if not ewin[0]<e<ewin[1]: weights[i] = 0.0
  v2s = [(np.conjugate(v)*v).real for v in eigvec] # square of the wavefunction
  ds = [[0.0 for i in range(eigvec.shape[1])] for e in es] # initialize
  ds = ldos_waves_jit(np.array(es),
          np.array(eigvec).T,np.array(eig),np.array(weights),
          np.array(v2s),np.array(ds),delta)
  return ds


from ..waves import get_waves




@jit(nopython=True)
def ldos_waves_jit(es,eigvec,eig,weights,v2s,ds,delta):
  for i in range(len(es)): # loop over energies
    energy = es[i] # energy
    d = ds[i]
    for j in range(len(eig)):
        v = eigvec[j]
        ie = eig[j]
        weight = weights[j]
        v2 = v2s[j]
        fac = delta/(np.abs(energy-ie)**2 + delta**2) # factor to create a delta
        d += weight*fac*v2 # add contribution
    d /= np.pi # normalize
    ds[i] = d # store
  return ds


