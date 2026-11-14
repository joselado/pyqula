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
  emean = np.mean(np.array(es)) # average energy
  eig,eigvec = get_waves(intra,e0=emean,**kwargs) # eigenvalues and eigenvectors
  return ldos_waves_from_eigsystem(eig,eigvec,es,delta,operator=operator,
          k=k,delta_discard=delta_discard)


def ldos_waves_from_eigsystem(eig,eigvec,es,delta,operator=None,
        k=None,delta_discard=None):
  """Same as ldos_waves, but starting from an already-diagonalized
  (eig, eigvec) pair -- rows of eigvec are the eigenvectors, as returned
  by waves.get_waves -- instead of diagonalizing internally. This lets
  callers batch the diagonalization step themselves (see ldos.ldosmap).

  With an operator O the weight of state n on orbital a is the local
  matrix element Re[conj(u_n(a)) (O u_n)(a)], which sums over the orbitals
  to <u_n|O|u_n> and over a degenerate multiplet to a trace, so it does
  not depend on the basis the diagonalization returns. Without an
  operator it is |u_n(a)|^2."""
  es = np.array(es) # array with energies
  ds = [] # empty list
  weights = np.ones(len(eig)) # one per state, zeroed by delta_discard
  if delta_discard is not None: # discard too far values
      ewin = [np.min(es)-delta_discard*delta,np.max(es)+delta_discard*delta]
      for i in range(len(weights)):
          e = eig[i]
          if not ewin[0]<e<ewin[1]: weights[i] = 0.0
  if operator is None: # square of the wavefunction
      v2s = [(np.conjugate(v)*v).real for v in eigvec]
  else: # local matrix element of the operator
      v2s = [np.real(np.conjugate(v)*np.asarray(operator(v,k=k)).reshape(-1))
              for v in eigvec]
  ds = [[0.0 for i in range(np.array(eigvec).shape[1])] for e in es] # initialize
  ds = ldos_waves_jit(np.array(es),
          np.array(eigvec).T,np.array(eig),np.array(weights),
          np.array(v2s),np.array(ds),delta)
  return ds


from ..waves import get_waves




@jit(nopython=True,cache=True)
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


