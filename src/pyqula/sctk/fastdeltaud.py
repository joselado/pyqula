import numpy as np
import scipy.sparse as sp
from scipy.sparse import bmat
from .reorder import reorder
from .. import algebra

def hopping2deltaud(H,T):
    """Given a hopping object, add a delta ud with the exact same
    form to the Hamiltonian. This can be fast for very large systems"""
    from ..superconductivity import reorder
    H.turn_nambu() # turn the Nambu spinor
    H.turn_multicell() # multicell mode
    T = T.copy() # make a dummy copy
    T.remove_spin() # remove the spin degree of freedom
    T.turn_multicell() # multicell mode
    n = len(H.geometry.r) # number of sites
    def t2h(m): # transform a single hopping to deltaud
      m = algebra.todense(m)
      pout = [[None for i in range(n)] for j in range(n)] # initialize
      iden = sp.identity(2) # identity
      for i in range(n): pout[i][i] = iden*0.
      for i in range(n): # loop over sites
        for j in range(n): # loop over sites
            if np.abs(m[i,j])<1e-6: continue # if the hopping is zero, skip
            else: pout[i][j] = m[i,j]*iden # get this pairing
      diag = sp.identity(2*n)*0. # zero matrix
      pout = bmat(pout) # convert to block matrix
      mout = [[diag,pout],[None,diag]] # output matrix
      mout = bmat(mout) # return full matrix
      return reorder(mout) # reorder the entries properly
    T.intra = t2h(T.intra) # intracell hopping
    for hop in T.hopping:
        hop.m = t2h(hop.m) # convert to pairing by hand
    T.has_eh = True # add eh by hand
    T.has_spin = True # add spin by hand
    # now the complementary
    T2 = T.copy()
    T2.intra = algebra.dagger(T2.intra)
    for hop in T2.hopping:
        hop.m = algebra.dagger(hop.m) # the other
        hop.dir = tuple(-np.array(hop.dir)) # the other
#    return T + T2 # summ both terms via brute force
    return H + T + T2 # summ both terms via brute force

