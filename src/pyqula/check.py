from __future__ import print_function
import numpy as np
from scipy.sparse import csc_matrix

def equal(m1,m2,tol=1e-4):
  """Check if two matrices are the same"""
  if np.max(np.abs(m1-m2))>tol:
#    print(csc_matrix(m1-m2))
    print("Maximum difference",np.max(np.abs(m1-m2)))
#    print("\n")
#    print(csc_matrix(m2))
    return False
  else: return True

def check_hamiltonian(h,tol=1e-5):
  """Do various checks in Hamiltonian, to ensure that nothing weird happens"""
  hk = h.get_hk_gen() # get generator
  m = hk(np.random.random(3)) # random k-point
  if not equal(m,np.conjugate(m).T):
    print("CHECK FAILED, Hamiltonian is not Hermitian")
    print(np.round(m,2))
    exit()
#    raise # not hermitian
  if h.has_eh: # if it has electron hole degree of freedom
    v = np.random.random(3) # random kpoint
    m1 = hk(v) # Hamiltonian
    m2 = hk(-v) # Hamiltonian in time reversal point
    from .superconductivity import eh_operator
    eh = eh_operator(m1) # get the function
    if not equal(m1,-eh(m2),tol=tol): 
      print("CHECK FAILED, Hamiltonian does not have electron-hole symmetry")
      exit()
    print("CHECKED that the Hamiltonian has electron-hole symmetry")


def check_dict(mf):
    """Check a dictionary with hopping, like the one used for mean field"""
    for key in mf:
        key2 = tuple([-i for i in key])
        m1 = mf[key]
        m2 = mf[key2]
        if not equal(m1,np.conjugate(m2).T):
            print(key,key2)
            print("First")
            print(m1)
            print("Second")
            print(m2)


