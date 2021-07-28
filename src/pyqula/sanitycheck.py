from __future__ import print_function
import numpy as np



# this library contains some checks for the hamiltonians


def electron_hole_symmetry(h,tol=1e-5,ntries=10):
   """Check that a system has electron-hole symmetry"""
   if not h.has_eh: return # skip if there is no e-h sector
   from superconductivity import eh_operator
   ehop = eh_operator(h.intra) # get the operator
   def ehsym(m1,m2):
     """Test eh symmetry"""
     diff = m1 + ehop(m2)
     diff = np.sum(np.abs(diff))
     if diff>tol: raise
   ehsym(h.intra,h.intra) # do it for the intra matrix
   if h.dimensionality>0:
     for i in range(ntries):
       hkgen = h.get_hk_gen() # generator
       k = np.random.random(h.dimensionality) # kpoint
       ehsym(hkgen(k),hkgen(-k))
       print("Checking k-point",k)
   print("Electron-hole symmetry is ok")
   


