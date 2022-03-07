# this library writes down the current that would be measured by placing
# a constact in certain atoms, and putting other contact anywhere
from __future__ import print_function
from scipy.sparse import csc_matrix
from . import correlator
import numpy as np

def write_correlator(h,index=None,e=0.0,delta=0.01):
  """Write the correlator in a file"""
  if h.dimensionality!=0: raise # only for 0d
  if h.has_spin: raise # not implemented
  intra = csc_matrix(h.intra) # hamiltonian 
  d = np.zeros(intra.shape[0]) # correlator
  if True:
    ii = 0
    for i in index: # loop over atoms
      d += correlator.gij(intra,i=i,e=e,delta=delta) # add contribution
      print("Done",ii,"of",len(index))
      ii += 1
  else:
    print("Full inversion")
    g = (np.identity(intra.shape[0])*(e+delta*1j)-intra.todense()).I
    for i in index: 
      gz = np.array(g[:,i]).reshape(g.shape[0])
      d += (gz*np.conjugate(gz)).real
  g = h.geometry
  np.savetxt("CORRELATOR.OUT",np.matrix([g.x,g.y,d]).T)
  print("Writen CORRELATOR.OUT")

