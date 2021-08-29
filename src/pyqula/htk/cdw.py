import numpy as np
from scipy.sparse import csc_matrix

def add_sublattice_imbalance(h,mass):
    """ Adds to the intracell matrix a sublattice imbalance """
    if h.geometry.has_sublattice:  # if has sublattice
      def ab(i):
        return h.geometry.sublattice[i]
    else:
      print("WARNING, no sublattice present")
      return 0. # if does not have sublattice
    natoms = len(h.geometry.r) # assume spinpolarized calculation 
    rows = range(natoms)
    if callable(mass):  # if mass is a function
      r = h.geometry.r
      data = [mass(r[i])*ab(i) for i in range(natoms)]
    else: data = [mass*ab(i) for i in range(natoms)]
    massterm = csc_matrix((data,(rows,rows)),shape=(natoms,natoms)) # matrix
    h.intra = h.intra + h.spinless2full(massterm)



