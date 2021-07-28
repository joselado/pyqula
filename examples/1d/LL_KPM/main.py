# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




import os
import sys
from pyqula import geometry
import numpy as np
from pyqula import ribbon
from pyqula import indexing

g = geometry.honeycomb_lattice() # get the 2D geometry of a honeycomb lattice
g = ribbon.bulk2ribbon(g,n=100) # generate a ribbon
h = g.get_hamiltonian(has_spin=False,is_sparse=True) # get the Hamiltonian
# Here you could use any 1d Hamiltonian you want
# Alternatively, you can generate 1D Hamiltonians from a 2D one using
# h = ribbon.hamiltonian_ribbon(h,n=10) # (more expensive though)

indb = indexing.bulk1d(h.geometry) # 1 for bulk sites, 0 for edges
def frand(): 
    """This function generates states located in the bulk"""
    return (np.random.random(len(h.geometry.r))-0.5)*indb

h0 = h.copy() # make a copy of the Hamiltonian


def getdos(b=0.03):
    """Compute DOS for a certain magnetic field"""
    h = h0.copy()
    h.add_peierls(b) # add magnetic field
    return h.get_dos(use_kpm=True, # use the KPM mode
            scale=4.0, # scale of the problem
            frand=frand, # random vector generator for KPM
            delta=0.04, # smearing of the distribution
            energies=np.linspace(-1.0,1.0,100),nk=20)


bs = np.linspace(0.0,0.1,30) # magnetic fields

fo = open("DOS_VS_FIELD.OUT","w") # output file

for b in bs: # loop over magnetic fields
  (xs,ys) = getdos(b=b)
  for (x,y) in zip(xs,ys):
    fo.write(str(b)+"  ")
    fo.write(str(x)+"  ")
    fo.write(str(y)+"\n")
  fo.flush()

fo.close()








