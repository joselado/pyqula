# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry,meanfield
g = geometry.chain()
ns = 20
g = g.supercell(ns)
h = g.get_hamiltonian() # create hamiltonian of the system
def J(r):
  return 2.*np.array([0.0,np.sin(r[0]*np.pi*2./ns),np.cos(r[0]*np.pi*2./ns)])
h.add_zeeman(J)
h.write_magnetization(nrep=2)
h.add_swave(0.0)
scf = meanfield.Vinteraction(h,V1=-1.0,nk=10,filling=1./3.,mf="random",
    verbosity=1)
print("Triplet",scf.order_parameter("odd_SC"))
print("Singlet",scf.order_parameter("even_SC"))
print("Average non-unitarity of the d-vector",sum(scf.hamiltonian.get_average_dvector(non_unitarity=True)))
from pyqula import scftypes
print("Symmetry breaking",scf.identify_symmetry_breaking()) 
scf.hamiltonian.get_bands(operator="electron") # get the Hamiltonian
from pyqula.sctk.dvector import dvector_non_unitarity_map
dvector_non_unitarity_map(scf.hamiltonian)






