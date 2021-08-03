# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry,meanfield
g = geometry.chain()
g = g.supercell(4)
h = g.get_hamiltonian() # create hamiltonian of the system
def J(r):
  return 2.*np.array([np.sin(r[0]*np.pi*2),np.cos(r[0]*np.pi*2),0.0])
h.add_zeeman(J)
h.write_magnetization()
#h.get_bands() ; exit()
h.add_swave(0.0)
scf = meanfield.Vinteraction(h,V1=-1.0,nk=20,filling=0.1,mf="random")
print("Triplet",scf.order_parameter("odd_SC"))
print("Singlet",scf.order_parameter("even_SC"))
print("Average d-vector",scf.hamiltonian.get_average_dvector())
from pyqula import scftypes
print("Symmetry breaking",scf.identify_symmetry_breaking()) 
scf.hamiltonian.get_bands(operator="electron") # get the Hamiltonian
from pyqula.sctk.dvector import dvector_times_rij_map
#dvector_times_rij_map(scf.hamiltonian)






