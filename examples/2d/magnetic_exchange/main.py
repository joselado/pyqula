# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
g = geometry.triangular_lattice()
h = g.get_hamiltonian(has_spin=True)
h.set_filling(0.05,nk=10)
from pyqula import magneticexchange
J = magneticexchange.NN_exchange(h,nk=100,J=1,mode="spiral",filling=0.5)
print("Exchange is",J)







