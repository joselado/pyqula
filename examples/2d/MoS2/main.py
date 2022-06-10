# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
g = geometry.honeycomb_lattice() # honeycomb lattice
h = g.get_hamiltonian() # Hamiltonian
h.add_sublattice_imbalance(1.0) # add a sublattice imbalance
h.add_soc(0.05) # add SOC
(k,e,s) = h.get_bands(operator="sz") # band structure







