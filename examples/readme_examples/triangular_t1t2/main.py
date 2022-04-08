# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry object
g = g.get_supercell(10)
h = g.get_hamiltonian() # get the Hamiltonian object
h.remove_spin() 

m = h.get_hk_gen()([0.,0.,0.]) # onsite matrix
print(m)




