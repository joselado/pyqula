# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import classicalspin
g = geometry.honeycomb_lattice() # generate the geometry
g = geometry.triangular_lattice() # generate the geometry
g = g.supercell(9)
sm = classicalspin.SpinModel(g) # generate a spin model
sm.add_heisenberg() # add heisenber exchange
classicalspin.use_jax = True
sm.minimize_energy() # minimize Hamiltonian
h = g.get_hamiltonian() # get the Hamiltonian
h.add_magnetism(sm.magnetization*2.0) # add magnetization
h.write_magnetization(nrep=2)






