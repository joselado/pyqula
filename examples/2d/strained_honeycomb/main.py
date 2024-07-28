# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import specialhopping
from pyqula import dos
g = geometry.honeycomb_lattice()
g = g.get_supercell(11)
h = g.get_hamiltonian(has_spin=False,is_sparse=True)


import numpy as np

omega = np.pi*2./np.sqrt(g.a1.dot(g.a1)) # modulation frequency

from pyqula.strain import graphene_buckling

pot = graphene_buckling(omega=omega,dt=0.2) # special function for buckled lattices

h.add_strain(pot,mode="non_uniform")
h.get_bands(num_bands=20)
h.get_ldos(e=0.)







