# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Strong and weak Z2 indices nu0;(nu1 nu2 nu3) of the Fu-Kane-Mele model, the
# diamond lattice with spin-orbit coupling between second neighbors, as the
# first-neighbor bond inside the unit cell is made weaker or stronger than
# the other three
import numpy as np
from pyqula import geometry

g = geometry.diamond_lattice_minimal() # diamond lattice, two sites per cell
scales = [0.5,0.7,0.9,1.1,1.3,2.0,2.5,3.5] # hopping of that bond, in units of t
for s in scales:
    h = g.get_hamiltonian() # first-neighbor hopping
    h.add_kane_mele(0.05) # spin-orbit coupling between second neighbors
    h.intra = s*h.intra # rescale the bond inside the unit cell
    (nu0,nus) = h.get_topological_invariant(nk=30,nt=30) # Z2 indices
    print("bond =",s,"t   Z2 indices =",str(nu0)+";"+"".join(map(str,nus)))
# a weaker bond gives a weak TI 0;(111), a stronger one a strong TI 1;(111),
# and a bond beyond 3t a band insulator 0;(000)
