# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import specialhamiltonian
# Hamiltonian
h = specialhamiltonian.multilayer_graphene(l=[0 for i in range(2)],ti=0.1)
h.add_onsite(lambda r: r[2])
from pyqula.symmetrytk.localsymmetry import all_permutations

Us = all_permutations(h,only_permutation=True)
for U in Us: print(np.round(U,3))
