# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import specialhamiltonian
# Hamiltonian
h = specialhamiltonian.multilayer_graphene(l=[0 for i in range(2)],ti=0.)
from pyqula.symmetrytk.localsymmetry import permutations

permutations(h)
