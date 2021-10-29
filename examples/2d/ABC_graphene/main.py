# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import specialhamiltonian
h = specialhamiltonian.multilayer_graphene(l=[-1,0,1])
h.get_bands()
