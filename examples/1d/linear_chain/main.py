# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(tij=[1.0,0.3])
h.get_bands()







