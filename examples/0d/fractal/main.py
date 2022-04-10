# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
import numpy as np

g = geometry.sierpinski(n=5,mode="honeycomb")
g.write()
h = g.get_hamiltonian(has_spin=False)
h.get_multildos(energies=np.linspace(-2.0,2.0,100),delta=1e-2)



