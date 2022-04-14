# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
import numpy as np

g = geometry.sierpinski(n=7,mode="triangular")
g.write()
#exit()
h = g.get_hamiltonian(has_spin=False)
h.get_multildos(es=np.linspace(-3.0,3.0,100),delta=1e-2)



