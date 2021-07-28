# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
import numpy as np

# read geometry from a file
g = geometry.read_xyz(input_file="input.xyz",species="C")

def fun(r1,r2):
    """Function defining NN hoppoing"""
    dr = r1-r2
    dr2 = np.sqrt(dr.dot(dr))
    if 0.8<dr2<1.2: return -1.0
    else: return 0.0

h = g.get_hamiltonian(fun=fun,has_spin=False)
h.set_filling(0.5) # set half filling in zero
g.write() # write the geometry just to check
h.get_bands() # compute electronic structure
h.get_multildos(projection="atomic") # compute the LDOS using atomic orbitals







