# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# this example is not finished

from pyqula import geometry
g = geometry.honeycomb_lattice()
g = g.supercell(1)
from pyqula import klist
import numpy as np

k = klist.label2k(g,"K")
kpath = [ik*k for ik in np.linspace(0.,1.0,100)]
kpath = klist.get_kpath(g,["G","K","M","G"])
h = g.get_hamiltonian()
h.get_bands(kpath=kpath)




