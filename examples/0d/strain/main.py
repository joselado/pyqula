# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=4,nedges=3,rot=np.pi/6) # get an island
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
from pyqula import potentials
fs = potentials.radial_decay(v0=2.0,voo=1.0,rl=3.0,mode="linear")
h.add_strain(fs)
h.geometry.write_profile(fs)
h.write_hopping()
h.get_bands()





