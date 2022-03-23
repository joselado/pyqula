# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian() # generate Hamiltonian
h.add_soc(0.15) # add intrinsic spin-orbit coupling
h.add_rashba(0.1) # add Rashba spin-orbit coupling
h.get_surface_kdos(delta=1e-2) # compute surface spectral function
