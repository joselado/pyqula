# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian() # get the Hamiltonian
h.add_onsite(2.0) # shift chemical potential
h.add_rashba(1.0) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,0.6]) # Zeeman field
h.add_swave(.3) # add superconductivity
(kx,ky,omega) = h.get_berry_curvature() # compute Berry curvature
(es,ks,ds,db) = h.get_surface_kdos(energies=np.linspace(-.4,.4,300)) # surface spectral function


