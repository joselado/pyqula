# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import potentials
import numpy as np
g0 = geometry.triangular_lattice() # create geometry
n = 5 # supercell
g = g0.get_supercell(n,store_primal=True) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian
fmoire = potentials.commensurate_potential(g,n=3,minmax=[0,1]) # morie potential
h.add_onsite(fmoire) # add onsite energy following the moire
kpath = np.array(g.get_kpath(nk=400))*n # enlarged k-path
h.get_kdos_bands(operator="unfold",delta=2e-2,kpath=kpath,
                  energies=np.linspace(-3,-1,300)) # unfolded bands

