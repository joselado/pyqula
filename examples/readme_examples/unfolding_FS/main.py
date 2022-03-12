# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g0 = geometry.triangular_lattice()
n = 3 # size of the supercell
g = g0.get_supercell(n,store_primal=True) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian
fons = lambda r: (np.sum((r - g.r[0])**2)<1e-2)*100 # onsite in the impurity
h.add_onsite(fons) # add onsite energy
kpath = np.array(g.get_kpath(nk=200))*n # enlarged k-path
h.get_multi_fermi_surface(nk=50,energies=np.linspace(-4,4,100),
        delta=0.1,nsuper=n,operator="unfold")

