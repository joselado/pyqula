# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

fo0 = open("DOSMAP.OUT","w")

import numpy as np
from pyqula import geometry
g = geometry.chain() # create a chain
g = g.supercell(100) # create a large supercell 
g.dimensionality = 0 # make it finite
h = g.get_hamiltonian() # create a new hamiltonian
h.add_onsite(2.0) # shift the chemical potential
h.add_rashba(.3) # add rashba spin-orbit coupling
h.add_exchange([0.,0.,0.15]) # add exchange coupling
h.add_swave(.1) # add s-wave superconductivity
energies = np.linspace(-.15,.15,200) # set of energies

for ri in g.r: # loop over exchange couplings
    edge = h.get_operator("location",r=ri) # projector on that site
    (e0,d0) = h.get_dos(operator=edge,energies=energies,delta=2e-3) # local DOS


    for (ei,di) in zip(e0,d0):
        fo0.write(str(ri[0])+" ")
        fo0.write(str(ei/0.1)+" ")
        fo0.write(str(di)+"\n")



