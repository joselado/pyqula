# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

fo0 = open("DOSMAP_EDGE.OUT","w")
fo1 = open("DOSMAP_BULK.OUT","w")

import numpy as np
from pyqula import geometry
g = geometry.chain() # create a chain
g = g.supercell(100) # create a large supercell 
g.dimensionality = 0 # make it finite

for J in np.linspace(0.,0.2,50): # loop over exchange couplings
    h = g.get_hamiltonian() # create a new hamiltonian
    h.add_onsite(2.0) # shift the chemical potential
    h.add_rashba(.3) # add rashba spin-orbit coupling
    h.add_exchange([0.,0.,J]) # add exchange coupling
    h.add_swave(.1) # add s-wave superconductivity
    edge = h.get_operator("location",r=g.r[0]) # projector on the edge
    energies = np.linspace(-.2,.2,200) # set of energies
    (e0,d0) = h.get_dos(operator=edge,energies=energies,delta=2e-3) # edge DOS


    print(J)
    for (ei,di) in zip(e0,d0):
        fo0.write(str(J/0.1)+" ")
        fo0.write(str(ei/0.1)+" ")
        fo0.write(str(di)+"\n")

    for (ei,di) in zip(e1,d1):
        fo1.write(str(J/0.1)+" ")
        fo1.write(str(ei/0.1)+" ")
        fo1.write(str(di)+"\n")


