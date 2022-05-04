# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry
g = geometry.chain() # geometry for a chain
g = g.get_supercell(10) ; g.dimensionality = 0 # make finite
# to build the Kitaev Hamiltonian we will isolate a spin polarized band at the
# Fermi energy and add spin triplet superconductivity
h = g.get_hamiltonian() # get the Hamiltonian
h.add_onsite(20) # make a large shift of the chemical potential
h.add_zeeman([0.,0.,20]) # put a single band at the chemical potential
# add spin triplet superconductivity with an in-plane dvector
# (only the moduli is important)
h.add_pairing(mode="pwave",delta=0.3,d=[1.0,0.,0.])

f = open("SWEEP.OUT","w") # open file
es = np.linspace(-1.0,1.0,40) # grid of energies
for i in range(len(g.r)): # loop over positions
    for e in es: # loop over energies
        T = 0.5 # transparency of the contact: 1e-5 tunneling, 1 contact
        z = h.didv(T=T,energy=e,i=i) # get the differential conductance
        f.write(str(i)+"  ")
        f.write(str(e)+"  ")
        f.write(str(z)+"\n")

f.close()
