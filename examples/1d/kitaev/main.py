# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.chain() # geometry for a chain

# to build the Kitaev Hamiltonian we will isolate a spin polarized band at the
# Fermi energy and add spin triplet superconductivity
h = g.get_hamiltonian() # get the Hamiltonian
h.add_onsite(20) # make a large shift of the chemical potential
h.add_zeeman([0.,0.,20]) # put a single band at the chemical potential
# add spin triplet superconductivity with an in-plane dvector
# (only the moduli is important)
h.add_pairing(mode="pwave",delta=0.3,d=[1.0,0.,0.]) 
h.get_bands() # get the band structure
