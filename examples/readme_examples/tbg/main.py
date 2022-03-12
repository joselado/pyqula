# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.twisted_bilayer_graphene() # TBG Hamiltonian
h.set_filling(.5) # put Fermi energy at half filling
(k,e) = h.get_bands() # compute band structure

