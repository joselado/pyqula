# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.excitonic_bilayer(gap=-1.,g="triangular") # Hamiltonian
h.geometry.write()
(k,e) = h.get_bands() # compute band structure








