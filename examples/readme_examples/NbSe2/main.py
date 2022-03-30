# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.NbSe2(soc=0.5) # NbSe2 Hamiltonian
(k,e,c) = h.get_bands(operator="sz",kpath=["G","K","M","G"]) # compute bands
(kx,ky,d) = h.get_fermi_surface() # get the Fermi surface
