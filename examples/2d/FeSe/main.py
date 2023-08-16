# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian
import numpy as np
h = specialhamiltonian.FeSe_GM() # get a toy model for FeSe
h.get_fermi_surface(nk=100,delta=4e-1,e=0.) # get the Fermi surface




