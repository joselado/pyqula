# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import specialhamiltonian
import numpy as np
h = specialhamiltonian.FeSe()
h.add_rashba(1.)
h.get_bands()
h.get_multi_fermi_surface(nk=100,delta=1e-1,energies=np.linspace(-6,8.100))





