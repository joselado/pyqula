# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import specialhamiltonian
from pyqula import geometry
import numpy as np
h = specialhamiltonian.TaS2_SOC()
h.get_bands(operator="sz") 
#; exit()
#h.get_fermi_surface(e=0.0,nk=60,delta=6e-1,mode="eigen") ; exit()
#h.get_multi_fermi_surface(energies=np.linspace(-6.,6.,100),
#        delta=3e-1,nk=100)







