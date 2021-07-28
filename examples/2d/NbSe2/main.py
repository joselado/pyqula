# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import specialhamiltonian
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice()
g = g.supercell(3)
#h = specialhamiltonian.valence_TMDC(soc=0.1,g=g)
h = specialhamiltonian.NbSe2(soc=0.2,cdw=0.3)
exit()
h.add_rashba(.4)
h.add_zeeman(.3)
h.get_bands(operator="sz")
#h.add_zeeman(.3)
#h.turn_dense()
#h.get_multi_fermi_surface(energies=np.linspace(-6.,6.,100),
#        delta=3e-2,nk=100)







