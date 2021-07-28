# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import spectrum
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
#h.get_qpi(delta=5e-2)
spectrum.multi_fermi_surface(h,nk=60,energies=np.linspace(-4,4,100),
        delta=0.1,nsuper=1)







