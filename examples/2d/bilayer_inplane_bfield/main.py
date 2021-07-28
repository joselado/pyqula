# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




import numpy as np
from pyqula import specialhamiltonian
h = specialhamiltonian.multilayer_graphene(l=[0,1],ti=0.0)
h.turn_spinful()
h.add_inplane_bfield(b=0.1,phi=0.5)
print(h.intra)
h.get_bands()
from pyqula import spectrum
spectrum.multi_fermi_surface(h,nk=40,energies=np.linspace(-4,4,100),
        delta=0.02,nsuper=1)








