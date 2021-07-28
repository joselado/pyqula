# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import hamiltonians
import numpy as np
from pyqula import specialhamiltonian
from pyqula import parallel
from pyqula import spectrum
parallel.cores = 7

h = specialhamiltonian.tbg(n=7,ti=0.4,is_sparse=True,has_spin=False)
h.set_filling(0.5,nk=2)
#h.get_bands(num_bands=20)
#exit()
spectrum.multi_fermi_surface(h,nk=60,energies=np.linspace(-0.05,0.05,100),
        delta=0.0005,nsuper=1)







