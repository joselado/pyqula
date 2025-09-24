# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry
from pyqula.specialhamiltonian import NbSe2
h = NbSe2(soc=0.9)
(kx,ky,fs) = h.get_fermi_surface(e=0.,nk=100,delta=3e-1,operator="sz")

# plot the spin Fermi surface
kx = np.unique(kx) ; ky = np.unique(ky) ; fs = fs.reshape((len(kx),len(ky))) # reshape
import matplotlib.pyplot as plt
plt.imshow(fs,cmap="bwr") ; plt.xticks([]) ; plt.yticks([])
plt.show()







