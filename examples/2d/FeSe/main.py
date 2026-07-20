# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian
import numpy as np
h = specialhamiltonian.FeSe_GM() # get a toy model for FeSe
nk = 100
(kx,ky,fs) = h.get_fermi_surface(nk=nk,delta=4e-1,e=0.) # get the Fermi surface

import matplotlib.pyplot as plt

plt.imshow(fs.reshape((nk,nk)))
plt.xticks([]) ; plt.yticks([])
plt.colorbar(label="Fermi surface weight")
plt.show()




