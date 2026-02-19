# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import specialhamiltonian

h = specialhamiltonian.square_altermagnet(am=.3)
ks = [[np.cos(k),np.sin(k),0.] for k in np.linspace(0.,1.,100)*np.pi*2]

(k,e,c) = h.get_bands(kpath=ks,operator="sz")

h.get_fermi_surface(operator="sz",e=1.0,nk=80)

import matplotlib.pyplot as plt

plt.scatter(k,e,c=c,cmap="bwr")

plt.show()

