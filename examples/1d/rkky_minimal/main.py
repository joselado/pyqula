# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian() # create hamiltonian of the system

h.add_onsite(1.)
from pyqula import rkky
out = rkky.rkky_map(h,n=10,mode="pm",fsuper=10,nk=1).T
x0,e0 = out[0],out[3]

out1 = rkky.rkky_map(h,n=10,mode="LR",nk=200).T
x1,e1 = out1[0],out1[3]

import matplotlib.pyplot as plt

plt.plot(x0,e0,marker="o")
plt.plot(x1,e1,marker="o")
plt.show()



