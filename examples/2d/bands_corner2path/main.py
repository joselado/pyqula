# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
from pyqula.kpointstk.locate import k2path
ks = [[0.,0.,0.],[1/3.,1/3.,0.],[0.,.5,0.]]
kpath = k2path(g,ks) # ks your list of corners
h = g.get_hamiltonian() # create hamiltonian of the system
(ks,es) = h.get_bands(kpath=kpath) # bands ina  custom kpath

import matplotlib.pyplot as plt

plt.scatter(ks,es)

plt.show()




