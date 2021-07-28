# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry  # library to create crystal geometries
from pyqula import hamiltonians  # library to work with hamiltonians
from pyqula import sculpt  # to modify the geometry
from pyqula import correlator
from pyqula import kpm
import numpy as np
import matplotlib.pyplot as plt
import time
g = geometry.chain()
g = g.supercell(100)
g.dimensionality = 0
h = g.get_hamiltonian()
h.add_rashba(.5)
h.add_zeeman([0.,1.,0.])
h.intra += np.diag(np.random.random(h.intra.shape[0]))
i = 0
j = 9
t1 = time.clock()
(x,y) = kpm.dm_ij_energy(h.intra,npol=200,i=i,j=j,ne=1000)
t2 = time.clock()
(x2,y2) = correlator.dm_ij_energy(h.intra,i=i,j=j,delta=0.1,ne=1000)
t3 = time.clock()
print("Time KPM = ",t2-t1)
print("Time in inversion = ",t3-t2)
#print(np.trapz(y,x=x,dx=x[1]-x[0]))
plt.subplot(1,2,1)
plt.plot(x,y.real,marker="o",label="KPM")
plt.plot(x2,y2.real,marker="o",label="Green")
plt.legend()
plt.subplot(1,2,2)
plt.plot(x,y.imag,marker="o",label="KPM")
plt.plot(x2,y2.imag,marker="o",label="Green")
plt.legend()
plt.show()







