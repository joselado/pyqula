# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry  # library to create crystal geometries
from pyqula import hamiltonians  # library to work with hamiltonians
from pyqula import sculpt  # to modify the geometry
from pyqula import correlator
from pyqula import kpm
from pyqula import densitymatrix
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
rand = lambda : np.random.randint(0,40)
pairs = [(i,i+np.random.randint(1,8)) for i in range(10,20)]
y = densitymatrix.restricted_dm(h,mode="KPM",pairs=pairs)
t2 = time.clock()
y2 = densitymatrix.restricted_dm(h,mode="full",pairs=pairs)
t3 = time.clock()
print("Time KPM mode = ",t2-t1)
print("Time in full mode = ",t3-t2)
#print(np.trapz(y,x=x,dx=x[1]-x[0]))
plt.subplot(1,2,1)
x = range(len(y))
plt.plot(x,y.real,marker="o",label="KPM")
plt.plot(x,y2.real,marker="o",label="Green")
plt.legend()
plt.subplot(1,2,2)
plt.plot(x,y.imag,marker="o",label="KPM")
plt.plot(x,y2.imag,marker="o",label="Green")
plt.legend()
plt.show()







