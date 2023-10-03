# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import specialhamiltonian

h = specialhamiltonian.square_2OTI(delta=0.3)
(k,e) = h.get_bands()
import numpy as np

print(np.round(h.intra,1))

h = h.set_finite_system(n=6)
(x,y,d) = h.get_ldos(e=0.,delta=0.1)


import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.scatter(k,e)
plt.subplot(1,2,2)
plt.scatter(x,y,c=d)

plt.show()
