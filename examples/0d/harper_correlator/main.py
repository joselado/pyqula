# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import densitymatrix
import numpy as np
n = 400
g = geometry.chain(n) # chain
g.dimensionality = 0
h = g.get_hamiltonian(has_spin=False)
h.add_onsite(lambda r: 2.3*np.cos(np.sqrt(2)/2.*r[0]))
pairs = [(n//2,n//2+i) for i in range(n//3)]
y = densitymatrix.restricted_dm(h,mode="full",pairs=pairs).real
import matplotlib.pyplot as plt
x = np.array(range(len(y)))
plt.plot(x,y*x)
plt.show()







