# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import disorder
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
import matplotlib.pyplot as plt
g = geometry.square_ribbon(6)
h = g.get_hamiltonian()
h.remove_spin()
h.get_bands()
hr = h.copy()
hl = h.copy()
hcs = [h for i in range(10)]
ht = heterostructures.build(hr,hl,central=[hl,hr])
es = np.linspace(-1.,1.,50)
ht.delta = 1e-3
#ts = np.array([ht.didv(energy=e) for e in es])
ts = np.array([ht.landauer(energy=e) for e in es])
m = np.genfromtxt("BANDS.OUT").transpose()
plt.subplot(2,1,1)
plt.plot(es,ts)
plt.xlim([min(es),max(es)])
plt.subplot(2,1,2)
plt.scatter(m[1],m[0])
plt.xlim([min(es),max(es)])
plt.show()







