# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import disorder
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
import matplotlib.pyplot as plt
g = geometry.square_ribbon(10)
g = g.supercell(2)
h = g.get_hamiltonian()
#h.add_peierls(.2)
h.remove_spin()
h = disorder.anderson(h,w=0.5)
h.get_bands(nk=2000)
hr = h.copy()
hl = h.copy()
hcs = [h for i in range(10)]
ht = heterostructures.create_leads_and_central_list(hr,hl,hcs)
es = np.linspace(-.5,.5,500)
ts = np.array([ht.didv(e,delta=1e-11,kwant=True) for e in es])
m = np.genfromtxt("BANDS.OUT").transpose()
plt.subplot(2,1,1)
plt.plot(es,ts)
plt.xlim([min(es),max(es)])
plt.subplot(2,1,2)
plt.scatter(m[1],m[0])
plt.xlim([min(es),max(es)])
plt.show()







