# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import dos
import matplotlib.pyplot as plt


g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
(es,ds) = h.get_dos(mode="ED",delta=0.01,nk=100)
plt.plot(es,ds)
plt.show()
#dos.dos(h,nk=100,use_kpm=True)







