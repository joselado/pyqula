# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import dos
import matplotlib.pyplot as plt


g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
import time
t0 = time.time()
import numpy as np
energies = np.linspace(-.5,.5,200)
(es,ds) = h.get_dos(energies=energies,delta=1e-2,nk=500
                    ,mode="adaptive"
                    )
print("Time in DOS",time.time()-t0)
plt.plot(es,ds)
plt.show()
#dos.dos(h,nk=100,use_kpm=True)







