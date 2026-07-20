# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
from pyqula import densitymatrix
import numpy as np
import time
g = islands.get_geometry(name="honeycomb",n=4,clean=False) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h.add_rashba(.2)
h.add_zeeman([0.,0.,0.3])
g.write()
t1 = time.time()
dm = densitymatrix.full_dm(h,dm_mode="accumulate")
t2 = time.time()
dmf = densitymatrix.full_dm(h,dm_mode="simultaneous")
t3 = time.time()
print("Error = ",np.sum(np.abs(dm-dmf)))
print("Time simultaneous = ",t3-t2)
print("Time accumulate = ",t2-t1)

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(np.abs(dm))
plt.colorbar(label="|Density matrix|")
plt.title("accumulate")
plt.subplot(1,2,2)
plt.imshow(np.abs(dm-dmf))
plt.colorbar(label="|Difference|")
plt.title("simultaneous - accumulate")
plt.show()







