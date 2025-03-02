# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry, embedding, potentials
import matplotlib.pyplot as plt
import numpy as np

g = geometry.honeycomb_lattice() # take a honeycomb lattice
h = g.get_hamiltonian() # generate Hamiltonian

hv = h.copy() # make a copy
hv.add_onsite(potentials.impurity(g.r[0],v=1e6)) # unit cell with a vacancy
# first the pristine
plt.subplot(1,2,1) ; plt.title("Without vacancy")
eb0 = embedding.Embedding(h,m=h) 
energies = np.linspace(-.8,.8,41) # energy grid
(e0,d0) = eb0.multidos(energies=energies,delta=3e-2) # compute LDOS
plt.plot(e0,d0,c="black") ; plt.xlabel("Energy") ; plt.ylabel("DOS") # plot
plt.xlim([min(e0),max(e0)]) ; plt.ylim([0.,max(d0)])

# create an embedding object (infinite pristine system with h, central impurity hv)
plt.subplot(1,2,2) ; plt.title("With vacancy")
eb = embedding.Embedding(h,m=hv.intra) 
#d = [eb.get_dos(e=e,delta=3e-2) for e in energies]
(e,d) = eb.multidos(energies=energies,delta=3e-2) # compute LDOS
plt.plot(e,d,c="black") ; plt.xlabel("Energy") ; plt.ylabel("DOS") # plot
plt.fill_between(e, d, where=d>=d, interpolate=True, color='lightyellow') # add a background
plt.xlim([min(e),max(e)]) ; plt.ylim([0.,max(d)])

plt.show()
