# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import latticegas # get_specific_heat/get_susceptibility are reused as-is
from pyqula import latticeising
import numpy as np

g = geometry.square_lattice()
g = g.get_supercell(12)
g.dimensionality = 0
n = len(g.r)

li = latticeising.LatticeIsing(g,m=0.0)
li.add_interaction(Jij=[1.]) # ferromagnetic coupling

# scan the temperature with fixed-magnetization-fluctuating (single-flip)
# Metropolis dynamics, cooling from high to low temperature and reusing
# the previous configuration at each step to reduce equilibration time
temps = np.linspace(6.,2.,20) # get_energy() double-counts bonds (see
    # latticeising module docstring), so the ferromagnetic transition
    # here sits near 2*2.269 for the 2d square lattice, not 2.269
ntries = int(60*n)
mags = [] ; specific_heats = []
for temp in temps:
    es,ms = li.optimize_energy(temp=temp,ntries=ntries)
    burn = len(es)//2 # discard the first half as equilibration
    mags.append(np.mean(np.abs(ms[burn:]))/n)
    specific_heats.append(latticegas.get_specific_heat(es[burn:],temp)/n)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(temps,mags,marker="o")
plt.xlabel("Temperature")
plt.ylabel("|Magnetization| per site")

plt.subplot(1,2,2)
plt.plot(temps,specific_heats,marker="o")
plt.xlabel("Temperature")
plt.ylabel("Specific heat per site")

plt.tight_layout()
plt.show()
