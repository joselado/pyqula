# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.square_lattice() # linear chain
h = g.get_hamiltonian() # get the Hamiltonian
h.shift_fermi(1.5)
o = h.get_rkky()

ns = range(1,10)

ds = [h.get_rkky(R=[i,0.,0.]) for i in ns]


import matplotlib.pyplot as plt
plt.plot(ns,ds,marker="o")

plt.show()



