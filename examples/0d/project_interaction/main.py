# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
import numpy as np

def get(n):
  g = islands.get_geometry(name="triangular",n=n,nedges=20,rot=0.0) # get an island
  # maximum distance to the origin
  h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
  h.set_filling(.5)
  m = h.project_interactions(n=4)
  return np.mean(np.abs(m))

ns = list(range(2,8))
ys = []
for n in ns:
    y = get(n)
    print(y)
    ys.append(y)

import matplotlib.pyplot as plt
plt.plot(ns,ys,marker="o")
plt.xlabel("n")
plt.ylabel("Mean |projected interaction|")
plt.show()







