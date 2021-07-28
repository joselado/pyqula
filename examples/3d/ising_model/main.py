# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import alloy
g = geometry.cubic_lattice()
#g = g.supercell(2)
a = alloy.Alloy(g)
n = len(a.r)
a.set_species([0 for i in range(n//2)]+[1 for i in range(n//2)])
def f(d,i,j):
    if i!=j and abs(d-a.d2[1])<1e-3: return -1
    return 0.0
a.fenergy = f
d = {(0,1,1):-1,(1,0,1):-1}
a.setup_interaction([[0,-1],[-1,0]])
a.setup_interaction(d)
a.supercell(2)
a.randomize([a.n//2,a.n//2])
a.minimize_energy()
#a.minimize_energy()
a.write()
print(a.get_energy())







