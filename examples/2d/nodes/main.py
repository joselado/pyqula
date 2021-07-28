# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import nodes
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
k = nodes.degenerate_points(h,n=len(g.r)//2-1) 
print(k)
nodes.dirac_points(h,n=len(g.r)//2-1) 







