# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import parallel
g = geometry.honeycomb_lattice()
g = geometry.chain()
g = g.get_supercell(30)
cores = [1,7]
import time
for core in cores:
    parallel.set_cores(core)
    h = g.get_hamiltonian() # create hamiltonian of the system
    t0 = time.time()
#    hmf = h.get_mean_field_hamiltonian(nk=5,U=3.,mf="antiferro") 
    h.get_dos(mode="Green")
    # mean field Hamiltonian
    t1 = time.time()
    print("Time with",core,"is",t1-t0)







