# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula importgeometry
import topology
import klist
#for i in range(3):
g = geometry.honeycomb_lattice()
g = g.supercell(3)
h = g.get_hamiltonian(has_spin=True)
import time
import parallel
t0 = time.clock()
parallel.cores = 1 # run in 1 core
h.get_dos(nk=10,use_kpm=False)
t1 = time.clock()
print("Time in parallel",t1-t0)
exit()
parallel.cores = 1 # run in 1 core
h.get_dos(nk=10,use_kpm=False)
t2 = time.clock()
print("Time in serial",t2-t1)
#dos.dos(h,nk=100,use_kpm=True)







