# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry  # library to create crystal geometries
import numpy as np
g = geometry.chain()
g = g.supercell(100)
g.dimensionality = 0
h = g.get_hamiltonian(is_sparse=True)

m = h.intra # get the matrix

m = m/2.

from pyqula.kpmtk import kpmnumba
from pyqula import kpm
import time

from pyqula.kpm import local_dos
#local_dos(m) ; exit()


v = np.zeros(m.shape[0],dtype=np.complex_) ; v[0] = 1.0
n = 1000
mus0 = kpmnumba.kpm_moments(v,m,n=n)
t0 = time.time()
mus0 = kpmnumba.kpm_moments(v,m,n=n)
t1 = time.time()
mus1 = kpm.python_kpm_moments(v,m,n=n)
t2 = time.time()

print(mus0-mus1)

print("Python",t2-t1)
print("Numba",t1-t0)

