# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.environ["PYQULAROOT"])

import numpy as np
from pyqula import geometry
import time

g = geometry.bichain()
g = g.get_supercell(2)
h = g.get_hamiltonian()
m = np.random.random(h.intra.shape) + 1j*np.random.random(h.intra.shape)
m = m + np.conjugate(m).T
h.intra = m # overwrite with a random matrix
energies = np.linspace(0.,5.0,400)
nk=3
import time
def compute(mode):
    from pyqula.chitk import rpa
    rpa.mode_rpa = mode
    t0 = time.time()
    (qs,es,chis) = h.get_qdos_iets(nk=1,nq=2,energies=energies)
    t1 = time.time()
    print("Compilation",mode,t1-t0)
    t0 = time.time()
    (qs,es,chis) = h.get_qdos_iets(nk=10,nq=20,energies=energies)
    t1 = time.time()
    print("Time in",mode,t1-t0)
    return chis

modes = ["sequential","vectorized"]
outs = np.array([compute(mode) for mode in modes]) # compute 6 times
mout = np.mean(outs,axis=0) # average
diffs = [np.sum(np.abs(o - mout)) for o in outs] # error
if np.max(diffs)>1e-4:
    print("Errors",diffs)
    raise
else:
    print("All methods consistent")


