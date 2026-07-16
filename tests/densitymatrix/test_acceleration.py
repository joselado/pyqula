# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.environ["PYQULAROOT"])

import numpy as np
from pyqula import geometry
import time

g = geometry.honeycomb_lattice()
g = g.get_supercell(4)
h = g.get_hamiltonian()
m = np.random.random(h.intra.shape) + 1j*np.random.random(h.intra.shape)
m = m + np.conjugate(m).T
h.intra = m # overwrite with a random matrix
nk=3
def compute(mode,use_ds=True):
    from pyqula.dmtk import fulldm
    fulldm.mode=mode[0]
    if use_ds: ds = [[i,0,0] for i in range(10)]
    else: ds = None
    o = h.get_density_matrix(nk=nk,ds=ds,dm_mode=mode[1]) # once to compile
    t0 = time.time()
    for i in range(4):
        o = h.get_density_matrix(nk=nk,ds=ds,dm_mode=mode[1]) # once to compile
    t1 = time.time()
    if use_ds: o = np.array([o[key] for key in o]) # as array
    print(mode,t1-t0)
    return o

for ud in [True, False]:
    print("Using ds",ud)
    modes1 = ["explicit","vectorized"]
    modes2 = ["accumulate","simultaneous"]
    modes = []
    for mode1 in modes1:
        for mode2 in modes2: modes.append([mode1,mode2])
    outs = np.array([compute(mode,use_ds=ud) for mode in modes]) # compute 6 times
    mout = np.mean(outs,axis=0) # average
    diffs = [np.sum(np.abs(o - mout)) for o in outs] # error
    if np.max(diffs)>1e-4:
        print("Errors",diffs)
        raise
    else:
        print("All methods consistent")


