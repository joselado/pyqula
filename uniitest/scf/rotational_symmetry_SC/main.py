# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.environ["PYQULAROOT"])

import numpy as np
from pyqula import geometry

g = geometry.bichain()
h0 = g.get_hamiltonian()
maxerror= 1e-8 # error is SCF
def compute():
    v = np.random.random(3)-.5 # random exchange
    v = 4*v/np.sqrt(v.dot(v)) # normalize
    h1 = h0.copy() ; h1.add_exchange(v) # add some zeeman field
    h1.turn_nambu()
    h,etot = h1.get_mean_field_hamiltonian(nk=20,mf="random",V1=-2.,
                                           filling = .3,
                                           maxerror=maxerror,
                                      return_total_energy=True) 
    return h.get_gap() # return the gap

etots = np.array([compute() for i in range(6)]) # compute 6 times
print(etots)
diff = etots-np.mean(etots) # differences
if np.max(np.abs(diff))>maxerror*10:
    print("Warning, no rotational symmetry")
    raise


