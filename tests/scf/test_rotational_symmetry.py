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
    v = 2*v/np.sqrt(v.dot(v)) # normalize
    vs = [v,-v]
    mf = h0.copy() ; mf.add_exchange(vs) # make this initial guess
    h1 = h0.copy() ; h1.add_exchange(.4*v) # add some bias
    h,etot = h1.get_mean_field_hamiltonian(nk=20,mf=mf,U=2.,
                                           maxerror=maxerror,
                                      return_total_energy=True) 
    return etot

etots = np.array([compute() for i in range(6)]) # compute 6 times
print(etots)
diff = etots-np.mean(etots) # differences
if np.max(np.abs(diff))>maxerror*10:
    print("Warning, no rotational symmetry")
    print(diff)
    raise


