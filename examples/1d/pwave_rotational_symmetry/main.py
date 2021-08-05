# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry,meanfield
def get():
    g = geometry.chain()
#    g = g.supercell(4) ; g.dimensionality = 0
    h = g.get_hamiltonian() # create hamiltonian of the system
    J = np.random.random(3) - 0.5
    J = np.array([0.,0.,1.])
    J = J/np.sqrt(J.dot(J))
#    print("Exchange",J)
#    h.add_zeeman(J)
    h.shift_fermi(1.0)
    dr = np.random.random(3) ; dr = dr/np.sqrt(dr.dot(dr)) # first vector
    di = np.random.random(3) ; di = np.cross(dr,di)
    di = di/np.sqrt(di.dot(di)) # second vector
    d = dr + 1j*di
    du0 = (1j*np.cross(np.conjugate(d),d)).real # non-unitarity
    du0 = du0/np.sqrt(du0.dot(du0)) # normalize
    print("Input non-unitarity",du0)
    h.add_pairing(d=d,mode="triplet",delta=1.0)
#    h.add_pairing(delta=0.3)
#    h.add_swave(delta=0.3)
    du = h.get_dvector_non_unitarity()[0]
    du = du/np.sqrt(du.dot(du)) # normalize
    print("Output non-unitarity",np.round(du,3))
#    print(np.round(h.get_hk_gen()([.25,0.,0.]),2))
    h.get_bands()
    

for i in range(1):
    print()
    print("Trial")
    get()



