# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import groundstate
from pyqula import meanfield
from scipy.sparse import csc_matrix
g = geometry.chain()
#g = g.supercell(2)

def get():
    h = g.get_hamiltonian() # create hamiltonian of the system
    m = np.random.random(3)-0.5
    m = m/np.sqrt(m.dot(m)) # random magnetic field
    h.add_zeeman(m)
    nk = 30
    filling = 0.25
    h.turn_nambu()
    mf = meanfield.guess(h,"random")

    scf = meanfield.Vinteraction(h,V1=-2.0,nk=nk,filling=filling,mf=mf,
            constrains =["no_charge"],verbose=0)
    h = scf.hamiltonian # get the Hamiltonian
    h.check()
    print(scf.identify_symmetry_breaking())
    d = np.round(h.get_average_dvector(),3)
    print("d-vector",d,np.sum(d))
    return h.get_gap()


for i in range(3):
    print(get())









