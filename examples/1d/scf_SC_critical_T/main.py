# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry,meanfield
def get(T):
    g = geometry.chain()
    h = g.get_hamiltonian() # create hamiltonian of the system
    h.turn_nambu()
    h = h.get_mean_field_hamiltonian(U=-.6,nk=100,T=T,mf="random",maxerror=1e-6)
    return h.get_gap()/2. # return the SC gap
    
Tmax = get(0.) # maximum gap
Ts = np.linspace(0.,Tmax,10)
gs = np.array([get(T) for T in Ts])

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(Ts,gs,marker="o")
plt.ylabel("$\\Delta$") ; plt.xlabel("T")

DT = 1./1.76 # BCS ratio
plt.subplot(1,2,2)
plt.plot(Ts/Tmax,gs/Tmax,marker="o")
plt.ylabel("$\\Delta/\\Delta_{T=0}$") ; plt.xlabel("$T/\\Delta_{T=0}$")
plt.plot([DT,DT],[0.,1.],linestyle="--")

plt.tight_layout()

plt.show()


