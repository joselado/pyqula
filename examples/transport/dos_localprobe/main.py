# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula.heterostructures import LocalProbe
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain()
#g = geometry.single_square_lattice()
h = g.get_hamiltonian()
h.add_onsite(2.0)
h.add_swave(0.1)

lp = LocalProbe(h) # create a local probe object
lp.delta = 1e-3
es = np.linspace(-0.2,0.2,101)
import matplotlib.pyplot as plt

Ts = [0.1,0.2,.5]

for T in Ts:
    lp.T = T
    ts = [lp.didv(energy=e) for e in es]
    ds = [lp.get_dos(energy=e) for e in es]
    ts = ts/np.mean(ts)
    ds = ds/np.mean(ds)
    
    plt.subplot(1,2,1)
    plt.plot(es,ts,label="T = "+str(np.round(T,2)))
    plt.xlabel("Energy [t]") ; plt.ylabel("Conductance")
    plt.yticks([])
    
    plt.subplot(1,2,2)
    plt.plot(es,ds,label="T = "+str(np.round(T,2)))
    plt.yticks([])
    plt.xlabel("Energy [t]") ; plt.ylabel("DOS")

plt.subplot(1,2,1) ; plt.legend() ; plt.ylim([0.,5])
plt.subplot(1,2,2) ; plt.legend() ; plt.ylim([0.,5])

plt.tight_layout()
plt.show()


