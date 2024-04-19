# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import topology
from pyqula import spectrum
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_onsite(0.3)
nk = 200
ks,ky,fs = h.get_fermi_surface(nk=nk,delta=1e-2,
        k0=[1./3.,1./3.], # location of the kpoint
        nsuper=0.15 # scale the size of the kmesh
        )
import matplotlib.pyplot as plt
plt.imshow(fs.reshape((nk,nk)))
plt.xticks([]) ; plt.yticks([])
plt.show()





