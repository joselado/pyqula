# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import meanfield
import numpy as np


omega = 1./np.sqrt(2)

g = geometry.bichain()
g = g.supercell(40) # get the geometry
g.dimensionality = 0

h = g.get_hamiltonian() # compute Hamiltonian
phi = 0.5
delta = 2.*np.cos(phi)
maf = 2.*np.sin(phi)*np.array([1.,0.,0.])
def fsc(r): return delta*np.cos(omega*np.pi*2*r[0])
def faf(r): return maf*np.sin(omega*np.pi*2*r[0])
h.add_antiferromagnetism(faf)
h.add_swave(fsc)
mf = None
filling=0.5
h.set_filling(filling)
mf = meanfield.guess(h,"random")
scf = meanfield.Vinteraction(h,V1=-2.0,mf=mf,
        filling=filling,mix=0.1,verbose=1,
        constrains = ["no_normal_term"]
        )

print(scf.identify_symmetry_breaking())
hscf = scf.hamiltonian


(inds,es) = h.get_bands()
(indsscf,esscf) = hscf.get_bands()


import matplotlib.pyplot as plt

plt.scatter(range(len(es)),es,c="red",label="Non-interacting")
plt.scatter(range(len(esscf)),esscf,c="blue",label="Interacting")

plt.legend()

plt.show()










