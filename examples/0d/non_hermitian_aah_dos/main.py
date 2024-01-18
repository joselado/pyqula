# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
import numpy as np


g = geometry.chain(100) # chain
g.dimensionality = 0
vs = np.linspace(0.0,4.0,30) # potentials

phis = 2*np.pi*np.linspace(0.,1.,101)

x,y,z = [],[],[]

es = np.linspace(-1,1,101)

for phi in phis:
    h = g.get_hamiltonian(has_spin=False,non_hermitian=True) # get 
    omega = 1./4. # frequency
    fun = lambda r: 1j*np.cos(2*np.pi*omega*r[0] + phi)
    h.add_onsite(fun) # add onsite energies
    (es,ds) = h.get_dos(energies=es,operator="edge")
    z.append(ds)

x = np.array(x)
y = np.array(y)
z = np.array(z)

import matplotlib.pyplot as plt

plt.imshow(np.array(z).T,cmap="inferno")
plt.colorbar()
plt.show()

