# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
import numpy as np


g = geometry.chain(100) # chain
g.dimensionality = 0
vs = np.linspace(0.0,4.0,30) # potentials

phis = 2*np.pi*np.linspace(0.,1.,100)

x,y,z = [],[],[]

for phi in phis:
    h = g.get_hamiltonian(has_spin=False,non_hermitian=True) # get 
    omega = 1./4. # frequency
    fun = lambda r: 1j*np.cos(2*np.pi*omega*r[0] + phi)
    h.add_onsite(fun) # add onsite energies
    (ks,es,cs) = h.get_bands(operator="edge")
    if len(x)!=0: x = np.concatenate([x,es.real*0+phi])
    else: x = es.real*0. + phi
    if len(y)!=0: y = np.concatenate([y,es])
    else: y = es
    if len(z)!=0: z = np.concatenate([z,cs])
    else: z = cs

x = np.array(x)
y = np.array(y)
z = np.array(z)

import matplotlib.pyplot as plt

plt.scatter(x,y.real,c=z,cmap="rainbow")
plt.colorbar()
plt.show()

