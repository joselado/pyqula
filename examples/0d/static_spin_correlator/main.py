# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")
sys.path.append(os.environ["DMRGROOT"])

import numpy as np
from pyqula import geometry


g = geometry.chain() # create the geometry
g = g.get_supercell(8) # make a supercell
g.dimensionality = 0 # zero dimensional
h = g.get_hamiltonian() # generate the Hamiltonian
h.add_onsite(0.7)
h.add_rashba(0.3)
h.add_exchange([0.2,0.5,1.0])

mu = -1. # chemical potential

h.add_onsite(mu)
i = 0 # first index

vevs = [] # storage for the expectation values
Pijs = [] # empty list for the operators
inds = range(len(g.r))

# these lines would be to compute charge-charge correlator
from pyqula.chi import static_sx_correlator
from pyqula.chi import static_charge_correlator
from pyqula.chi import static_sy_correlator
from pyqula.chi import static_sz_correlator
vevd = [static_charge_correlator(h,i=0,j=j) for j in inds]
vevsx = [static_sx_correlator(h,i=0,j=j) for j in inds]
vevsy = [static_sy_correlator(h,i=0,j=j) for j in inds]
vevsz = [static_sz_correlator(h,i=0,j=j) for j in inds]

from pyqula import pyqula2dmrgpy

fc = pyqula2dmrgpy.generate_fermionic_chain(h)


def vevtn(A,B):
    return fc.vev(A*B) - fc.vev(A)*fc.vev(B)

vevtnd = [vevtn(fc.Ntot[i],fc.Ntot[j]) for j in inds]
vevtnsx = [vevtn(fc.Sx[i],fc.Sx[j]) for j in inds]
vevtnsy = [vevtn(fc.Sy[i],fc.Sy[j]) for j in inds]
vevtnsz = [vevtn(fc.Sz[i],fc.Sz[j]) for j in inds]


import matplotlib.pyplot as plt

plt.subplot(1,4,1)
plt.title("XX")
plt.scatter(inds,vevsx,marker="o",c="red",s=300,label="pyqula")
plt.scatter(inds,vevtnsx,marker="o",c="blue",s=100,label="dmrgpy")
plt.legend()
plt.subplot(1,4,2)
plt.title("YY")
plt.scatter(inds,vevsy,marker="o",c="red",s=300,label="pyqula")
plt.scatter(inds,vevtnsy,marker="o",c="blue",s=100,label="dmrgpy")
plt.legend()
plt.subplot(1,4,3)
plt.title("ZZ")
plt.scatter(inds,vevsz,marker="o",c="red",s=300,label="pyqula")
plt.scatter(inds,vevtnsz,marker="o",c="blue",s=100,label="dmrgpy")
plt.legend()
plt.subplot(1,4,4)
plt.title("Density")
plt.scatter(inds,vevd,marker="o",c="red",s=300,label="pyqula")
plt.scatter(inds,vevtnd,marker="o",c="blue",s=100,label="dmrgpy")
plt.legend()
plt.tight_layout()
plt.show()
