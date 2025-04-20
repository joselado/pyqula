# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry


g = geometry.chain() # create the geometry
g = g.get_supercell(10) # make a supercell
g.dimensionality = 0 # zero dimensional
h = g.get_hamiltonian() # generate the Hamiltonian

mu = -1. # chemical potential

h.add_onsite(mu)
i = 0 # first index

vevs = [] # storage for the expectation values
Pijs = [] # empty list for the operators
inds = range(len(g.r))
for j in inds: # loop over sites
    Pij = h.get_operator("correlator",i=i,j=j) # correlator between i and j
    Pijs.append(Pij) # store this operator

vevs = h.get_several_vev(Pijs) # compute several expectation values


# these lines would be to compute charge-charge correlator
#from pyqula.chi import static_charge_correlator
#vevs2 = [static_charge_correlator(h,i=0,j=j).real for j in inds]


import matplotlib.pyplot as plt

plt.scatter(inds,vevs,c="red",label="VEV")
#plt.plot(inds,vevs2,c="blue",label="charge-charge")
plt.xlabel("Site j")
plt.ylabel("$\\langle C^\\dagger_i C_j \\rangle$")
plt.show()
