# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import heterostructures
import numpy as np
import matplotlib.pyplot as plt

def get_conductance(W=0.0,L=10):
    """Get the conductance for a value of disorder W
    and length L"""
    g = geometry.chain()
    g = geometry.honeycomb_zigzag_ribbon()
    h = g.get_hamiltonian()
    h.add_soc(0.05)
    hr = h.copy() # right lead
    hl = h.copy() # left lead
#    hr.add_onsite(1.5) # shift chemical potential for the lead
#    hl.add_onsite(1.5) # shift chemical potential for the lead
    hcs = [] # empty list for the scattering region
    for i in range(L): # loop over cells of the scattering region
        hi = h.copy() # make a copy of the Hamiltonian
        hi.add_onsite(lambda r: W*(np.random.random()-0.5)) # add disorder
        hcs.append(hi.copy()) # add Hamiltonian to the list
    ht = heterostructures.build(hr,hl,central=hcs) # create the scattering region
    ht.use_minimal_selfenergy = True
    ht.minimal_selfenergy_gamma = 1.
    ht.delta = 1e-7 # accurate analytic continuation
    energy = 0.0 # energy of the transport calculation
    return ht.didv(energy=energy) # return conductance




Ws = [0.,1.] # values of disorder
Ls = range(4,40,4) # length of the system
ntries = 1 # number of realizations
for W in Ws: # loop over disorders
    Gs = [get_conductance(W=W,L=L) for L in Ls] # compute conductance
    plt.plot(Ls,Gs,label="W = "+str(W),marker="o")
plt.xlabel("Length")
plt.ylabel("Conductance")
plt.legend()
plt.show()



