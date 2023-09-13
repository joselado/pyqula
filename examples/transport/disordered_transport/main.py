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
    h = g.get_hamiltonian()
    hr = h.copy() # right lead
    hl = h.copy() # left lead
    hr.add_onsite(1.5) # shift chemical potential for the lead
    hl.add_onsite(1.5) # shift chemical potential for the lead
    hcs = [] # empty list for the scattering region
    for i in range(L): # loop over cells of the scattering region
        hi = h.copy() # make a copy of the Hamiltonian
        hi.add_onsite(lambda r: W*(np.random.random()-0.5)) # add disorder
        hcs.append(hi.copy()) # add Hamiltonian to the list
    ht = heterostructures.build(hr,hl,central=hcs) # create the scattering region
    ht.delta = 1e-7 # accurate analytic continuation
    energy = 0.2 # energy of the transport calculation
    return ht.didv(energy=energy) # return conductance




Ws = [0.,1.,2.] # values of disorder
Ls = range(4,100,4) # length of the system
ntries = 4 # number of realizations
for W in Ws: # loop over disorders
    get_G = lambda W,L: np.mean([get_conductance(W=W,L=L) for i in range(ntries)])
    Gs = [get_G(W,L) for L in Ls] # compute conductance
   # Gs = [get_conductance(W=W,L=L) for L in Ls] # compute conductance
    plt.plot(Ls,Gs,label="W = "+str(W),marker="o")
plt.xlabel("Length")
plt.ylabel("Conductance")
plt.legend()
plt.show()



