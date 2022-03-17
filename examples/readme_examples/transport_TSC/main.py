# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

fo = open("DOSMAP.OUT","w")

from pyqula import geometry
from pyqula import heterostructures
import numpy as np

g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create teh Hamiltonian
h1 = h.copy() # first lead
h2 = h.copy() # second lead
h2.add_onsite(2.0) # shift chemical potential in the second lead
h2.add_exchange([0.,0.,.3]) # add exchange in the second lead
h2.add_rashba(.3) # add Rashba SOC in the second lead
h2.add_swave(.05) # add s-wave SC in the second lead
es = np.linspace(-.1,.1,100) # grid of energies
for T in np.linspace(1e-3,0.5,10): # loop over transparencies
    HT = heterostructures.build(h1,h2) # create the junction
    HT.set_coupling(T) # set the coupling between the leads 
    Gs = [HT.didv(energy=e) for e in es] # calculate transmission


    print(T)
    for (ei,gi) in zip(es,Gs):
        fo.write(str(T)+" ")
        fo.write(str(ei)+" ")
        fo.write(str(gi/Gs[0])+"\n")

fo.close()
