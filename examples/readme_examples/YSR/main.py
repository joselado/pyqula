# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

fo = open("DOSMAP.OUT","w")



from pyqula import geometry
from pyqula import embedding
import numpy as np

g = geometry.square_lattice() # create geometry 
for J in np.linspace(0.,4.0,100): # loop over exchange
    h = g.get_hamiltonian() # get the Hamiltonian,spinless
    h.add_onsite(3.0) # shift chemical potential
    h.add_swave(0.2) # add s-wave superconductivity
    hv = h.copy() # copy Hamiltonian to create a defective one
    # add magnetic site
    hv.add_exchange(lambda r: [0.,0.,(np.sum((r - g.r[0])**2)<1e-2)*J]) 
    eb = embedding.Embedding(h,m=hv) # create an embedding object
    energies = np.linspace(-0.4,0.4,100) # energies
    d = [eb.dos(nsuper=2,delta=1e-2,e=ei) for ei in energies] # compute DOS


    print(J)
    for (ei,di) in zip(energies,d):
        fo.write(str(J)+" ")
        fo.write(str(ei/0.2)+" ")
        fo.write(str(di)+"\n")

fo.close()
