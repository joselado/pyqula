# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import embedding
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=4,nedges=4) # get a square
h = g.get_hamiltonian() # get the Hamiltonian
# perform a calculation for the isotaled system

if False: # make it True as a sanity check
    hscf = h.get_mean_field_hamiltonian(U=1.0,filling=0.5,
                   mf="ferro",
                   constrains=["no_charge"],
                   verbose=1) # perform SCF
    print("Finished mean-field of the isolated system")
    h.write_magnetization() # get the magnetization in each site
    exit()

# create a selfenergy in a single edge (the left one) to kill magnetism 
hs = h.copy() ; hs = hs*0.0 ; xmin = np.min(g.r[:,0])
hs.add_onsite(lambda r: (np.abs(r[0]-xmin)<1e-2)*1.0)
selfe = -1j*hs.intra # selfenergy on the left side, beware of the minus sign


# create an embedding object with that selfenergy
eb = embedding.embed_hamiltonian(h,selfenergy=selfe) # create embedding object
eb = eb.get_mean_field_hamiltonian(U=1.0,verbose=1,
           constrains=["no_charge"], # this ignores chemical potential renorm
           mf="antiferro", # initial ferro guess
           mix=0.9 # aggresive mixing
           )
# eb.H is the selfconsistent Hamiltonian object
eb.H.write_magnetization() # get the magnetization in each site




