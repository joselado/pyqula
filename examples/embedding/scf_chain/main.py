# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import embedding
from pyqula import islands
g = geometry.bichain()
h = g.get_hamiltonian() # get the Hamiltonian
# perform a calculation for the isotaled system

if True: # make it True as a sanity check
    hscf = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,
                   mf="antiferro",
                   constrains=["no_charge"],
                   nk=40,
                   verbose=1) # perform SCF
    print("Finished mean-field of the isolated system")
    hscf.get_bands()
    print("Gap = ",hscf.get_gap())
    exit()


# create an embedding object with that selfenergy
eb = embedding.embed_hamiltonian(h) # create embedding object
eb = eb.get_mean_field_hamiltonian(U=2.0,verbose=1,
           constrains=["no_charge"], # this ignores chemical potential renorm
           mf="antiferro", # initial antiferro guess
           mix=0.9, # aggresive mixing
           nk = 40
           )
# eb.H is the selfconsistent Hamiltonian object
eb.H.get_bands()
print("Gap = ",eb.H.get_gap())




