# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
import numpy as np
g = geometry.chain() # create a 1D chain
g = g.supercell(60) # create a supercell
g.dimensionality = 0 # and make it finite
h = g.get_hamiltonian() # generate a Hamiltonian with first neighbor hopping
# now create an artificial TSC combining Zeeman, Rashba and SC
h.add_rashba(.5) # add Rashba SOC

# if you want a spatially dependent potential, just add a function as
# fr = lambda r: np.cos(r[0]) # this will modulate energies in a cosine-like fashion
# where "r" refers to the location of that site
# h.add_onsite(fr) # shift the chemical potential
# fyi, the positions of the 1d chain can be accessed as g.r

h.add_onsite(2.0) # shift the chamical potential
h.add_exchange([0.,0.,.5]) # add exchange field
h.add_swave(0.2) # add swave SC order
# compute LDOS in an energy window
energies = np.linspace(-0.4,0.4,100)
f = open("DOS_MAP.OUT","w") # file to write the results
for e in energies: # loop over energies
    x,y,d = h.get_ldos(delta=1e-2,e=e) # compute LDOS
    for i in range(len(d)): # loop over locations
        f.write(str(e)+"  ") # write energy
        f.write(str(x[i])+"  ") # write x position
        f.write(str(d[i])+"\n") # write DOS
f.close()






