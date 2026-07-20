# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

fo = open("BAND.OUT","w")

from pyqula import geometry
import numpy as np
g = geometry.chain() # geometry of a chain
g = g.get_supercell(100) # generate a 100 supercell
g.dimensionality = 0 # set finite
phis_all,es_all,cs_all = [],[],[] # storage for the plot
for phi in np.linspace(0.,2*np.pi,400): # loop over phases
    h = g.get_hamiltonian() # create Hamiltonian of the system
    omega = (1 + np.sqrt(5))/2.*2.*np.pi # frequency
    h.add_onsite(lambda r: 6*np.cos(omega*r[0]+phi)) # quasiperiodicity
    (k,e,c) = h.get_bands(operator="xposition") # calculate eigenvalues


    for (ie,ic) in zip(e,c):
        fo.write(str(phi)+"    ")
        fo.write(str(ie)+"    ")
        fo.write(str(ic)+"  \n")
        print(phi,ie,ic)
    fo.flush()
    phis_all += [phi]*len(e) # store the phase
    es_all += list(e) # store the energies
    cs_all += list(c) # store the position operator

import matplotlib.pyplot as plt

plt.scatter(phis_all,es_all,c=cs_all,cmap="inferno",s=2)
plt.colorbar(label="x-position")
plt.xlabel("Phase") ; plt.ylabel("Energy")
plt.show()




