# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import potentials
import numpy as np


g = geometry.bichain(100) # bipartitie chain geometry
g.dimensionality = 0 # zero dimensional (finite system)

def get_energies(omega=0.0,phi=0.0):
    """Compute the energies"""
    h = g.get_hamiltonian(has_spin=True) # generate Hamiltonian
    def fm(r):
        """Function that add the antiferromagnetism"""
        return .5 + .5*np.cos(2*np.pi*(omega*r[0]+phi)) # local AF magnetization
    h.add_antiferromagnetism(fm)
    inds,es = h.get_bands() # compute eigenstates
    return es # return eigenstates

# compute the Hofstadter butterfly
f = open("HOFSTADTER.OUT","w") # output file
omegas = np.linspace(0.,1.,100) # different frequencies
omegas_plot,es_hofstadter = [],[]
for omega in omegas:
    es = get_energies(omega=omega) # return energies
    for e in es:
        f.write(str(omega)+"  "+str(e)+"\n") # write in file
        omegas_plot.append(omega)
        es_hofstadter.append(e)
f.close()


# compute the spectrum as a function of the phason phi
f = open("PUMPING.OUT","w") # output file
phis = np.linspace(0.,1.,100) # different frequencies
phis_plot,es_pumping = [],[]
for phi in phis:
    es = get_energies(omega=0.1,phi=phi) # return energies
    for e in es:
        f.write(str(phi)+"  "+str(e)+"\n") # write in file
        phis_plot.append(phi)
        es_pumping.append(e)
f.close()

import matplotlib.pyplot as plt
plt.subplot(121)
plt.scatter(omegas_plot,es_hofstadter,s=1)
plt.xlabel("omega")
plt.ylabel("Energy")
plt.subplot(122)
plt.scatter(phis_plot,es_pumping,s=1)
plt.xlabel("phi")
plt.ylabel("Energy")
plt.show()










