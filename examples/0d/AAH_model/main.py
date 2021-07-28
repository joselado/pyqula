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
for omega in omegas:
    es = get_energies(omega=omega) # return energies
    for e in es: f.write(str(omega)+"  "+str(e)+"\n") # write in file
f.close()


# compute the spectrum as a function of the phason phi
f = open("PUMPING.OUT","w") # output file
phis = np.linspace(0.,1.,100) # different frequencies
for phi in phis:
    es = get_energies(omega=0.1,phi=phi) # return energies
    for e in es: f.write(str(phi)+"  "+str(e)+"\n") # write in file
f.close()










