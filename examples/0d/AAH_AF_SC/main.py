# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
import numpy as np

g = geometry.bichain()
g = g.supercell(20) # get the geometry
g.dimensionality = 0

def get_energies(omega):
    h = g.get_hamiltonian() # compute Hamiltonian
    delta = .0
    maf = .5
    def fsc(r): return delta*np.cos(omega*np.pi*2*r[0])
    def faf(r): return maf*np.sin(omega*np.pi*2*r[0])
    h.add_antiferromagnetism(faf)
    h.add_swave(fsc)
    return h.get_bands()[1]

fo = open("HOFSTADTER.OUT","w")
for omega in np.linspace(0.,1.,100):
    es = get_energies(omega)
    for e in es: fo.write(str(omega)+"  "+str(e)+"\n")

fo.close()







