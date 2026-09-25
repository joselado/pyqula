# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Z2 invariant of a time-reversal-symmetric (class DIII) superconductor:
# helical p-wave pairing on the square lattice, spin up pairing as px-ipy
# and spin down as px+ipy, swept across the band (between -4 and 4)
import numpy as np
from pyqula import geometry

def helical(r1,r2): # helical p-wave pairing, d-vector along the bond
    dr = r1-r2 # bond vector
    if abs(dr.dot(dr)-1.)>1e-4: return np.zeros((2,2)) # first neighbors only
    return 1j*np.array([[0.,dr[0]-1j*dr[1]],[dr[0]+1j*dr[1],0.]]) # i d.sigma

g = geometry.square_lattice() # square lattice
mus = np.linspace(-5.5,5.5,12) # chemical potentials, avoiding the gap closing at 0
z2s = [] # storage
for mu in mus:
    h = g.get_hamiltonian() # spinful first-neighbor Hamiltonian
    h.add_onsite(-mu) # chemical potential
    h.add_pairing(mode=helical,delta=0.3) # BdG Hamiltonian with helical pairing
    z2s.append(h.get_topological_invariant(nk=30,nt=30)) # Z2 invariant
    print("mu =",mu,"Z2 =",z2s[-1])

import matplotlib.pyplot as plt
plt.plot(mus,z2s,marker="o")
plt.xlabel("Chemical potential") ; plt.ylabel("Z2 invariant")
plt.show()
