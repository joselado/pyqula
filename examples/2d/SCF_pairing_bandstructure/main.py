# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




import numpy as np
from pyqula import geometry
from pyqula import meanfield

g = geometry.triangular_lattice() # square lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_zeeman([0.,0.,6.0]) # add exchange to promote triplet
h.add_swave(0.0) # activate a BdG Hamiltonian


mf = meanfield.guess(h,mode="random")
scf = meanfield.Vinteraction(h,nk=20,V1 = -2.0,filling=0.2,mf=mf,
    constrains = ["no_normal_term"],
    mix = 0.8,verbose=1)

hscf = scf.hamiltonian # SCF Hamiltonian
# now extract only the anomalous part
h0 = hscf.copy()
h0.remove_nambu() # remove SC (and Nambu as side effect)
h0.add_swave(0.0) # restore Nambu basis 

print("SCF symmetry breaking",scf.identify_symmetry_breaking())

ha = hscf-h0 # this yields a Hamiltonian only with anomalous term
ha.get_bands() # band structure of only the anomalous term








