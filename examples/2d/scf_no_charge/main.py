# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import meanfield

g = geometry.honeycomb_lattice()
g = g.supercell(1)
h = g.get_hamiltonian() # create hamiltonian of the system
U = 3.0
filling = 0.5
mf = meanfield.guess(h,mode="random")
scf = meanfield.Vinteraction(h,nk=10,U=U,filling=filling,mf=mf,
        constrains=["no_charge"])
h = scf.hamiltonian # get the Hamiltonian
h.get_bands() # calculate band structure







