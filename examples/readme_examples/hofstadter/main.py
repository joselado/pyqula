# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

fo = open("DOSMAP.OUT","w")

import numpy as np
from pyqula import geometry
g = geometry.square_ribbon(40) # create square ribbon geometry

for B in np.linspace(0.,1.0,300): # loop over magnetic field
    h = g.get_hamiltonian() # create a new hamiltonian
    h.add_orbital_magnetic_field(B) # add an orbital magnetic field
    # calculate DOS projected on the bulk
    (e,d) = h.get_dos(operator="bulk",energies=np.linspace(-4.5,4.5,200)) 


    print(B)
    for (ei,di) in zip(e,d):
        fo.write(str(B)+" ")
        fo.write(str(ei)+" ")
        fo.write(str(di)+"\n")

fo.close()
