# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import meanfield
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_swave(0.001) # add swave
mf = meanfield.guess(h,mode="swave",fun=0.02)
scf = meanfield.hubbardscf(h,nk=5,mf=mf,U=-2.0,filling=0.7)
h = scf.hamiltonian
print("Delta",h.extract("swave"))
print("Onsite",h.extract("density"))
h.write_swave()
h.get_bands()
#scf.hamiltonian.get_bands(operator="electron")








