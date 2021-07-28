# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import scftypes
g = geometry.honeycomb_lattice()
g = g.supercell(5)
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_multicell()
h.shift_fermi(1.0)
#h.add_swave(0.0)
#mf = scftypes.guess(h,mode="swave",fun=0.02)
#scf = scftypes.selfconsistency(h,nkp=5,
#              mix=0.9,mf=mf,mode={"U":-2})
h.remove_spin()
from pyqula import parallel
parallel.cores = 4
scf = scftypes.attractive_hubbard(h,nk=4,mix=0.9,mf=None,g=-1.0)
h = scf.hamiltonian             
#h = scf.hamiltonian
h.write_swave()
#h.write_swave()
#scf.hamiltonian.get_bands(operator="electron")








