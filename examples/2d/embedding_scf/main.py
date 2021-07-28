# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import scftypes
from scipy.sparse import csc_matrix
g = geometry.honeycomb_lattice()
g = g.supercell(1)
h = g.get_hamiltonian() # create hamiltonian of the system
mf = scftypes.guess(h,mode="antiferro")
U = 3.0


from pyqula import embedding
mb = embedding.Embedding(h) # embedding object

from pyqula import meanfield
hubbard = meanfield.hubbardscf
#hubbard = scftypes.hubbardscf
filling = 0.5
mf = meanfield.guess(h,mode="antiferro")
scf = hubbard(mb,nk=20,U=U,mf=mf,mu=0.0,verbose=1,
        constrains=["no_charge"])
#h = scf.hamiltonian # get the Hamiltonian
#h.write_magnetization()
#print(scf.identify_symmetry_breaking())
#h.get_bands() # calculate band structure







