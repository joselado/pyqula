# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import groundstate
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
nk = 10
filling = 0.5

from pyqula import meanfield
#scf = scftypes.selfconsistency(h,nk=nk,filling=filling,g=g,mode="V")
h.add_zeeman([0.,0.,1.0])
mf = meanfield.guess(h,"dimerization")
mf = None
scf = meanfield.Vinteraction(h,V1=1.0,V2=1.0,nk=nk,filling=filling,
        mf=mf,mix=0.2)
h = scf.hamiltonian # get the Hamiltonian
h.get_bands(operator="sz") # calculate band structure
groundstate.hopping(h)
topology.write_berry(h)
h.write_onsite()







