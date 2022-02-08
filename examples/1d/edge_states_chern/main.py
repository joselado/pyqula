# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


# zigzag ribbon
from pyqula import geometry
g = geometry.triangular_lattice() # create geometry of a triagnular lattice
h = g.get_hamiltonian() # create hamiltonian of the system

h.add_rashba(1.0)
h.add_zeeman([0.,0.,1.0])
h.add_onsite(-6.0)
h.add_swave(0.4)

# insert here your Hamiltonian

#print(h.get_chern(nk=40))
#h.get_bands() ; exit()
from pyqula import multicell
hr = multicell.bulk2ribbon(h, # initial Hamiltonian
                           n=60 # width
                           )

hr.get_bands(operator="yposition")
hr.get_ldos(e=0.0,delta=1e-3,nk=100,nrep=30)
#hr.geometry.write(nrep=20) # see the geometry


