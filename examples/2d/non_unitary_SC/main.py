# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import scftypes
from pyqula import meanfield
import os
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
ds = []
Us = np.linspace(0.0,10.0,10)
h.add_swave(0.0)
os.system("rm -rf *.pkl")
U = 2.0
# SC and CDW guess
mf = 10*(meanfield.guess(h,"swave") + meanfield.guess(h,"CDW")) 
scf = meanfield.Vinteraction(h,nk=10,mix=0.5,U=-U,mf=mf,filling=0.45,
        verbose=1,
        V1=0.25, # first neighbor repulsion
#        constrains=["no_charge"] # this forces the system to ignore CDW
#        constrains=["no_SC"] # this forces the system to ignore SC
        )
hscf = scf.hamiltonian
hscf.get_bands(operator="electron",nk=4000)
print("CDW order",np.abs(hscf.extract("CDW")))
print("SC order",np.abs(hscf.extract("swave")))
print("Gap",np.abs(hscf.get_gap()))







