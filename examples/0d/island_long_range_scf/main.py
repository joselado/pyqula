# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=3,nedges=3,rot=0.0) # get an island
# maximum distance to the origin
h = g.get_hamiltonian(has_spin=True) # get the Hamiltonian
from pyqula import meanfield
g.write()

mf = meanfield.guess(h,mode="ferro")
scf = meanfield.Vinteraction(h,filling=0.5,U=3.0,V1=1.0,mf=mf,maxerror=1e-9)
scf.hamiltonian.write_hopping(spin_imbalance=True)
scf.hamiltonian.write_magnetization()
#scf.hamiltonian.get_bands()

mag = scf.hamiltonian.get_magnetization()

import matplotlib.pyplot as plt
plt.scatter(g.x,g.y,c=mag[:,2])
plt.colorbar(label="$m_z$")
plt.xlabel("x")
plt.ylabel("y")
plt.show()







