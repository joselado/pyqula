# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import meanfield
g = geometry.honeycomb_lattice()
g = g.supercell(5)
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_multicell()
h.shift_fermi(1.0)
h.setup_nambu_spinor() # electron-hole degree of freedom
from pyqula import parallel
parallel.cores = 4
# attractive Hubbard at the fixed chemical potential set above
scf = meanfield.hubbardscf(h,U=-1.0,nk=4,mix=0.9,mu=0.0,
        mf=meanfield.guess(h,"swave"))
h = scf.hamiltonian
h.write_swave()
# write_swave does not return arrays, it writes AMPLITUDE_SWAVE.OUT (x,y,z,amplitude)
m = np.genfromtxt("AMPLITUDE_SWAVE.OUT").T
x,y,amplitude = m[0],m[1],m[3]

import matplotlib.pyplot as plt

plt.scatter(x,y,c=amplitude,cmap="inferno")
plt.colorbar(label="|swave order parameter|")
plt.xlabel("x")
plt.ylabel("y")
plt.show()








