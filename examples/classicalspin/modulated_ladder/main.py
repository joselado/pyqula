# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import classicalspin
from pyqula.specialhopping import NNG
import numpy as np
g = geometry.ladder() # generate the geometry
g = g.get_supercell(30)
g.dimensionality = 0
Jij = NNG(g,[-1.0]) # first neighbor
Jij_tb = Jij.apply(lambda r: abs(r[1])>0.2) # top and bottom
Jij_I = Jij.apply(lambda r: 0.1*(abs(r[1])<0.2)*np.sign(r[0])) # inter
Jij_T = Jij_tb + Jij_I # sum both contributions
sm = classicalspin.SpinModel(g) # generate a spin model
sm.add_heisenberg(Jij=Jij_T) # add heisenberg exchange
classicalspin.use_jax = True
sm.minimize_energy() # minimize Hamiltonian
h = g.get_hamiltonian() # get the Hamiltonian
h.add_magnetism(sm.magnetization*2.0) # add magnetization
h.write_magnetization(nrep=2)






