# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import scftypes
import os
g = geometry.honeycomb_lattice()
h0 = g.get_hamiltonian() # create hamiltonian of the system
h0.remove_spin()
ds = []
mus = np.linspace(-0.,.5,10)
ts = np.linspace(0.0,0.15,10)
f = open("DELTA_VS_T_VS_MU.OUT","w")
for mu in mus:
  for t in ts:
    h = h0.copy()
    h.shift_fermi(mu)
    os.system("rm -rf *.pkl")
    scf = scftypes.attractive_hubbard(h,nk=10,mix=0.9,mf=None,g=-2.0,T=t)
    hscf = scf.hamiltonian
  #  rho = hscf.get_filling()
    d = -np.mean(hscf.extract("swave").real)
    f.write(str(mu)+"  ")
    f.write(str(t)+"  ")
    f.write(str(d)+"\n")
    f.flush()
f.close()







