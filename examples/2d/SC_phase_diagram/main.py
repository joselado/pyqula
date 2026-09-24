# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import meanfield
import os
import glob
g = geometry.honeycomb_lattice()
h0 = g.get_hamiltonian() # create hamiltonian of the system
ds = []
mus = np.linspace(-0.,.5,10)
ts = np.linspace(0.0,0.15,10)
f = open("DELTA_VS_T_VS_MU.OUT","w")
for mu in mus:
  for t in ts:
    h = h0.copy()
    h.shift_fermi(mu)
    h.setup_nambu_spinor() # electron-hole degree of freedom
    for name in glob.glob("*.pkl"): os.remove(name)
    # attractive Hubbard at fixed chemical potential and temperature t
    scf = meanfield.hubbardscf(h,U=-2.0,nk=10,mix=0.9,mu=0.0,T=t,
            mf=meanfield.guess(h,"swave"))
    hscf = scf.hamiltonian
  #  rho = hscf.get_filling()
    d = np.abs(np.mean(hscf.extract("swave")))
    ds.append(d)
    f.write(str(mu)+"  ")
    f.write(str(t)+"  ")
    f.write(str(d)+"\n")
    f.flush()
f.close()

import matplotlib.pyplot as plt

dgrid = np.array(ds).reshape((len(mus),len(ts)))
plt.contourf(ts,mus,dgrid,levels=100,cmap="inferno")
plt.colorbar(label="SC order parameter")
plt.xlabel("T")
plt.ylabel("mu")
plt.show()







