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
h0 = g.get_hamiltonian() # create hamiltonian of the system
ds = []
Us = np.linspace(0.0,10.0,10)
f = open("DELTA_VS_T_VS_MU.OUT","w")
h0.add_swave(0.0)
for U in Us:
    h = h0.copy()
    os.system("rm -rf *.pkl")
    #scf = scftypes.attractive_hubbard(h,nk=10,mix=0.9,mf=None,g=-2.0,T=t)
    mf = 10*(meanfield.guess(h,"swave") + meanfield.guess(h,"CDW"))
    mf = meanfield.guess(h,"swave") #+ meanfield.guess(h,"CDW"))
    mf = meanfield.guess(h,"random") #+ meanfield.guess(h,"CDW"))
    scf = meanfield.hubbardscf(h,nk=10,mix=0.5,U=-U,mf=mf,filling=0.5,
            verbose=1,
            constrains=["no_charge"] # this forces the system to ignore CDW
            )
    hscf = scf.hamiltonian
  #  rho = hscf.get_filling()
    d = np.abs(np.mean(hscf.extract("swave")))
    f.write(str(U)+"  ")
    f.write(str(d)+"\n")
    print("Doing",U)
    f.flush()
f.close()







