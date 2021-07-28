# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
import os
from pyqula import groundstate
from pyqula.selfconsistency import densitydensity
g = geometry.honeycomb_lattice(3)
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
nk = 3
filling = 0.5
g = 2.0 # interaction

V1s = np.linspace(0.,3.0,10)
V2s = np.linspace(0.,3.0,10)

#scf = scftypes.selfconsistency(h,nk=nk,filling=filling,g=g,mode="V")
def get_gap(V1,V2):
    os.system("rm -f MF.pkl")
    scf = densitydensity.Vinteraction(h,V1=V1,V2=V2,nk=nk,filling=filling)
    return scf.hamiltonian.get_gap() # get the Hamiltonian

f = open("MAP.OUT","w")
for V1 in V1s:
   for V2 in V2s:
       g = get_gap(V1,V2)
       f.write(str(V1)+"  ")
       f.write(str(V2)+"  ")
       f.write(str(g)+"\n")
       f.flush()
f.close()









