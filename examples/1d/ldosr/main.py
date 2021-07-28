# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_zigzag_ribbon(20) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system

from pyqula import ldos

f = ldos.ldosr_generator(h,nk=20,es=np.linspace(-.5,.5,200)) # function to compute the DOS at position r
g.write()
fo = open("MAP.OUT","w")
for y in np.linspace(-30,30,80):
    (es,ds) = f([0.,y,0.])
    for (ei,di) in zip(es,ds):
        fo.write(str(y)+"  ")
        fo.write(str(ei)+"  ")
        fo.write(str(di)+"\n")
fo.close()








