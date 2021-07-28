# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=6,nedges=3,rot=0.0) # get an island
# maximum distance to the origin
h = g.get_hamiltonian(has_spin=True) # get the Hamiltonian

from pyqula import ldos

f = ldos.ldosr_generator(h) # function to compute the DOS at position r
g.write()
fo = open("MAP.OUT","w")
for x in np.linspace(-9.,9.,80):
    (es,ds) = f([x,0.,0.])
    for (ei,di) in zip(es,ds):
        fo.write(str(x)+"  ")
        fo.write(str(ei)+"  ")
        fo.write(str(di)+"\n")
fo.close()







