# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=4,nedges=7,rot=0.0) # get an island
cs = g.get_connections()
pot = [int(len(c)<3)-0.5 for c in cs] # atoms in the edge
np.savetxt("POTENTIAL.OUT",np.matrix([g.x,g.y,pot]).T)







