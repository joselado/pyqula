# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=2.6,nedges=6,rot=np.pi/2.) # get an island
g = g.rotate(90)

from pyqula import ribbon
g = ribbon.island2ribbon(g)
g = g.get_supercell(2)

import matplotlib.pyplot as plt


plt.scatter(g.x,g.y) ; plt.axis("equal") ; plt.show()

