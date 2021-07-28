# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import neighbor
from pyqula import multiterminal
from pyqula import geometry
from pyqula import sculpt
from pyqula import skeleton
g = geometry.honeycomb_lattice()
imfile = "island.png"
g = sculpt.image2island(imfile,g,size=20,color="black")
g.write()







